import torch
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import VisionEncoderDecoderModel, ViTImageProcessor
from transformers import AutoModelForSeq2SeqLM

from fastapi import FastAPI, Request
from ray import serve
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from slack_bolt.adapter.starlette.handler import SlackRequestHandler
import requests
from slack_sdk.signature import SignatureVerifier
import ray
import asyncio

import logging
# Configure the logger.

from atlassian import Confluence
import os

# Set up Confluence API connection
confluence = Confluence(
url='https://advendio.atlassian.net',
)

space_key = "SO"
pages = confluence.get_all_pages_from_space(space_key)

# Create a directory to store the downloaded pages
if not os.path.exists('advendio_pages'):
    os.makedirs('advendio_pages')
# Download each page
for page in pages:
    page_id = page['id']
    page_title = page['title']
    page_filename = page_title.replace(' ', '_') + '.html'
    page_content = confluence.get_page_by_id(page_id, expand='body.storage')['body']['storage']['value']
    try:
        with open('advendio_pages/' + page_filename, 'w') as f:
            f.write(page_content)
    except:
        pass
    print('Downloaded:', page_filename)

import numpy as np

import torch
from typing import List
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from bs4 import BeautifulSoup
import os
import faiss

@serve.deployment(ray_actor_options={"num_gpus": 0.5})
class DocumentVectorDB:
    def __init__(self, 
                 question_encoder_model: str = "facebook/dpr-question_encoder-single-nq-base",
                 context_encoder: str = "facebook/dpr-ctx_encoder-single-nq-base"
                 ):
        self.token_limit = 512

        self.documents = self.format_documents()
        self.question_encoder = DPRQuestionEncoder.from_pretrained(
            question_encoder_model)
        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            question_encoder_model)
        self.context_encoder = DPRContextEncoder.from_pretrained(
            context_encoder)
        self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            context_encoder)
        count = self.index_documents(self.documents)
        print("document count:{}".format(count))
      
    def index_documents(self, documents: List[str]) -> int:
        # Encode the documents
        encoded_documents = self.context_tokenizer(
            self.documents, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.token_limit)
        document_embeddings = self.context_encoder(**encoded_documents).pooler_output

        document_embeddings = document_embeddings.detach().numpy()
        document_embeddings=np.ascontiguousarray(document_embeddings)
        # Create Faiss Index
        vector_dimension = document_embeddings.shape[1]
        print("vector dimension:{}".format(vector_dimension))
        self.index = faiss.IndexFlatL2(vector_dimension)
        faiss.normalize_L2(document_embeddings)
        self.index.add(document_embeddings)
        return self.index.ntotal

    def insert_documents(self) -> List[str]:
        """
        store some whre
        """
        pass

    def encode_questions(self, query: str) -> List[torch.Tensor]:
        encoded_query = self.question_tokenizer(query, return_tensors="pt")
        query_embedding = self.question_encoder(
            **encoded_query).pooler_output.detach().numpy()
        query_embedding = np.ascontiguousarray(query_embedding)
        return query_embedding

    def query_documents(self, query: str) -> str:
        # Encode the query
        query_embedding = self.encode_questions(query)
        _, idx = self.index.search(query_embedding, 4)
        doc = self.documents[idx[0][0]]
        print("query result: index={}, doc={}".format(idx[0][0], doc))
        return doc

    def format_documents(self):
        documents = []
        for filename in os.listdir('advendio_pages'):
            f = os.path.join('advendio_pages', filename)
            with open(f, 'r', encoding='utf-8') as file:
                html_content = file.read()
                soup = BeautifulSoup(html_content, "lxml")

                text_content = soup.get_text(separator=" ", strip=True)
                documents.append(text_content)
        return documents

@serve.deployment(ray_actor_options={"num_gpus": 0.5})
class RAGConversationBot:
    def __init__(self, 
                 db: DocumentVectorDB, 
                 model: str = "databricks/dolly-v2-3b"):
        self.model = model
        self.db = db

    async def prompt(self, input: str) ->str:
        context_ref = await self.db.query_documents.remote(input)
        context = await context_ref
        assert isinstance(context, str)
        return "{} \n context: {}\n output:".format(input, context)
         
    # Change this method to be an async function
    async def generate_text(self, thread_ts, input_text: str) -> str:
        generator = pipeline(model=self.model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
        prompt_text = await self.prompt(input_text)
        assert isinstance(prompt_text, str)
        return generator(prompt_text)[0]

    async def __call__(self, http_request: Request) -> str:
        input_text: str = await http_request.json()
        return await self.generate_text(input_text)

@serve.deployment()
class ImageCaptioningBot:
    def __init__(self):
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        self.feature_extractor = ViTImageProcessor.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # self.image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

    def caption_image(self, image_url):
        max_length = 16
        num_beams = 4
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
        token = os.environ["SLACK_BOT_TOKEN"]
        response = requests.get(
            image_url, headers={"Authorization": "Bearer %s" % token}
        )
        image = Image.open(BytesIO(response.content))
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        images = []
        images.append(image)

        pixel_values = self.feature_extractor(
            images=images, return_tensors="pt"
        ).pixel_values
        pixel_values = pixel_values.to(self.device)

        output_ids = self.model.generate(pixel_values, **gen_kwargs)

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds[0]

fastapi_app = FastAPI()
from fastapi import Body
from pydantic import BaseModel

class LLMQuery(BaseModel):
    prompt: str
import os

@serve.deployment(route_prefix="/")
@serve.ingress(fastapi_app)
class SlackAgent:
    def __init__(self, conversation_bot, image_captioning_bot):  # , summarization_bot):
        self.conversation_bot = conversation_bot
        self.caption_bot = image_captioning_bot
        # self.summarization_bot = summarization_bot
        self.slack_app = AsyncApp(
            token=os.environ["SLACK_BOT_TOKEN"],
            signing_secret=os.environ["SLACK_SIGNING_SECRET"],
            request_verification_enabled=False,
        )
        self.app_handler = AsyncSlackRequestHandler(self.slack_app)
        self.slack_app.event("app_mention")(self.handle_app_mention)
        self.slack_app.event("file_shared")(self.handle_file_shared_events)
        self.slack_app.event("message")(self.handle_message_events)

    async def handle_app_mention(self, event, say):
        human_text = event["text"]  # .replace("<@U04MGTBFC7J>", "")
        thread_ts = event.get("thread_ts", None) or event["ts"]

        print("event:{}".format(event))

        if "files" in event:
            if "summarize" in event["text"].lower():
                response_ref = await self.caption_bot.caption_image.remote(
                    event["files"][0]["url_private"]
                )
        else:
            response_ref = await self.conversation_bot.generate_text.remote(
                thread_ts, human_text
            )

        await say(await response_ref, thread_ts=thread_ts)

    async def handle_file_shared_events(self, event, logger):
        logger.info(event)

    async def handle_message_events(self, event, say):
        if '<@U04UTNRPEM9>' in event['text']:
            # will get handled in app_mention
            pass
        else:
            print("message event:{}".format(event))
            await self.handle_app_mention(event, say)
            logger.info(event)

    @fastapi_app.post("/slack/events")
    async def events_endpoint(self, req: Request) -> None:
        resp = await self.app_handler.handle(req)
        return resp

    @fastapi_app.post("/slack/interactions")
    async def interactions_endpoint(self, req: Request) -> None:
        return await self.app_handler.handle(req)

    @fastapi_app.get("/hello")
    def say_hello(self, name: str):
        return f"Hello {name}!"

    @fastapi_app.post("/test")
    async def handle_test(self, user_input: LLMQuery):
        result = await self.conversation_bot.generate_text.remote(user_input.prompt)
        return {'message': result}

# model deployment
rag_bot = RAGConversationBot.bind(DocumentVectorDB.bind())
image_captioning_bot = ImageCaptioningBot.bind()

# ingress deployment
slack_agent_deployment = SlackAgent.bind(rag_bot, image_captioning_bot)

#serve run async_rag_app:slack_agent_deployment -p 3000
