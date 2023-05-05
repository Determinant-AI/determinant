import os
import random
from io import BytesIO
from typing import List

import faiss
import numpy as np
import requests
import spacy
import torch
from atlassian import Confluence
from bs4 import BeautifulSoup
from fastapi import Body, FastAPI, Request
from PIL import Image
from pydantic import BaseModel
from ray import serve
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DPRContextEncoder, DPRContextEncoderTokenizer,
                          DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
                          StoppingCriteria, StoppingCriteriaList,
                          VisionEncoderDecoderModel, ViTImageProcessor,
                          pipeline)

from logger import create_logger

# Configure the logger.
logger = create_logger(__name__)

# Load a pre-trained SpaCy model for NER
nlp = spacy.load('en_core_web_sm')


def download_confluence(url_str: str = 'https://advendio.atlassian.net', space_key="SO"):
    # Set up Confluence API connection
    confluence = Confluence(
        url=url_str,
    )
    pages = confluence.get_all_pages_from_space(space_key)

    # Create a directory to store the downloaded pages
    if os.path.exists('advendio_pages'):
        logger.info(
            "Detected existing advendio_pages directory. Skipping download.")
        return

    os.makedirs('advendio_pages')

    # Download each page
    for page in pages:
        page_id = page['id']
        page_title = page['title']
        page_filename = page_title.replace(' ', '_') + '.html'
        page_content = confluence.get_page_by_id(page_id, expand='body.storage')[
            'body']['storage']['value']
        try:
            with open(f'advendio_pages/{page_filename}', 'w') as f:
                f.write(page_content)
        except Exception as e:
            logger.error(
                f'Failed to download: {page_filename}, eerror: {str(e)}')
        logger.info(f'Downloaded: {page_filename}')


download_confluence()


@serve.deployment(num_replicas=1)  # ray_actor_options={"num_gpus": 0.5})
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
        document_embeddings = self.context_encoder(
            **encoded_documents).pooler_output

        document_embeddings = document_embeddings.detach().numpy()
        document_embeddings = np.ascontiguousarray(document_embeddings)
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


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


@serve.deployment(num_replicas=1)  # ray_actor_options={"num_gpus": 0.5})
class RAGConversationBot(object):
    def __init__(self,
                 db: DocumentVectorDB,
                 model_name: str = "StabilityAI/stablelm-tuned-alpha-7b"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.db = db
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name).to(self.device)

    def to_retrieve(self, s: str) -> bool:
        # Perform NER on user input
        doc = nlp(s)
        entities = [ent.label_ for ent in doc.ents]

        # Determine if retrieval is needed
        if 'ORG' in entities:
            return True
        else:
            return False

    async def prompt(self, input: str) -> str:
        if self.to_retrieve(input):
            context_ref = await self.db.query_documents.remote(input)
            context = await context_ref
            assert isinstance(context, str)
            user_input = f"Answer this question with context:{input} \n Context: {context}"
        else:
            context = ""
            user_input = f"Answer this question:{input}"

        system_prompt = """
        <|SYSTEM|># StableLM Tuned (Alpha version)
        - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
        - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
        - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
        - StableLM will refuse to participate in anything that could harm a human.
        """
        result_prompt = f"{system_prompt}<|USER|>{user_input}#end<|ASSISTANT|>"
        return result_prompt

    def remove_prefix_until_tag(self, s: str, tag="#end"):
        # Find the index of the tag in the string
        tag_index = s.find(tag)
        if tag_index != -1:
            # Calculate the end index of the tag
            end_index = tag_index + len(tag)
            # Slice the string starting from the end index of the tag
            return s[end_index:]
        return s  # Return the original string if the tag is not found

    async def generate_text(self, thread_ts, input_text: str):
        prompt_text = await self.prompt(input_text)
        assert isinstance(prompt_text, str)
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        tokens = self.model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.8,
            do_sample=True,
            stopping_criteria=StoppingCriteriaList([StopOnTokens()])
        )
        response = self.tokenizer.decode(tokens[0], skip_special_tokens=True)
        response = self.remove_prefix_until_tag(response)
        return response

    async def __call__(self, http_request: Request) -> str:
        input_text: str = await http_request.json()
        return await self.generate_text(input_text)


@serve.deployment(num_replicas=1)
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

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
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

        preds = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds[0]


class LLMQuery(BaseModel):
    prompt: str


app = AsyncApp(
    token=os.environ["SLACK_BOT_TOKEN"],
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
    request_verification_enabled=True,
)


@serve.deployment(route_prefix="/", num_replicas=1)
class SlackAgent:
    # , summarization_bot):
    async def __init__(self, conversation_bot, image_captioning_bot):
        self.conversation_bot = conversation_bot
        self.caption_bot = image_captioning_bot
        # self.summarization_bot = summarization_bot
        self.register()
        self.app_handler = AsyncSocketModeHandler(
            app, os.environ["SLACK_APP_TOKEN"])
        await self.app_handler.start_async()

    def register(self):
        @app.event("reaction_added")
        async def handle_reaction_added_events(event, say):
            # TODO: log events with feedback label
            logger.info(f"event: {event}")
            say(f"reaction added: {event['reaction']}")

        @app.event("app_mention")
        async def handle_app_mention(event, say):
            human_text = ''
            if "text" in event:
                human_text = event["text"]  # .replace("<@U04MGTBFC7J>", "")
            thread_ts = event.get("thread_ts", None) or event["ts"]

            # TODO: log events with task label
            logger.info(f"event: {event}")

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

        @app.event("file_shared")
        async def handle_file_shared_events(event):
            logger.info(event)

        @app.event("user_change")
        async def handle_user_change_events(body):
            logger.info(body)

        @app.event("file_public")
        async def handle_file_public_events(body):
            logger.info(body)

        @app.event("message")
        async def handle_message_events(event, say):
            if '<@U04UTNRPEM9>' in event.get('text', ''):
                # will get handled in app_mention
                pass
            elif random.random() < 0.5:
                pass
            else:
                # TODO: write a event handler to produce events.
                logger.info(f"message event:{event}")
                await handle_app_mention(event, say)


# model deployment
rag_bot = RAGConversationBot.bind(DocumentVectorDB.bind())
image_captioning_bot = ImageCaptioningBot.bind()

# ingress deployment
slack_agent_deployment = SlackAgent.bind(rag_bot, image_captioning_bot)
# serve.run(slack_agent_deployment)

# response = requests.post("http://127.0.0.1:3000/test", json={"prompt": "what is an ad event in advendio"})
# # Print the response status code and JSON response body
# print(response.status_code)
# print(response.json())

# serve run async_rag_app:slack_agent_deployment
