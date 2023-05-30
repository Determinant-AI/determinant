import os
from typing import List

import faiss
import numpy as np
import requests
import torch
from atlassian import Confluence
from fastapi import FastAPI, Request
from ray import serve
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from slack_bolt.async_app import AsyncApp
from starlette.requests import Request
from transformers import (DPRContextEncoder, DPRContextEncoderTokenizer,
                          DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
                          pipeline, set_seed)
# from src.ingestor.confluence_ingestor import ConfluenceIngester

from bs4 import BeautifulSoup


class ConfluenceIngester(object):
    def __init__(self, data_path='advendio_pages'):
        # data_path can be local or remote
        self.data_path = data_path
        self.documents = []

    def ingest_confluence_data(self, seed_url='https://advendio.atlassian.net', page_name="SO"):
        # Set up Confluence API connection
        confluence = Confluence(url=seed_url)
        pages = confluence.get_all_pages_from_space(page_name)
        # Create a directory to store the downloaded pages
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        # Download each page
        for page in pages:
            page_id = page['id']
            page_title = page['title']
            page_filename = page_title.replace(' ', '_') + '.html'
            page_content = confluence.get_page_by_id(page_id, expand='body.storage')[
                'body']['storage']['value']
            try:
                with open(f'{self.data_path}/{page_filename}', 'w') as f:
                    f.write(page_content)
            except:
                pass

    def format_documents(self):
        for filename in os.listdir('advendio_pages'):
            f = os.path.join('advendio_pages', filename)
            with open(f, 'r', encoding='utf-8') as file:
                html_content = file.read()
                soup = BeautifulSoup(html_content, "lxml")
                text_content = soup.get_text(separator=" ", strip=True)
                self.documents.append(text_content)


# make it integration with Slack Ingestor
# @serve.deployment()
class DocumentLoader():
    def __init__(self, ingestor: ConfluenceIngester, 
                 question_encoder_model: str = "facebook/dpr-question_encoder-single-nq-base", 
                 context_encoder_model: str = "facebook/dpr-ctx_encoder-single-nq-base"):
        self.question_encoder = DPRQuestionEncoder.from_pretrained(
            question_encoder_model)
        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            question_encoder_model)

        self.context_encoder = DPRContextEncoder.from_pretrained(
            context_encoder_model)
        self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            context_encoder_model)
        self.token_limit = 512

        ingestor.format_documents()
        self.documents = ingestor.documents
        self.index_documents()

    def index_documents(self) -> int:
        # Encode the documents
        encoded_documents = self.context_tokenizer(
            self.documents, return_tensors="pt", padding=True, truncation=True, max_length=self.token_limit)
        document_embeddings = self.context_encoder(
            **encoded_documents).pooler_output

        document_embeddings = document_embeddings.detach().numpy()
        document_embeddings = np.ascontiguousarray(document_embeddings)
        # Create Faiss Index
        vector_dimension = document_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(vector_dimension)
        faiss.normalize_L2(document_embeddings)
        self.index.add(document_embeddings)
        return self.index.ntotal

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
        return self.documents[idx[0][0]]


# @serve.deployment(route_prefix="/")
class ConversationBot:
    def __init__(self, db: DocumentLoader, model: str = "databricks/dolly-v2-3b"):
        self.model = model
        self.db = db

    # async def prompt(self, input: str):
    #     context = await self.db.query_documents.remote(input)
    #     result = "{input} \n context: {}\n".format(context)
    #     return result

    def prompt(self, input: str):
        context = self.db.query_documents(input)
        result = f"{input} \n context: {context}\n"
        return result

    # async def generate_text(self, input_text: str) -> str:
    #     generator = pipeline(model=self.model,
    #                          torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    #     return generator(await self.prompt(input_text))[0]["generated_text"]

    def generate_text(self, input_text: str) -> str:
        generator = pipeline(model=self.model,
                             torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

        # generator = pipeline('text-generation', model='gpt2',
        #                      max_length=800, num_return_sequences=5)
        output_text = generator(self.prompt(input_text))[0]["generated_text"]
        return output_text

    # async def __call__(self, http_request: Request) -> str:
    #     input_text: str = await http_request.json()
    #     return self.generate_text(input_text)


# serve.run(ConversationBot.bind(DocumentLoader.bind()))

english_text = "what is an ads event?"

dolly = ConversationBot(DocumentLoader(ConfluenceIngester()))
print(dolly.generate_text(english_text))

# 3: Query the deployment and print the result.
# response = requests.post("http://127.0.0.1:8000/", json=english_text)
# output_text = response.text


# fastapi_app = FastAPI()

# @serve.deployment(route_prefix="/")
# @serve.ingress(fastapi_app)
# class SlackAgent:
#     def __init__(self, conversation_bot: ConversationBot):
#         self.conversation_bot = conversation_bot
#         self.slack_app = AsyncApp(
#             token=os.environ["SLACK_BOT_TOKEN"],
#             signing_secret=os.environ["SLACK_SIGNING_SECRET"],
#             request_verification_enabled=False,
#         )
#         self.app_handler = AsyncSlackRequestHandler(self.slack_app)
#         self.slack_app.event("app_mention")(self.handle_app_mention)
#         self.slack_app.event("file_shared")(self.handle_file_shared_events)
#         self.slack_app.event("message")(self.handle_message_events)

#     async def handle_app_mention(self, event, say):
#         human_text = event["text"]  # .replace("<@U04MGTBFC7J>", "")
#         print("event:{}".format(event))
#         if 'files' in event:
#             if 'summarize' in event['text'].lower():
#                 response_ref = await self.caption_bot.caption_image.remote(event['files'][0]['url_private'])
#         else:
#             response_ref = await self.conversation_bot.generate_next.remote(human_text)
#         await say(await response_ref)

#     async def handle_file_shared_events(self, event):
#         print(event)

#     async def handle_message_events(self, event):
#         print(event)

#     @fastapi_app.post("/slack/events")
#     async def events_endpoint(self, req: Request) -> None:
#         resp = await self.app_handler.handle(req)
#         return resp

#     @fastapi_app.post("/slack/interactions")
#     async def interactions_endpoint(self, req: Request) -> None:
#         return await self.app_handler.handle(req)
