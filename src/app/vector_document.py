
import faiss
import numpy as np
import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from ray import serve
from typing import Dict, List
from starlette.requests import Request
from bs4 import BeautifulSoup
import os


@serve.deployment()
class DocumentVectorDB(object):
    def __init__(self, question_encoder_model: str = "facebook/dpr-question_encoder-single-nq-base"):
        self.token_limit = 512

        self.documents = self.format_documents()
        self.question_encoder = DPRQuestionEncoder.from_pretrained(
            question_encoder_model)
        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            question_encoder_model)
        self.context_encoder = DPRContextEncoder.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base")
        self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base")
        count = self.index_documents(self.documents)
        print(count)

    # def __call__(self, request: Request) -> Dict:
    #     print("hellp")
    #     req_str = "what is ads campaign?"
    #     msg = self.documents[self.query_documents(req_str)]
    #     return {"result": msg}

    def index_documents(self, documents: List[str]) -> int:
        # Encode the documents
        encoded_documents = self.context_tokenizer(
            self.documents, return_tensors="pt", padding=True, truncation=True, max_length=self.token_limit)
        document_embeddings = self.context_encoder(
            **encoded_documents).pooler_output

        document_embeddings = document_embeddings.detach().numpy()
        document_embeddings = np.ascontiguousarray(document_embeddings)
        # Create Faiss Index
        vector_dimension = document_embeddings.shape[1]
        # print(vector_dimension)
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
