from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DistilBertTokenizerFast, DistilBertModel
from sentence_transformers import SentenceTransformer

from vertexai.preview.language_models import TextEmbeddingModel

import os
import openai
from typing import Union, List

class TextEmbeddingFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def create_embedding(self):
        if self.model_name == "text-embedding-ada-002":
            return TextEmbeddingAda002()
        elif self.model_name == "distilbert":
            return TextEmbeddingDistilBERT()
        elif self.model_name == "dpr":
            return TextEmbeddingDPR()
        elif self.model_name == "sentence-transformer":
            return TextEmbeddingSentenceTransformer()
        else:
            raise ValueError("Invalid model name")

class TextEmbedding:
    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        raise NotImplementedError("Subclasses must implement embed()")

class TextEmbeddingAda002(TextEmbedding):
    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        if isinstance(text, str):
            text = [text]

        text = [t.replace("\n", " ") for t in text]
        return [data['embedding'] for data in openai.Embedding.create(input=text, model="text-embedding-ada-002")['data']]

class TextEmbeddingDistilBERT(TextEmbedding):
    def __init__(self):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        if isinstance(text, str):
            text = [text]

        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        return embeddings

class TextEmbeddingDPR(TextEmbedding):
    def __init__(self):
        self.tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
        self.model = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        if isinstance(text, str):
            text = [text]

        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        outputs = self.model(**inputs)
        embeddings = outputs.pooler_output.squeeze().tolist()
        return embeddings

class TextEmbeddingSentenceTransformer(TextEmbedding):
    def __init__(self):
        self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')

    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        if isinstance(text, str):
            text = [text]

        embeddings = self.model.encode(text)
        return embeddings.tolist()


def test_usage():
    # Usage example
    model_name = "distilbert"
    factory = TextEmbeddingFactory(model_name)
    embedding = factory.create_embedding()
    result = embedding.embed("Example text")
    print(result)


if __name__ == "__main__":
    test_usage()
