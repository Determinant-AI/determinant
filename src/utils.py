from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import os
import openai

class TextEmbeddingFactory:
    def __init__(self, model_name):
        self.model_name = model_name

    def create_embedding(self):
        if self.model_name == "text-embedding-ada-002":
            return TextEmbeddingAda002()
        elif self.model_name == "distilbert":
            return TextEmbeddingDistilBERT()
        elif self.model_name == "dpr":
            return TextEmbeddingDPR()
        else:
            raise ValueError("Invalid model name")

class TextEmbedding:
    def embed(self, text):
        raise NotImplementedError("Subclasses must implement embed()")

class TextEmbeddingAda002(TextEmbedding):
    def embed(self, text):
        if isinstance(text, str):
           text = [text]

        text = [t.replace("\n", " ") for t in text]
        return [data['embedding']for data in openai.Embedding.create(input = text, model=model)['data']]

class TextEmbeddingDistilBERT(TextEmbedding):
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    def embed(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        return embeddings

class TextEmbeddingDPR(TextEmbedding):
    def __init__(self):
        self.tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
        self.model = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

    def embed(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        outputs = self.model(**inputs)
        embeddings = outputs.pooler_output.squeeze().tolist()
        return embeddings

      
def test_usage():
    # Usage example
    model_name = "dpr"
    factory = TextEmbeddingFactory(model_name)
    embedding = factory.create_embedding()
    result = embedding.embed("Example text")
    print(result)


if __name__ == "__main__":
    test_usage()
