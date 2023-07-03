from deeplake.core.vectorstore.deeplake_vectorstore import DeepLakeVectorStore
import openai
import os
from embeddings import *
import subprocess
import shutil
import faiss


class ActiveLoopVectorDB:
    def __init__(self, vector_store_path, embedding_model_name="distilbert"):
        self.vector_store = DeepLakeVectorStore(
            path = vector_store_path,
        )

        self.documents = []
        self.vectors = []
        embedding = TextEmbeddingFactory(embedding_model_name).create_embedding()
        self.embedding_func = embedding.embed

    def add_documents(self, documents, metadata, use_batch=True, batch_size=100):
        if use_batch:
            num_chunks = len(documents) // batch_size + (len(documents) % batch_size > 0)

            for i in range(5):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size

                batch_text = documents[start_idx:end_idx]
                batch_metadata = metadata[start_idx:end_idx]

                self.vector_store.add(text=batch_text, 
                    embedding_function=self.embedding_func, 
                    embedding_data=batch_text, 
                    metadata=batch_metadata
                )

    def query(self, prompt, topk=5):
        """
        Queries the database with a prompt and returns the top k results.

        Args:
            prompt (str): The query prompt.
            k (int, optional): The number of results to retrieve. Defaults to 5.

        Returns:
            str: The top k results from the query.
        """
        search_results = self.vector_store.search(
            embedding_data=prompt, embedding_function=self.embedding_func, k=topk
        )
        return search_results["text"][0]
    

class FAISSVectorDB:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)

    def add_documents(self, document):
        raise NotImplementedError("Add method is not implemented yet.")

    def query(self, prompt, topk=5):
        # Perform FAISS search here
        pass
    

def test_usage():
    def remove_content_from_path(path):
        if os.path.exists(path) and os.path.isdir(path):
            shutil.rmtree(path)
            print(f'{path} removed')
        return path

    # Example usage
    user_path = os.path.expanduser("~")
    vector_store_path = f"{user_path}/active_loop_vector"
    repo_path = f'{user_path}/repos/the-algorithm/'
    remove_content_from_path(vector_store_path)
    remove_content_from_path(repo_path)


    def git_clone(repo_url, destination_path):
        # Run the 'git clone' command
        result = subprocess.run(['git', 'clone', repo_url, destination_path], capture_output=True, text=True)
        
        # Check the return code to see if the command was successful
        if result.returncode == 0:
            print("Git clone successful")
        else:
            print("Git clone failed")
            print("Error message:", result.stderr)


    repository_url = "https://github.com/twitter/the-algorithm"
    destination = repo_path
    git_clone(repository_url, destination)


    CHUNK_SIZE = 1000

    chunked_text = []
    metadata = []
    for dirpath, dirnames, filenames in os.walk(repo_path):
        for file in filenames:
            try: 
                full_path = os.path.join(dirpath,file)
                with open(full_path, 'r') as f:
                    text = f.read()
                    new_chunkned_text = [text[i:i+1000] for i in range(0,len(text), CHUNK_SIZE)]
                    chunked_text += new_chunkned_text
                    metadata += [{'filepath': full_path} for i in range(len(new_chunkned_text))]
            except Exception as e: 
                print(e)
                pass




