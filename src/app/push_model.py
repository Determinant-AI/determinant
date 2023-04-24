from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, pipeline, TextClassificationPipeline
from huggingface_hub import Repository


def push_model_to_hub(model_name, tokenizer_name, model_dir, repo_name, commit_message):
    """
    Uploads a model and tokenizer to the Hugging Face Model Hub and creates a new repository for it.

    Args:
        model_name (str): Name of the model.
        tokenizer_name (str): Name of the tokenizer.
        model_dir (str): Path to the directory containing the model and tokenizer files.
        repo_name (str): Name of the repository to be created on the Hugging Face Model Hub.
        commit_message (str): Commit message to be used when pushing the model to the repository.
    """
    # Load the tokenizer and model from the specified directory
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # Create a text classification pipeline
    nlp = TextClassificationPipeline(model=model, tokenizer=tokenizer)

    # Save the model and tokenizer to the specified directory
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Create a new repository on the Model Hub
    repo = Repository.create(repo_name=repo_name)

    # Push the model and tokenizer to the repository
    repo.push_to_hub(model_path=model_dir, message=commit_message)
