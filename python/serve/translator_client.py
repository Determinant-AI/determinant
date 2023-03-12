import requests
from ray import serve
from translator import Translator

url = "http://127.0.0.1:8000/"

# bind Translator deployment to arguments that Ray Serve can pass to its constructor
translator = Translator.bind()
serve.run(translator)


english_text = "Hello world!"
response = requests.post(url, json=english_text)
french_text = response.text


print(french_text)

# python model_client.py
