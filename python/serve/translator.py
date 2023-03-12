from starlette.requests import Request
import argparse
import ray
from ray import serve

from transformers import pipeline


@serve.deployment
class Translator:
    def __init__(self):
        # Load model
        self.model = pipeline("translation_en_to_fr", model="t5-small")

    def translate(self, text: str) -> str:
        # Run inference
        model_output = self.model(text)

        # Post-process output to return only the translation text
        translation = model_output[0]["translation_text"]

        return translation

    # __call__ processes the incoming HTTP request by reading its JSON data
    # and forwarding it to the translate method
    async def __call__(self, http_request: Request) -> str:
        english_text: str = await http_request.json()
        return self.translate(english_text)


# ray job submit -- python3 translator.py


# def main():
# parser = argparse.ArgumentParser(
#     description='translate english to french')
# parser.add_argument('english', type=str,
#                     help="input string to be translated to French")
# args = parser.parse_args()

# translator = Translator()

# french = translator.translate(args.english)

# print(f"The translated French is: {french}")


# if __name__ == '__main__':
#     main()
