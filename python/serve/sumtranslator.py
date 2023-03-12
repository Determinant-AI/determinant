from transformers import pipeline
import requests
from ray import serve
from starlette.requests import Request
from summarizer import Summarizer
from translator import Translator


@serve.deployment
class Summarizer2(Summarizer):
    def __init__(self, translator):
        # Load model
        self.model = pipeline("summarization", model="t5-small")
        self.translator = translator

    async def __call__(self, http_request: Request) -> str:
        english_text: str = await http_request.json()
        summary = self.summarize(english_text)

        translation_ref = await self.translator.translate.remote(summary)
        translation = await translation_ref

        return translation


# bind Translator deployment to arguments that Ray Serve can pass to its constructor
sumtranslator = Summarizer2.bind(Translator.bind())
serve.run(sumtranslator)


english_text = """
There are two main methods to elicit chain-of-thought reasoning: few-shot prompting and zero-shot prompting. The initial proposition of CoT prompting demonstrated few-shot prompting, wherein at least one example of a question paired with proper human-written CoT reasoning is prepended to the prompt.[2] It has been discovered since, however, that it is also possible to elicit similar reasoning and performance gain with zero-shot prompting, which can be as simple as appending to the prompt the words "Let's think step-by-step".[15] This allows for better scaling as one no longer needs to prompt engineer specific CoT prompts for each task to get the corresponding boost in performance.[16]
"""

url = "http://127.0.0.1:8000/"
response = requests.post(url, json=english_text)
summarized_text = response.text


print(summarized_text)
