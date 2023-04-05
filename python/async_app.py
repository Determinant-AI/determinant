import logging
import os
from typing import Dict, List
from PIL import Image
import requests
from io import BytesIO

logging.basicConfig(level=logging.INFO)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from fastapi import FastAPI, Request
from ray import serve
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from slack_bolt.adapter.starlette.handler import SlackRequestHandler
import requests
from slack_sdk.signature import SignatureVerifier

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import asyncio

from transformers import pipeline
import ray

from langchain.chains.conversation.memory import ConversationBufferMemory
from collections import deque


from slack_bolt.async_app import AsyncApp
from typing import Dict, Any, Optional
from memory.redis_manager import RedisManager

fastapi_app = FastAPI()


@serve.deployment(route_prefix="/")
@serve.ingress(fastapi_app)
class SlackAgent:
    def __init__(self, conversation_bot, image_captioning_bot):  # , summarization_bot):
        self.conversation_bot = conversation_bot
        self.caption_bot = image_captioning_bot
        # self.summarization_bot = summarization_bot
        self.slack_app = AsyncApp(
            token=os.environ["SLACK_BOT_TOKEN"],
            signing_secret=os.environ["SLACK_SIGNING_SECRET"],
            request_verification_enabled=True,
        )
        self.app_handler = AsyncSlackRequestHandler(self.slack_app)
        self.slack_app.event("app_mention")(self.handle_app_mention)
        self.slack_app.event("file_shared")(self.handle_file_shared_events)
        self.slack_app.event("message")(self.handle_message_events)

    async def handle_app_mention(self, event, say):
        human_text = event["text"]  # .replace("<@U04MGTBFC7J>", "")
        thread_ts = event.get("thread_ts", None) or event["ts"]

        print("event:{}".format(event))

        if "files" in event:
            if "summarize" in event["text"].lower():
                response_ref = await self.caption_bot.caption_image.remote(
                    event["files"][0]["url_private"]
                )
        else:
            response_ref = await self.conversation_bot.generate_next.remote(
                thread_ts, human_text
            )

        await say(await response_ref, thread_ts=thread_ts)

    async def handle_file_shared_events(self, event, logger):
        logger.info(event)

    async def handle_message_events(self, event, say):
        if '<@U04UTNRPEM9>' in event['text']:
            # will get handled in app_mention
            pass
        else:
            print("message event:{}".format(event))
            await self.handle_app_mention(event, say)
            logger.info(event)

    @fastapi_app.post("/slack/events")
    async def events_endpoint(self, req: Request) -> None:
        resp = await self.app_handler.handle(req)
        return resp

    @fastapi_app.post("/slack/interactions")
    async def interactions_endpoint(self, req: Request) -> None:
        return await self.app_handler.handle(req)


@serve.deployment(num_replicas=1)
class ConversationBot:
    def __init__(self, memory_buffer: RedisManager):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/GODEL-v1_1-large-seq2seq"
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "microsoft/GODEL-v1_1-large-seq2seq"
        )
        self.memory = memory_buffer
        self.instruction = (
            f"Instruction: given a dialog context, you need to response empathically."
        )
        # Leave the knowldge empty
        self.knowledge = ""

    def generate_next(self, key: str, human_text: str) -> str:
        if self.knowledge != "":
            self.knowledge = "[KNOWLEDGE] " + self.knowledge
        self.memory.insert_list_strings(key, [human_text])
        dialog = self.memory.get_list_strings(key)
        logger.debug("dialog:{}".format(dialog))
        dialog = " EOS ".join(dialog)
        query = f"{self.instruction} [CONTEXT] {dialog} {self.knowledge}"
        input_ids = self.tokenizer(f"{query}", return_tensors="pt").input_ids
        outputs = self.model.generate(
            input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True
        )
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.memory.insert_list_strings(key, output)
        return output


@serve.deployment()
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds[0]


# model deployment
memory_buffer = RedisManager()
conversation_bot = ConversationBot.bind(memory_buffer)
image_captioning_bot = ImageCaptioningBot.bind()

# ingress deployment
slack_agent_deployment = SlackAgent.bind(conversation_bot, image_captioning_bot)

# serve run async_app:fast_api_deployment -p 3000
