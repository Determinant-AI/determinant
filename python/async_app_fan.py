import logging
import os
from io import BytesIO

import requests
import torch
from fastapi import FastAPI, Request
from PIL import Image
from ray import serve
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from slack_bolt.async_app import AsyncApp
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          VisionEncoderDecoderModel, ViTImageProcessor)

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


fastapi_app = FastAPI()


@serve.deployment(num_replicas=2)
class ConversationBot:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/DialoGPT-medium")
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/DialoGPT-medium")

    def generate_next(self, human_text):
        new_user_input_ids = self.tokenizer.encode(
            human_text, self.tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([new_user_input_ids], dim=-1)
        model_output = self.model.generate(
            bot_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
        response_text = self.tokenizer.decode(
            model_output[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response_text


@serve.deployment()
class ImageCaptioningBot:
    def __init__(self):
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTImageProcessor.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # self.image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

    def caption_image(self, image_url):
        max_length = 16
        num_beams = 4
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
        token = os.environ["SLACK_BOT_TOKEN"]
        response = requests.get(
            image_url, headers={'Authorization': 'Bearer %s' % token})
        image = Image.open(BytesIO(response.content))
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        images = []
        images.append(image)

        pixel_values = self.feature_extractor(
            images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        output_ids = self.model.generate(pixel_values, **gen_kwargs)

        preds = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds[0]


@serve.deployment(route_prefix="/")
@serve.ingress(fastapi_app)
class FastAPIDeployment:
    def __init__(self, conversation_bot: ConversationBot = None, image_captioning_bot: ImageCaptioningBot = None):
        self.conversation_bot = conversation_bot
        self.caption_bot = image_captioning_bot
        # self.summarization_bot = summarization_bot
        self.slack_app = AsyncApp(
            token=os.environ["SLACK_BOT_TOKEN"],
            signing_secret=os.environ["SLACK_SIGNING_SECRET"],
            request_verification_enabled=False,
        )
        self.app_handler = AsyncSlackRequestHandler(self.slack_app)
        self.slack_app.event("app_mention")(self.handle_app_mention)
        self.slack_app.event("file_shared")(self.handle_file_shared_events)
        self.slack_app.event("message")(self.handle_message_events)

    async def handle_app_mention(self, event, say):
        human_text = event["text"]  # .replace("<@U04MGTBFC7J>", "")
        print("event:{}".format(event))
        if 'files' in event:
            if 'summarize' in event['text'].lower():
                response_ref = await self.caption_bot.caption_image.remote(event['files'][0]['url_private'])
        else:
            response_ref = await self.conversation_bot.generate_next.remote(human_text)
        await say(await response_ref)

    async def handle_file_shared_events(self, event):
        logger.info(event)

    async def handle_message_events(self, event):
        logger.info(event)

    @fastapi_app.post("/slack/events")
    async def events_endpoint(self, req: Request) -> None:
        resp = await self.app_handler.handle(req)
        return resp

    @fastapi_app.post("/slack/interactions")
    async def interactions_endpoint(self, req: Request) -> None:
        return await self.app_handler.handle(req)

 # model deployment
conversation_bot = ConversationBot.bind()
image_captioning_bot = ImageCaptioningBot.bind()

# ingress deployment
fast_api_deployment = FastAPIDeployment.bind(
    conversation_bot, image_captioning_bot)

# serve build async_app_fan:fast_api_deployment -o ../deployment/async_app_serve.yaml
# serve build async_app_fan:fast_api_deployment -k
# serve run async_app_fan:fast_api_deployment -p 3000

# serve.run(target=fast_api_deployment, port=3000)
