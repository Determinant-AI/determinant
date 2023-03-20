import logging
import os 
from typing import Dict

logging.basicConfig(level=logging.DEBUG)
from slack_bolt import BoltRequest, App, BoltResponse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fastapi import FastAPI, Request
from ray import serve
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from slack_bolt.adapter.starlette.handler import SlackRequestHandler
import requests
from slack_sdk.signature import SignatureVerifier
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
import asyncio

from transformers import pipeline
import ray

@ray.remote
class MemoryStore(object):
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value

    def get_counter(self):
        return self.value

# Create an actor from this class.
memory = MemoryStore.remote()


from slack_bolt.async_app import AsyncApp, AsyncBoltRequest
from slack_bolt.oauth.async_oauth_flow import AsyncOAuthFlow
from typing import Dict, Any, Optional


local_deployment = os.environ.get("DB_HOST") == "localhost"
verify_requests = True
if local_deployment:
    verify_requests = False


fastapi_app = FastAPI()

@serve.deployment(route_prefix="/")
@serve.ingress(fastapi_app)
class FastAPIDeployment:
    def __init__(self, conversation_bot): #, summarization_bot):
        self.conversation_bot = conversation_bot
        # self.summarization_bot = summarization_bot
        self.memory = memory
        self.slack_app = AsyncApp(
            token=os.environ["SLACK_BOT_TOKEN"],
            signing_secret=os.environ["SLACK_SIGNING_SECRET"],
            request_verification_enabled=False,
        )
        self.app_handler = AsyncSlackRequestHandler(self.slack_app)
        self.slack_app.event("app_mention")(self.handle_app_mention)

    async def handle_app_mention(self, event, say):
        human_text = event["text"] # .replace("<@U04MGTBFC7J>", "")
        response_ref = await self.conversation_bot.generate_next.remote(human_text)
        await say(await response_ref)
    
    @fastapi_app.post("/slack/events")
    async def events_endpoint(self, req: Request) -> None:
        resp = await self.app_handler.handle(req)
        obj_ref = self.memory.increment.remote()
        return resp

    @fastapi_app.post("/slack/interactions")
    async def interactions_endpoint(self, req: Request) -> None:
        return await self.app_handler.handle(req)


@serve.deployment
class ConversationBot:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    def generate_next(self, human_text):
        new_user_input_ids = self.tokenizer.encode(human_text, self.tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([new_user_input_ids], dim=-1)
        model_output = self.model.generate(bot_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
        response_text = self.tokenizer.decode(model_output[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response_text

# model deployment
conversation_bot = ConversationBot.bind()

# ingress deployment
fast_api_deployment = FastAPIDeployment.bind(conversation_bot)

# serve run async_app:fast_api_deployment -p 3000