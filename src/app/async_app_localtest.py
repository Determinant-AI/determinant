import os

from ray import serve
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.signature import SignatureVerifier

from logger import create_logger

# Configure the logger.
logger = create_logger(__name__)

app = AsyncApp(
            token=os.environ["SLACK_BOT_TOKEN"],
            signing_secret=os.environ["SLACK_SIGNING_SECRET"],
            request_verification_enabled=True,
        )

@app.event("app_mention")
async def event_test(event, say):
    logger.info(event)
    text = event["text"]
    await say(f"you mentioned me, you said {text}")

@app.event("message")
async def handle_message_events(event, say):
    logger.info(event)
    text = event["text"]
    await say(f"What's up? you said {text}")

@app.event("file_shared")
async def handle_file_shared_events(event, logger):
    logger.info(event)


@serve.deployment(route_prefix="/")
class SlackAgent:
    async def __init__(self):
        self.app_handler = AsyncSocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
        await self.app_handler.start_async()
    
    @app.event("app_mention")
    async def event_test(self, event, say):
        logger.info(event)
        text = event["text"]
        await say(f"you mentioned me, you said {text}")

    @app.event("message")
    async def handle_message_events(self, event, say):
        logger.info(event)
        text = event["text"]
        await say(f"What's up? you said {text}")

    @app.event("file_shared")
    async def handle_file_shared_events(self, event, logger):
        logger.info(event)


test_slack_bot = SlackAgent.bind()
# serve run async_app:test_slack_bot