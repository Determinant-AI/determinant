import json
import os
import random

import requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from data.event import Feedback, LLMAnswerContext
from data.handler import SQSPublisher
from logger import DEBUG, create_logger

# Initializes your app with your bot token and signing secret
app = App(
    token=os.environ["SLACK_BOT_TOKEN"],
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
    request_verification_enabled=True,
)
HF_API_URL = "https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"

event_handler: SQSPublisher = SQSPublisher()
logger = create_logger(__name__, log_level=DEBUG)
answerContext = LLMAnswerContext(model="oasst-sft-4-pythia-12b-epoch-3.5")


@app.event("app_mention")
def handle_app_mention(event, say):
    human_text = ''
    if "text" in event:
        human_text = event["text"]
    thread_ts = event.get("thread_ts", None) or event["ts"]
    logger.info(f"[Human Task] Handling the pinged event: {event}")
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}",
               "Content-Type": "application/json"}
    prompt = f'<|prompter|>{event["text"]}<|endoftext|><|assistant|>'

    answerContext.raw_text = event["text"]
    answerContext.prompt = prompt

    def query(payload):
        data = json.dumps(
            {"inputs": payload, "wait_for_model": True})
        response = requests.request(
            "POST", HF_API_URL, headers=headers, data=data)
        # return response
        resp = json.loads(response.content.decode("utf-8"))
        text = resp[0]["generated_text"]
        if text == payload:
            return text
        else:
            return query(text)

    response = query(prompt)
    answerContext.output_text = response
    logger.info(
        f"[Bot Response]: {response}")
    say(response.removeprefix(prompt), thread_ts=thread_ts)


@app.event("reaction_added")
async def handle_reaction_added_events(event, say) -> None:
    POSTIVE_EMOJIS = set(
        ["thumbsup", "+1", "white_check_mark", "raise_hand", "laughing", "point_up"])
    NEGATIVE_EMOJIS = set(["thumbsdown", "-1"])

    logger.info(f"[Reaction] Reaction event: {event}")

    reaction = event['reaction']
    reaction_author = event['user']
    if answerContext.is_empty() or 'client_msg_id' in event:
        # Note: reaction to app's message (Claude, Determinant, etc.) won't have `client_msg_id`
        logger.info(f"[Human-Human Reaction]: {reaction}")
        return

    feedback = Feedback(answerContext, reaction, reaction_author)
    thread_ts = feedback.thread_ts
    if reaction in NEGATIVE_EMOJIS:
        # TODO: revisit the answer or add self-criticism prompt
        feedback.is_positive = False
        # TODO: Fix bug thread_ts doesn't work
        await say("You seemed to be unhappy with the answer.", thread_ts=thread_ts)
    elif reaction in POSTIVE_EMOJIS:
        feedback.is_positive = True
        await say(f"<@{reaction_author}> Thank you for your positive feedback!",
                  thread_ts=thread_ts)

    logger.info("[Logging] Sending feedback event to SQS...")
    response = event_handler.send_message(feedback.toJson())
    status = response.get('HTTPStatusCode', None)
    if response and status == 200:
        length = response['content-length']
        logger.info(
            f"[Logging] Feedback event sent to SQS successfully, content length {length}.")
    else:
        logger.error(
            "[Logging] Failed to send feedback event to SQS, status {status}.")
    logger.info(f"[Feedback]: {feedback.toJson()}")


@app.event("file_shared")
def handle_file_shared_events(event):
    logger.info(event)


@app.event("user_change")
def handle_user_change_events(body):
    logger.info(body)


@app.event("pin_added")
def handle_pin_added_events(body, logger):
    logger.info(body)


@app.event("app_home_opened")
def handle_app_home_opened_events(body, logger):
    logger.info(body)


@app.event("file_public")
def handle_file_public_events(body):
    logger.info(body)


@app.event("group_left")
def handle_group_left_events(body, logger):
    logger.info(body)


@app.event("message")
def handle_message_events(event, say):
    handle_app_mention(event, say)


# Start your app
if __name__ == "__main__":
    app_handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    app_handler.start()
