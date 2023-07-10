import os
import random
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_bolt import App
from logger import create_logger, DEBUG
import json
import requests

# Initializes your app with your bot token and signing secret
app = App(
    token=os.environ["SLACK_BOT_TOKEN"],
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
    request_verification_enabled=True,
)


logger = create_logger(__name__, log_level=DEBUG)


@app.event("reaction_added")
def handle_reaction_added_events(event, say) -> None:
    logger.info(f"[Reaction] Reaction event: {event}")


@app.event("app_mention")
def handle_app_mention(event, say):
    human_text = ''
    if "text" in event:
        human_text = event["text"]
    thread_ts = event.get("thread_ts", None) or event["ts"]

    logger.info(f"[Human Task] Handling the pinged event: {event}")

    # if "files" in event and "summarize" in event["text"].lower():
    #     logger.info("graph!")
    #     response = agent.run(event["text"], picture=event["files"][0]["url_private"])
    # else:
    #     response = agent.run(event["text"])
    API_URL = "https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}",
               "Content-Type": "application/json"}
    prompt = f'<|prompter|>{event["text"]}<|endoftext|><|assistant|>'

    def query(payload):
        data = json.dumps(
            {"inputs": payload, "wait_for_model": True})
        response = requests.request(
            "POST", API_URL, headers=headers, data=data)
        # return response
        resp = json.loads(response.content.decode("utf-8"))
        text = resp[0]["generated_text"]
        if text == payload:
            return text
        else:
            return query(text)
    response = query(prompt)

    logger.info(
        f"[Bot Response]: {response}")
    # res = response[0]["generated_text"]
    say(response.removeprefix(prompt), thread_ts=thread_ts)


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
