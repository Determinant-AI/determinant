import os
# Use the package we installed
from slack_bolt import App
import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Start a Ray cluster
ray.init()

# Load the model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
conversation_state = {}

# Initializes your app with your bot token and signing secret
app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)

# Add functionality here
# @app.event("app_home_opened") etc
@app.event("app_mention")
def reply_mention(event, say):
    human_text = event["text"].replace("<@U04MGTBFC7J>", "")
    new_user_input_ids = tokenizer.encode(human_text, tokenizer.eos_token, return_tensors='pt')
    print(new_user_input_ids)
    bot_input_ids = torch.cat([new_user_input_ids], dim=-1)
    model_output = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(model_output[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    say(text=response_text, channel=event["channel"])

@app.event("message")
def reply_message_channel(event, say):
    counter = 0
    human_text = event["text"].replace("<@U04MGTBFC7J>", "")
    new_user_input_ids = tokenizer.encode(human_text, tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if counter > 0 else new_user_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    say(text=response_text, channel=event["channel"])

# Start your app
if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 3000)))

