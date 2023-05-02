# How it works in GCP VM

## clone the git repo

setup ssh key via https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent

## create virtual env

```
python3 -m venv venv
```

## export env var PYTHONPATH

```
PYTHONPATH="path/to/determinant/src"
export PYTHONPATH
```

## export env var SLACK_BOT_TOKEN, SLACK_APP_TOKEN, SLACK_SIGNING_SECRET

## run

```
cd src/app
serve run async_rag_app:slack_agent_deployment
```

# How does slack connection work?

It's now using slack app [socket mode](https://api.slack.com/apis/connections/socket). We don't need to expose ports now. 

If you want to add more functions for other slack events, please add it within func `def register(self)`.