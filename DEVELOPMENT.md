# Develop in AWS VM
There's an AWS instance that constantly runs determinant app:

### 1. Create Key pair: 
Click the [link](https://us-west-2.console.aws.amazon.com/ec2/home?region=us-west-2#CreateKeyPair), with default to RSA and .pem, it will automatically download the private key `.perm` file to your local

### 2. Generate a public RSA key with the pem file
```
chmod 400 /path_to_key_pair/my-key-pair.pem
ssh-keygen -y -f /path_to_key_pair/my-key-pair.pem
```

### 3. Add public key into the instance
Delegate @kulama or @rebellama to add your public key to the existing instance.

### 4. SSH to the instance:
Reference the [link](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/replacing-key-pair.html)
```
ssh -i "my-key-pair.pem" ec2-user@ec2-18-237-192-233.us-west-2.compute.amazonaws.com
```

### 5. Develop on the remote instance:
After you SSH to the machine, you can develop on the remote instance.

a) Attach to the Ray serve running session to view the live logs:
tmux to the Ray serve session by running
```
tmux attach
```

b) Test the latest change pushed into main branch
```
git pull
serve run async_rag_app:slack_agent_deployment
```

# Develop in GCP VM

### 1. clone the git repo

setup ssh key via [this link](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh)

```

### 2. create virtual env

```
python3 -m venv venv
```

### 3. export env var PYTHONPATH

```
PYTHONPATH="path/to/determinant/src"
export PYTHONPATH
```

### 4. export env var SLACK_BOT_TOKEN, SLACK_APP_TOKEN, SLACK_SIGNING_SECRET

### 5. run

```
cd src/app
serve run async_rag_app:slack_agent_deployment
```

## Connect your local VSCode to the remote VM
Install the exention [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) in your local VSCode. 

Then follow the
[doc](https://code.visualstudio.com/docs/remote/troubleshooting#_configuring-key-based-authentication)
## How does slack connection work?

It's now using slack app [socket mode](https://api.slack.com/apis/connections/socket). We don't need to expose ports now. 

If you want to add more functions for other slack events, please add it within function `def register(self)`.