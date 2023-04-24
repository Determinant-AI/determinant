from fastapi import FastAPI
import asyncio
import starlette
import requests
from starlette.requests import Request
from typing import Dict
import ray
from ray import serve


# 1: Define a Ray Serve deployment.
@serve.deployment(route_prefix="/")
class MyModelDeployment:
    def __init__(self, msg: str):
        # Initialize model state: could be very large neural net weights.
        self._msg = msg

    def __call__(self, request: Request) -> Dict:
        return {"result": self._msg}


# 2: Deploy the model.
handle = serve.run(MyModelDeployment.bind(msg="Hello world!"))

# 3: Query the deployment and print the result.
print(requests.get("http://localhost:8000/").json())

# {'result': 'Hello world!'}


# 1: Define a Ray Serve deployment.
@serve.deployment(route_prefix="/")
class MyModelDeployment2:
    def __init__(self, msg: str):
        # Initialize model state: could be very large neural net weights.
        self._msg = msg

    def __call__(self) -> Dict:
        return {"result": self._msg}


handle = serve.run(MyModelDeployment2.bind(msg="Good Morning!"))
print(ray.get(handle.remote()))

# python3.9 hello_world_deploy.py


# RayServeHandle object (compose deployments)
@serve.deployment
class Driver:
    def __init__(self, model_a_handle, model_b_handle):
        self._model_a_handle = model_a_handle
        self._model_b_handle = model_b_handle

    async def __call__(self, request):
        ref_a = await self._model_a_handle.remote(request)
        ref_b = await self._model_b_handle.remote(request)
        return (await ref_a) + (await ref_b)


@serve.deployment
class ModelA:
    def __init__(self):
        pass

    async def __call__(self):
        await asyncio.sleep(1)
        return 1


@serve.deployment
class ModelB:
    def __init__(self):
        pass

    async def __call__(self):
        await asyncio.sleep(2)
        return 2


model_a = ModelA.bind()
model_b = ModelB.bind()
# model_a and model_b will be passed to the Driver constructor as ServeHandles
driver = Driver.bind(model_a, model_b)

# deploys model_a, model_b, and driver
serve.run(driver)

# Ingress Deployment (top-level/entrypoint deployment)


@serve.deployment
class MostBasicIngress:
    async def __call__(self, request: starlette.requests.Request) -> str:
        name = await request.json()["name"]
        return f"Hello {name}"


ingressor = MostBasicIngress.bind()
serve.run(ingressor)

print(requests.get("http://127.0.0.1:8000/",
                   json={"name": "Fan AI"}).text)

# TypeError: 'coroutine' object is not subscriptable.


app = FastAPI()


@serve.deployment
@serve.ingress(app)
class MostBasicIngress2:
    @app.get("/{name}")
    async def say_hi(self, name: str) -> str:
        return f"Hello {name}"
