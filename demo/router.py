import asyncio
from itertools import cycle

import httpx
from fastapi import FastAPI, Request
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse


app = FastAPI()

BACKENDS = cycle(
    [
        ("A", "http://127.0.0.1:8000"),
        ("B", "http://127.0.0.1:8001"),
    ]
)

backend_lock = asyncio.Lock()

client = httpx.AsyncClient(
    timeout=None,
    limits=httpx.Limits(
        max_connections=20,
        max_keepalive_connections=20,
    ),
)


async def choose_backend():
    async with backend_lock:
        return next(BACKENDS)


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST"],
)
async def proxy(path: str, request: Request):
    # Health/model discovery stays on A so it does not disturb
    # the deterministic A/B sequence used for inference requests.
    if request.method == "POST":
        backend_name, backend_url = await choose_backend()
    else:
        backend_name, backend_url = (
            "A",
            "http://127.0.0.1:8000",
        )

    destination = f"{backend_url}/{path}"

    body = await request.body()

    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower()
        not in {
            "host",
            "content-length",
            "connection",
        }
    }

    # SSE is easier to reason about when an intermediary isn't
    # helpfully compressing things nobody asked it to compress.
    headers["accept-encoding"] = "identity"

    upstream_request = client.build_request(
        request.method,
        destination,
        headers=headers,
        content=body,
    )

    upstream = await client.send(
        upstream_request,
        stream=True,
    )

    print(
        f"{request.method:4s} /{path} "
        f"-> instance {backend_name} "
        f"({backend_url})",
        flush=True,
    )

    response_headers = {
        key: value
        for key, value in upstream.headers.items()
        if key.lower()
        not in {
            "content-length",
            "transfer-encoding",
            "connection",
        }
    }

    response_headers["x-demo-backend"] = backend_name

    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        headers=response_headers,
        background=BackgroundTask(upstream.aclose),
    )