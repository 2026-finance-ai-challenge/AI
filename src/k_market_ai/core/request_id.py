from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any
from uuid import uuid4

from starlette.types import Message, Receive, Scope, Send

ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


class RequestIdMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid4())
        state = _state(scope)
        state["request_id"] = request_id

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_request_id)


def _state(scope: Scope) -> MutableMapping[str, Any]:
    state = scope.setdefault("state", {})
    if not isinstance(state, MutableMapping):
        raise TypeError("ASGI state must be mutable")
    return state
