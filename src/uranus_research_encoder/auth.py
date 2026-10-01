"""ASGI ingress: authenticate before body parsing, bound memory and active work."""

import asyncio
import hmac

from starlette.types import ASGIApp, Receive, Scope, Send

from .contracts import MAX_BODY_BYTES
from .errors import error, event


class Ingress:
    def __init__(self, app: ASGIApp, state):
        self.app = app
        self.state = state
        self.active = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if scope["path"] == "/health" and scope["method"] == "GET":
            await self.app(scope, receive, send)
            return
        settings = self.state.settings
        if settings is None:
            await error(503, "not_ready")(scope, receive, send)
            return
        headers = scope.get("headers", [])
        credentials = [v for k, v in headers if k.lower() == b"authorization"]
        expected = ("Bearer " + settings.api_key.get_secret_value()).encode("ascii")
        credential = credentials[0] if len(credentials) == 1 else b""
        scheme, separator, token = credential.partition(b" ")
        valid = hmac.compare_digest(b"Bearer " + token, expected)
        if not (valid and separator and scheme.lower() == b"bearer"):
            await error(401, "unauthorized")(scope, receive, send)
            return
        # No waiting queue: a slot covers body reception, validation, work and response.
        limited = scope["method"] not in ("GET", "HEAD")
        if limited and self.active >= settings.max_concurrent_requests:
            await error(503, "busy")(scope, receive, send)
            return
        if limited:
            self.active += 1
        started = False
        try:
            lengths = [v for k, v in headers if k.lower() == b"content-length"]
            if len(lengths) > 1 or (lengths and (not lengths[0].isdigit())):
                await error(400, "invalid_request")(scope, receive, send)
                return
            if lengths and (len(lengths[0]) > 10 or int(lengths[0]) > MAX_BODY_BYTES):
                await error(413, "payload_too_large")(scope, receive, send)
                return
            if any(
                k.lower() == b"content-encoding" and v.lower() != b"identity" for k, v in headers
            ):
                await error(415, "unsupported_encoding")(scope, receive, send)
                return
            body = bytearray()
            async with asyncio.timeout(15):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    body.extend(message.get("body", b""))
                    if len(body) > MAX_BODY_BYTES:
                        await error(413, "payload_too_large")(scope, receive, send)
                        return
                    if not message.get("more_body", False):
                        break
            delivered = False

            async def replay():
                nonlocal delivered
                if delivered:
                    return await receive()
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}

            async def no_store(message):
                nonlocal started
                if message["type"] == "http.response.start":
                    started = True
                    message.setdefault("headers", []).append((b"cache-control", b"no-store"))
                await send(message)

            await self.app(scope, replay, no_store)
        except TimeoutError:
            await error(408, "request_timeout")(scope, receive, send)
        except Exception:
            event("request_failed")
            if not started:
                await error(500, "internal_error")(scope, receive, send)
        finally:
            if limited:
                self.active -= 1
