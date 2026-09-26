"""Bound work before multipart decoding; no payloads or credentials in logs."""
import asyncio
import logging
import time
from uuid import uuid4
from starlette.responses import JSONResponse

log = logging.getLogger('uvicorn.error')


class RequestLimits:
    def __init__(self, app, max_bytes=40 * 1024 * 1024, heavy_limit=2):
        self.app, self.max_bytes, self.heavy_limit = app, max_bytes, heavy_limit
        self.active = 0

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        operation = uuid4().hex
        start = time.monotonic()
        status = 500
        heavy = scope['method'] in ('POST', 'PUT', 'PATCH')
        async def reply_send(message):
            nonlocal status
            if message['type'] == 'http.response.start':
                status = message['status']
                message = {**message, 'headers': list(message.get('headers', [])) +
                           [(b'x-request-id', operation.encode())]}
            await send(message)
        if heavy and self.active >= self.heavy_limit:
            return await JSONResponse({'detail': 'Service busy; retry the same operation later'},
                status_code=429, headers={'Retry-After': '5'})(scope, receive, reply_send)
        headers = dict(scope.get('headers', []))
        try:
            length = int(headers.get(b'content-length', b'0'))
        except ValueError:
            length = -1
        if length < 0 or length > self.max_bytes:
            return await JSONResponse({'detail': 'Request exceeds 40 MiB limit'},
                status_code=413)(scope, receive, reply_send)
        consumed = 0
        async def bounded_receive():
            nonlocal consumed
            from starlette.exceptions import HTTPException
            try:
                message = await asyncio.wait_for(receive(), timeout=30)
            except asyncio.TimeoutError:
                raise HTTPException(408, 'Upload stalled; retry the same operation') from None
            consumed += len(message.get('body', b''))
            if consumed > self.max_bytes:
                raise HTTPException(413, 'Request exceeds 40 MiB limit')
            return message
        if heavy:
            self.active += 1
        try:
            await self.app(scope, bounded_receive, reply_send)
        finally:
            if heavy:
                self.active -= 1
            # Only generated ID, status and time. Never URLs, body, user or token.
            log.info('request=%s status=%s elapsed_ms=%d', operation, status,
                     (time.monotonic()-start)*1000)


def dependency_status():
    """Read-only readiness. Failure details intentionally contain no secrets."""
    from .corrections_api import _config, _require_private_bucket
    from .correction_workflow import WorkflowStore
    checks = {}
    store = WorkflowStore(_config)
    for name, rpc in [('contours', 'contour_workflow_version'),
                      ('history', 'server_history_version'),
                      ('quality', 'model_quality_version')]:
        try:
            checks[name] = store.request('POST', 'rpc/'+rpc, json={}) == 1
        except Exception:
            checks[name] = False
    try:
        _require_private_bucket()
        checks['private_storage'] = True
    except Exception:
        checks['private_storage'] = False
    return checks
