"""Keep credentials from compatibility query transports out of ASGI logs.

This only changes the logged path, never the request scope. Older clients may
still authenticate using their query transport while newer clients use headers.
Reverse-proxy access logs have a separate configuration and retention policy.
"""
import logging


class AccessPathFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Uvicorn's access record is (client, method, full_path, HTTP version,
        # status). Fail closed if a future formatter changes that contract;
        # formatting an unknown record could expose a credential again.
        args = record.args
        if not isinstance(args, tuple) or len(args) != 5 or not isinstance(args[2], str):
            return False
        record.args = (*args[:2], args[2].partition('?')[0], *args[3:])
        return True


def install_access_log_filter() -> None:
    logger = logging.getLogger('uvicorn.access')
    if not any(isinstance(item, AccessPathFilter) for item in logger.filters):
        logger.addFilter(AccessPathFilter())
