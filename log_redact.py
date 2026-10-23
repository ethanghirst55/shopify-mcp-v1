"""Keep MCP access keys out of the process logs.

Clients connect with ``/mcp?auth=<MCP_AUTH_TOKEN>``.  uvicorn writes the full
request line, query string included, to its access log (stdout), and a
WebSocket handshake line to ``uvicorn.error`` (stderr).  Railway keeps both
streams, so the key ends up in plaintext in the deploy logs.

This module redacts the *value* of any ``auth=``, ``key=``, ``token=`` (and
look-alike) parameter before a log record is written, and leaves everything
else on the line alone.  Authentication itself is untouched: the request still
carries ``?auth=`` and the middleware still compares it.

Use it by passing ``log_config=uvicorn_log_config()`` to ``uvicorn.run``.
"""

import copy
import logging
import re

REDACTED = "REDACTED"

# name=value, where the name is auth / key / sig or ends in auth, token,
# secret, password, apikey or api_key (door_token, access_token, ...).
# "author=" and "monkey=" do not match: the name has to END on one of these.
_SECRET_PARAM = re.compile(
    r"(?i)(?<![\w-])"
    r"((?:[\w-]*(?:auth|token|secret|passwd|password|apikey|api_key))|key|sig)"
    r"=([^&\s\"'#]*)"
)


def redact(text):
    """Return ``text`` with the value of every secret-looking parameter replaced."""

    if not isinstance(text, str) or "=" not in text:
        return text
    return _SECRET_PARAM.sub(lambda m: f"{m.group(1)}={REDACTED}", text)


class RedactSecrets(logging.Filter):
    """Logging filter that scrubs secret query parameters from a record in place.

    The record keeps its shape (``msg`` stays a format string, ``args`` stay a
    tuple), because uvicorn's access formatter unpacks ``args`` itself.
    """

    def filter(self, record):
        record.msg = redact(record.msg)
        args = record.args
        if isinstance(args, tuple):
            record.args = tuple(redact(a) for a in args)
        elif isinstance(args, dict):
            record.args = {k: redact(v) for k, v in args.items()}
        return True


def uvicorn_log_config():
    """uvicorn's default logging config, with the filter on every handler.

    A handler-level filter sees every record the handler writes, whichever
    logger it came from (``uvicorn.access`` for HTTP request lines,
    ``uvicorn.error`` for WebSocket handshakes).
    """

    from uvicorn.config import LOGGING_CONFIG

    config = copy.deepcopy(LOGGING_CONFIG)
    config.setdefault("filters", {})["redact_secrets"] = {"()": RedactSecrets}
    for handler in config["handlers"].values():
        handler.setdefault("filters", []).append("redact_secrets")
    return config


def harden_root_logging():
    """Also put the filter on handlers the app's own libraries installed.

    Covers the root logger (``logging.basicConfig``) and FastMCP's logger, so a
    library that logs a URL cannot carry a key into the output either.
    """

    flt = RedactSecrets()
    for name in ("", "fastmcp", "mcp"):
        for handler in logging.getLogger(name).handlers:
            if not any(isinstance(f, RedactSecrets) for f in handler.filters):
                handler.addFilter(flt)
