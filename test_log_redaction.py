"""The MCP access key must never reach the process log.

Clients connect with ``/mcp?auth=<key>``.  uvicorn writes the request line,
query string included, to stdout and a WebSocket handshake line to stderr, and
Railway keeps both streams.  These tests prove the live entry point does not
write the key (or any key=/token= value) anywhere, while authentication still
works exactly as before.

Every secret below is a fake value made up for the test.

Run:  python -m pytest -q test_log_redaction.py
"""

import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import time

import httpx
import pytest

import log_redact

HERE = os.path.dirname(os.path.abspath(__file__))

FAKE_KEY = "SECRET123"  # the server's configured MCP key in these tests (fake)
WRONG_KEY = "WRONGKEY456"
OTHER_KEY = "KEYVALUE789"
OTHER_TOKEN = "TOKENVALUE012"
ALL_SECRETS = (FAKE_KEY, WRONG_KEY, OTHER_KEY, OTHER_TOKEN)


def _server_env(key):
    return {
        "MCP_AUTH_TOKEN": key,
        "SHOPIFY_STORE": "fake-store",
        "SHOPIFY_ACCESS_TOKEN": "fake-shopify-token",
    }


# ---------------------------------------------------------------------------
# The redaction itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line, expected",
    [
        ("POST /mcp?auth=SECRET123 HTTP/1.1", "POST /mcp?auth=REDACTED HTTP/1.1"),
        ("/mcp?auth=SECRET123&door_token=abc", "/mcp?auth=REDACTED&door_token=REDACTED"),
        ("/x?a=1&key=K1&token=T1&b=2", "/x?a=1&key=REDACTED&token=REDACTED&b=2"),
        ("/x?access_token=T1&api_key=K1&AUTH=Z", "/x?access_token=REDACTED&api_key=REDACTED&AUTH=REDACTED"),
        ("/x?author=bob&monkey=1&page=2", "/x?author=bob&monkey=1&page=2"),
        ("nothing to hide here", "nothing to hide here"),
    ],
)
def test_redact_replaces_only_secret_values(line, expected):
    assert log_redact.redact(line) == expected


def test_filter_scrubs_a_uvicorn_access_record_and_keeps_its_shape():
    from uvicorn.logging import AccessFormatter

    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:5000", "POST", f"/mcp?auth={FAKE_KEY}", "1.1", 200),
        None,
    )
    assert log_redact.RedactSecrets().filter(record) is True
    # uvicorn's own formatter unpacks record.args; it must still work.
    line = AccessFormatter('%(client_addr)s - "%(request_line)s" %(status_code)s').format(record)
    assert FAKE_KEY not in line
    assert "auth=REDACTED" in line
    assert record.args[-1] == 200


def test_uvicorn_log_config_puts_the_filter_on_every_handler():
    config = log_redact.uvicorn_log_config()
    assert config["handlers"], "uvicorn defines handlers"
    for name, handler in config["handlers"].items():
        assert "redact_secrets" in handler["filters"], name


# ---------------------------------------------------------------------------
# The live entry point: `python server.py`, the command Railway runs
# ---------------------------------------------------------------------------


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _run_server_and_collect_output(exercise):
    port = _free_port()
    env = {
        **os.environ,
        **_server_env(FAKE_KEY),
        "PORT": str(port),
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "NO_PROXY": "127.0.0.1,localhost",
    }
    base = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(
            [sys.executable, "server.py"],
            cwd=HERE,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.time() + 60
            while True:
                assert process.poll() is None, "server.py exited during startup"
                try:
                    # Any HTTP answer (200 or 401) means it is up.
                    httpx.get(f"{base}/health", timeout=2)
                    break
                except httpx.HTTPError:
                    pass
                assert time.time() < deadline, "server.py never answered"
                time.sleep(0.1)
            exercise(base, port)
            time.sleep(0.5)  # the log line is written before the response; this covers slow disks
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
        output.seek(0)
        return output.read().decode("utf-8", "replace")


_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "log-redaction-test", "version": "0"},
    },
}


def test_live_server_logs_no_key_but_still_authenticates():
    seen = {}

    def exercise(base, port):
        headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
        # 1. The configured key: must still be accepted exactly as before.
        right = httpx.post(f"{base}/mcp?auth={FAKE_KEY}", content=json.dumps(_INITIALIZE), headers=headers, timeout=20)
        seen["right"] = right.status_code
        # 2. A wrong key plus other secret-looking parameters: must still be refused.
        wrong = httpx.post(
            f"{base}/mcp?auth={WRONG_KEY}&key={OTHER_KEY}&token={OTHER_TOKEN}",
            content=json.dumps(_INITIALIZE),
            headers=headers,
            timeout=20,
        )
        seen["wrong"] = wrong.status_code
        # 3. A path that does not exist, still carrying the key.
        seen["stray"] = httpx.get(f"{base}/nope?auth={FAKE_KEY}", timeout=10).status_code
        # 4. A raw WebSocket upgrade: uvicorn logs its path and query on stderr.
        with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
            sock.sendall(
                (
                    f"GET /mcp?auth={FAKE_KEY}&key={OTHER_KEY} HTTP/1.1\r\n"
                    f"Host: 127.0.0.1:{port}\r\n"
                    "Upgrade: websocket\r\n"
                    "Connection: Upgrade\r\n"
                    "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                    "Sec-WebSocket-Version: 13\r\n"
                    "\r\n"
                ).encode("ascii")
            )
            sock.recv(4096)

    output = _run_server_and_collect_output(exercise)

    # The check can fail: the server really ran, really authenticated, and the
    # capture really holds what it wrote.
    assert seen["right"] == 200, seen
    assert seen["wrong"] == 401, seen
    assert "Application startup complete" in output, output
    leaked = [line for line in output.splitlines() if any(s in line for s in ALL_SECRETS)]
    assert not leaked, "an access key reached the log:\n" + "\n".join(leaked)
    # The request lines are still logged, with the value replaced.
    assert "auth=REDACTED" in output, output
