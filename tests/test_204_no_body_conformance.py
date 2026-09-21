# RFC 9110 s15.3.5: a 204 response MUST NOT carry content. On EVERY path.
"""
The sibling of test_head_no_body_conformance.py, for the other status whose
body the spec forbids.

Python did NOT behave correctly: with TINA4_DEBUG=true the dev toolbar was
appended to every 204, because _stage_dev_toolbar_inject gated on dev mode and
text/html and never on the status - and a 204 IS text/html, since Response()
sets that content type unconditionally and `response(None, 204)` (what
`tina4 make crud` scaffolds for every delete handler) leaves it alone.
Measured 2026-09-18 at 3.13.136: a scaffolded DELETE returned 1037 bytes and
declared Content-Length: 1037; an OPTIONS on a known path returned 1032.

Why it matters beyond conformance: under uvicorn's h11 the body is refused
("Too much data for declared Content-Length") and the connection is torn down,
so the NEXT request over it dies with RemoteDisconnected. Under httptools, and
under Tina4's own dev bridge, the bytes simply go out and are discarded - and
with Accept-Encoding: gzip the 204 even goes out gzip-encoded, because the
toolbar pushes the body past the compression threshold in build_headers.

The strip is unconditional and late, for the same reason the HEAD strip is:
whoever put the body there - an injector, a handler that passed content
alongside a 204 - it does not leave. A guard inside one injector protects one
injector.

Driven through the REAL ASGI app: the body is assembled from the
http.response.body messages, which is where a leak would actually show.
"""
import asyncio

import pytest

from tina4_python.core.router import Router, get as route_get, delete as route_delete
from tina4_python.core.server import app


@pytest.fixture(autouse=True)
def _workspace(tmp_path, monkeypatch):
    (tmp_path / "src" / "public").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TINA4_DEBUG", "true")
    Router.clear()

    @route_get("/page")
    async def _page(request, response):
        return response("<html><body><h1>hi</h1></body></html>")

    @route_delete("/item/1", auth_required=False)
    async def _delete(request, response):
        return response(None, 204)

    @route_get("/handmade")
    async def _handmade(request, response):
        return response("<b>a 204 the handler gave a body to</b>", 204)

    yield tmp_path
    Router.clear()


def drive(method, path, headers=None):
    """Return (status, body_bytes, headers) from the real ASGI app."""
    sent = []
    scope = {"type": "http", "method": method, "path": path, "query_string": b"",
             "headers": headers or [], "client": ("127.0.0.1", 1)}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    hdrs = {k.decode().lower(): v.decode() for k, v in start.get("headers", [])}
    return start["status"], body, hdrs


def test_a_scaffolded_delete_returns_a_204_with_no_body():
    status, body, _ = drive("DELETE", "/item/1")
    assert status == 204
    assert len(body) == 0, (
        f"the 204 carried {len(body)} bytes - RFC 9110 s15.3.5 forbids content "
        f"in a 204 response"
    )


def test_the_204_declares_content_length_zero():
    _status, _body, headers = drive("DELETE", "/item/1")
    assert headers.get("content-length") == "0", (
        "a 204 declaring a non-zero Content-Length is refused by uvicorn's h11 "
        "and kills the connection"
    )


def test_an_options_on_a_known_path_returns_a_204_with_no_body():
    """s9.3.7: OPTIONS answers 204 + Allow. It is a fallback stage, so unlike
    the CORS preflight it DOES fall through the response stages."""
    status, body, headers = drive("OPTIONS", "/item/1")
    assert status == 204
    assert "allow" in {k.lower() for k in headers}, "the Allow header was lost"
    assert len(body) == 0


def test_a_204_the_handler_gave_a_body_to_is_stripped_too():
    """The HEAD strip removes a body an explicit Router.head() handler returned
    by mistake. This holds the same line."""
    status, body, _ = drive("GET", "/handmade")
    assert status == 204
    assert len(body) == 0


def test_a_204_is_never_gzip_encoded():
    """build_headers compresses above 1024 bytes; the toolbar pushed a 204 past
    it, so a no-content response went out Content-Encoding: gzip."""
    _status, body, headers = drive(
        "DELETE", "/item/1", [(b"accept-encoding", b"gzip")]
    )
    assert "content-encoding" not in headers
    assert len(body) == 0


def test_an_ordinary_html_200_still_gets_the_toolbar():
    """NEGATIVE: the strip must not have cost the toolbar its actual job."""
    status, body, _ = drive("GET", "/page")
    assert status == 200
    assert b"__dev" in body, "the dev toolbar stopped being injected at all"
    assert b"<h1>hi</h1>" in body


def test_a_head_still_carries_no_body_and_still_reports_the_length():
    """NEGATIVE: the 204 strip runs beside the HEAD strip; neither may break
    the other."""
    status, body, headers = drive("HEAD", "/page")
    assert status == 200
    assert len(body) == 0
    assert int(headers["content-length"]) > 0


def test_a_head_on_a_204_reports_content_length_zero():
    """ORDER: the 204 strip must run BEFORE the HEAD strip.

    HEAD maps to the GET route, so this is a HEAD on a 204. The HEAD strip
    records the length the equivalent GET would have sent - correct for a 200,
    wrong for a 204, where the answer is always zero. Run the other way round
    and this response carries Content-Length: 39 for a body that is forbidden.
    """
    status, body, headers = drive("HEAD", "/handmade")
    assert status == 204
    assert len(body) == 0
    assert headers.get("content-length") == "0", (
        "the HEAD strip reported the length of a body a 204 may not have - "
        "_stage_no_content_strip is registered after it instead of before"
    )


def test_the_dev_dashboard_records_the_size_that_went_on_the_wire():
    """ORDER: the 204 strip must run BEFORE the inspector capture.

    _RESPONSE_STAGES says body_size is meant to be what actually went on the
    wire - that is why the inspector is placed after the toolbar injection.
    Strip the 204 after the inspector has already looked and the dashboard
    reports 1037 bytes for a response that ships none.
    """
    from tina4_python.dev_admin import RequestInspector

    RequestInspector.clear()
    _status, body, _ = drive("DELETE", "/item/1")
    assert len(body) == 0
    recorded = RequestInspector.get()
    assert recorded, "the dev inspector recorded nothing at all"
    assert recorded[0]["body_size"] == 0, (
        f"the dashboard reports {recorded[0]['body_size']} bytes for a 204 that "
        f"put {len(body)} on the wire"
    )
