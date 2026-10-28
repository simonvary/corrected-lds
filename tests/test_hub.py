"""``fetch-banks`` sends a Hugging Face token only to the Hub over https, never past the first
redirect.  urlopen is stubbed; test_contracts.py covers the download and hash checks.
"""

import hashlib

from corrected_lds.banks import hub
from corrected_lds.banks.hub import download

_HF_URL = "https://huggingface.co/datasets/org/banks/resolve/abc123/demo/pool.npz"
_PAYLOAD = b"pool bytes"
_SHA = hashlib.sha256(_PAYLOAD).hexdigest()


def _capturing_urlopen(monkeypatch):
    """Stub hub's urlopen: record every Request, stream ``_PAYLOAD`` once each."""
    seen = []

    class _Resp:
        def __init__(self):
            self._left = _PAYLOAD

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n):
            b, self._left = self._left, b""
            return b

    def fake(req):
        seen.append(req)
        return _Resp()

    monkeypatch.setattr(hub.urllib.request, "urlopen", fake)
    return seen


def test_token_rides_hf_https_requests_unredirected(tmp_path, monkeypatch):
    seen = _capturing_urlopen(monkeypatch)
    download(_HF_URL, str(tmp_path / "pool.npz"), _SHA, token="hf_secret")
    (req,) = seen
    assert req.get_header("Authorization") == "Bearer hf_secret"
    # unredirected: urllib forwards ordinary headers across hosts, and the Hub 302s payloads to
    # signed CDN URLs, so the bearer must die at the first hop
    assert "Authorization" in req.unredirected_hdrs
    assert "Authorization" not in req.headers


def test_token_withheld_from_non_hf_and_plaintext_urls(tmp_path, monkeypatch):
    seen = _capturing_urlopen(monkeypatch)
    urls = ("https://example.com/pool.npz",
            "https://huggingface.co.evil.example/pool.npz",    # lookalike host
            "http://huggingface.co/pool.npz")                  # plaintext would expose it
    for i, url in enumerate(urls):
        download(url, str(tmp_path / f"pool_{i}.npz"), _SHA, token="hf_secret")
    assert [r.get_header("Authorization") for r in seen] == [None] * len(urls)
