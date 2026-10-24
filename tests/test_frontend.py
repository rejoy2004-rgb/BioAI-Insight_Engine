"""Static checks for the frontend: contract with the backend, DOM ids, safe DOM usage, JS syntax."""
import re
import shutil
import subprocess
from html.parser import HTMLParser

import pytest

from src import config

STATIC = config.STATIC_DIR
JS = (STATIC / "app.js").read_text(encoding="utf-8")
HTML = (STATIC / "index.html").read_text(encoding="utf-8")


class _Collect(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.stack, self.errors = [], [], []
        self.void = {"meta", "link", "input", "br", "img", "circle"}

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if "id" in d:
            self.ids.append(d["id"])
        if tag not in self.void:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.void:
            self.stack.pop()

    def handle_endtag(self, tag):
        if tag in self.void:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unbalanced </{tag}> (open: {self.stack[-3:]})")
        else:
            self.stack.pop()


def test_html_is_well_formed_with_unique_ids():
    p = _Collect()
    p.feed(HTML)
    assert not p.errors, p.errors
    assert not p.stack
    dupes = {i for i in p.ids if p.ids.count(i) > 1}
    assert not dupes, dupes


def test_every_id_used_by_js_exists_in_html_or_is_generated():
    p = _Collect()
    p.feed(HTML)
    html_ids = set(p.ids)
    used = set(re.findall(r'\$\("([\w-]+)"\)', JS))
    generated_prefix = ("val-", "desc-")
    missing = {u for u in used if u not in html_ids and not u.startswith(generated_prefix)}
    assert not missing, missing


def test_slider_ids_generated_from_feature_list():
    ids = re.findall(r'id: "(\w+)", label', JS)
    assert len(ids) == 9


def test_no_unsafe_dom_insertion():
    for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert banned not in JS, banned
    assert "onclick=" not in HTML and "<script>" not in HTML    # no inline scripts (CSP script-src 'self')


def test_endpoints_called_by_frontend_exist_in_backend():
    from src.main import app
    routes = {r.path for r in app.routes}
    called = set(re.findall(r'api\("(/[\w/]+)"', JS)) | set(re.findall(r'fetch\("(/[\w/]+)"', JS))
    assert called, "no endpoints found"
    for path in called:
        # /train is registered only when BIOAI_ENABLE_TRAIN_ENDPOINT=1
        assert path in routes or path == "/train", path


def test_response_fields_used_by_frontend_match_schema():
    from src.schemas import Prediction
    fields = set(Prediction.model_fields)
    for f in re.findall(r"\bd\.(\w+)", JS.split("function renderPrediction")[1].split("function renderShap")[0]):
        assert f in fields, f


def test_vendored_assets_present():
    for f in ("vendor/chart.umd.min.js", "vendor/fontawesome/css/all.min.css", "vendor/fontawesome/webfonts/fa-solid-900.woff2"):
        assert (STATIC / f).stat().st_size > 1000
    assert "cdn.jsdelivr" not in HTML and "cdnjs" not in HTML


def test_accessibility_basics():
    assert 'lang="en"' in HTML and 'name="viewport"' in HTML
    assert HTML.count('role="tab"') == 4 and HTML.count('role="tabpanel"') == 4
    assert "skip-link" in HTML and "prefers-reduced-motion" in (STATIC / "style.css").read_text(encoding="utf-8")
    assert "not a validated clinical diagnostic tool" in HTML


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_syntax():
    r = subprocess.run(["node", "--check", str(STATIC / "app.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
