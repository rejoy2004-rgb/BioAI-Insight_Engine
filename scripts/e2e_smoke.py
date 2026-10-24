"""End-to-end browser check + screenshot capture (requires `playwright install chromium`).

    python scripts/e2e_smoke.py [--shots docs/screenshots]

Starts the real server on a free port (against the real trained artifacts in
data/models), drives each user flow in headless Chromium, fails on any console
error / failed request / missing expected element, and saves genuine screenshots.
"""
import argparse
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(base: str, timeout: float = 90) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            if urllib.request.urlopen(base + "/ready", timeout=2).status == 200:
                return
        except Exception:
            time.sleep(0.5)
    raise SystemExit("server did not become ready")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default=str(ROOT / "docs" / "screenshots"))
    args = ap.parse_args()
    shots = Path(args.shots)
    shots.mkdir(parents=True, exist_ok=True)
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.main:app", "--port", str(port), "--log-level", "warning"],
                              cwd=ROOT)
    problems: list[str] = []
    try:
        wait_ready(base)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            ctx = browser.new_context(viewport={"width": 1440, "height": 900})
            page = ctx.new_page()
            page.on("console", lambda m: m.type in ("error", "warning") and problems.append(f"console {m.type}: {m.text}"))
            page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
            page.on("requestfailed", lambda r: problems.append(f"request failed: {r.url}") if "fonts.g" not in r.url else None)
            page.on("response", lambda r: problems.append(f"HTTP {r.status}: {r.url}") if r.status >= 400 else None)

            page.goto(base)
            page.wait_for_selector("#risk-badge:has-text('prediction')")
            assert "API ready" in page.inner_text("#api-status-text")
            page.set_viewport_size({"width": 1440, "height": 1250})
            page.screenshot(path=shots / "01-classifier-default.png")
            page.set_viewport_size({"width": 1440, "height": 900})

            page.click("#btn-example-malignant")
            page.wait_for_selector("#risk-badge:has-text('Malignant')")
            assert page.locator(".shap-bar-item").count() == 9
            page.screenshot(path=shots / "02-classifier-malignant-like.png")

            page.click("#btn-example-benign")
            page.wait_for_selector("#risk-badge:has-text('Benign')")

            page.click("#tab-btn-model-metrics")
            page.wait_for_selector("#metrics-table tbody tr")
            page.wait_for_selector("#metrics-status", state="hidden")
            assert page.locator("#metrics-table tbody tr").count() == 4
            page.wait_for_timeout(400)
            page.set_viewport_size({"width": 1440, "height": 2100})
            page.wait_for_timeout(300)
            page.screenshot(path=shots / "03-model-evaluation.png")
            page.set_viewport_size({"width": 1440, "height": 900})

            page.click("#tab-btn-rag-search")
            page.fill("#rag-query", "What are the risk factors for breast cancer?")
            page.click("#btn-rag-search")
            page.wait_for_selector(".answer-card")
            assert page.locator("blockquote.quote").count() >= 1 and page.locator("sup.cite").count() >= 1
            page.screenshot(path=shots / "04-evidence-answer.png")

            page.fill("#rag-query", "quantum chromodynamics lattice")
            page.click("#btn-rag-search")
            page.wait_for_selector("text=No supporting evidence found")
            page.screenshot(path=shots / "05-evidence-no-result.png")

            page.fill("#rag-query", "<img src=x onerror=window.__xss=1>")
            page.click("#btn-rag-search")
            page.wait_for_timeout(500)
            assert page.evaluate("window.__xss") is None, "XSS payload executed"

            page.click("#tab-btn-batch-inference")
            bad = ROOT / "scripts" / "_bad.csv"
            bad.write_text("Cl.thickness,Cell.size\n1,2\n", encoding="utf-8")
            try:
                page.set_input_files("#csv-file-input", str(bad))
                page.wait_for_selector("#batch-error:not([hidden])")
                assert "Missing column" in page.inner_text("#batch-error")
            finally:
                bad.unlink()
            tmpl = ROOT / "scripts" / "_ok.csv"
            tmpl.write_text("Cl.thickness,Cell.size,Cell.shape,Marg.adhesion,Epith.c.size,Bare.nuclei,Bl.cromatin,"
                            "Normal.nucleoli,Mitoses\n5,3,3,1,2,1,3,1,1\n8,8,8,5,7,10,9,7,1\n", encoding="utf-8")
            try:
                page.set_input_files("#csv-file-input", str(tmpl))
                page.wait_for_selector("#batch-table tbody tr")
                assert page.locator("#batch-table tbody tr").count() == 2
            finally:
                tmpl.unlink()
            page.screenshot(path=shots / "06-batch.png")

            # keyboard: tabs reachable and operable with arrow keys
            page.focus("#tab-btn-batch-inference")
            page.keyboard.press("ArrowDown")
            assert page.get_attribute("#tab-btn-patient-predict", "aria-selected") == "true"

            # mobile layout: no horizontal overflow
            mobile = browser.new_context(viewport={"width": 390, "height": 844}).new_page()
            mobile.goto(base)
            mobile.wait_for_selector("#risk-badge:has-text('prediction')")
            overflow = mobile.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
            if overflow > 1:
                problems.append(f"mobile horizontal overflow: {overflow}px")
            mobile.screenshot(path=shots / "07-mobile.png", full_page=True)

            # API disconnect: stop server -> UI must show an offline state, not crash
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=15)
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        return 1
    print(f"E2E OK. Screenshots in {shots}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
