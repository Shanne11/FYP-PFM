"""Local, dependency-free dashboard for PocketIQ research outputs.

Run from the repository root:
    python results_dashboard.py

The server binds to localhost only. No experiment data are uploaded.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse


REPO_ROOT = Path(__file__).resolve().parent


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def number(value: object, digits: int = 4) -> str:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return "—" if value in (None, "") else html.escape(str(value))
    return f"{result:.{digits}f}"


def percent(value: object, digits: int = 1) -> str:
    try:
        return f"{100 * float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def mean_sd(row: dict[str, str], stem: str) -> str:
    return f"{number(row.get(stem + '_mean'))} ± {number(row.get(stem + '_std'))}"


def resolve_asset(roots: dict[str, Path], root_key: str, relative: str) -> Path | None:
    base = roots.get(root_key)
    if base is None:
        return None
    base = base.resolve()
    candidate = (base / relative).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return '<p class="empty">No matching output was found.</p>'
    head = "".join(f"<th>{html.escape(cell)}</th>" for cell in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def card(label: str, value: str, note: str = "") -> str:
    return (
        '<article class="card">'
        f'<span>{html.escape(label)}</span><strong>{html.escape(value)}</strong>'
        f'<small>{html.escape(note)}</small></article>'
    )


def asset_url(root_key: str, relative: str) -> str:
    return f"/asset?root={quote(root_key)}&path={quote(relative)}"


def build_page(outputs: Path, deployment: Path) -> str:
    baseline = read_csv(outputs / "comparison" / "baseline_comparison.csv")
    repeated = read_csv(outputs / "repeated" / "repeated_summary.csv")
    ablations = read_csv(outputs / "ablations" / "ablation_summary.csv")
    actm_rows = read_csv(outputs / "actm_evaluation" / "actm_summary.csv")
    heuristic_rows = read_csv(outputs / "utility_diagnostics" / "utility_diagnostic_summary.csv")
    components = read_csv(outputs / "utility_diagnostics" / "utility_component_summary.csv")
    balance = read_csv(outputs / "class_balance" / "class_balance_summary.csv")
    statistics = read_csv(outputs / "statistics" / "paired_statistics.csv")
    deploy_metrics = read_json(deployment / "deployment_metrics.json")
    parity = read_json(deployment / "parity_report.json")

    proposed = next((r for r in repeated if r.get("method", "").lower() == "proposed"), {})
    actm = actm_rows[-1] if actm_rows else {}
    heuristic_all = next((r for r in heuristic_rows if r.get("run", "").upper() == "ALL"), {})
    if not heuristic_all and heuristic_rows:
        heuristic_all = heuristic_rows[-1]

    heldout = deploy_metrics.get("held_out", deploy_metrics.get("heldout", {}))
    if not isinstance(heldout, dict):
        heldout = {}
    parity_passed = parity.get("passed")

    files = sorted(p for p in outputs.rglob("*") if p.is_file()) if outputs.exists() else []
    total_size = sum(p.stat().st_size for p in files)
    cards = "".join(
        [
            card("Output files", f"{len(files):,}", f"{total_size / (1024 ** 2):.1f} MiB local evidence"),
            card("Proposed Macro F1", number(proposed.get("macro_f1_mean")), "Tier A, mean across repeated seeds"),
            card("ACTM prompt precision", percent(actm.get("prompt_precision_mean")), "Tier A retrospective proxy outcome"),
            card("Deployment held-out Macro F1", number(heldout.get("macro_f1")), "Separate 14-category ONNX model"),
            card("ONNX parity", "Passed" if parity_passed is True else "Unavailable" if parity_passed is None else "Failed", "Python versus ONNX Runtime"),
        ]
    )

    baseline_table = table(
        ["Model", "Accuracy", "Macro F1", "Weighted F1", "ECE", "Brier"],
        [[html.escape(r.get("model", "")), number(r.get("accuracy")), number(r.get("macro_f1")), number(r.get("weighted_f1")), number(r.get("ece")), number(r.get("brier_score"))] for r in baseline],
    )
    repeated_table = table(
        ["Method", "Runs", "Accuracy (mean ± SD)", "Macro F1 (mean ± SD)", "Weighted F1 (mean ± SD)", "ECE (mean ± SD)"],
        [[html.escape(r.get("method", "").title()), html.escape(r.get("runs", "")), mean_sd(r, "accuracy"), mean_sd(r, "macro_f1"), mean_sd(r, "weighted_f1"), mean_sd(r, "ece")] for r in repeated],
    )
    ablation_table = table(
        ["Variant", "Accuracy", "Macro F1", "Weighted F1", "ECE", "Brier"],
        [[html.escape(r.get("variant", "").replace("_", " ").title()), mean_sd(r, "accuracy"), mean_sd(r, "macro_f1"), mean_sd(r, "weighted_f1"), mean_sd(r, "ece"), mean_sd(r, "brier_score")] for r in ablations],
    )
    actm_table = table(
        ["Runs", "Eligible", "Prompts per 100", "Prompt precision", "Note acceptance", "Mean uncertainty reduction"],
        [[html.escape(actm.get("runs", "")), percent(actm.get("eligible_rate_mean")), number(actm.get("prompts_per_100_mean"), 2), percent(actm.get("prompt_precision_mean")), percent(actm.get("note_acceptance_rate_mean")), number(actm.get("mean_uncertainty_reduction_mean"), 6)]] if actm else [],
    )
    heuristic_table = table(
        ["Client rounds", "Fallback rate", "Near-identical weights", "Mean heuristic range", "Multiplier range", "Mean absolute weight change"],
        [[
            html.escape(heuristic_all.get("client_rounds", "")),
            percent(heuristic_all.get("fallback_rate")),
            percent(heuristic_all.get("near_identical_rate")),
            f"{number(heuristic_all.get('client_mean_utility_min'))}–{number(heuristic_all.get('client_mean_utility_max'))}",
            f"{number(heuristic_all.get('multiplier_min'))}–{number(heuristic_all.get('multiplier_max'))}",
            number(heuristic_all.get("absolute_weight_change_mean"), 7),
        ]] if heuristic_all else [],
    )
    component_all = [r for r in components if r.get("run", "").upper() == "ALL"]
    if not component_all:
        component_all = components
    component_table = table(
        ["Component", "Mean", "SD", "Median", "Range"],
        [[html.escape(r.get("component", "").replace("_", " ").title()), number(r.get("value_mean")), number(r.get("value_std")), number(r.get("value_median")), f"{number(r.get('value_min'))}–{number(r.get('value_max'))}"] for r in component_all],
    )
    balance_table = table(
        ["Method", "Loss", "Macro F1", "Weighted F1", "Zero-recall classes", "Active predicted classes"],
        [[html.escape(r.get("method", "").title()), html.escape(r.get("loss", "").replace("_", " ").title()), mean_sd(r, "macro_f1"), mean_sd(r, "weighted_f1"), mean_sd(r, "zero_recall_classes"), mean_sd(r, "active_prediction_classes")] for r in balance],
    )
    macro_stats = [r for r in statistics if r.get("metric") == "macro_f1"]
    stats_table = table(
        ["Comparison", "Mean difference", "95% CI", "Wins", "Exact p"],
        [[html.escape(r.get("comparison", "")), number(r.get("mean_difference_a_minus_b")), f"[{number(r.get('ci95_lower'))}, {number(r.get('ci95_upper'))}]", html.escape(r.get("wins_a", "")), number(r.get("exact_sign_flip_p"), 2)] for r in macro_stats],
    )

    validation = deploy_metrics.get("validation", {})
    if not isinstance(validation, dict):
        validation = {}
    deploy_table = table(
        ["Split / check", "Accuracy", "Macro F1", "Weighted F1", "NLL", "ECE"],
        [
            ["Validation", number(validation.get("accuracy")), number(validation.get("macro_f1")), number(validation.get("weighted_f1")), number(validation.get("negative_log_likelihood")), number(validation.get("ece_10_equal_width"))],
            ["Held-out", number(heldout.get("accuracy")), number(heldout.get("macro_f1")), number(heldout.get("weighted_f1")), number(heldout.get("negative_log_likelihood")), number(heldout.get("ece_10_equal_width"))],
        ] if deploy_metrics else [],
    )
    parity_table = table(
        ["Rows", "Maximum absolute difference", "Mean absolute difference", "Tolerance", "Status"],
        [[html.escape(str(parity.get("rows", parity.get("fixture_count", "")))), number(parity.get("max_absolute_logit_difference", parity.get("max_abs_difference")), 9), number(parity.get("mean_absolute_logit_difference", parity.get("mean_abs_difference")), 9), number(parity.get("tolerance"), 7), "Passed" if parity_passed else "Failed"]] if parity else [],
    )

    chart_candidates = [
        ("comparison/baseline_comparison.png", "Single-seed baseline comparison"),
        ("repeated/macro_f1_mean_std.png", "Repeated-seed Macro F1"),
        ("ablations/macro_f1_ablation.png", "Ablation comparison"),
        ("actm_evaluation/actm_rates.png", "ACTM rates"),
        ("utility_diagnostics/utility_diagnostics.png", "Heuristic diagnostics"),
        ("class_analysis/class_f1_comparison.png", "Per-class F1"),
        ("class_balance/class_balance_macro_f1.png", "Class-balance comparison"),
    ]
    charts = "".join(
        f'<figure><a href="{asset_url("outputs", rel)}" target="_blank"><img loading="lazy" src="{asset_url("outputs", rel)}" alt="{html.escape(title)}"></a><figcaption>{html.escape(title)}</figcaption></figure>'
        for rel, title in chart_candidates if (outputs / rel).is_file()
    ) or '<p class="empty">No chart images were found.</p>'

    catalog_rows: list[list[str]] = []
    for path in files:
        rel = path.relative_to(outputs).as_posix()
        suffix = path.suffix.lower()
        if suffix not in {".csv", ".json", ".txt", ".png"}:
            continue
        link = f'<a href="{asset_url("outputs", rel)}" target="_blank">{html.escape(rel)}</a>'
        catalog_rows.append([link, html.escape(suffix.lstrip(".").upper()), f"{path.stat().st_size / 1024:.1f} KiB"])
    catalog = table(["Output file", "Type", "Size"], catalog_rows)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PocketIQ Results Dashboard</title>
<style>
:root{{--ink:#172033;--muted:#64748b;--blue:#2563eb;--line:#dbe3ef;--bg:#f5f7fb;--panel:#fff;--warn:#fff8e6}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
header{{background:linear-gradient(135deg,#13213a,#234f95);color:#fff;padding:34px max(24px,calc((100vw - 1240px)/2)) 28px}}h1{{margin:0;font-size:clamp(28px,4vw,44px)}}header p{{max-width:850px;margin:8px 0 0;color:#dbeafe}}
nav{{position:sticky;top:0;z-index:5;display:flex;gap:8px;overflow:auto;padding:10px max(18px,calc((100vw - 1240px)/2));background:#fffffff2;border-bottom:1px solid var(--line);backdrop-filter:blur(10px)}}nav a{{white-space:nowrap;text-decoration:none;color:#334155;padding:7px 10px;border-radius:8px}}nav a:hover{{background:#eaf1ff;color:var(--blue)}}main{{max-width:1240px;margin:auto;padding:24px}}section{{scroll-margin-top:68px;margin:0 0 26px;background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:22px;box-shadow:0 7px 22px #1720330a}}h2{{margin:0 0 6px;font-size:23px}}h3{{margin:22px 0 8px;font-size:17px}}.sub{{margin:0 0 16px;color:var(--muted)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin-bottom:18px}}.card{{border:1px solid var(--line);border-radius:12px;padding:15px;background:#fbfdff}}.card span,.card small{{display:block;color:var(--muted)}}.card strong{{display:block;font-size:25px;margin:3px 0}}.notice{{padding:14px 16px;border-left:4px solid #d69e2e;background:var(--warn);border-radius:8px}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:10px}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}}th{{position:sticky;top:0;background:#edf3fb;font-size:13px}}tbody tr:hover{{background:#f8fbff}}.empty{{color:var(--muted);font-style:italic}}.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:16px}}figure{{margin:0;border:1px solid var(--line);border-radius:12px;overflow:hidden;background:#fff}}figure img{{display:block;width:100%;aspect-ratio:16/10;object-fit:contain;background:#f8fafc}}figcaption{{padding:10px 12px;font-weight:600}}#catalog .table-wrap{{max-height:560px}}footer{{max-width:1240px;margin:0 auto 30px;padding:0 24px;color:var(--muted)}}
@media(max-width:650px){{main{{padding:14px}}section{{padding:15px}}th,td{{padding:8px;font-size:13px}}}}
</style></head><body>
<header><h1>PocketIQ Results Dashboard</h1><p>Local view of regenerated Tier A research evidence and the separate PocketIQ ONNX deployment package. Refresh the page after rerunning experiments.</p></header>
<nav><a href="#overview">Overview</a><a href="#tier-a">Tier A</a><a href="#actm">ACTM & heuristic</a><a href="#class-balance">Class balance</a><a href="#deployment">Deployment</a><a href="#charts">Charts</a><a href="#catalog">Files</a></nav>
<main>
<section id="overview"><h2>Overview</h2><p class="sub">Generated directly from the current local CSV and JSON files.</p><div class="cards">{cards}</div><div class="notice"><strong>Evidence boundary:</strong> Tier A evaluates the 105-feature, 13-category federated research method. PocketIQ uses a separately trained 4,096-feature, 14-category frozen ONNX model. The dashboard keeps these results separate.</div></section>
<section id="tier-a"><h2>Tier A research results</h2><p class="sub">Controlled BudgetWise evaluation. Three-seed summaries are stronger evidence than a single run, but three seeds remain exploratory.</p><h3>Single comparison</h3>{baseline_table}<h3>Repeated seeds</h3>{repeated_table}<h3>Claim-dependent ablations</h3>{ablation_table}<h3>Paired Macro F1 comparisons</h3>{stats_table}</section>
<section id="actm"><h2>ACTM and combined heuristic</h2><p class="sub">The Tier A heuristic includes local-adaptation effects and must not be interpreted as an isolated causal Smart Note benefit.</p>{actm_table}<h3>Aggregation-effect diagnosis</h3>{heuristic_table}<h3>Heuristic components</h3>{component_table}</section>
<section id="class-balance"><h2>Class-balance analysis</h2><p class="sub">Class weighting improves category coverage and Macro F1 while reducing overall accuracy in these runs.</p>{balance_table}</section>
<section id="deployment"><h2>PocketIQ deployment model</h2><p class="sub">Separate 14-category TF-IDF/linear ONNX model used by Flutter; not the Tier A checkpoint.</p>{deploy_table}<h3>ONNX parity</h3>{parity_table}</section>
<section id="charts"><h2>Charts</h2><p class="sub">Select any chart to open the full local image.</p><div class="gallery">{charts}</div></section>
<section id="catalog"><h2>Output file browser</h2><p class="sub">CSV, JSON, text, and PNG files under the selected outputs directory. Use the browser search command (Ctrl+F) to find a filename.</p>{catalog}</section>
</main><footer>Served from <code>127.0.0.1</code>. No output is sent to an external service.</footer>
</body></html>"""


class DashboardServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], outputs: Path, deployment: Path):
        super().__init__(address, DashboardHandler)
        self.outputs = outputs.resolve()
        self.deployment = deployment.resolve()


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardServer

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            payload = build_page(self.server.outputs, self.server.deployment).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path == "/asset":
            query = parse_qs(parsed.query)
            root_key = query.get("root", [""])[0]
            relative = query.get("path", [""])[0]
            target = resolve_asset(
                {"outputs": self.server.outputs, "deployment": self.server.deployment},
                root_key,
                relative,
            )
            if target is None:
                self.send_error(404)
                return
            payload = target.read_bytes()
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Content-Disposition", f'inline; filename="{target.name}"')
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404)

    def log_message(self, format: str, *args: object) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View PocketIQ experiment outputs locally.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--outputs", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--deployment", type=Path, default=REPO_ROOT / "deployment" / "pocketiq_deployment_package")
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = DashboardServer(("127.0.0.1", args.port), args.outputs, args.deployment)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"PocketIQ results dashboard: {url}")
    print(f"Research outputs: {server.outputs}")
    print(f"Deployment package: {server.deployment}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
