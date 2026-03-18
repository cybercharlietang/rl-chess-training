"""Generate HTML report from diagnostic test results (best corrected versions)."""

import html as html_mod
import json

import chess
import chess.svg


def render_board_svg(fen, size=220):
    """Render a chess board as SVG."""
    try:
        board = chess.Board(fen)
        flipped = board.turn == chess.BLACK
        return chess.svg.board(board, flipped=flipped, size=size)
    except Exception:
        return ""


def generate_report():
    """Combine the best diagnostic results into one HTML report."""

    # Load the corrected results
    sources = {
        "Declarative Rules Knowledge": ("results/diagnostics_rules/chess_diagnostics.json", None),
        "FEN → Piece Identification": ("results/diagnostics_v2/chess_diagnostics.json", "FEN → Piece Identification"),
        "Legal Move Generation": ("results/diagnostics_v2/chess_diagnostics.json", "Legal Move Generation"),
        "Move Legality Judgment": ("results/diagnostics_v3/chess_diagnostics.json", None),
        "One-Move Consequence": ("results/diagnostics_v3_cons/chess_diagnostics.json", None),
    }

    tests = []
    all_samples = {}

    for test_name, (path, filter_name) in sources.items():
        data = json.load(open(path))

        # Find the right metric entry
        for m in data["metrics"]:
            if filter_name and m.get("test_name") != filter_name:
                continue
            tests.append({
                "name": test_name,
                "score": m.get("accuracy", m.get("mean_f1", 0)),
                "num_samples": m.get("num_samples", 0),
                "baseline": m.get("baseline", 0),
                "details": {k: v for k, v in m.items()
                           if k not in ("test_name", "score", "num_samples", "baseline", "samples")},
            })
            break

        # Collect samples for this test
        if filter_name:
            samples = [s for s in data.get("samples", []) if s.get("test_name") == filter_name]
        else:
            samples = data.get("samples", [])
        all_samples[test_name] = samples

    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           max-width: 1200px; margin: 0 auto; padding: 20px; background: #0d1117; color: #c9d1d9; }
    h1, h2, h3 { color: #58a6ff; }
    table { border-collapse: collapse; width: 100%; margin: 16px 0; }
    th, td { border: 1px solid #30363d; padding: 10px 14px; text-align: left; }
    th { background: #161b22; color: #58a6ff; }
    tr:nth-child(even) { background: #161b22; }
    .correct { color: #3fb950; font-weight: bold; }
    .wrong { color: #f85149; font-weight: bold; }
    .bar-container { background: #21262d; border-radius: 4px; height: 24px; width: 200px; display: inline-block; vertical-align: middle; }
    .bar-fill { height: 100%; border-radius: 4px; display: inline-block; }
    .bar-good { background: #3fb950; }
    .bar-ok { background: #d29922; }
    .bar-bad { background: #f85149; }
    .metric-card { display: inline-block; background: #161b22; border: 1px solid #30363d;
                   border-radius: 8px; padding: 16px 24px; margin: 6px; text-align: center; min-width: 160px; }
    .metric-value { font-size: 32px; font-weight: bold; }
    .metric-label { font-size: 13px; color: #8b949e; margin-top: 4px; }
    .completion-box { white-space: pre-wrap; max-height: 400px; overflow-y: auto;
                      background: #161b22; color: #8b949e; padding: 12px; border-radius: 6px;
                      font-size: 12px; font-family: monospace; margin: 8px 0; border: 1px solid #30363d; }
    .sample-header { cursor: pointer; padding: 10px; background: #161b22; border: 1px solid #30363d;
                     border-radius: 6px; margin: 4px 0; font-size: 13px; }
    .sample-header:hover { background: #1c2128; }
    .sample-detail { display: none; padding: 12px; border: 1px solid #30363d;
                     border-top: none; border-radius: 0 0 6px 6px; margin-bottom: 8px; }
    .test-section { margin: 30px 0; padding: 20px; border: 1px solid #30363d; border-radius: 8px; }
    .breakdown-table { width: auto; }
    .breakdown-table td, .breakdown-table th { padding: 6px 12px; font-size: 13px; }
    """

    # Summary table
    summary_html = """<h2>Summary</h2>
    <table>
    <tr><th>Test</th><th>Score</th><th>Random Baseline</th><th>Samples</th><th></th></tr>"""

    for t in tests:
        score_pct = t["score"] * 100
        baseline_pct = t["baseline"] * 100
        bar_class = "bar-good" if score_pct > 70 else "bar-ok" if score_pct > 50 else "bar-bad"
        score_color = "#3fb950" if score_pct > 70 else "#d29922" if score_pct > 50 else "#f85149"

        summary_html += f"""<tr>
            <td>{t['name']}</td>
            <td style="color: {score_color}; font-weight: bold;">{score_pct:.0f}%</td>
            <td>{baseline_pct:.1f}%</td>
            <td>{t['num_samples']}</td>
            <td><div class="bar-container"><div class="bar-fill {bar_class}" style="width: {min(score_pct, 100)}%;"></div></div></td>
        </tr>"""
    summary_html += "</table>"

    # Metric cards
    cards_html = '<div style="display: flex; flex-wrap: wrap; justify-content: center; margin: 20px 0;">'
    for t in tests:
        score_pct = t["score"] * 100
        color = "#3fb950" if score_pct > 70 else "#d29922" if score_pct > 50 else "#f85149"
        short_name = t["name"].split("→")[-1].strip() if "→" in t["name"] else t["name"].split("(")[0].strip()
        if len(short_name) > 20:
            short_name = short_name[:18] + "..."
        cards_html += f"""<div class="metric-card">
            <div class="metric-value" style="color: {color};">{score_pct:.0f}%</div>
            <div class="metric-label">{short_name}</div></div>"""
    cards_html += "</div>"

    # Interpretation
    interp_html = """<h2>Interpretation</h2>
    <table>
    <tr><th>Finding</th><th>Evidence</th></tr>
    <tr><td>Strong declarative knowledge</td><td>88% on chess rules questions — the model knows how pieces move in theory</td></tr>
    <tr><td>FEN parsing works but is expensive</td><td>76% piece identification, but requires ~1000+ tokens of chain-of-thought</td></tr>
    <tr><td>Can enumerate moves procedurally</td><td>66% F1 on listing legal moves — amenable to step-by-step reasoning</td></tr>
    <tr><td>Cannot judge legality compositionally</td><td>56% on move legality (near random 50%) — requires reasoning about blocking, pins, checks</td></tr>
    <tr><td>Can track board state after one move</td><td>87% on consequence tracking — improved dramatically with color-explicit prompts</td></tr>
    <tr><td>Color context in prompts is critical</td><td>Consequence test: 40% → 87% after adding "it is black's turn, after black plays..."</td></tr>
    </table>"""

    # Per-test detail sections
    detail_html = ""
    for t in tests:
        test_name = t["name"]
        samples = all_samples.get(test_name, [])

        detail_html += f'<div class="test-section">'
        detail_html += f'<h2>{test_name}</h2>'
        score_pct = t["score"] * 100
        detail_html += f'<p>Score: <b>{score_pct:.0f}%</b> (baseline: {t["baseline"]*100:.1f}%) | {t["num_samples"]} samples</p>'

        # Show breakdown if available
        breakdown = t["details"].get("category_breakdown") or t["details"].get("piece_breakdown") or t["details"].get("type_breakdown")
        if breakdown:
            detail_html += '<h3>Breakdown</h3><table class="breakdown-table"><tr><th>Category</th><th>Score</th></tr>'
            for cat, score in sorted(breakdown.items()):
                score_val = score * 100
                color = "#3fb950" if score_val > 70 else "#d29922" if score_val > 50 else "#f85149"
                detail_html += f'<tr><td>{cat}</td><td style="color: {color};">{score_val:.0f}%</td></tr>'
            detail_html += '</table>'

        # F1 details for legal move gen
        if "mean_precision" in t["details"]:
            d = t["details"]
            detail_html += f'<p>Precision: {d["mean_precision"]*100:.0f}% | Recall: {d["mean_recall"]*100:.0f}% | F1: {d["mean_f1"]*100:.0f}%</p>'

        # Sample browser
        if samples:
            detail_html += f'<h3>Samples ({len(samples)})</h3>'
            for j, s in enumerate(samples):
                is_correct = s.get("is_correct", s.get("correct", False))
                status = '<span class="correct">CORRECT</span>' if is_correct else '<span class="wrong">WRONG</span>'

                question = s.get("question", s.get("prompt", ""))
                if isinstance(question, str) and len(question) > 120:
                    q_short = question[:120] + "..."
                else:
                    q_short = str(question)

                expected = s.get("correct_answer", s.get("expected", ""))
                # Extract predicted answer from raw output
                raw = s.get("raw_answer", "")
                # Try to get answer after </think> or from last line
                import re as _re
                _after_think = _re.split(r'</think>', raw)
                if len(_after_think) > 1:
                    predicted = _after_think[-1].strip()[:100]
                else:
                    predicted = raw.strip()[-100:] if raw else ""
                # Truncate for display in header
                predicted_short = predicted[:60] + "..." if len(predicted) > 60 else predicted

                sid = f"{test_name.replace(' ', '_')}_{j}"

                detail_html += f"""
                <div class="sample-header" onclick="document.getElementById('{sid}').style.display = document.getElementById('{sid}').style.display === 'none' ? 'block' : 'none'">
                    {status} | Expected: <code>{html_mod.escape(str(expected))}</code> | Answer: <code>{html_mod.escape(str(predicted_short))}</code>
                </div>
                <div class="sample-detail" id="{sid}">"""

                # Board if FEN available
                fen = s.get("fen", "")
                if fen:
                    svg = render_board_svg(fen)
                    detail_html += f'<div style="display: flex; gap: 20px;"><div>{svg}</div><div>'
                    detail_html += f'<p><b>FEN:</b> <code style="font-size:11px">{html_mod.escape(fen)}</code></p>'
                else:
                    detail_html += '<div>'

                detail_html += f'<p><b>Question:</b> {html_mod.escape(str(question))}</p>'
                detail_html += f'<p><b>Expected:</b> <code>{html_mod.escape(str(expected))}</code></p>'
                detail_html += f'<p><b>Model Answer:</b> <code>{html_mod.escape(str(predicted))}</code></p>'

                if fen:
                    detail_html += '</div></div>'
                else:
                    detail_html += '</div>'

                # Full model output
                output = s.get("raw_answer", s.get("model_output", s.get("completion", "")))
                if output:
                    detail_html += f'<h4>Full Model Output</h4><div class="completion-box">{html_mod.escape(str(output))}</div>'

                detail_html += '</div>'

        detail_html += '</div>'

    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Chess LLM Diagnostic Report</title>
<style>{css}</style></head>
<body>
<h1>Chess LLM Diagnostic Report</h1>
<p>Model: <b>DeepSeek-R1-Distill-Qwen-14B</b> | 5 tests | 4096 max tokens | Color-explicit prompts</p>
<p style="color: #8b949e;">These are the corrected results after fixing scoring (final-answer-only), token budget (512→4096), move parsing, and color disambiguation. See LESSONS.md for details.</p>
{cards_html}
{summary_html}
{interp_html}
{detail_html}
</body></html>"""

    with open("diagnostics_report.html", "w") as f:
        f.write(full_html)
    print(f"Report saved to diagnostics_report.html")


if __name__ == "__main__":
    generate_report()
