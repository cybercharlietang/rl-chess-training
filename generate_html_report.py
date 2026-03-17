"""Generate a standalone HTML comparison report for baseline evaluations."""

import base64
import html
import json
import sys

import chess
import chess.svg


def render_board_svg(fen, move_san=None, size=280):
    """Render a chess board as SVG string."""
    board = chess.Board(fen)
    arrows = []
    if move_san:
        try:
            move = board.parse_san(move_san)
            arrows = [chess.svg.Arrow(move.from_square, move.to_square, color="#22c55e")]
        except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
            pass
    flipped = board.turn == chess.BLACK
    return chess.svg.board(board, arrows=arrows, flipped=flipped, size=size)


def generate_report(baseline_path, output_path, baseline_label="DeepSeek-R1-Distill-Qwen-14B",
                    trained_path=None, trained_label="After GRPO", training_log=None):
    """Generate HTML report from eval result JSONL files."""

    baseline = [json.loads(l) for l in open(baseline_path)]

    def compute_stats(results):
        n = len(results)
        correct = sum(1 for r in results if r.get("correct", False))
        legal = sum(1 for r in results if r.get("rewards", {}).get("legal", 0) > 0)
        fmt = sum(1 for r in results if r.get("rewards", {}).get("format", 0) > 0)
        finished = sum(1 for r in results if "</answer>" in r.get("completion", ""))
        avg_len = sum(len(r.get("completion", "")) for r in results) / n if n else 0
        return {
            "n": n, "correct": correct, "legal": legal, "format": fmt,
            "finished": finished, "avg_len": avg_len,
        }

    b_stats = compute_stats(baseline)
    trained = None
    t_stats = None
    if trained_path:
        trained = [json.loads(l) for l in open(trained_path)]
        t_stats = compute_stats(trained)

    # Build HTML
    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           max-width: 1200px; margin: 0 auto; padding: 20px; background: #0d1117; color: #c9d1d9; }
    h1, h2, h3 { color: #58a6ff; }
    table { border-collapse: collapse; width: 100%; margin: 16px 0; }
    th, td { border: 1px solid #30363d; padding: 10px 14px; text-align: left; }
    th { background: #161b22; color: #58a6ff; }
    tr:nth-child(even) { background: #161b22; }
    tr:hover { background: #1c2128; }
    .correct { color: #3fb950; font-weight: bold; }
    .wrong { color: #f85149; font-weight: bold; }
    .finished { color: #3fb950; }
    .truncated { color: #d29922; }
    .filter-btn { background: #21262d; color: #c9d1d9; border: 1px solid #30363d;
                  padding: 6px 14px; margin: 3px; cursor: pointer; border-radius: 6px; }
    .filter-btn:hover, .filter-btn.active { background: #1f6feb; color: #fff; border-color: #1f6feb; }
    .completion-box { white-space: pre-wrap; max-height: 400px; overflow-y: auto;
                      background: #161b22; color: #8b949e; padding: 12px; border-radius: 6px;
                      font-size: 12px; font-family: monospace; margin: 8px 0; border: 1px solid #30363d; }
    .sample-header { cursor: pointer; padding: 10px; background: #161b22; border: 1px solid #30363d;
                     border-radius: 6px; margin: 4px 0; }
    .sample-header:hover { background: #1c2128; }
    .sample-detail { display: none; padding: 12px; border: 1px solid #30363d;
                     border-top: none; border-radius: 0 0 6px 6px; margin-bottom: 8px; }
    .metric-card { display: inline-block; background: #161b22; border: 1px solid #30363d;
                   border-radius: 8px; padding: 16px 24px; margin: 6px; text-align: center; min-width: 140px; }
    .metric-value { font-size: 28px; font-weight: bold; color: #58a6ff; }
    .metric-label { font-size: 13px; color: #8b949e; margin-top: 4px; }
    .change-pos { color: #3fb950; }
    .change-neg { color: #f85149; }
    """

    # Summary stats section
    def pct(count, total):
        return f"{count/total*100:.0f}%" if total else "0%"

    summary_html = "<h2>Summary Statistics</h2>"

    if t_stats:
        summary_html += f"""
        <table>
        <tr><th>Metric</th><th>{baseline_label}</th><th>{trained_label}</th><th>Change</th></tr>
        """
        metrics = [
            ("Accuracy", "correct"), ("Legal Move Rate", "legal"),
            ("Format Compliance", "format"), ("Finished Reasoning", "finished"),
        ]
        for label, key in metrics:
            b_pct = b_stats[key] / b_stats["n"] * 100
            t_pct = t_stats[key] / t_stats["n"] * 100
            diff = t_pct - b_pct
            change_class = "change-pos" if diff > 0 else "change-neg" if diff < 0 else ""
            summary_html += f"""<tr>
                <td>{label}</td><td>{b_pct:.0f}%</td><td>{t_pct:.0f}%</td>
                <td class="{change_class}">{diff:+.0f}%</td></tr>"""
        summary_html += f"""<tr><td>Avg Completion Length</td>
            <td>{b_stats['avg_len']:.0f} chars</td><td>{t_stats['avg_len']:.0f} chars</td><td>-</td></tr>"""
        summary_html += "</table>"
    else:
        summary_html += """<div style="display: flex; flex-wrap: wrap;">"""
        for label, key in [("Accuracy", "correct"), ("Legal Move Rate", "legal"),
                           ("Format Compliance", "format"), ("Finished Reasoning", "finished")]:
            val = pct(b_stats[key], b_stats["n"])
            summary_html += f"""<div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div></div>"""
        summary_html += f"""<div class="metric-card">
            <div class="metric-value">{b_stats['avg_len']:.0f}</div>
            <div class="metric-label">Avg Completion Length (chars)</div></div>"""
        summary_html += "</div>"

    # Truncation cross-tab (only if trained)
    crosstab_html = ""
    if trained:
        n_pairs = min(len(baseline), len(trained))
        bf_tf = bf_tt = bt_tf = bt_tt = 0
        for i in range(n_pairs):
            b_fin = "</answer>" in baseline[i].get("completion", "")
            t_fin = "</answer>" in trained[i].get("completion", "")
            if b_fin and t_fin: bf_tf += 1
            elif b_fin and not t_fin: bf_tt += 1
            elif not b_fin and t_fin: bt_tf += 1
            else: bt_tt += 1
        crosstab_html = f"""
        <h2>Truncation Cross-tab</h2>
        <table style="width: auto;">
        <tr><th></th><th>Trained Finished</th><th>Trained Truncated</th></tr>
        <tr><td><b>Baseline Finished</b></td><td>{bf_tf}</td><td>{bf_tt}</td></tr>
        <tr><td><b>Baseline Truncated</b></td><td>{bt_tf}</td><td>{bt_tt}</td></tr>
        </table>"""

    # Training curve (if log provided)
    training_html = ""
    if training_log:
        training_html = "<h2>Training Curve</h2><p><i>Training log integration coming soon.</i></p>"

    # Accuracy by rating bucket
    buckets = [(200, 800), (800, 1200), (1200, 1600), (1600, 2000), (2000, 2400), (2400, 2800)]
    def bucket_acc(results):
        out = {}
        for lo, hi in buckets:
            in_b = [r for r in results if lo <= r.get("puzzle_rating", 0) < hi]
            out[f"{lo}-{hi}"] = (sum(1 for r in in_b if r.get("correct")) / len(in_b) * 100) if in_b else 0
        return out

    b_buck = bucket_acc(baseline)
    rating_html = "<h2>Accuracy by Rating Bucket</h2><table><tr><th>Rating</th><th>Baseline</th>"
    if trained:
        t_buck = bucket_acc(trained)
        rating_html += f"<th>{trained_label}</th>"
    rating_html += "</tr>"
    for k in b_buck:
        rating_html += f"<tr><td>{k}</td><td>{b_buck[k]:.1f}%</td>"
        if trained:
            rating_html += f"<td>{t_buck[k]:.1f}%</td>"
        rating_html += "</tr>"
    rating_html += "</table>"

    # Sample browser
    samples_html = "<h2>Side-by-Side Comparison</h2>" if trained else "<h2>Sample Browser</h2>"

    # Pre-render board SVGs and build sample data
    sample_data = []
    board_svgs = {}  # idx -> svg string (kept separate to avoid JSON bloat)
    for i, b in enumerate(baseline):
        b_fin = "</answer>" in b.get("completion", "")
        fen = b.get("fen", "")
        board_svgs[i] = render_board_svg(fen, b["solution_move"])
        entry = {
            "idx": i,
            "rating": b.get("puzzle_rating", 0),
            "solution": b["solution_move"],
            "b_predicted": b.get("predicted_move") or "(none)",
            "b_correct": b.get("correct", False),
            "b_finished": b_fin,
            "b_completion": b.get("completion", ""),
            "fen": fen,
        }
        if trained and i < len(trained):
            t = trained[i]
            t_fin = "</answer>" in t.get("completion", "")
            entry["t_predicted"] = t.get("predicted_move") or "(none)"
            entry["t_correct"] = t.get("correct", False)
            entry["t_finished"] = t_fin
            entry["t_completion"] = t.get("completion", "")
        sample_data.append(entry)

    has_trained_js = "true" if trained else "false"

    # Embed board SVGs as hidden divs
    svg_divs = ""
    for idx, svg in board_svgs.items():
        svg_divs += f'<div id="board-svg-{idx}" style="display:none">{svg}</div>\n'

    samples_html += f"""
    {svg_divs}
    <div id="filters">
        <button class="filter-btn active" onclick="filterSamples('all')">All ({len(baseline)})</button>
        <button class="filter-btn" onclick="filterSamples('finished')">Finished</button>
        <button class="filter-btn" onclick="filterSamples('truncated')">Truncated</button>
        <button class="filter-btn" onclick="filterSamples('correct')">Correct</button>
        <button class="filter-btn" onclick="filterSamples('incorrect')">Incorrect</button>
    </div>
    <div id="sample-list"></div>

    <script>
    const samples = {json.dumps(sample_data)};
    const hasTrained = {has_trained_js};

    function filterSamples(mode) {{
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        event.target.classList.add('active');

        let filtered = samples;
        if (mode === 'finished') filtered = samples.filter(s => s.b_finished);
        else if (mode === 'truncated') filtered = samples.filter(s => !s.b_finished);
        else if (mode === 'correct') filtered = samples.filter(s => s.b_correct);
        else if (mode === 'incorrect') filtered = samples.filter(s => !s.b_correct);

        renderSamples(filtered);
    }}

    function escapeHtml(str) {{
        return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }}

    function toggleDetail(idx) {{
        const el = document.getElementById('detail-' + idx);
        el.style.display = el.style.display === 'none' ? 'block' : 'none';
    }}

    function getBoardSvg(idx) {{
        const el = document.getElementById('board-svg-' + idx);
        return el ? el.innerHTML : '';
    }}

    function renderSamples(list) {{
        const container = document.getElementById('sample-list');
        let html = '<p>' + list.length + ' samples</p>';
        list.forEach(s => {{
            const bStatus = s.b_correct ? '<span class="correct">CORRECT</span>' : '<span class="wrong">WRONG</span>';
            const bTrunc = s.b_finished ? '<span class="finished">FINISHED</span>' : '<span class="truncated">TRUNCATED</span>';

            let label = 'Sample ' + s.idx + ' | Rating: ' + s.rating + ' | Solution: ' + s.solution + ' | ' + bStatus + ' ' + bTrunc;

            if (hasTrained && s.t_predicted !== undefined) {{
                const tStatus = s.t_correct ? '<span class="correct">CORRECT</span>' : '<span class="wrong">WRONG</span>';
                const tTrunc = s.t_finished ? '<span class="finished">FINISHED</span>' : '<span class="truncated">TRUNCATED</span>';
                label += ' -> ' + tStatus + ' ' + tTrunc;
            }}

            html += '<div class="sample-header" onclick="toggleDetail(' + s.idx + ')">' + label + '</div>';
            html += '<div class="sample-detail" id="detail-' + s.idx + '">';

            // Board + info layout
            html += '<div style="display: flex; gap: 20px; align-items: flex-start; margin-bottom: 12px;">';
            html += '<div style="flex-shrink: 0;">' + getBoardSvg(s.idx) + '</div>';
            html += '<div>';
            html += '<p><b>FEN:</b> <code style="font-size:11px">' + escapeHtml(s.fen) + '</code></p>';
            html += '<p><b>Solution:</b> <code>' + s.solution + '</code></p>';
            html += '<p><b>Predicted:</b> <code>' + escapeHtml(s.b_predicted) + '</code></p>';
            html += '<p><b>Rating:</b> ' + s.rating + '</p>';
            html += '</div></div>';

            html += '<h4>Model Output</h4>';
            html += '<div class="completion-box">' + escapeHtml(s.b_completion) + '</div>';

            if (hasTrained && s.t_completion !== undefined) {{
                html += '<h4>Trained Output</h4>';
                html += '<p><b>Predicted:</b> <code>' + escapeHtml(s.t_predicted) + '</code></p>';
                html += '<div class="completion-box">' + escapeHtml(s.t_completion) + '</div>';
            }}

            html += '</div>';
        }});
        container.innerHTML = html;
    }}

    renderSamples(samples);
    </script>
    """

    # Assemble full HTML
    title = f"Baseline Evaluation: {baseline_label}" if not trained else f"GRPO Results: {baseline_label}"
    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>{css}</style></head>
<body>
<h1>{title}</h1>
<p>Evaluated on <b>{b_stats['n']}</b> puzzles | Max completion: 4096 tokens</p>
{summary_html}
{rating_html}
{crosstab_html}
{training_html}
{samples_html}
</body></html>"""

    with open(output_path, "w") as f:
        f.write(full_html)
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--trained", default=None)
    parser.add_argument("--baseline_label", default="Baseline")
    parser.add_argument("--trained_label", default="After GRPO")
    parser.add_argument("--output", default="comparison.html")
    args = parser.parse_args()
    generate_report(args.baseline, args.output, args.baseline_label, args.trained, args.trained_label)
