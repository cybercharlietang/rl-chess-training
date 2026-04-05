"""Generate an interactive HTML report from evaluation results.

Shows chess boards (via chessboard.js), full reasoning traces,
and aggregate statistics.
"""

import json
import html
import sys

def load_results(path: str) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def compute_stats(results: list[dict]) -> dict:
    n = len(results)
    correct = sum(r["correct"] for r in results)
    legal = sum(r["rewards"]["legal"] > 0 for r in results)
    has_think = sum(r["rewards"]["format"] > 0 for r in results)

    buckets = {}
    for r in results:
        rating = r["puzzle_rating"]
        for lo, hi in [(200,800),(800,1200),(1200,1600),(1600,2000),(2000,2400),(2400,2800)]:
            if lo <= rating < hi:
                key = f"{lo}-{hi}"
                buckets.setdefault(key, {"correct": 0, "total": 0, "legal": 0})
                buckets[key]["total"] += 1
                buckets[key]["correct"] += r["correct"]
                buckets[key]["legal"] += (r["rewards"]["legal"] > 0)
                break

    return {
        "n": n,
        "accuracy": correct / n if n else 0,
        "legal_rate": legal / n if n else 0,
        "think_rate": has_think / n if n else 0,
        "correct": correct,
        "legal": legal,
        "buckets": buckets,
    }


def generate_html(results: list[dict], title: str = "Baseline Evaluation") -> str:
    stats = compute_stats(results)

    # Build per-sample cards
    cards_html = []
    for i, r in enumerate(results):
        fen = html.escape(r["fen"])
        solution = html.escape(r["solution_move"])
        predicted = html.escape(str(r.get("predicted_move", "None")))
        rating = r["puzzle_rating"]
        is_correct = r["correct"]
        is_legal = r["rewards"]["legal"] > 0

        # Status badge
        if is_correct:
            badge = '<span class="badge correct">CORRECT</span>'
        elif is_legal:
            badge = '<span class="badge legal">LEGAL (wrong)</span>'
        else:
            badge = '<span class="badge illegal">ILLEGAL/NONE</span>'

        # Full completion - escape HTML but preserve newlines
        completion = r.get("completion", "")
        # Split at </think> for display
        if "</think>" in completion:
            think_part = completion.split("</think>")[0]
            answer_part = completion.split("</think>")[-1].strip()
            completion_html = (
                '<div class="think-block">'
                '<div class="think-label">REASONING</div>'
                f'<pre class="think-text">{html.escape(think_part)}</pre>'
                '</div>'
                '<div class="answer-block">'
                '<div class="answer-label">ANSWER</div>'
                f'<pre class="answer-text">{html.escape(answer_part)}</pre>'
                '</div>'
            )
        else:
            completion_html = f'<pre class="completion-text">{html.escape(completion[:3000])}</pre>'

        card = f'''
        <div class="card" data-correct="{str(is_correct).lower()}" data-legal="{str(is_legal).lower()}" data-rating="{rating}">
            <div class="card-header">
                <span class="card-num">#{i+1}</span>
                {badge}
                <span class="rating">Rating: {rating}</span>
                <span class="moves">Solution: <b>{solution}</b> | Predicted: <b>{predicted}</b></span>
                <button class="toggle-btn" onclick="toggleCard(this)">Show reasoning</button>
            </div>
            <div class="card-body" style="display:none;">
                <div class="board-container">
                    <div id="board-{i}" class="chess-board"></div>
                </div>
                <div class="completion-container">
                    {completion_html}
                </div>
            </div>
            <script>
                boardConfigs.push({{ elementId: 'board-{i}', fen: '{fen}' }});
            </script>
        </div>'''
        cards_html.append(card)

    # Bucket stats rows
    bucket_rows = ""
    for key in sorted(stats["buckets"].keys()):
        b = stats["buckets"][key]
        acc = b["correct"] / b["total"] if b["total"] else 0
        leg = b["legal"] / b["total"] if b["total"] else 0
        bucket_rows += f'<tr><td>{key}</td><td>{b["total"]}</td><td>{acc:.1%}</td><td>{leg:.1%}</td></tr>\n'

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title} — Chess-GRPO</title>
<link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0d1117; color: #e6edf3; padding: 20px; }}
    h1 {{ color: #58a6ff; margin-bottom: 5px; }}
    h2 {{ color: #7ee787; margin: 20px 0 10px; }}
    .subtitle {{ color: #8b949e; margin-bottom: 20px; }}

    /* Stats grid */
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin: 20px 0; }}
    .stat-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; text-align: center; }}
    .stat-value {{ font-size: 2em; font-weight: bold; color: #58a6ff; }}
    .stat-label {{ color: #8b949e; font-size: 0.9em; margin-top: 4px; }}

    /* Table */
    table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
    th, td {{ border: 1px solid #30363d; padding: 8px 12px; text-align: left; }}
    th {{ background: #161b22; color: #58a6ff; }}
    tr:hover {{ background: #161b22; }}

    /* Filter bar */
    .filter-bar {{ margin: 15px 0; display: flex; gap: 10px; flex-wrap: wrap; }}
    .filter-btn {{ padding: 6px 14px; border-radius: 20px; border: 1px solid #30363d; background: #161b22; color: #e6edf3; cursor: pointer; font-size: 0.85em; }}
    .filter-btn.active {{ background: #1f6feb; border-color: #1f6feb; }}

    /* Cards */
    .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; margin: 8px 0; overflow: hidden; }}
    .card-header {{ display: flex; align-items: center; gap: 10px; padding: 10px 15px; flex-wrap: wrap; }}
    .card-num {{ color: #8b949e; font-weight: bold; min-width: 35px; }}
    .badge {{ padding: 3px 10px; border-radius: 12px; font-size: 0.8em; font-weight: bold; }}
    .badge.correct {{ background: #238636; color: #fff; }}
    .badge.legal {{ background: #9e6a03; color: #fff; }}
    .badge.illegal {{ background: #da3633; color: #fff; }}
    .rating {{ color: #8b949e; font-size: 0.85em; }}
    .moves {{ color: #c9d1d9; font-size: 0.85em; }}
    .toggle-btn {{ margin-left: auto; padding: 4px 12px; border-radius: 6px; border: 1px solid #30363d; background: #21262d; color: #58a6ff; cursor: pointer; font-size: 0.8em; }}
    .toggle-btn:hover {{ background: #30363d; }}

    .card-body {{ padding: 15px; display: flex; gap: 20px; flex-wrap: wrap; }}
    .board-container {{ flex: 0 0 350px; }}
    .chess-board {{ width: 350px; }}
    .completion-container {{ flex: 1; min-width: 300px; max-height: 500px; overflow-y: auto; }}

    .think-block, .answer-block {{ margin-bottom: 10px; }}
    .think-label, .answer-label {{ font-size: 0.75em; font-weight: bold; padding: 3px 8px; border-radius: 4px 4px 0 0; display: inline-block; }}
    .think-label {{ background: #1f6feb; color: #fff; }}
    .answer-label {{ background: #238636; color: #fff; }}
    .think-text, .answer-text, .completion-text {{
        background: #0d1117; border: 1px solid #30363d; border-radius: 0 6px 6px 6px;
        padding: 12px; font-size: 0.82em; white-space: pre-wrap; word-break: break-word;
        line-height: 1.5; max-height: 400px; overflow-y: auto; color: #c9d1d9;
    }}
    .answer-text {{ border-color: #238636; }}
</style>
</head>
<body>

<h1>{title}</h1>
<p class="subtitle">DeepSeek-R1-Distill-Qwen-14B — {stats["n"]} puzzles — max 8192 tokens</p>

<div class="stats-grid">
    <div class="stat-box">
        <div class="stat-value">{stats["accuracy"]:.1%}</div>
        <div class="stat-label">Puzzle Accuracy ({stats["correct"]}/{stats["n"]})</div>
    </div>
    <div class="stat-box">
        <div class="stat-value">{stats["legal_rate"]:.1%}</div>
        <div class="stat-label">Legal Move Rate ({stats["legal"]}/{stats["n"]})</div>
    </div>
    <div class="stat-box">
        <div class="stat-value">{stats["think_rate"]:.1%}</div>
        <div class="stat-label">Has Reasoning</div>
    </div>
    <div class="stat-box">
        <div class="stat-value">{stats["n"]}</div>
        <div class="stat-label">Total Puzzles</div>
    </div>
</div>

<h2>Accuracy by Rating</h2>
<table>
<tr><th>Rating Bucket</th><th>Count</th><th>Accuracy</th><th>Legal Rate</th></tr>
{bucket_rows}
</table>

<h2>All Problems</h2>
<div class="filter-bar">
    <button class="filter-btn active" onclick="filterCards('all', this)">All ({stats["n"]})</button>
    <button class="filter-btn" onclick="filterCards('correct', this)">Correct ({stats["correct"]})</button>
    <button class="filter-btn" onclick="filterCards('legal', this)">Legal but wrong ({stats["legal"] - stats["correct"]})</button>
    <button class="filter-btn" onclick="filterCards('illegal', this)">Illegal/None ({stats["n"] - stats["legal"]})</button>
</div>

<script>var boardConfigs = [];</script>
<div id="cards-container">
{"".join(cards_html)}
</div>

<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
<script>
var boards = {{}};

function toggleCard(btn) {{
    var body = btn.closest('.card').querySelector('.card-body');
    var showing = body.style.display !== 'none';
    body.style.display = showing ? 'none' : 'flex';
    btn.textContent = showing ? 'Show reasoning' : 'Hide reasoning';

    if (!showing) {{
        // Init board if not yet created
        var boardDiv = body.querySelector('.chess-board');
        if (boardDiv && !boards[boardDiv.id]) {{
            var cfg = boardConfigs.find(c => c.elementId === boardDiv.id);
            if (cfg) {{
                boards[boardDiv.id] = Chessboard(boardDiv.id, {{
                    position: cfg.fen,
                    pieceTheme: 'https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/img/chesspieces/wikipedia/{{piece}}.png'
                }});
            }}
        }}
    }}
}}

function filterCards(filter, btn) {{
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.card').forEach(card => {{
        var correct = card.dataset.correct === 'true';
        var legal = card.dataset.legal === 'true';
        var show = false;
        if (filter === 'all') show = true;
        else if (filter === 'correct') show = correct;
        else if (filter === 'legal') show = legal && !correct;
        else if (filter === 'illegal') show = !legal;
        card.style.display = show ? '' : 'none';
    }});
}}
</script>
</body>
</html>'''


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to eval results JSONL")
    parser.add_argument("--output", required=True, help="Path to output HTML")
    parser.add_argument("--title", default="Baseline Evaluation", help="Report title")
    args = parser.parse_args()

    results = load_results(args.input)
    report = generate_html(results, title=args.title)

    with open(args.output, "w") as f:
        f.write(report)
    print(f"Report saved to {args.output} ({len(results)} samples)")
