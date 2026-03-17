"""Generate HTML report for diagnostic test results."""

import json

import chess
import chess.svg


def render_board_svg(fen: str, size: int = 240) -> str:
    """Render a chess board as SVG string."""
    board = chess.Board(fen)
    flipped = board.turn == chess.BLACK
    return chess.svg.board(board, flipped=flipped, size=size)


def generate_html_report(
    model_name: str,
    all_samples: list[dict],
    all_metrics: list[dict],
    total_time: float,
    output_path: str,
) -> None:
    """Generate a standalone HTML diagnostic report."""

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
    .sample-header { cursor: pointer; padding: 10px; background: #161b22; border: 1px solid #30363d;
                     border-radius: 6px; margin: 4px 0; font-family: monospace; font-size: 13px; }
    .sample-header:hover { background: #1c2128; }
    .sample-detail { display: none; padding: 16px; border: 1px solid #30363d;
                     border-top: none; border-radius: 0 0 6px 6px; margin-bottom: 8px;
                     background: #0d1117; }
    .completion-box { white-space: pre-wrap;
                      background: #161b22; color: #8b949e; padding: 12px; border-radius: 6px;
                      font-size: 12px; font-family: monospace; border: 1px solid #30363d; margin: 8px 0; }
    .metric-card { display: inline-block; background: #161b22; border: 1px solid #30363d;
                   border-radius: 8px; padding: 16px 24px; margin: 6px; text-align: center; min-width: 120px; }
    .metric-value { font-size: 28px; font-weight: bold; color: #58a6ff; }
    .metric-label { font-size: 12px; color: #8b949e; margin-top: 4px; }
    .above-baseline { color: #3fb950; }
    .at-baseline { color: #d29922; }
    .below-baseline { color: #f85149; }
    .filter-btn { background: #21262d; color: #c9d1d9; border: 1px solid #30363d;
                  padding: 6px 12px; margin: 2px; cursor: pointer; border-radius: 6px; font-size: 12px; }
    .filter-btn:hover, .filter-btn.active { background: #1f6feb; color: #fff; border-color: #1f6feb; }
    .board-and-info { display: flex; gap: 16px; align-items: flex-start; margin: 8px 0; }
    """

    # Summary table
    summary_rows = ""
    for m in all_metrics:
        acc = m.get("accuracy", m.get("mean_f1", 0))
        baseline = m.get("baseline", 0)
        acc_class = "above-baseline" if acc > baseline * 1.5 else "at-baseline" if acc > baseline else "below-baseline"
        metric_label = "F1" if "mean_f1" in m else "Accuracy"

        summary_rows += f"""<tr>
            <td>{m['test_name']}</td>
            <td class="{acc_class}">{acc:.1%}</td>
            <td>{baseline:.1%}</td>
            <td>{m['num_samples']}</td>
            <td>{metric_label}</td>
        </tr>"""

        # Extra rows for precision/recall if available
        if "mean_precision" in m:
            summary_rows += f"""<tr>
                <td style="padding-left:30px">- Precision</td>
                <td>{m['mean_precision']:.1%}</td><td>-</td><td>-</td><td></td></tr>"""
            summary_rows += f"""<tr>
                <td style="padding-left:30px">- Recall</td>
                <td>{m['mean_recall']:.1%}</td><td>-</td><td>-</td><td></td></tr>"""

    # Per-test sample sections
    test_sections = ""
    tests = {}
    for s in all_samples:
        tests.setdefault(s["test"], []).append(s)

    test_labels = {
        "fen_parsing": "Test 1: FEN Piece Identification",
        "legal_moves": "Test 2: Legal Move Generation",
        "legality": "Test 3: Move Legality Judgment",
        "consequences": "Test 4: One-Move Consequence",
        "rules_knowledge": "Test 5: Declarative Rules Knowledge",
    }

    # Pre-render board SVGs
    board_svgs = {}
    for s in all_samples:
        key = s.get("fen", "")
        if key and key not in board_svgs:
            board_svgs[key] = render_board_svg(key)

    # Build sample data for JS
    sample_data_by_test = {}
    for test_key, test_samples in tests.items():
        js_samples = []
        for i, s in enumerate(test_samples):
            entry = {
                "idx": i,
                "fen": s.get("fen", ""),
                "question": s["question"],
                "raw_answer": s.get("raw_answer", ""),
                "is_correct": s.get("is_correct", False),
                "test": test_key,
            }
            # Test-specific fields
            if test_key == "fen_parsing":
                entry["correct_answer"] = s["correct_answer"]
                entry["square"] = s["square"]
            elif test_key == "legal_moves":
                entry["correct_moves"] = s.get("correct_moves", [])
                entry["piece_type"] = s.get("piece_type", "")
                entry["square"] = s.get("square", "")
                score = s.get("score", {})
                entry["precision"] = score.get("precision", 0)
                entry["recall"] = score.get("recall", 0)
                entry["f1"] = score.get("f1", 0)
                entry["predicted_moves"] = score.get("predicted_moves", [])
            elif test_key == "legality":
                entry["move"] = s.get("move", "")
                entry["correct_answer"] = s["correct_answer"]
                entry["category"] = s.get("category", "")
            elif test_key == "consequences":
                entry["move"] = s.get("move", "")
                entry["target_square"] = s.get("target_square", "")
                entry["correct_answer"] = s["correct_answer"]
                entry["move_type"] = s.get("move_type", "")
            elif test_key == "rules_knowledge":
                entry["correct_answer"] = s["correct_answer"]
            js_samples.append(entry)
        sample_data_by_test[test_key] = js_samples

    # Build SVG lookup div
    svg_divs = ""
    for fen, svg in board_svgs.items():
        safe_id = fen.replace("/", "_").replace(" ", "__")
        svg_divs += f'<div id="svg-{safe_id}" style="display:none">{svg}</div>\n'

    sections_html = ""
    for test_key in ["fen_parsing", "legal_moves", "legality", "consequences", "rules_knowledge"]:
        if test_key not in tests:
            continue
        label = test_labels.get(test_key, test_key)
        n_correct = sum(1 for s in tests[test_key] if s.get("is_correct", False))
        n_total = len(tests[test_key])
        sections_html += f"""
        <h2>{label}</h2>
        <p>{n_correct}/{n_total} correct</p>
        <div>
            <button class="filter-btn active" onclick="filterTest('{test_key}', 'all', this)">All ({n_total})</button>
            <button class="filter-btn" onclick="filterTest('{test_key}', 'correct', this)">Correct ({n_correct})</button>
            <button class="filter-btn" onclick="filterTest('{test_key}', 'wrong', this)">Wrong ({n_total - n_correct})</button>
        </div>
        <div id="samples-{test_key}"></div>
        """

    html_content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Chess Diagnostics: {model_name}</title>
<style>{css}</style></head>
<body>
<h1>Chess LLM Diagnostic Results</h1>
<p><b>Model:</b> {model_name} | <b>Total time:</b> {total_time:.1f} min | <b>Samples:</b> {len(all_samples)}</p>

<h2>Summary</h2>
<table>
<tr><th>Test</th><th>Score</th><th>Random Baseline</th><th>Samples</th><th>Metric</th></tr>
{summary_rows}
</table>

{svg_divs}
{sections_html}

<script>
const samplesByTest = {json.dumps(sample_data_by_test)};

function getSvg(fen) {{
    const safeId = fen.replace(/\\//g, '_').replace(/ /g, '__');
    const el = document.getElementById('svg-' + safeId);
    return el ? el.innerHTML : '';
}}

function escapeHtml(str) {{
    return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}}

function toggleDetail(id) {{
    const el = document.getElementById(id);
    el.style.display = el.style.display === 'none' ? 'block' : 'none';
}}

function filterTest(testKey, mode, btn) {{
    btn.parentElement.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    renderTest(testKey, mode);
}}

function renderTest(testKey, mode) {{
    const samples = samplesByTest[testKey] || [];
    let filtered = samples;
    if (mode === 'correct') filtered = samples.filter(s => s.is_correct);
    else if (mode === 'wrong') filtered = samples.filter(s => !s.is_correct);

    const container = document.getElementById('samples-' + testKey);
    let html = '';

    filtered.forEach(s => {{
        const status = s.is_correct ? '<span class="correct">CORRECT</span>' : '<span class="wrong">WRONG</span>';
        const detailId = 'detail-' + testKey + '-' + s.idx;

        let headerExtra = '';
        if (testKey === 'fen_parsing') {{
            headerExtra = ' | Square: ' + s.square + ' | Answer: ' + s.correct_answer;
        }} else if (testKey === 'legal_moves') {{
            headerExtra = ' | ' + s.piece_type + ' on ' + s.square + ' | F1: ' + (s.f1 * 100).toFixed(0) + '%';
        }} else if (testKey === 'legality') {{
            headerExtra = ' | Move: ' + s.move + ' | ' + s.category;
        }} else if (testKey === 'consequences') {{
            headerExtra = ' | Move: ' + s.move + ' | ' + s.move_type + ' | Square: ' + s.target_square;
        }} else if (testKey === 'rules_knowledge') {{
            headerExtra = ' | Answer: ' + s.correct_answer;
        }}

        html += '<div class="sample-header" onclick="toggleDetail(\\'' + detailId + '\\')">'
            + status + ' Sample ' + s.idx + headerExtra + '</div>';
        html += '<div class="sample-detail" id="' + detailId + '">';

        // Board + info layout (skip board for rules_knowledge)
        if (testKey !== 'rules_knowledge' && s.fen) {{
            html += '<div class="board-and-info">';
            html += '<div>' + getSvg(s.fen) + '</div>';
            html += '<div>';
        }} else {{
            html += '<div>';
        }}
        if (s.fen) {{
            html += '<p><b>FEN:</b> <code style="font-size:11px">' + escapeHtml(s.fen) + '</code></p>';
        }}
        html += '<p><b>Question:</b> ' + escapeHtml(s.question) + '</p>';

        if (testKey === 'fen_parsing' || testKey === 'consequences' || testKey === 'rules_knowledge') {{
            html += '<p><b>Correct:</b> ' + s.correct_answer + '</p>';
        }} else if (testKey === 'legal_moves') {{
            html += '<p><b>Correct moves:</b> ' + s.correct_moves.join(', ') + '</p>';
            html += '<p><b>Predicted:</b> ' + (s.predicted_moves || []).join(', ') + '</p>';
            html += '<p><b>Precision:</b> ' + (s.precision * 100).toFixed(0) + '% | <b>Recall:</b> ' + (s.recall * 100).toFixed(0) + '%</p>';
        }} else if (testKey === 'legality') {{
            html += '<p><b>Correct:</b> ' + s.correct_answer + ' (' + s.category + ')</p>';
        }}

        if (testKey !== 'rules_knowledge' && s.fen) {{
            html += '</div></div>';  // close board-and-info
        }} else {{
            html += '</div>';  // close info-only
        }}
        html += '<h4>Model Output</h4>';
        html += '<div class="completion-box">' + escapeHtml(s.raw_answer || '') + '</div>';
        html += '</div>';
    }});

    container.innerHTML = html;
}}

// Initial render
['fen_parsing', 'legal_moves', 'legality', 'consequences', 'rules_knowledge'].forEach(t => renderTest(t, 'all'));
</script>
</body></html>"""

    with open(output_path, "w") as f:
        f.write(html_content)
    print(f"HTML report saved to {output_path}")
