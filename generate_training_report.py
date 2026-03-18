"""Generate HTML report for GRPO training results."""

import html
import json
import re
import sys

import chess
import chess.svg
import pandas as pd


def parse_training_log(log_path):
    """Extract training metrics from the log file."""
    metrics = []
    with open(log_path) as f:
        content = f.read()

    # Find all metric dictionaries logged by TRL
    pattern = r"\{[^{}]*'reward':[^{}]*\}"
    matches = re.findall(pattern, content)

    for match in matches:
        # Convert single quotes to double quotes for JSON parsing
        m = match.replace("'", '"')
        try:
            d = json.loads(m)
            # Convert string values to float where possible
            for k, v in d.items():
                if isinstance(v, str):
                    try:
                        d[k] = float(v)
                    except ValueError:
                        pass
            metrics.append(d)
        except json.JSONDecodeError:
            continue

    return metrics


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


def extract_answer(text):
    """Extract move from <answer> tags."""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def generate_report(log_path, completions_dir, output_path):
    """Generate training report HTML."""

    # Parse training metrics
    metrics = parse_training_log(log_path)

    # Load completions
    completions = {}
    for step in [10, 20]:
        path = f"{completions_dir}/completions_000{step}.parquet"
        try:
            df = pd.read_parquet(path)
            completions[step] = df
        except FileNotFoundError:
            pass

    # Prior baseline numbers (from BACKGROUND.md, 14B at 4096 tokens)
    baseline_4096 = {
        "accuracy": 5, "legal_move_rate": 51,
        "format_compliance": 51, "finished_reasoning": 62,
    }

    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           max-width: 1200px; margin: 0 auto; padding: 20px; background: #0d1117; color: #c9d1d9; }
    h1, h2, h3 { color: #58a6ff; }
    table { border-collapse: collapse; width: 100%; margin: 16px 0; }
    th, td { border: 1px solid #30363d; padding: 10px 14px; text-align: left; }
    th { background: #161b22; color: #58a6ff; }
    tr:nth-child(even) { background: #161b22; }
    .metric-card { display: inline-block; background: #161b22; border: 1px solid #30363d;
                   border-radius: 8px; padding: 16px 24px; margin: 6px; text-align: center; min-width: 140px; }
    .metric-value { font-size: 28px; font-weight: bold; color: #58a6ff; }
    .metric-label { font-size: 13px; color: #8b949e; margin-top: 4px; }
    .metric-sub { font-size: 11px; color: #8b949e; }
    .change-pos { color: #3fb950; font-weight: bold; }
    .change-neg { color: #f85149; font-weight: bold; }
    .neutral { color: #8b949e; }
    .completion-box { white-space: pre-wrap; max-height: 500px; overflow-y: auto;
                      background: #161b22; color: #8b949e; padding: 12px; border-radius: 6px;
                      font-size: 12px; font-family: monospace; margin: 8px 0; border: 1px solid #30363d; }
    .sample-header { cursor: pointer; padding: 10px; background: #161b22; border: 1px solid #30363d;
                     border-radius: 6px; margin: 4px 0; }
    .sample-header:hover { background: #1c2128; }
    .sample-detail { display: none; padding: 12px; border: 1px solid #30363d;
                     border-top: none; border-radius: 0 0 6px 6px; margin-bottom: 8px; }
    .correct { color: #3fb950; font-weight: bold; }
    .wrong { color: #f85149; }
    canvas { background: #161b22; border: 1px solid #30363d; border-radius: 6px; }
    .config-box { background: #161b22; border: 1px solid #30363d; border-radius: 6px;
                  padding: 16px; font-family: monospace; font-size: 13px; margin: 16px 0; }
    """

    # Config section
    config_html = """
    <h2>Training Configuration</h2>
    <div class="config-box">
    Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B (LoRA rank 64)<br>
    Training data: Easy puzzles only (&lt;1200 rating, 2043 samples)<br>
    Reward mode: Dense (Stockfish depth 15)<br>
    Batch size: 4 | Num generations: 2 | Max tokens: 8192<br>
    Learning rate: 1e-5 | Steps: 20 | Total time: 3.6 hours<br>
    Reward weights: move=1.0, format=0.5, legal=0.5
    </div>
    """

    # Training metrics table
    metrics_html = "<h2>Training Metrics</h2>"
    if metrics:
        metrics_html += """<table>
        <tr><th>Metric</th><th>Step 10</th><th>Step 20</th><th>Trend</th></tr>"""

        rows = [
            ("Total Reward", "reward", ""),
            ("Move Reward (Dense Stockfish)", "rewards/reward_fn/mean", ""),
            ("Format Compliance", "rewards/format_reward_fn/mean", ""),
            ("Legal Move Rate", "rewards/legal_move_reward_fn/mean", ""),
            ("Avg Completion Length (tokens)", "completions/mean_length", ""),
            ("Max Completion Length", "completions/max_length", ""),
            ("Loss", "loss", ""),
            ("Entropy", "entropy", ""),
            ("Grad Norm", "grad_norm", ""),
        ]

        for label, key, fmt in rows:
            vals = [m.get(key, "N/A") for m in metrics]
            if len(vals) >= 2:
                v0, v1 = vals[0], vals[-1]
                if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
                    diff = v1 - v0
                    trend_class = "change-pos" if diff > 0 else "change-neg" if diff < 0 else "neutral"
                    # For loss and entropy, lower is not necessarily bad
                    if key in ("loss", "entropy", "grad_norm"):
                        trend_class = "neutral"
                    trend = f'<span class="{trend_class}">{diff:+.4f}</span>'
                    metrics_html += f"<tr><td>{label}</td><td>{v0:.4f}</td><td>{v1:.4f}</td><td>{trend}</td></tr>"
                else:
                    metrics_html += f"<tr><td>{label}</td><td>{v0}</td><td>{v1}</td><td>-</td></tr>"

        metrics_html += "</table>"

    # Comparison with prior baseline
    if metrics:
        last = metrics[-1]
        metrics_html += """<h2>Comparison with Prior 14B Baseline (4096 tokens)</h2>
        <table>
        <tr><th>Metric</th><th>14B Baseline (4096 tok)</th><th>After 20 GRPO Steps (8192 tok)</th><th>Change</th></tr>"""

        comparisons = [
            ("Format Compliance", baseline_4096["format_compliance"], last.get("rewards/format_reward_fn/mean", 0) * 100),
            ("Legal Move Rate", baseline_4096["legal_move_rate"], last.get("rewards/legal_move_reward_fn/mean", 0) * 100),
        ]
        for label, base_val, trained_val in comparisons:
            diff = trained_val - base_val
            cls = "change-pos" if diff > 0 else "change-neg"
            metrics_html += f"<tr><td>{label}</td><td>{base_val:.0f}%</td><td>{trained_val:.0f}%</td><td class='{cls}'>{diff:+.0f}%</td></tr>"

        metrics_html += "</table>"
        metrics_html += "<p class='metric-sub'>Note: Baseline was evaluated on mixed-difficulty puzzles. Training used only easy puzzles (&lt;1200 rating). Direct comparison is indicative, not apples-to-apples.</p>"

    # Training curve chart
    chart_html = ""
    if len(metrics) >= 2:
        steps_json = json.dumps([10, 20])
        reward_json = json.dumps([m.get("reward", 0) for m in metrics])
        move_json = json.dumps([m.get("rewards/reward_fn/mean", 0) for m in metrics])
        fmt_json = json.dumps([m.get("rewards/format_reward_fn/mean", 0) for m in metrics])
        legal_json = json.dumps([m.get("rewards/legal_move_reward_fn/mean", 0) for m in metrics])
        length_json = json.dumps([m.get("completions/mean_length", 0) for m in metrics])

        chart_html = f"""
        <h2>Training Curves</h2>
        <div style="display: flex; flex-wrap: wrap; gap: 20px;">
        <div>
            <h3>Rewards</h3>
            <canvas id="rewardChart" width="500" height="300"></canvas>
        </div>
        <div>
            <h3>Completion Length</h3>
            <canvas id="lengthChart" width="500" height="300"></canvas>
        </div>
        </div>
        <script>
        function drawChart(canvasId, datasets, yLabel) {{
            const canvas = document.getElementById(canvasId);
            const ctx = canvas.getContext('2d');
            const W = canvas.width, H = canvas.height;
            const pad = {{left: 60, right: 20, top: 20, bottom: 40}};
            const plotW = W - pad.left - pad.right;
            const plotH = H - pad.top - pad.bottom;

            // Find y range
            let yMin = Infinity, yMax = -Infinity;
            datasets.forEach(ds => ds.data.forEach(v => {{ yMin = Math.min(yMin, v); yMax = Math.max(yMax, v); }}));
            const yPad = (yMax - yMin) * 0.1 || 0.1;
            yMin -= yPad; yMax += yPad;

            const steps = {steps_json};

            // Axes
            ctx.strokeStyle = '#30363d';
            ctx.beginPath();
            ctx.moveTo(pad.left, pad.top);
            ctx.lineTo(pad.left, H - pad.bottom);
            ctx.lineTo(W - pad.right, H - pad.bottom);
            ctx.stroke();

            // Y labels
            ctx.fillStyle = '#8b949e';
            ctx.font = '11px monospace';
            ctx.textAlign = 'right';
            for (let i = 0; i <= 4; i++) {{
                const y = pad.top + plotH * (1 - i/4);
                const val = yMin + (yMax - yMin) * i/4;
                ctx.fillText(val.toFixed(2), pad.left - 8, y + 4);
                ctx.strokeStyle = '#21262d';
                ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
            }}

            // X labels
            ctx.textAlign = 'center';
            steps.forEach((s, i) => {{
                const x = pad.left + plotW * i / (steps.length - 1);
                ctx.fillText('Step ' + s, x, H - pad.bottom + 20);
            }});

            // Lines
            datasets.forEach(ds => {{
                ctx.strokeStyle = ds.color;
                ctx.lineWidth = 2;
                ctx.beginPath();
                ds.data.forEach((v, i) => {{
                    const x = pad.left + plotW * i / (steps.length - 1);
                    const y = pad.top + plotH * (1 - (v - yMin) / (yMax - yMin));
                    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                }});
                ctx.stroke();

                // Points
                ds.data.forEach((v, i) => {{
                    const x = pad.left + plotW * i / (steps.length - 1);
                    const y = pad.top + plotH * (1 - (v - yMin) / (yMax - yMin));
                    ctx.fillStyle = ds.color;
                    ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI*2); ctx.fill();
                }});
            }});

            // Legend
            let lx = pad.left + 10;
            datasets.forEach(ds => {{
                ctx.fillStyle = ds.color;
                ctx.fillRect(lx, pad.top + 5, 12, 12);
                ctx.fillStyle = '#c9d1d9';
                ctx.textAlign = 'left';
                ctx.fillText(ds.label, lx + 16, pad.top + 15);
                lx += ctx.measureText(ds.label).width + 36;
            }});
        }}

        drawChart('rewardChart', [
            {{label: 'Total Reward', data: {reward_json}, color: '#58a6ff'}},
            {{label: 'Move Reward', data: {move_json}, color: '#f85149'}},
            {{label: 'Format', data: {fmt_json}, color: '#3fb950'}},
            {{label: 'Legal', data: {legal_json}, color: '#d29922'}},
        ], 'Reward');

        drawChart('lengthChart', [
            {{label: 'Mean Length', data: {length_json}, color: '#bc8cff'}},
        ], 'Tokens');
        </script>
        """

    # Sample completions
    samples_html = "<h2>Sample Completions (Step 20)</h2>"
    if 20 in completions:
        df = completions[20]
        samples_html += f"<p>{len(df)} completions logged at step 20</p>"

        for i, row in df.iterrows():
            prompt_text = row.get("prompt", "")
            if isinstance(prompt_text, list):
                # Extract FEN and solution from chat messages
                user_msg = [m for m in prompt_text if m.get("role") == "user"]
                prompt_display = user_msg[0]["content"][:200] + "..." if user_msg else str(prompt_text)[:200]
            else:
                prompt_display = str(prompt_text)[:200]

            completion_text = row.get("completion", "")
            if isinstance(completion_text, list):
                completion_text = completion_text[-1].get("content", "") if completion_text else ""

            move_reward = row.get("reward_fn", "N/A")
            fmt_reward = row.get("format_reward_fn", "N/A")
            legal_reward = row.get("legal_move_reward_fn", "N/A")
            predicted = extract_answer(completion_text)

            # Try to extract FEN from prompt
            fen_match = re.search(r"FEN is:?\s*`?([rnbqkpRNBQKP1-8/]+ [wb] [KQkq-]+ [a-h1-8-]+ \d+ \d+)", str(prompt_text))
            board_html = ""
            if fen_match:
                try:
                    board_html = render_board_svg(fen_match.group(1), predicted)
                except Exception:
                    pass

            reward_color = "correct" if isinstance(move_reward, (int, float)) and move_reward > 0 else "wrong"

            samples_html += f"""
            <div class="sample-header" onclick="document.getElementById('sample-{i}').style.display = document.getElementById('sample-{i}').style.display === 'none' ? 'block' : 'none'">
                Sample {i} | Move: <span class="{reward_color}">{move_reward:.3f}</span> | Format: {fmt_reward:.3f} | Legal: {legal_reward:.3f} | Predicted: <code>{html.escape(str(predicted or '(none)'))}</code>
            </div>
            <div class="sample-detail" id="sample-{i}">
                <div style="display: flex; gap: 20px; align-items: flex-start;">
                    <div style="flex-shrink: 0;">{board_html}</div>
                    <div>
                        <p><b>Prompt:</b></p>
                        <div class="completion-box" style="max-height: 150px;">{html.escape(str(prompt_display))}</div>
                        <p><b>Predicted Move:</b> <code>{html.escape(str(predicted or '(none)'))}</code></p>
                        <p><b>Rewards:</b> Move={move_reward:.3f}, Format={fmt_reward:.3f}, Legal={legal_reward:.3f}</p>
                    </div>
                </div>
                <h4>Full Completion</h4>
                <div class="completion-box">{html.escape(completion_text)}</div>
            </div>
            """

    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>GRPO Training Report - Easy Puzzles</title>
<style>{css}</style></head>
<body>
<h1>GRPO Training Report: 14B on Easy Puzzles</h1>
<p>DeepSeek-R1-Distill-Qwen-14B | 20 GRPO steps | Easy puzzles (&lt;1200 rating) | 8192 max tokens</p>
{config_html}
{metrics_html}
{chart_html}
{samples_html}
</body></html>"""

    with open(output_path, "w") as f:
        f.write(full_html)
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    generate_report(
        log_path="outputs/grpo_easy.log",
        completions_dir="outputs/grpo_easy/completions",
        output_path="grpo_easy_report.html",
    )
