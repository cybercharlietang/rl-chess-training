"""Generate HTML reports for chess eval JSONL files (multiple schemas)."""
import json, sys, os, html as htmllib
import chess, chess.svg
sys.path.insert(0, "/workspace/rl-chess-training")
from rewards.format_reward import extract_move, has_valid_tags, is_legal_move


def board_svg(fen, move_san=None, size=300):
    board = chess.Board(fen)
    arrows = []
    if move_san:
        try:
            m = board.parse_san(move_san)
            arrows = [chess.svg.Arrow(m.from_square, m.to_square, color="#22c55e")]
        except Exception:
            pass
    return chess.svg.board(board, arrows=arrows, flipped=(board.turn == chess.BLACK), size=size)


def load_and_normalize(path, max_tokens_fallback):
    """Return (stats, samples). Samples are dicts with uniform fields."""
    with open(path) as f:
        lines = [l for l in f if l.strip()]
    first = json.loads(lines[0])
    # vllm_eval format: first line is stats dict
    if "accuracy" in first and "n" in first:
        stats = first
        rows = [json.loads(l) for l in lines[1:]]
        samples = []
        for r in rows:
            samples.append({
                "fen": r["fen"],
                "rating": r.get("rating"),
                "solution": r["solution"],
                "predicted": r.get("predicted"),
                "correct": r.get("correct", False),
                "truncated": r.get("truncated", False),
                "tokens": r.get("completion_tokens"),
                "completion": r["completion"],
            })
        return stats, samples
    # evaluate.py format: each line is a sample
    rows = [json.loads(l) for l in lines]
    samples = []
    n = len(rows)
    correct = legal = fmt = trunc = 0
    for r in rows:
        comp = r["completion"]
        sol = r["solution_move"]
        fen = r["fen"]
        pred = extract_move(comp)
        is_c = (pred == sol) if pred else False
        correct += int(is_c)
        legal += int(is_legal_move(comp, fen) > 0)
        fmt += int(has_valid_tags(comp) > 0)
        # Truncation: we don't know token count exactly, approximate by char length
        # 14B evaluate.py used 8192 max tokens → assume char-cutoff as truncation flag if no </think>
        is_trunc = ("</think>" not in comp)
        trunc += int(is_trunc)
        samples.append({
            "fen": fen, "rating": r.get("puzzle_rating"),
            "solution": sol, "predicted": pred, "correct": is_c,
            "truncated": is_trunc, "tokens": None, "completion": comp,
        })
    stats = {
        "n": n, "accuracy": correct/n, "legal_rate": legal/n,
        "format_rate": fmt/n, "truncation_rate": trunc/n,
    }
    return stats, samples


def render_html(title, stats, samples, meta=""):
    stats_html = f"""
      <tr><td>Samples</td><td>{stats['n']}</td></tr>
      <tr><td>Accuracy</td><td><b>{stats['accuracy']*100:.1f}%</b></td></tr>
      <tr><td>Legal move rate</td><td>{stats['legal_rate']*100:.1f}%</td></tr>
      <tr><td>Format (has &lt;/think&gt;)</td><td>{stats['format_rate']*100:.1f}%</td></tr>
      <tr><td>Truncation rate</td><td>{stats['truncation_rate']*100:.1f}%</td></tr>
    """
    if "max_tokens" in stats:
        stats_html += f"<tr><td>Max tokens</td><td>{stats['max_tokens']}</td></tr>"
    if "model" in stats:
        stats_html += f"<tr><td>Model</td><td>{htmllib.escape(stats['model'])}</td></tr>"
    if "wall_s" in stats:
        stats_html += f"<tr><td>Wall time</td><td>{stats['wall_s']:.1f}s</td></tr>"

    # Rating buckets
    buckets = [(200,800),(800,1200),(1200,1600),(1600,2000),(2000,2400),(2400,2800)]
    bh = "<table><tr><th>Rating</th><th>N</th><th>Accuracy</th></tr>"
    for lo, hi in buckets:
        inb = [s for s in samples if s["rating"] is not None and lo <= s["rating"] < hi]
        if not inb: continue
        acc = sum(1 for s in inb if s["correct"]) / len(inb)
        bh += f"<tr><td>{lo}-{hi}</td><td>{len(inb)}</td><td>{acc*100:.1f}%</td></tr>"
    bh += "</table>"

    rows_html = []
    for i, s in enumerate(samples):
        status_color = "#22c55e" if s["correct"] else "#ef4444"
        trunc_badge = '<span class="badge bad">TRUNCATED</span>' if s["truncated"] else '<span class="badge ok">FINISHED</span>'
        correct_badge = f'<span class="badge" style="background:{status_color};color:white">{"CORRECT" if s["correct"] else "WRONG"}</span>'
        svg = board_svg(s["fen"], s["solution"], size=260)
        comp_escaped = htmllib.escape(s["completion"])
        rows_html.append(f"""
        <div class="sample">
          <h3>#{i+1} · Rating {s['rating']} · {correct_badge} {trunc_badge}</h3>
          <div class="row">
            <div>{svg}</div>
            <div class="meta">
              <div><b>FEN:</b> <code>{htmllib.escape(s['fen'])}</code></div>
              <div><b>Solution:</b> {htmllib.escape(s['solution'])}</div>
              <div><b>Predicted:</b> {htmllib.escape(str(s['predicted']))}</div>
              <div><b>Tokens:</b> {s['tokens']}</div>
            </div>
          </div>
          <details><summary>Chain of thought ({len(s['completion'])} chars)</summary>
          <pre>{comp_escaped}</pre></details>
        </div>""")

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{htmllib.escape(title)}</title>
<style>
  body{{font-family:system-ui,sans-serif;max-width:1100px;margin:20px auto;padding:0 20px;background:#fafafa;color:#1a1a1a}}
  h1{{color:#1e40af}} h2{{color:#4338ca;margin-top:32px}}
  table{{border-collapse:collapse;margin:8px 0}} td,th{{padding:6px 14px;border:1px solid #ddd}}
  th{{background:#e0e7ff}}
  .sample{{background:white;border:1px solid #e5e7eb;border-radius:8px;padding:14px;margin:14px 0}}
  .row{{display:flex;gap:20px;align-items:flex-start}}
  .meta{{font-family:monospace;font-size:13px;line-height:1.8}}
  .meta code{{background:#f3f4f6;padding:2px 6px;border-radius:3px}}
  pre{{background:#0f172a;color:#e2e8f0;padding:14px;border-radius:6px;overflow-x:auto;white-space:pre-wrap;word-break:break-word;max-height:500px;overflow-y:auto}}
  .badge{{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600}}
  .badge.ok{{background:#d1fae5;color:#065f46}}
  .badge.bad{{background:#fee2e2;color:#991b1b}}
  details summary{{cursor:pointer;font-weight:600;margin:6px 0}}
</style></head><body>
<h1>{htmllib.escape(title)}</h1>
<p>{htmllib.escape(meta)}</p>
<h2>Summary</h2><table>{stats_html}</table>
<h2>Accuracy by rating bucket</h2>{bh}
<h2>Per-sample details ({len(samples)} puzzles)</h2>
{''.join(rows_html)}
</body></html>"""


def main():
    jobs = [
        ("outputs/baseline_eval.jsonl", "report_14b_baseline.html",
         "DeepSeek-R1-Distill-Qwen-14B — Baseline (100 puzzles, 8192 tokens)"),
        ("outputs/qwen3_4b_baseline_think_8k.jsonl", "report_qwen3_4b_8k.html",
         "Qwen3-4B thinking=True — Baseline (100 puzzles, 8192 tokens)"),
        ("outputs/qwen3_4b_baseline_think_16k.jsonl", "report_qwen3_4b_16k.html",
         "Qwen3-4B thinking=True — Baseline (100 puzzles, 16384 tokens)"),
    ]
    for path, out, title in jobs:
        full = f"/workspace/rl-chess-training/{path}"
        if not os.path.exists(full):
            print(f"MISSING: {full}")
            continue
        stats, samples = load_and_normalize(full, max_tokens_fallback=8192)
        html_str = render_html(title, stats, samples)
        out_path = f"/workspace/rl-chess-training/{out}"
        with open(out_path, "w") as f:
            f.write(html_str)
        print(f"Wrote {out_path}  ({len(samples)} samples, acc {stats['accuracy']*100:.1f}%)")

if __name__ == "__main__":
    main()
