"""Streamlit app for inspecting chess puzzle data and model outputs."""

import json
import os

import chess
import chess.svg
import streamlit as st

# ── Data loading ─────────────────────────────────────────────────────────


@st.cache_data
def load_jsonl(path: str) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


@st.cache_data
def load_results(path: str) -> list[dict] | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── Board rendering ──────────────────────────────────────────────────────


def render_board(fen: str, last_move_san: str | None = None, size: int = 350) -> str:
    """Render a chess board as SVG from a FEN string."""
    board = chess.Board(fen)

    # Highlight the solution/predicted move if provided
    arrows = []
    if last_move_san:
        try:
            move = board.parse_san(last_move_san)
            arrows = [chess.svg.Arrow(move.from_square, move.to_square, color="#22c55e")]
        except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
            pass

    # Orient board from the perspective of the side to move
    flipped = board.turn == chess.BLACK

    svg = chess.svg.board(board, arrows=arrows, flipped=flipped, size=size)
    return svg


# ── Helpers ───────────────────────────────────────────────────────────────


def _std(values: list[float]) -> float:
    """Standard deviation of a list of floats."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def _histogram(values: list[float], bins: int = 20) -> dict[str, int]:
    """Simple histogram: bin labels → counts."""
    if not values:
        return {}
    lo, hi = min(values), max(values)
    if lo == hi:
        return {f"{lo:.2f}": len(values)}
    width = (hi - lo) / bins
    counts: dict[str, int] = {}
    for v in values:
        idx = min(int((v - lo) / width), bins - 1)
        label = f"{lo + idx * width:.2f}"
        counts[label] = counts.get(label, 0) + 1
    return counts


# ── Pages ────────────────────────────────────────────────────────────────


def page_data_explorer():
    """Browse raw puzzle data."""
    st.header("Puzzle Data Explorer")

    # File selection
    data_dir = "data"
    available = [f for f in os.listdir(data_dir) if f.endswith(".jsonl")]
    if not available:
        st.warning("No JSONL files found in data/. Run preprocessing first.")
        return

    selected_file = st.selectbox("Dataset", available)
    samples = load_jsonl(os.path.join(data_dir, selected_file))

    # Stats
    st.subheader("Dataset Statistics")
    ratings = [s["puzzle_rating"] for s in samples]
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", len(samples))
    col2.metric("Rating Range", f"{min(ratings)}-{max(ratings)}")
    col3.metric("Avg Legal Moves", f"{sum(len(s['legal_moves']) for s in samples) / len(samples):.1f}")

    # Rating distribution
    st.subheader("Rating Distribution")
    st.bar_chart({r: ratings.count(r) for r in sorted(set(ratings))})

    # Browse individual puzzles
    st.subheader("Browse Puzzles")

    # Filters
    col1, col2 = st.columns(2)
    rating_range = col1.slider("Rating Range", min(ratings), max(ratings),
                                (min(ratings), max(ratings)))
    filtered = [s for s in samples
                if rating_range[0] <= s["puzzle_rating"] <= rating_range[1]]
    col2.write(f"{len(filtered)} puzzles in range")

    if not filtered:
        return

    idx = st.number_input("Puzzle index", 0, len(filtered) - 1, 0)
    sample = filtered[idx]

    col_board, col_info = st.columns([1, 1])

    with col_board:
        svg = render_board(sample["fen"], sample["solution_move"])
        st.markdown(svg, unsafe_allow_html=True)

    with col_info:
        st.markdown(f"**Puzzle ID:** `{sample.get('puzzle_id', 'N/A')}`")
        st.markdown(f"**Rating:** {sample['puzzle_rating']}")
        board = chess.Board(sample["fen"])
        side = "White" if board.turn == chess.WHITE else "Black"
        st.markdown(f"**Side to move:** {side}")
        st.markdown(f"**FEN:** `{sample['fen']}`")
        st.markdown(f"**Solution:** `{sample['solution_move']}`")
        st.markdown(f"**Legal moves ({len(sample['legal_moves'])}):**")
        st.code(", ".join(sample["legal_moves"]))


def page_model_outputs():
    """Inspect model completions and reward breakdowns."""
    st.header("Model Output Inspector")

    results_path = st.text_input("Results JSONL path", "outputs/eval_results.jsonl")

    if not os.path.exists(results_path):
        st.info(f"No results file at `{results_path}`. Run evaluation first to generate model outputs.")
        st.markdown("**Expected format** (one JSON per line):")
        st.code(json.dumps({
            "fen": "...",
            "legal_moves": ["e4", "d4"],
            "solution_move": "e4",
            "puzzle_rating": 1500,
            "completion": "<think>...</think><answer>e4</answer>",
            "predicted_move": "e4",
            "correct": True,
            "rewards": {"move": 1.0, "format": 1.0, "legal": 1.0, "total": 2.0},
        }, indent=2))
        return

    results = load_jsonl(results_path)
    st.success(f"Loaded {len(results)} results.")

    # Aggregate stats
    st.subheader("Aggregate Metrics")
    n = len(results)
    correct = sum(1 for r in results if r.get("correct", False))
    legal = sum(1 for r in results if r.get("rewards", {}).get("legal", 0) > 0)
    format_ok = sum(1 for r in results if r.get("rewards", {}).get("format", 0) > 0.5)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{correct / n:.1%}")
    col2.metric("Legal Move Rate", f"{legal / n:.1%}")
    col3.metric("Format Compliance", f"{format_ok / n:.1%}")
    col4.metric("Avg Total Reward", f"{sum(r.get('rewards', {}).get('total', 0) for r in results) / n:.3f}")

    # Accuracy by rating bucket
    st.subheader("Accuracy by Rating")
    buckets = [(200, 800), (800, 1200), (1200, 1600),
               (1600, 2000), (2000, 2400), (2400, 2800)]
    bucket_data = {}
    for lo, hi in buckets:
        in_bucket = [r for r in results if lo <= r.get("puzzle_rating", 0) < hi]
        if in_bucket:
            acc = sum(1 for r in in_bucket if r.get("correct")) / len(in_bucket)
            bucket_data[f"{lo}-{hi}"] = acc
    if bucket_data:
        st.bar_chart(bucket_data)

    # Reward distribution histogram
    st.subheader("Reward Distribution")
    total_rewards = [r.get("rewards", {}).get("total", 0) for r in results]
    st.bar_chart(_histogram(total_rewards, bins=20))

    # Browse individual outputs
    st.subheader("Browse Outputs")

    filter_mode = st.radio("Filter", ["All", "Correct only", "Incorrect only"], horizontal=True)
    if filter_mode == "Correct only":
        filtered = [r for r in results if r.get("correct")]
    elif filter_mode == "Incorrect only":
        filtered = [r for r in results if not r.get("correct")]
    else:
        filtered = results

    if not filtered:
        st.warning("No results match the filter.")
        return

    idx = st.number_input("Result index", 0, len(filtered) - 1, 0, key="result_idx")
    result = filtered[idx]

    col_board, col_info = st.columns([1, 1])

    with col_board:
        predicted = result.get("predicted_move")
        svg = render_board(result["fen"], predicted)
        st.markdown(svg, unsafe_allow_html=True)

    with col_info:
        st.markdown(f"**Rating:** {result.get('puzzle_rating', '?')}")
        board = chess.Board(result["fen"])
        side = "White" if board.turn == chess.WHITE else "Black"
        st.markdown(f"**Side to move:** {side}")
        st.markdown(f"**Solution:** `{result['solution_move']}`")
        st.markdown(f"**Predicted:** `{predicted or '(none)'}`")
        is_correct = result.get("correct", False)
        st.markdown(f"**Correct:** {'Yes' if is_correct else 'No'}")

        rewards = result.get("rewards", {})
        if rewards:
            st.markdown("**Reward breakdown:**")
            for k, v in rewards.items():
                st.markdown(f"- {k}: `{v:.3f}`")

    # Full model output
    st.subheader("Model Output")
    completion = result.get("completion", "(no completion recorded)")
    st.text_area("Raw completion", completion, height=300, disabled=True)


def page_grpo_training():
    """Inspect GRPO training: per-prompt groups, reward signals, advantages."""
    st.header("GRPO Training Inspector")

    log_path = st.text_input("Training log JSONL path", "outputs/grpo_training_log.jsonl")

    if not os.path.exists(log_path):
        st.info(f"No training log at `{log_path}`. Will be generated during GRPO training.")
        st.markdown("**Expected format** (one JSON per line, one per prompt group):")
        st.code(json.dumps({
            "step": 10,
            "prompt_idx": 0,
            "fen": "...",
            "solution_move": "e4",
            "puzzle_rating": 1500,
            "completions": [
                {
                    "text": "<think>...</think><answer>e4</answer>",
                    "predicted_move": "e4",
                    "reward": 1.5,
                    "reward_breakdown": {"move": 1.0, "format": 0.33, "legal": 1.0},
                    "advantage": 0.82,
                }
            ],
        }, indent=2))
        return

    groups = load_jsonl(log_path)
    st.success(f"Loaded {len(groups)} prompt groups.")

    # ── Aggregate training curves ────────────────────────────────────────
    st.subheader("Training Curves")

    steps = sorted(set(g["step"] for g in groups))
    step_rewards = {}
    step_accuracy = {}
    step_reward_std = {}
    for step in steps:
        step_groups = [g for g in groups if g["step"] == step]
        all_rewards = [c["reward"] for g in step_groups for c in g["completions"]]
        correct = sum(
            1 for g in step_groups for c in g["completions"]
            if c.get("predicted_move") == g["solution_move"]
        )
        total = sum(len(g["completions"]) for g in step_groups)
        step_rewards[step] = sum(all_rewards) / len(all_rewards)
        step_accuracy[step] = correct / total if total else 0
        step_reward_std[step] = _std(all_rewards)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Mean Reward per Step**")
        st.line_chart(step_rewards)
    with col2:
        st.markdown("**Accuracy per Step**")
        st.line_chart(step_accuracy)

    st.markdown("**Intra-group Reward Std (should be > 0 for GRPO to learn)**")
    # Per-group std averaged over groups at each step
    step_intra_std = {}
    for step in steps:
        step_groups = [g for g in groups if g["step"] == step]
        group_stds = [
            _std([c["reward"] for c in g["completions"]])
            for g in step_groups if len(g["completions"]) > 1
        ]
        step_intra_std[step] = sum(group_stds) / len(group_stds) if group_stds else 0
    st.line_chart(step_intra_std)

    # ── Per-prompt group browser ─────────────────────────────────────────
    st.subheader("Browse Prompt Groups")

    selected_step = st.select_slider("Training step", options=steps)
    step_groups = [g for g in groups if g["step"] == selected_step]

    group_idx = st.number_input("Group index", 0, len(step_groups) - 1, 0,
                                key="grpo_group_idx")
    group = step_groups[group_idx]

    col_board, col_meta = st.columns([1, 1])
    with col_board:
        svg = render_board(group["fen"], group["solution_move"])
        st.markdown(svg, unsafe_allow_html=True)

    with col_meta:
        st.markdown(f"**Solution:** `{group['solution_move']}`")
        st.markdown(f"**Rating:** {group.get('puzzle_rating', '?')}")
        completions = group["completions"]
        rewards = [c["reward"] for c in completions]
        correct_count = sum(
            1 for c in completions
            if c.get("predicted_move") == group["solution_move"]
        )
        st.markdown(f"**Completions:** {len(completions)}")
        st.markdown(f"**Correct in group:** {correct_count}/{len(completions)}")
        st.markdown(f"**Reward range:** [{min(rewards):.3f}, {max(rewards):.3f}]")
        st.markdown(f"**Reward std:** {_std(rewards):.3f}")

    # Show all completions in the group
    st.markdown("**All completions in group:**")
    for i, comp in enumerate(completions):
        reward = comp["reward"]
        adv = comp.get("advantage", "?")
        predicted = comp.get("predicted_move", "(none)")
        correct = predicted == group["solution_move"]
        label = f"{'OK' if correct else 'WRONG'} | move={predicted} | reward={reward:.3f} | adv={adv}"

        with st.expander(f"Completion {i}: {label}"):
            # Reward breakdown
            breakdown = comp.get("reward_breakdown", {})
            if breakdown:
                cols = st.columns(len(breakdown))
                for col, (k, v) in zip(cols, breakdown.items()):
                    col.metric(k, f"{v:.3f}")

            st.text_area(f"Output {i}", comp["text"], height=200, disabled=True,
                         key=f"grpo_comp_{selected_step}_{group_idx}_{i}")


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    st.set_page_config(page_title="Chess-GRPO Inspector", layout="wide")
    st.title("Chess-GRPO Inspector")

    page = st.sidebar.radio("Page", ["Data Explorer", "Model Outputs", "GRPO Training"])

    if page == "Data Explorer":
        page_data_explorer()
    elif page == "Model Outputs":
        page_model_outputs()
    else:
        page_grpo_training()


if __name__ == "__main__":
    main()
