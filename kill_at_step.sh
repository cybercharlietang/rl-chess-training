#!/bin/bash
# Monitor and kill training after step 20, then save won't happen automatically
# so we just let it finish step 20 and kill — the checkpoint-20 will exist if save_steps divides 20
TARGET=20
LOG=/root/rl-chess-training/outputs/overnight_full.log
PID=$1

while true; do
    # Check for step completion line (e.g. "20/50 [...")
    STEP=$(grep -oP "\d+(?=/50 \[)" "$LOG" 2>/dev/null | tail -1)
    if [ -n "$STEP" ] && [ "$STEP" -ge "$TARGET" ]; then
        echo "$(date): Step $STEP reached. Sending SIGINT to PID $PID for graceful save..."
        kill -2 $PID
        # Wait up to 2 min for graceful shutdown
        for i in $(seq 1 24); do
            if ! kill -0 $PID 2>/dev/null; then
                echo "$(date): Process exited gracefully."
                exit 0
            fi
            sleep 5
        done
        echo "$(date): Force killing..."
        kill -9 $PID 2>/dev/null
        exit 0
    fi
    sleep 30
done
