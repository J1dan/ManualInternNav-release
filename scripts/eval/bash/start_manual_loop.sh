#!/bin/bash
# Quick start script for manual control mode
# 
# Usage Examples:
#   ./scripts/eval/bash/start_manual_loop.sh                          # Run all episodes (default)
#   ./scripts/eval/bash/start_manual_loop.sh --episode 5593_571       # Run ONE specific episode
#   ./scripts/eval/bash/start_manual_loop.sh --scene 17DRP5sb8fy      # Run ALL episodes in ONE scene
#   ./scripts/eval/bash/start_manual_loop.sh --list-episodes          # List available episodes
#   ./scripts/eval/bash/start_manual_loop.sh --split val_unseen       # Filter by dataset split
#
# Scene vs Episode:
#   - Scene: A 3D environment (e.g., a house). Contains multiple episodes.
#   - Episode: A navigation task (start position → goal + instruction).
#   - One scene can have 5-20+ different episodes (different tasks in same house).

echo "🚀 Starting Manual Control Mode for VLN-PE"
echo "=========================================="
echo ""
echo "This will:"
echo "  ✅ Launch Isaac Sim with GUI"
echo "  ✅ Start H1 robot in visible mode"
echo "  ✅ Save RGB frames automatically"
echo "  ✅ Enable manual control via JSON file"
echo ""

# Check if conda.sh exists (Docker vs local)
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate internutopia
elif [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "✅ Already in conda environment: $CONDA_DEFAULT_ENV"
else
    echo "⚠️  Warning: Could not activate conda environment"
    echo "   Make sure you've run: conda activate internutopia"
fi

# Check if config exists
CONFIG="scripts/eval/configs/manual_control_cfg.py"
if [ ! -f "$CONFIG" ]; then
    echo "❌ Config file not found: $CONFIG"
    exit 1
fi

# Create logs directory
mkdir -p manual_logs

echo ""
echo "📝 Control file will be: manual_logs/manual_control/action_command.json"
echo "📸 Frames will be saved to: manual_logs/manual_control/frames/"
echo ""

# Kill any existing agent server
echo "🧹 Cleaning up any existing agent server..."
processes=$(ps -ef | grep 'internnav/agent/utils/server.py' | grep -v grep | awk '{print $2}')
if [ -n "$processes" ]; then
    for pid in $processes; do
        kill -9 $pid
        echo "   Killed process: $pid"
    done
fi

# Start agent server in background
echo "🚀 Starting agent server..."
python internnav/utils/comm_utils/server.py --config "$CONFIG" > logs/manual_control_server.log 2>&1 &
SERVER_PID=$!
echo "   Server PID: $SERVER_PID"
echo "   Server log: logs/manual_control_server.log"

# Wait for server to be ready
echo "⏳ Waiting for server to initialize..."
sleep 8

# Check if server started successfully
if ! ps -p $SERVER_PID > /dev/null 2>&1; then
    echo "❌ Server failed to start. Check logs/manual_control_server.log"
    exit 1
fi

echo "✅ Server ready"
echo ""

# Parse command line arguments to pass to manual_loop.py
MANUAL_ARGS=""
DISPLAY_MODE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --episode)
            MANUAL_ARGS="$MANUAL_ARGS --episode $2"
            DISPLAY_MODE="Episode: $2"
            shift 2
            ;;
        --scene)
            MANUAL_ARGS="$MANUAL_ARGS --scene $2"
            DISPLAY_MODE="Scene: $2 (all episodes)"
            shift 2
            ;;
        --split)
            MANUAL_ARGS="$MANUAL_ARGS --split $2"
            shift 2
            ;;
        --list-episodes)
            MANUAL_ARGS="$MANUAL_ARGS --list-episodes"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--episode <id>] [--scene <id>] [--split <name>] [--list-episodes]"
            exit 1
            ;;
    esac
done

if [ -z "$DISPLAY_MODE" ]; then
    DISPLAY_MODE="All episodes (sequential)"
    echo "💡 Tip: To explore a specific episode or scene, use:"
    echo "   ./start_manual_control.sh --episode 5593_571"
    echo "   ./start_manual_control.sh --scene 17DRP5sb8fy"
    echo "   ./start_manual_control.sh --list-episodes"
fi

echo ""
echo "🎯 Mode: $DISPLAY_MODE"
echo ""
echo "To control the robot in another terminal, use:"
echo "  python scripts/utils/send_action.py --forward 0.5"
echo "  python scripts/utils/send_action.py --yaw 0.3"
echo "  python scripts/utils/send_action.py --stop"
echo ""
echo "Press Ctrl+C to stop."
echo ""

# Trap to cleanup on exit
trap 'echo ""; echo "🧹 Cleaning up..."; kill -9 $SERVER_PID 2>/dev/null || true; exit' INT TERM EXIT

# Run manual control loop with parsed arguments
python scripts/eval/manual_loop.py --config "$CONFIG" $MANUAL_ARGS
