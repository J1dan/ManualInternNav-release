#!/bin/bash
# Launch script for NavDP evaluation in InternNav
# 
# Usage Examples:
#   ./scripts/eval/bash/start_navdp_eval.sh                     # Run all episodes
#   ./scripts/eval/bash/start_navdp_eval.sh --episode 5593_571  # Run specific episode
#   ./scripts/eval/bash/start_navdp_eval.sh --scene 17DRP5sb8fy # Run all episodes in scene
#   ./scripts/eval/bash/start_navdp_eval.sh --gpu 0             # Use GPU 0
#   ./scripts/eval/bash/start_navdp_eval.sh --headless          # Run without GUI

echo "🎬 Starting NavDP Evaluation in InternNav"
echo "=============================================="
echo ""
echo "Pipeline: Gemini (Reasoning) → NavDP (Navigation) → Controller (Action)"
echo ""

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check Gemini API key
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "❌ GOOGLE_API_KEY not set"
    echo "   Please export your Gemini API key:"
    echo "   export GOOGLE_API_KEY='your-key-here'"
    exit 1
fi
echo "✅ Gemini API key found"

# Check NavDP directory
if [ ! -d "ImagiNav/NavDP" ]; then
    echo "❌ NavDP directory not found"
    exit 1
fi

# Check if conda.sh exists
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate internutopia
elif [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "✅ Already in conda environment: $CONDA_DEFAULT_ENV"
else
    echo "⚠️  Warning: Could not activate conda environment"
fi

# # Create logs directory
# mkdir -p logs/navdp_eval

# Parse command line arguments
EVAL_ARGS=""
DISPLAY_MODE=""
GPU_DEVICES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --episode)
            EVAL_ARGS="$EVAL_ARGS --episode $2"
            DISPLAY_MODE="Episode: $2"
            shift 2
            ;;
        --scene)
            EVAL_ARGS="$EVAL_ARGS --scene $2"
            DISPLAY_MODE="Scene: $2 (all episodes)"
            shift 2
            ;;
        --split)
            EVAL_ARGS="$EVAL_ARGS --split $2"
            shift 2
            ;;
        --gpu)
            GPU_DEVICES="$2"
            shift 2
            ;;
        --headless)
            EVAL_ARGS="$EVAL_ARGS --headless"
            shift
            ;;
        --gui)
            EVAL_ARGS="$EVAL_ARGS --gui"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--episode <id>] [--scene <id>] [--split <name>] [--gpu <id>] [--headless] [--gui]"
            exit 1
            ;;
    esac
done

# Set GPU devices if specified
if [ -n "$GPU_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
    echo "🎮 Using GPU(s): $GPU_DEVICES (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
else
    echo "🎮 Using default GPU configuration"
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        echo "   Current CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    fi
fi

if [ -z "$DISPLAY_MODE" ]; then
    DISPLAY_MODE="All episodes (sequential)"
fi

echo "🎯 Mode: $DISPLAY_MODE"
echo ""

# Extract task name from config
echo "📋 Reading configuration..."
CONFIG="scripts/eval/configs/navdp_eval_cfg.py"
if [ -f "$CONFIG" ]; then
    TASK_NAME=$(python3 -c "import sys; sys.path.insert(0, 'scripts/eval/configs'); from navdp_eval_cfg import TASK_NAME; print(TASK_NAME)" 2>/dev/null)
    if [ -z "$TASK_NAME" ]; then
        # Fallback: try to extract from task_name field
        TASK_NAME=$(grep -oP "task_name\s*=\s*['\"]\K[^'\"]+" "$CONFIG" | head -1)
    fi
    if [ -z "$TASK_NAME" ]; then
        TASK_NAME="navdp_eval"
        echo "⚠️  Could not extract task name from config, using default: $TASK_NAME"
    else
        echo "✅ Task name: $TASK_NAME"
    fi
else
    TASK_NAME="navdp_eval"
    echo "⚠️  Config not found, using default task name: $TASK_NAME"
fi
LOG_DIR="logs/$TASK_NAME"
mkdir -p "$LOG_DIR"
echo "📁 Log directory: $LOG_DIR"
echo ""

# Start NavDP Server
echo "🚀 Starting NavDP server..."
cd ImagiNav/NavDP
# Using the checkpoint found in the directory
# Pass log_dir to navdp_server.py so it knows where to save decision-making frames
python baselines/navdp/navdp_server.py --port 1234 --checkpoint ckpts/navdp-cross-modal.ckpt --log_dir "../../$LOG_DIR/video/navdp_decision_making" > "../../$LOG_DIR/navdp_server.log" 2>&1 &
NAVDP_PID=$!
echo "   NavDP Server PID: $NAVDP_PID"
cd ../..

# Wait for NavDP server
echo "⏳ Waiting for NavDP server to initialize..."
sleep 10

# Check if NavDP server started
if ! ps -p $NAVDP_PID > /dev/null 2>&1; then
    echo "❌ NavDP Server failed to start. Check $LOG_DIR/navdp_server.log"
    tail -n 20 "$LOG_DIR/navdp_server.log"
    exit 1
fi
echo "✅ NavDP Server ready"

# Start Agent Server (InternNav)
echo "🚀 Starting InternNav agent server..."
CONFIG="scripts/eval/configs/navdp_eval_cfg.py"
python internnav/utils/comm_utils/server.py --config "$CONFIG" > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "   InternNav Server PID: $SERVER_PID"

# Wait for InternNav server
sleep 10
if ! ps -p $SERVER_PID > /dev/null 2>&1; then
    echo "❌ InternNav Server failed to start. Check $LOG_DIR/server.log"
    tail -n 20 "$LOG_DIR/server.log"
    kill -9 $NAVDP_PID 2>/dev/null
    exit 1
fi
echo "✅ InternNav Server ready"

echo ""
echo "🎯 Starting Evaluation Loop..."
echo "Press Ctrl+C to stop."
echo ""

# Trap to cleanup on exit
trap 'echo ""; echo "🧹 Cleaning up..."; kill -9 $SERVER_PID 2>/dev/null; kill -9 $NAVDP_PID 2>/dev/null; exit' INT TERM EXIT

# Run Evaluation Loop
# We reuse imaginav_loop.py as it is generic
python scripts/eval/imaginav_loop.py --config "$CONFIG" $EVAL_ARGS
