#!/bin/bash
# Launch script for ImagiNav evaluation in InternNav
# 
# Usage Examples:
#   ./scripts/eval/bash/start_imaginav_eval.sh                     # Run all episodes
#   ./scripts/eval/bash/start_imaginav_eval.sh --episode 5593_571  # Run specific episode
#   ./scripts/eval/bash/start_imaginav_eval.sh --scene 17DRP5sb8fy # Run all episodes in scene
#   ./scripts/eval/bash/start_imaginav_eval.sh --gpu 0             # Use GPU 0
#   ./scripts/eval/bash/start_imaginav_eval.sh --gpu 1,2           # Use GPUs 1 and 2
#
# Prerequisites:
#   1. VGGT installed: pip install -e /path/to/VGGT
#   2. Gemini API key: export GOOGLE_API_KEY="your-key-here"
#   3. LTX-Video checkpoint available

echo "🎬 Starting ImagiNav Evaluation in InternNav"
echo "=============================================="
echo ""
echo "Pipeline: Gemini (Reasoning) → LTX-Video (Imagination) → VGGT (Navigation)"
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

# Check ImagiNav directory
if [ ! -d "ImagiNav" ]; then
    echo "❌ ImagiNav directory not found"
    echo "   Expected at: $(pwd)/ImagiNav"
    exit 1
fi
echo "✅ ImagiNav directory found"

# Check ImagiNav config
IMAGINAV_CONFIG="ImagiNav/configs/default_config.yaml"
if [ ! -f "$IMAGINAV_CONFIG" ]; then
    echo "❌ ImagiNav config not found: $IMAGINAV_CONFIG"
    exit 1
fi
echo "✅ ImagiNav config found"

# Check if conda.sh exists (Docker vs local)
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate internutopia
elif [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "✅ Already in conda environment: $CONDA_DEFAULT_ENV"
else
    echo "⚠️  Warning: Could not activate conda environment"
fi

# Check if config exists
CONFIG="scripts/eval/configs/imaginav_eval_cfg.py"
if [ ! -f "$CONFIG" ]; then
    echo "❌ Config file not found: $CONFIG"
    exit 1
fi
echo "✅ InternNav config found"

# Extract task name from config
echo "📋 Reading configuration..."
TASK_NAME=$(python3 -c "import sys; sys.path.insert(0, 'scripts/eval/configs'); from imaginav_eval_cfg import TASK_NAME; print(TASK_NAME)" 2>/dev/null)
if [ -z "$TASK_NAME" ]; then
    TASK_NAME=$(grep -oP "task_name\s*=\s*['\"]\K[^'\"]+" "$CONFIG" | head -1)
fi
if [ -z "$TASK_NAME" ]; then
    TASK_NAME="imaginav_eval"
    echo "⚠️  Could not extract task name from config, using default: $TASK_NAME"
else
    echo "✅ Task name: $TASK_NAME"
fi
LOG_DIR="logs/$TASK_NAME"
mkdir -p "$LOG_DIR"
echo "📁 Log directory: $LOG_DIR"

echo ""

# Parse command line arguments FIRST (before starting server)
EVAL_ARGS=""
DISPLAY_MODE=""
GPU_DEVICES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --episode)
            if [[ -z "$2" || "$2" == --* ]]; then echo "❌ Error: Missing argument for --episode"; exit 1; fi
            EVAL_ARGS="$EVAL_ARGS --episode $2"
            DISPLAY_MODE="Episode: $2"
            shift 2
            ;;
        --scene)
            if [[ -z "$2" || "$2" == --* ]]; then echo "❌ Error: Missing argument for --scene"; exit 1; fi
            EVAL_ARGS="$EVAL_ARGS --scene $2"
            DISPLAY_MODE="Scene: $2 (all episodes)"
            shift 2
            ;;
        --split)
            if [[ -z "$2" || "$2" == --* ]]; then echo "❌ Error: Missing argument for --split"; exit 1; fi
            EVAL_ARGS="$EVAL_ARGS --split $2"
            shift 2
            ;;
        --gpu)
            if [[ -z "$2" || "$2" == --* ]]; then echo "❌ Error: --gpu requires a value (e.g. 0)."; exit 1; fi
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

# Set GPU devices if specified (BEFORE starting server)
if [ -n "$GPU_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
    echo "🎮 Using GPU(s): $GPU_DEVICES (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
else
    echo "🎮 Using default GPU configuration"
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        echo "   Current CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    fi
fi
echo ""
echo "📂 Output locations:"
echo "   - Frames: $LOG_DIR/frames/"
echo "   - ImagiNav outputs: $LOG_DIR/<timestamp>/"
echo "   - Evaluation results: $LOG_DIR/results/"
echo ""

# Kill any existing agent server
echo "🧹 Cleaning up any existing agent server..."
processes=$(ps -ef | grep 'internnav/utils/comm_utils/server.py' | grep -v grep | awk '{print $2}')
if [ -n "$processes" ]; then
    for pid in $processes; do
        kill -9 $pid 2>/dev/null
        echo "   Killed process: $pid"
    done
fi

# Start agent server in background
echo "🚀 Starting ImagiNav agent server..."
python internnav/utils/comm_utils/server.py --config "$CONFIG" > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "   Server PID: $SERVER_PID"
echo "   Server log: $LOG_DIR/server.log"

# Wait for server to be ready
echo "⏳ Waiting for server to initialize..."
sleep 10

# Check if server started successfully
if ! ps -p $SERVER_PID > /dev/null 2>&1; then
    echo "❌ Server failed to start. Check $LOG_DIR/server.log"
    tail -n 50 "$LOG_DIR/server.log"
    exit 1
fi

echo "✅ Server ready"
echo ""

if [ -z "$DISPLAY_MODE" ]; then
    DISPLAY_MODE="All episodes (sequential)"
fi

echo "🎯 Mode: $DISPLAY_MODE"
echo ""
echo "💡 This will take time as ImagiNav generates videos for each navigation decision."
echo "   - Gemini reasoning: ~1-2 seconds"
echo "   - LTX-Video generation: ~30-60 seconds"
echo "   - VGGT trajectory extraction: ~2-4 minutes"
echo "   - Total per episode: ~3-5 minutes"
echo ""
echo "Press Ctrl+C to stop."
echo ""

# Trap to cleanup on exit
trap 'echo ""; echo "🧹 Cleaning up..."; kill -9 $SERVER_PID 2>/dev/null || true; exit' INT TERM EXIT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # This prevents CUDA OOM errors for video generation
# Run ImagiNav evaluation loop with parsed arguments
python scripts/eval/imaginav_loop.py --config "$CONFIG" $EVAL_ARGS

