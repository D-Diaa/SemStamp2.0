#!/bin/bash

# =============================================================================
# SemStamp Experiment Runner
# Runs LSH and KMeans experiments (fixed and non-fixed variants)
# Distributes tasks across available GPUs (one task per GPU at a time)
# =============================================================================

# Don't use set -e as it breaks the process monitoring logic

# Default parameters
CUDA_DEVICES="4,5,6,7"
DATA_FOLDER="data/c4-val-1000"
LSH_DIM=3
KMEANS_DIM=8
CC_PATH="data/c4-train/cc.pt"
DELTA=0.02
HUMAN_TEXT="data/c4-human"
PARAPHRASER="custom"
PARA_BSZ=32
CUSTOM_MODEL="DDiaa/WM-Removal-Unigram-Qwen2.5-3B"
CUSTOM_PROMPT="standard"
FORCE_SAMPLE=false

# Parse command line arguments
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --gpus DEVICES        Comma-separated list of GPU IDs to use (default: 3,5,6,7)"
    echo "  --data FOLDER         Data folder path (default: data/c4-val-200)"
    echo "  --lsh-dim DIM         LSH dimension (default: 3)"
    echo "  --kmeans-dim DIM      KMeans dimension (default: 8)"
    echo "  --cc-path PATH        Path to cluster centers (default: data/c4-train/cc.pt)"
    echo "  --delta DELTA         Delta/margin parameter (default: 0.02)"
    echo "  --human-text PATH     Path to human text data (default: data/c4-human)"
    echo "  --paraphraser NAME    Paraphraser model to use (default: pegasus)"
    echo "  --custom-model PATH   HuggingFace model path for custom paraphraser (default: Qwen/Qwen2.5-3B-Instruct)"
    echo "  --custom-prompt NAME  Prompt style for custom paraphraser: standard, shuffle, combine (default: combine)"
    echo "  --para-bsz BATCH_SIZE    Batch size for paraphrasing (default: 32)"
    echo "  --modes MODES         Comma-separated modes to run (default: lsh,lsh_fixed,kmeans,kmeans_fixed)"
    echo "  --force-sample        Force re-sampling even if data already exists"
    echo "  -h, --help            Show this help message"
    exit 1
}

MODES="lsh,lsh_fixed,kmeans,kmeans_fixed"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --data)
            DATA_FOLDER="$2"
            shift 2
            ;;
        --lsh-dim)
            LSH_DIM="$2"
            shift 2
            ;;
        --kmeans-dim)
            KMEANS_DIM="$2"
            shift 2
            ;;
        --cc-path)
            CC_PATH="$2"
            shift 2
            ;;
        --delta)
            DELTA="$2"
            shift 2
            ;;
        --human-text)
            HUMAN_TEXT="$2"
            shift 2
            ;;
        --paraphraser)
            PARAPHRASER="$2"
            shift 2
            ;;
        --custom-model)
            CUSTOM_MODEL="$2"
            shift 2
            ;;
        --custom-prompt)
            CUSTOM_PROMPT="$2"
            shift 2
            ;;
        --para-bsz)
            PARA_BSZ="$2"
            shift 2
            ;;
        --modes)
            MODES="$2"
            shift 2
            ;;
        --force-sample)
            FORCE_SAMPLE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Convert GPU list to array
IFS=',' read -ra GPU_ARRAY <<< "$CUDA_DEVICES"
NUM_GPUS=${#GPU_ARRAY[@]}

# Convert modes to array
IFS=',' read -ra MODE_ARRAY <<< "$MODES"
NUM_TASKS=${#MODE_ARRAY[@]}

echo "=============================================="
echo "SemStamp Experiment Configuration"
echo "=============================================="
echo "GPUs: ${CUDA_DEVICES} (${NUM_GPUS} devices)"
echo "Data folder: ${DATA_FOLDER}"
echo "LSH dimension: ${LSH_DIM}"
echo "KMeans dimension: ${KMEANS_DIM}"
echo "Cluster centers: ${CC_PATH}"
echo "Delta: ${DELTA}"
echo "Human text: ${HUMAN_TEXT}"
echo "Modes: ${MODES} (${NUM_TASKS} tasks)"
echo "=============================================="

# Create logs directory
LOG_DIR="${DATA_FOLDER}/logs"
mkdir -p "$LOG_DIR"

# Function to run a single experiment pipeline (sampling -> paraphrase -> detection)
run_experiment() {
    local mode=$1
    local gpu=$2
    local sp_dim=$3
    local extra_args=$4

    local log_file="${LOG_DIR}/${mode}.log"
    local detection_mode="${mode}"

    {
        echo "=============================================="
        echo "Starting experiment: ${mode} on GPU ${gpu}"
        echo "=============================================="
        local gen_path="${DATA_FOLDER}/${mode}-generated"


        echo "=== SAMPLING: ${mode} ==="
        echo "Started at: $(date)"

        # Skip sampling if data already exists (unless --force-sample)
        if [[ "${FORCE_SAMPLE}" == false ]] && ls "${gen_path}"/data*.arrow 1>/dev/null 2>&1; then
            echo "Found existing data in ${gen_path}, skipping sampling."
        else
            # Run sampling
            CUDA_VISIBLE_DEVICES=${gpu} python sampling.py "${DATA_FOLDER}" \
                --sp_mode "${mode}" \
                --sp_dim "${sp_dim}" \
                --delta "${DELTA}" \
                ${extra_args}
        fi

        echo "Sampling completed at: $(date)"

        # Run paraphrase generation
        echo ""
        echo "=== PARAPHRASING: ${mode} ==="
        echo "Input: ${gen_path}"

        local para_args="--paraphraser ${PARAPHRASER} --bsz ${PARA_BSZ}"
        if [[ "${PARAPHRASER}" == "custom" && -n "${CUSTOM_MODEL}" ]]; then
            para_args="${para_args} --custom_model ${CUSTOM_MODEL} --custom_prompt ${CUSTOM_PROMPT}"
        fi
        CUDA_VISIBLE_DEVICES=${gpu} python paraphrase_gen.py "${gen_path}" ${para_args}

        echo "Paraphrasing completed at: $(date)"

        IS_BIGRAM="False"
        if [[ "${PARAPHRASER}" == *"bigram"* ]]; then
            IS_BIGRAM="True"
        fi

        # Run detection
        local para_path
        if [[ "${PARAPHRASER}" == "custom" ]]; then
            local model_short="${CUSTOM_MODEL##*/}"
            para_path="${gen_path}-custom-${model_short}-${CUSTOM_PROMPT}"
        elif [[ "${PARAPHRASER}" == "t5" || "${PARAPHRASER}" == "t5-bigram" ]]; then
            para_path="${gen_path}-t5-bigram=${IS_BIGRAM}-threshold=0.0"
        else
            para_path="${gen_path}-${PARAPHRASER}-bigram=${IS_BIGRAM}-threshold=0.0"
        fi
        echo ""
        echo "=== DETECTION: ${mode} ==="
        echo "Input: ${para_path}"

        CUDA_VISIBLE_DEVICES=${gpu} python detection.py "${para_path}" \
            --detection_mode "${detection_mode}" \
            --sp_dim "${sp_dim}" \
            --human_text "${HUMAN_TEXT}" \
            ${extra_args}

        echo "Detection completed at: $(date)"
        echo ""
        echo "=== EXPERIMENT ${mode} COMPLETE ==="

    } 2>&1 | tee "${log_file}"
}

# Build task list with parameters
declare -a TASK_MODES
declare -a TASK_DIMS
declare -a TASK_EXTRA_ARGS

for mode in "${MODE_ARRAY[@]}"; do
    case $mode in
        lsh|lsh_fixed)
            TASK_MODES+=("$mode")
            TASK_DIMS+=("$LSH_DIM")
            TASK_EXTRA_ARGS+=("")
            ;;
        kmeans|kmeans_fixed)
            TASK_MODES+=("$mode")
            TASK_DIMS+=("$KMEANS_DIM")
            TASK_EXTRA_ARGS+=("--cc_path ${CC_PATH}")
            ;;
        *)
            echo "Unknown mode: $mode, skipping..."
            ;;
    esac
done

NUM_TASKS=${#TASK_MODES[@]}

echo ""
echo "Distributing ${NUM_TASKS} tasks across ${NUM_GPUS} GPUs..."
echo ""

# Track PIDs and their associated GPU index
declare -a GPU_PIDS      # PID running on each GPU (0 if free)
declare -a GPU_TASK_IDX  # Task index running on each GPU
declare -a TASK_STATUS   # 0=pending, 1=running, 2=done, 3=failed

# Initialize
for ((i=0; i<NUM_GPUS; i++)); do
    GPU_PIDS[$i]=0
done
for ((i=0; i<NUM_TASKS; i++)); do
    TASK_STATUS[$i]=0
done

NEXT_TASK=0
COMPLETED=0
FAILED=0

# Function to start a task on a GPU
start_task_on_gpu() {
    local gpu_idx=$1
    local task_idx=$2
    local gpu="${GPU_ARRAY[$gpu_idx]}"
    local mode="${TASK_MODES[$task_idx]}"
    local sp_dim="${TASK_DIMS[$task_idx]}"
    local extra_args="${TASK_EXTRA_ARGS[$task_idx]}"

    echo "[$(date +%H:%M:%S)] Starting task ${task_idx} (${mode}) on GPU ${gpu}"

    run_experiment "$mode" "$gpu" "$sp_dim" "$extra_args" &
    GPU_PIDS[$gpu_idx]=$!
    GPU_TASK_IDX[$gpu_idx]=$task_idx
    TASK_STATUS[$task_idx]=1
}

# Function to check if a GPU is free (process finished)
check_gpu_free() {
    local gpu_idx=$1
    local pid=${GPU_PIDS[$gpu_idx]}

    if [[ $pid -eq 0 ]]; then
        return 0  # GPU is free
    fi

    if ! kill -0 "$pid" 2>/dev/null; then
        # Process finished, check exit status
        wait "$pid"
        local exit_code=$?
        local task_idx=${GPU_TASK_IDX[$gpu_idx]}
        local mode="${TASK_MODES[$task_idx]}"

        if [[ $exit_code -eq 0 ]]; then
            echo "[$(date +%H:%M:%S)] Task ${task_idx} (${mode}) completed successfully on GPU ${GPU_ARRAY[$gpu_idx]}"
            TASK_STATUS[$task_idx]=2
            ((COMPLETED++))
        else
            echo "[$(date +%H:%M:%S)] Task ${task_idx} (${mode}) FAILED on GPU ${GPU_ARRAY[$gpu_idx]} (exit code: ${exit_code})"
            TASK_STATUS[$task_idx]=3
            ((FAILED++))
        fi

        GPU_PIDS[$gpu_idx]=0
        return 0  # GPU is now free
    fi

    return 1  # GPU still busy
}

# Main scheduling loop
# Continue while there are tasks still running or pending
while [[ $((COMPLETED + FAILED)) -lt $NUM_TASKS ]]; do
    # Check each GPU and start new tasks on free ones
    for ((gpu_idx=0; gpu_idx<NUM_GPUS; gpu_idx++)); do
        # Check if GPU became free
        if check_gpu_free $gpu_idx; then
            # GPU is free, assign next pending task if available
            if [[ $NEXT_TASK -lt $NUM_TASKS ]]; then
                start_task_on_gpu $gpu_idx $NEXT_TASK
                ((NEXT_TASK++))
            fi
        fi
    done

    # Brief sleep to avoid busy-waiting
    sleep 5
done

# Wait for any remaining tasks
for ((gpu_idx=0; gpu_idx<NUM_GPUS; gpu_idx++)); do
    pid=${GPU_PIDS[$gpu_idx]}
    if [[ $pid -ne 0 ]]; then
        wait "$pid"
        exit_code=$?
        task_idx=${GPU_TASK_IDX[$gpu_idx]}
        mode="${TASK_MODES[$task_idx]}"

        if [[ $exit_code -eq 0 ]]; then
            echo "[$(date +%H:%M:%S)] Task ${task_idx} (${mode}) completed successfully"
            ((COMPLETED++))
        else
            echo "[$(date +%H:%M:%S)] Task ${task_idx} (${mode}) FAILED (exit code: ${exit_code})"
            ((FAILED++))
        fi
    fi
done

echo ""
echo "=============================================="
echo "All experiments finished"
echo "=============================================="
echo "Total: ${NUM_TASKS} experiments"
echo "Completed: ${COMPLETED}"
echo "Failed: ${FAILED}"
echo "Logs saved to: ${LOG_DIR}"
echo ""

# Print summary of results
echo "Results summary:"
for mode in "${TASK_MODES[@]}"; do
    if [[ "${PARAPHRASER}" == "custom" ]]; then
        model_short="${CUSTOM_MODEL##*/}"
        results_file="${DATA_FOLDER}/${mode}-generated-custom-${model_short}-${CUSTOM_PROMPT}/results.csv"
    elif [[ "${PARAPHRASER}" == "t5" || "${PARAPHRASER}" == "t5-bigram" ]]; then
        is_bigram="False"
        [[ "${PARAPHRASER}" == "t5-bigram" ]] && is_bigram="True"
        results_file="${DATA_FOLDER}/${mode}-generated-t5-bigram=${is_bigram}-threshold=0.0/results.csv"
    else
        is_bigram="False"
        [[ "${PARAPHRASER}" == *"bigram"* ]] && is_bigram="True"
        results_file="${DATA_FOLDER}/${mode}-generated-${PARAPHRASER}-bigram=${is_bigram}-threshold=0.0/results.csv"
    fi
    if [[ -f "$results_file" ]]; then
        echo ""
        echo "=== ${mode} ==="
        cat "$results_file"
    fi
done

exit $FAILED
