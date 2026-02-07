#!/bin/bash

# =============================================================================
# SemStamp Experiment Runner
# Runs LSH and KMeans experiments (fixed and non-fixed variants)
# Distributes tasks across available GPUs (one task per GPU at a time)
# Supports multiple paraphrasers in a single run
# =============================================================================

# Don't use set -e as it breaks the process monitoring logic

# Default parameters
CUDA_DEVICES="4,5,6,7"
DATA_FOLDER="data/c4-val-10"
LSH_DIM=3
KMEANS_DIM=8
CC_PATH="data/c4-train/cc.pt"
DELTA=0.02
HUMAN_TEXT="data/c4-human"
MODEL="meta-llama/Llama-3.1-8B"
EMBEDDER="AbeHou/SemStamp-c4-sbert"
PARAPHRASERS="custom,parrot"
PARA_BSZ=32
CUSTOM_MODELS="DDiaa/WM-Removal-Unigram-Qwen2.5-3B,Qwen/Qwen2.5-3B-Instruct"
CUSTOM_PROMPTS="standard,shuffle,combine"
FORCE_SAMPLE=false
FORCE_PARAPHRASE=false
FORCE_EVALUATION=false
QUALITY_CORPUS="data/c4-human"
SOURCE_DATA="data/c4-val"
NUM_SAMPLES=10

MODES="lsh,lsh_fixed,kmeans,kmeans_fixed"

# Parse command line arguments
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --gpus DEVICES           Comma-separated list of GPU IDs (default: 4,5,6,7)"
    echo "  --data FOLDER            Data folder path (default: data/c4-val-10)"
    echo "  --source-data PATH       Source dataset for auto-creating subset if --data doesn't exist (default: data/c4-val)"
    echo "  --num-samples N          Number of samples for subset creation (default: 10)"
    echo "  --model MODEL            HuggingFace causal LM model (default: meta-llama/Llama-3.1-8B)"
    echo "  --embedder MODEL         Sentence embedder model (default: AbeHou/SemStamp-c4-sbert)"
    echo "  --lsh-dim DIM            LSH dimension (default: 3)"
    echo "  --kmeans-dim DIM         KMeans dimension (default: 8)"
    echo "  --cc-path PATH           Path to cluster centers (default: data/c4-train/cc.pt)"
    echo "  --delta DELTA            Delta/margin parameter (default: 0.02)"
    echo "  --human-text PATH        Path to human text data (default: data/c4-human)"
    echo "  --paraphrasers LIST      Comma-separated paraphraser types: parrot,t5,custom,etc. (default: custom,parrot)"
    echo "  --custom-models LIST     Comma-separated HuggingFace model paths for custom paraphraser"
    echo "  --custom-prompts LIST    Comma-separated prompt styles: standard,shuffle,combine (default: standard,shuffle,combine)"
    echo "  --para-bsz BATCH_SIZE    Batch size for paraphrasing (default: 32)"
    echo "  --modes MODES            Comma-separated modes to run (default: lsh,lsh_fixed,kmeans,kmeans_fixed)"
    echo "  --corpus PATH            Corpus path for quality evaluation semantic entropy (default: data/c4-human)"
    echo "  --force-sample           Force re-sampling even if data already exists"
    echo "  --force-paraphrase       Force re-paraphrasing even if data already exists"
    echo "  --force-evaluation       Force re-running detection and quality evaluation even if results exist"
    echo "  -h, --help               Show this help message"
    exit 1
}

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
        --source-data)
            SOURCE_DATA="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --embedder)
            EMBEDDER="$2"
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
        --paraphrasers)
            PARAPHRASERS="$2"
            shift 2
            ;;
        --custom-models)
            CUSTOM_MODELS="$2"
            shift 2
            ;;
        --custom-prompts)
            CUSTOM_PROMPTS="$2"
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
        --corpus)
            QUALITY_CORPUS="$2"
            shift 2
            ;;
        --force-sample)
            FORCE_SAMPLE=true
            shift
            ;;
        --force-paraphrase)
            FORCE_PARAPHRASE=true
            shift
            ;;
        --force-evaluation)
            FORCE_EVALUATION=true
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

# =============================================================================
# Auto-create dataset if it doesn't exist
# =============================================================================
if [[ ! -d "${DATA_FOLDER}" ]]; then
    if [[ -z "${SOURCE_DATA}" ]]; then
        echo "ERROR: Data folder '${DATA_FOLDER}' does not exist."
        echo "Provide --source-data to auto-create a subset."
        exit 1
    fi

    # Try to extract N from path like data/c4-val-1000 -> 1000
    BASENAME=$(basename "${DATA_FOLDER}")
    if [[ "${BASENAME}" =~ -([0-9]+)$ ]]; then
        SUBSET_N="${BASH_REMATCH[1]}"
        echo "Extracted subset size N=${SUBSET_N} from data path '${DATA_FOLDER}'"
    else
        SUBSET_N="${NUM_SAMPLES}"
        echo "No size found in path, using --num-samples=${SUBSET_N}"
    fi

    echo "Creating dataset subset: ${DATA_FOLDER} (${SUBSET_N} samples from ${SOURCE_DATA})"
    python scripts/build_subset.py "${SOURCE_DATA}" --n "${SUBSET_N}"

    # build_subset.py creates ${SOURCE_DATA}-${SUBSET_N}
    CREATED_PATH="${SOURCE_DATA}-${SUBSET_N}"
    if [[ "${CREATED_PATH}" != "${DATA_FOLDER}" ]]; then
        mv "${CREATED_PATH}" "${DATA_FOLDER}"
    fi

    if [[ ! -d "${DATA_FOLDER}" ]]; then
        echo "ERROR: Failed to create dataset at '${DATA_FOLDER}'"
        exit 1
    fi
    echo "Dataset created successfully at '${DATA_FOLDER}'"
fi

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
echo "Model: ${MODEL}"
echo "Embedder: ${EMBEDDER}"
echo "LSH dimension: ${LSH_DIM}"
echo "KMeans dimension: ${KMEANS_DIM}"
echo "Cluster centers: ${CC_PATH}"
echo "Delta: ${DELTA}"
echo "Human text: ${HUMAN_TEXT}"
echo "Modes: ${MODES} (${NUM_TASKS} tasks)"
echo "Paraphrasers: ${PARAPHRASERS}"
echo "Custom models: ${CUSTOM_MODELS}"
echo "Custom prompts: ${CUSTOM_PROMPTS}"
echo "Quality corpus: ${QUALITY_CORPUS}"
echo "Force sample: ${FORCE_SAMPLE}"
echo "Force paraphrase: ${FORCE_PARAPHRASE}"
echo "Force evaluation: ${FORCE_EVALUATION}"
echo "=============================================="

# Create logs directory
LOG_DIR="${DATA_FOLDER}/logs"
mkdir -p "$LOG_DIR"

# =============================================================================
# Build paraphraser configuration list
# =============================================================================
# Each config is stored as: PARA_TYPE|CUSTOM_MODEL|CUSTOM_PROMPT
IFS=',' read -ra PARA_ARRAY <<< "${PARAPHRASERS}"
IFS=',' read -ra CUSTOM_MODEL_ARRAY <<< "${CUSTOM_MODELS}"
IFS=',' read -ra CUSTOM_PROMPT_ARRAY <<< "${CUSTOM_PROMPTS}"

declare -a PARA_CONFIGS
for para in "${PARA_ARRAY[@]}"; do
    if [[ "${para}" == "custom" ]]; then
        for cm in "${CUSTOM_MODEL_ARRAY[@]}"; do
            for cp in "${CUSTOM_PROMPT_ARRAY[@]}"; do
                PARA_CONFIGS+=("custom|${cm}|${cp}")
            done
        done
    else
        PARA_CONFIGS+=("${para}||")
    fi
done

echo ""
echo "Paraphraser configurations (${#PARA_CONFIGS[@]} total):"
for cfg in "${PARA_CONFIGS[@]}"; do
    IFS='|' read -r ptype pmodel pprompt <<< "${cfg}"
    if [[ "${ptype}" == "custom" ]]; then
        echo "  custom: model=${pmodel}, prompt=${pprompt}"
    else
        echo "  ${ptype}"
    fi
done
echo ""

# =============================================================================
# Pre-download all HuggingFace models (ensures TRANSFORMERS_OFFLINE=1 works)
# =============================================================================
echo "=== Pre-downloading models ==="

# Build list of models to download
MODELS_TO_DOWNLOAD=("${MODEL}" "${EMBEDDER}" "roberta-large")

for para in "${PARA_ARRAY[@]}"; do
    case "${para}" in
        t5|t5-bigram)
            MODELS_TO_DOWNLOAD+=("humarin/chatgpt_paraphraser_on_T5_base")
            ;;
        parrot|parrot-bigram)
            MODELS_TO_DOWNLOAD+=("prithivida/parrot_paraphraser_on_T5")
            ;;
        custom)
            for cm in "${CUSTOM_MODEL_ARRAY[@]}"; do
                MODELS_TO_DOWNLOAD+=("${cm}")
            done
            ;;
    esac
done

# De-duplicate
declare -A SEEN_MODELS
UNIQUE_MODELS=()
for m in "${MODELS_TO_DOWNLOAD[@]}"; do
    if [[ -z "${SEEN_MODELS[$m]+x}" ]]; then
        SEEN_MODELS[$m]=1
        UNIQUE_MODELS+=("$m")
    fi
done

echo "Models to ensure cached: ${UNIQUE_MODELS[*]}"

for model_id in "${UNIQUE_MODELS[@]}"; do
    echo "  Checking: ${model_id}"
    python -c "
from huggingface_hub import snapshot_download
try:
    snapshot_download('${model_id}', local_files_only=True)
    print('    Already cached.')
except Exception:
    print('    Downloading...')
    snapshot_download('${model_id}')
    print('    Done.')
" || echo "  WARNING: Failed to download ${model_id}, pipeline may fail later"
done

echo "=== Model pre-download complete ==="
echo ""

# =============================================================================
# Helper functions
# =============================================================================

# Compute the paraphrased dataset output path
# Usage: get_para_path GEN_PATH PARA_TYPE CUSTOM_MODEL CUSTOM_PROMPT
get_para_path() {
    local gen_path=$1
    local para_type=$2
    local custom_model=$3
    local custom_prompt=$4

    local is_bigram="False"
    [[ "${para_type}" == *"bigram"* ]] && is_bigram="True"

    if [[ "${para_type}" == "custom" ]]; then
        local model_short="${custom_model##*/}"
        echo "${gen_path}-custom-${model_short}-${custom_prompt}"
    elif [[ "${para_type}" == "t5" || "${para_type}" == "t5-bigram" ]]; then
        echo "${gen_path}-t5-bigram=${is_bigram}-threshold=0.0"
    else
        echo "${gen_path}-${para_type}-bigram=${is_bigram}-threshold=0.0"
    fi
}

# Compute the visualization suffix from a paraphrased path
# Given mode="lsh" and para_path="data/x/lsh-generated-parrot-bigram=False-threshold=0.0"
# Returns "-generated-parrot-bigram=False-threshold=0.0"
get_viz_suffix() {
    local para_path=$1
    local mode=$2
    local para_basename
    para_basename=$(basename "${para_path}")
    echo "${para_basename#${mode}}"
}

# =============================================================================
# Experiment pipeline function
# =============================================================================

# Function to run a single experiment pipeline for one mode across all paraphrasers
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
        echo "Paraphraser configs: ${#PARA_CONFIGS[@]}"
        echo "=============================================="
        local gen_path="${DATA_FOLDER}/${mode}-generated"

        # === SAMPLING ===
        echo "=== SAMPLING: ${mode} ==="
        echo "Started at: $(date)"

        if [[ "${FORCE_SAMPLE}" == false ]] && ls "${gen_path}"/data*.arrow 1>/dev/null 2>&1; then
            echo "Found existing data in ${gen_path}, skipping sampling."
        else
            HF_HUB_DISABLE_TELEMETRY=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu} python -m sampling "${DATA_FOLDER}" \
                --model "${MODEL}" \
                --embedder "${EMBEDDER}" \
                --sp_mode "${mode}" \
                --sp_dim "${sp_dim}" \
                --delta "${DELTA}" \
                ${extra_args}
        fi

        echo "Sampling completed at: $(date)"

        # === WATERMARK-ONLY DETECTION ===
        echo ""
        echo "=== DETECTION (watermark-only): ${mode} ==="
        echo "Input: ${gen_path}"

        if [[ "${FORCE_EVALUATION}" == false ]] && [[ -f "${gen_path}/results_wm.csv" ]]; then
            echo "Found existing results in ${gen_path}/results_wm.csv, skipping watermark detection."
        else
            HF_HUB_DISABLE_TELEMETRY=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu} python -m detection "${gen_path}" \
                --detection_mode "${detection_mode}" \
                --sp_dim "${sp_dim}" \
                --embedder "${EMBEDDER}" \
                --human_text "${HUMAN_TEXT}" \
                ${extra_args}
        fi

        echo "Watermark-only detection completed at: $(date)"

        # === WATERMARK-ONLY QUALITY ===
        echo ""
        echo "=== QUALITY EVALUATION (watermark-only): ${mode} ==="
        echo "Input: ${gen_path} [text column]"

        if [[ "${FORCE_EVALUATION}" == false ]] && [[ -f "${gen_path}/eval_quality.csv" ]]; then
            echo "Found existing results in ${gen_path}/eval_quality.csv, skipping watermark quality evaluation."
        else
            HF_HUB_DISABLE_TELEMETRY=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu} python -m quality "${gen_path}" 512 \
                --model_path "${MODEL}" \
                --reference "${DATA_FOLDER}" \
                --corpus "${QUALITY_CORPUS}" \
                --column text
        fi

        echo "Watermark-only quality evaluation completed at: $(date)"

        # === LOOP OVER PARAPHRASER CONFIGS ===
        for cfg in "${PARA_CONFIGS[@]}"; do
            IFS='|' read -r ptype pmodel pprompt <<< "${cfg}"

            local para_path
            para_path=$(get_para_path "${gen_path}" "${ptype}" "${pmodel}" "${pprompt}")

            echo ""
            echo "----------------------------------------------"
            echo "Paraphraser: ${ptype} (model=${pmodel:-N/A}, prompt=${pprompt:-N/A})"
            echo "----------------------------------------------"

            # === PARAPHRASING (with smart skip) ===
            echo "=== PARAPHRASING: ${mode} with ${ptype} ==="
            if [[ "${FORCE_PARAPHRASE}" == false ]] && ls "${para_path}"/data*.arrow 1>/dev/null 2>&1; then
                echo "Found existing paraphrased data in ${para_path}, skipping."
            else
                local para_args="--paraphraser ${ptype} --bsz ${PARA_BSZ}"
                if [[ "${ptype}" == "custom" && -n "${pmodel}" ]]; then
                    para_args="${para_args} --custom_model ${pmodel} --custom_prompt ${pprompt}"
                fi
                HF_HUB_DISABLE_TELEMETRY=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu} python -m paraphrasing "${gen_path}" ${para_args}
            fi
            echo "Paraphrasing completed at: $(date)"

            # === PARAPHRASED DETECTION ===
            echo ""
            echo "=== DETECTION (paraphrased): ${mode} with ${ptype} ==="
            echo "Input: ${para_path}"

            if [[ "${FORCE_EVALUATION}" == false ]] && [[ -f "${para_path}/results.csv" ]]; then
                echo "Found existing results in ${para_path}/results.csv, skipping paraphrased detection."
            else
                HF_HUB_DISABLE_TELEMETRY=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu} python -m detection "${para_path}" \
                    --detection_mode "${detection_mode}" \
                    --sp_dim "${sp_dim}" \
                    --embedder "${EMBEDDER}" \
                    --human_text "${HUMAN_TEXT}" \
                    ${extra_args}
            fi

            echo "Paraphrased detection completed at: $(date)"

            # === PARAPHRASED QUALITY ===
            echo ""
            echo "=== QUALITY EVALUATION (paraphrased): ${mode} with ${ptype} ==="
            echo "Input: ${para_path} [para_text column]"

            if [[ "${FORCE_EVALUATION}" == false ]] && [[ -f "${para_path}/eval_quality.csv" ]]; then
                echo "Found existing results in ${para_path}/eval_quality.csv, skipping paraphrased quality evaluation."
            else
                HF_HUB_DISABLE_TELEMETRY=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu} python -m quality "${para_path}" 512 \
                    --model_path "${MODEL}" \
                    --reference "${DATA_FOLDER}" \
                    --corpus "${QUALITY_CORPUS}"
            fi

            echo "Paraphrased quality evaluation completed at: $(date)"

        done  # end paraphraser config loop

        echo ""
        echo "=== EXPERIMENT ${mode} COMPLETE ==="

    } 2>&1 | tee "${log_file}"
}

# =============================================================================
# Build task list with parameters
# =============================================================================
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

# =============================================================================
# GPU scheduling infrastructure
# =============================================================================

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

# =============================================================================
# Run visualizations (CPU only, no GPU needed)
# =============================================================================
echo "=== Running visualizations ==="

# 1. Watermark quality (no paraphrasing needed)
echo ""
echo "--- Watermark quality visualization ---"
python -m visualization watermark_quality "${DATA_FOLDER}" || echo "WARNING: watermark_quality visualization failed"

# 2. Robustness visualization for each unique paraphraser suffix
echo ""
echo "--- Robustness visualizations (per paraphraser) ---"

declare -A VIZ_SUFFIXES
for mode in "${TASK_MODES[@]}"; do
    _gen_path="${DATA_FOLDER}/${mode}-generated"
    for cfg in "${PARA_CONFIGS[@]}"; do
        IFS='|' read -r _ptype _pmodel _pprompt <<< "${cfg}"
        _para_path=$(get_para_path "${_gen_path}" "${_ptype}" "${_pmodel}" "${_pprompt}")
        _suffix=$(get_viz_suffix "${_para_path}" "${mode}")
        VIZ_SUFFIXES["${_suffix}"]=1
    done
done

for suffix in "${!VIZ_SUFFIXES[@]}"; do
    echo "  Visualizing robustness for suffix: ${suffix}"
    python -m visualization robustness "${DATA_FOLDER}" --suffix="${suffix}" || \
        echo "WARNING: robustness visualization failed for suffix=${suffix}"
done

# 3. Cross-paraphraser robustness vs quality
echo ""
echo "--- Robustness-quality cross-paraphraser visualization ---"
python -m visualization robustness_quality "${DATA_FOLDER}" --force || \
    echo "WARNING: robustness_quality visualization failed"

echo ""
echo "=== Visualizations complete ==="

# =============================================================================
# Results summary
# =============================================================================
echo ""
echo "Results summary:"
for mode in "${TASK_MODES[@]}"; do
    gen_path="${DATA_FOLDER}/${mode}-generated"

    # Watermark-only results
    wm_results="${gen_path}/results_wm.csv"
    if [[ -f "$wm_results" ]]; then
        echo ""
        echo "=== ${mode} (watermark-only detection) ==="
        cat "$wm_results"
    fi
    wm_quality="${gen_path}/eval_quality_wm.csv"
    if [[ -f "$wm_quality" ]]; then
        echo "=== ${mode} (watermark-only quality) ==="
        cat "$wm_quality"
    fi

    # Per-paraphraser results
    for cfg in "${PARA_CONFIGS[@]}"; do
        IFS='|' read -r ptype pmodel pprompt <<< "${cfg}"
        para_path=$(get_para_path "${gen_path}" "${ptype}" "${pmodel}" "${pprompt}")

        para_results="${para_path}/results.csv"
        if [[ -f "$para_results" ]]; then
            echo ""
            echo "=== ${mode} + ${ptype}$([ -n "${pmodel}" ] && echo " (${pmodel##*/}/${pprompt})") (paraphrased detection) ==="
            cat "$para_results"
        fi
        para_quality="${para_path}/eval_quality.csv"
        if [[ -f "$para_quality" ]]; then
            echo "=== ${mode} + ${ptype}$([ -n "${pmodel}" ] && echo " (${pmodel##*/}/${pprompt})") (paraphrased quality) ==="
            cat "$para_quality"
        fi
    done
done

exit $FAILED
