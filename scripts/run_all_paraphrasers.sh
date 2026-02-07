#!/bin/bash

# =============================================================================
# Run experiments for all paraphrasers on c4-val-1000
# GPUs: 5,6,7 | No force resample
# =============================================================================

GPUS="5,6,7"
DATA="data/c4-val-1000"
COMMON_ARGS="--gpus ${GPUS} --data ${DATA}"
# PARAPHRASERS=(parrot parrot-bigram t5 t5-bigram)
PARAPHRASERS=(parrot t5)
CUSTOM_MODELS=(
    "DDiaa/WM-Removal-Unigram-Qwen2.5-3B"
    "Qwen/Qwen2.5-3B-Instruct"
)
PROMPTS=(standard shuffle combine)

# --- Non-custom paraphrasers ---
for para in "${PARAPHRASERS[@]}"; do
    echo ""
    echo "########################################################"
    echo "# Running: ${para}"
    echo "########################################################"
    bash scripts/run_experiments.sh ${COMMON_ARGS} --paraphraser "${para}"
done

# --- Custom paraphrasers × all prompts ---

for model in "${CUSTOM_MODELS[@]}"; do
    for prompt in "${PROMPTS[@]}"; do
        echo ""
        echo "########################################################"
        echo "# Running: custom | model=${model} | prompt=${prompt}"
        echo "########################################################"
        bash scripts/run_experiments.sh ${COMMON_ARGS} \
            --paraphraser custom \
            --custom-model "${model}" \
            --custom-prompt "${prompt}"
    done
done

echo ""
echo "========================================"
echo "All paraphraser runs complete."
echo "========================================"
