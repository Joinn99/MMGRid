export MODEL_ZOO_PATH=""
export CKPT_PATH=""
export PROJECT_PATH=$(pwd)

MODE="hllm"
SOURCE_DOMAIN="Movies_and_TV"
SPLITS="pretrain"
TARGET_DOMAINS="Video_Games"
METHOD="average_merging"
BASE_MODEL_PATH="${MODEL_ZOO_PATH}/Qwen3-0.6B"

python merge_example.py \
    --mode ${MODE} \
    --source_domain ${SOURCE_DOMAIN} \
    --splits ${SPLITS} \
    --target_domains ${TARGET_DOMAINS} \
    --method ${METHOD} \
    --base_model_path ${BASE_MODEL_PATH}