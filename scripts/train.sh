export CURDIR=$(pwd)
export MODEL_ZOO_PATH=""
export DATA_DIR=""
export CKPT_PATH=""


DOMAINS=(
    # "Video_Games"
    "Movies_and_TV"
    # "Cell_Phones_and_Accessories"
    # "Books"
    # "Sports_and_Outdoors"
    # "Health_and_Household"
)

# CUR_DOMAIN=${DOMAINS[0]}
GPU_ID="3"
export MODE="title"

if [ ${MODE} == "hllm" ]; then
    for DOMAIN_NAME in ${DOMAINS[@]}; do
        for SPLIT_NAME in "pretrain" "phase1" "phase2"; do
        export master_port=29506
        export DOMAIN=${DOMAIN_NAME}
        export SPLIT=${SPLIT_NAME}
        cd src/genrec/hllm && CUDA_VISIBLE_DEVICES=1 torchrun \
            --master_port=$master_port \
            --node_rank=0 \
            --nproc_per_node=1 \
            --nnodes=1 \
            run.py \
            --config_file overall/LLM_ddp_full.yaml HLLM/HLLM-${SPLIT}.yaml
        cd ..
        done
    done

else
    for CUR_DOMAIN in ${DOMAINS[@]}; do
        for CUR_SPLIT in "pretrain" "phase1" "phase2"; do
            export DOMAIN=${CUR_DOMAIN}
            export SPLIT=${CUR_SPLIT}
            export DATA_PATH=$DATA_DIR

            if [ ${SPLIT} == "pretrain" ]; then
                export CHECKPOINT_PATH="${MODEL_ZOO_PATH}/Qwen3-0.6B"
            elif [ ${SPLIT} == "phase1" ]; then
                export CHECKPOINT_PATH="${CKPT_PATH}/${DOMAIN}-pretrain-${MODE}/epoch_2"
            elif [ ${SPLIT} == "phase2" ]; then
                export CHECKPOINT_PATH="${CKPT_PATH}/${DOMAIN}-phase1-${MODE}/epoch_1"
            fi

            export OUTPUT_DIR="${CKPT_PATH}/${DOMAIN}-${SPLIT}-${MODE}"
            export BATCH_SIZE=8
            mkdir -p ${OUTPUT_DIR}

            if [ ${SPLIT} == "pretrain" ]; then
                export EPOCHS=3
                export LEARNING_RATE=5e-5
                envsubst <config/${MODE}.yml > ${OUTPUT_DIR}/fine_tune_config.yml
            else
                export EPOCHS=2
                export LEARNING_RATE=1e-5
                envsubst <config/${MODE}_cont.yml > ${OUTPUT_DIR}/fine_tune_config.yml
            fi

            CUDA_VISIBLE_DEVICES=${GPU_ID} tune run full_finetune_single_device \
                --config ${OUTPUT_DIR}/fine_tune_config.yml
        done
    done
fi