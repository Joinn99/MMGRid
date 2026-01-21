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

SPLITS=(
    "pretrain"
    "phase1"
    "phase2"
)

GPU_ID="0"

export MODE="hllm"
export BEAM_WIDTH=5
export DATA_PATH=$DATA_DIR
export CHECKPOINT_DIR="${CKPT_PATH}"

if [ ${MODE} == "hllm" ]; then
    for DOMAIN_NAME in ${DOMAINS[@]}; do
        for SPLIT_NAME in ${SPLITS[@]}; do
        export master_port=29506
        export DOMAIN=${DOMAIN_NAME}
        export SPLIT=${SPLIT_NAME}
        export EVAL_NAME="${DOMAIN}-${SPLIT}"
        cd src/genrec/hllm && CUDA_VISIBLE_DEVICES=1 torchrun \
            --master_port=$master_port \
            --node_rank=0 \
            --nproc_per_node=1 \
            --nnodes=1 \
            run.py \
            --config_file overall/LLM_ddp_full.yaml HLLM/HLLM-test.yaml
        cd ..
        done
    done

else

    if [ ${MODE} == "sem_id" ]; then
        export BEAM_WIDTH=20
    else
        export BEAM_WIDTH=5
    fi

    RECOMPUTE=true

    if [ ${RECOMPUTE} == true ]; then
    for CUR_DOMAIN in ${DOMAINS[@]}; do
        export DOMAIN=${CUR_DOMAIN}
        for SPLIT_ID in ${SPLITS[@]}; do
            export SPLIT=${SPLIT_ID}

            if [ ${SPLIT} == "pretrain" ]; then
                export EPOCH=2
            else
                export EPOCH=1
            fi
            
            export CHECKPOINT_PATH="${CHECKPOINT_DIR}/${DOMAIN}-${SPLIT}-${MODE}/epoch_${EPOCH}"
            # export CHECKPOINT_PATH="${CHECKPOINT_DIR}/merged"

            if [ ${MODE} == "sem_id" ]; then
            python -m src.genrec.semid.add_tokens \
                --checkpoint_dir ${CHECKPOINT_DIR} \
                --domain ${DOMAIN} \
                --split ${SPLIT} \
                --epoch ${EPOCH} \
                --checkpoint_path ${CHECKPOINT_PATH} \
                --base_model_path ${MODEL_ZOO_PATH}/Qwen3-0.6B
            fi

            CUDA_VISIBLE_DEVICES=${GPU_ID} python -m src.eval.generate \
                --model_path ${CHECKPOINT_PATH} \
                --mode ${MODE} \
                --split ${SPLIT} \
                --domain ${DOMAIN} \
                --beam_width ${BEAM_WIDTH} \
                --sample_num 5000
        done
    done
    fi

    python -m src.eval.eval --mode ${MODE} \
        --domain $(echo ${DOMAINS[@]} | tr ' ' ' ') \
        --split $(echo ${SPLITS[@]} | tr ' ' ' ') \
        --beam_size ${BEAM_WIDTH} \
        --gpu_id ${GPU_ID} \
        --embed_model_path ${MODEL_ZOO_PATH}/Qwen3-Embedding-8B \
        --rescale
fi