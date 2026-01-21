export CURDIR=$(pwd)
export DATA_DIR=${CURDIR}/data      # path to the data directory
export MODEL_ZOO_PATH=""            # path to the model zoo directory

DOMAINS=(
    # "Video_Games"
    "Movies_and_TV"
    # "Cell_Phones_and_Accessories"
    # "Books"
    # "Sports_and_Outdoors"
    # "Health_and_Household"
)

mkdir -p $DATA_DIR/dataset/Amazon

for DOMAIN in ${DOMAINS[@]}; do
    ## Check file path
    RATINGS_FILE_PATH="$DATA_DIR/dataset/Amazon/${DOMAIN}.csv.gz"
    INFORMATION_FILE_PATH="$DATA_DIR/dataset/Amazon/meta_${DOMAIN}.jsonl.gz"

    if [ ! -f ${RATINGS_FILE_PATH} ]; then
        echo "Error: ${RATINGS_FILE_PATH} not found, try to download..."
        wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/${DOMAIN}.csv.gz
        mv ${DOMAIN}.csv.gz ${RATINGS_FILE_PATH} || true
    fi
    if [ ! -f ${INFORMATION_FILE_PATH} ]; then
        echo "Error: ${INFORMATION_FILE_PATH} not found, try to download..."
        wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_${DOMAIN}.jsonl.gz
        mv meta_${DOMAIN}.jsonl.gz ${INFORMATION_FILE_PATH} || true
    fi

    # Preprocessing
    python -m src.dataset.preprocess --file_path $DATA_DIR/dataset/Amazon --domain $DOMAIN --tokenizer_path ${MODEL_ZOO_PATH}/Qwen3-0.6B --min_date 2017-07-01
    python -m src.dataset.embed --domain $DOMAIN  --model_path ${MODEL_ZOO_PATH}/Qwen3-Embedding-0.6B --gpu_id 0

    # Text-grounded: Formulate dataset
    python -m src.genrec.common.formulator --domain $DOMAIN --index title

    # Semantic ID: Item tokenization
    python -m src.genrec.semid.item_tokenize --domain $DOMAIN --gpu_id 0 --n_layers 4 --cluster_sizes 256 256 256 256
    python -m src.genrec.common.formulator --domain $DOMAIN --index sem_id
    python -m src.genrec.semid.item_formulator --domain $DOMAIN
done