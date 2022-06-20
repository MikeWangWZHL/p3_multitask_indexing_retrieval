K=$1
MAX_LENGTH=974 # n_token = 50
OUTPUT_DIR="/cephfs/user/mikeeewang/summer_22/workspace/data/p3_finetuning/max_length_${MAX_LENGTH}"

SHARD_NAMES="mulcqa_mixture_k-1_6-10_n-2"
INTERMEDIATE_DIR="/cephfs/user/mikeeewang/summer_22/code/p3_retrieval_and_processing/evaluation/retrieved_dataset_train_validation/mulcqa_mixture_k-1_6-10_n-2"


python evaluation/get_retrieved_dataset_from_disk.py \
    --dataset_name openbookqa \
    --dataset_config_name main \
    --k $K \
    --max_length $MAX_LENGTH \
    --output_dir $OUTPUT_DIR \
    --intermediate_dir ${INTERMEDIATE_DIR} \
    --shard_names ${SHARD_NAMES}
echo "done openbookqa_main"

python evaluation/get_retrieved_dataset_from_disk.py \
    --dataset_name piqa \
    --dataset_config_name None \
    --k $K \
    --max_length $MAX_LENGTH \
    --output_dir $OUTPUT_DIR \
    --intermediate_dir ${INTERMEDIATE_DIR} \
    --shard_names ${SHARD_NAMES}
echo "done piqa"

# python evaluation/get_retrieved_dataset_from_disk.py \
#     --dataset_name race \
#     --dataset_config_name high \
#     --k $K
# echo "done race_high"

# python evaluation/get_retrieved_dataset_from_disk.py \
#     --dataset_name race \
#     --dataset_config_name middle \
#     --k $K
# echo "done race_middle"

# python evaluation/get_retrieved_dataset_from_disk.py \
#     --dataset_name super_glue \
#     --dataset_config_name boolq \
#     --k $K
# echo "done boolq"

# python evaluation/get_retrieved_dataset_from_disk.py \
#     --dataset_name super_glue \
#     --dataset_config_name multirc \
#     --k $K
# echo "done multirc"


