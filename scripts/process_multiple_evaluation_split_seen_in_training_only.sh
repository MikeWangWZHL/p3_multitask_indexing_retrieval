K=$1
SHARD_NAME=$2

python evaluation/get_retrieved_dataset_from_disk_for_eval_seen_in_training_only.py \
    --dataset_name openbookqa \
    --dataset_config_name main \
    --k $K \
    --shard_name $SHARD_NAME
echo "done openbookqa_main"

python evaluation/get_retrieved_dataset_from_disk_for_eval_seen_in_training_only.py \
    --dataset_name piqa \
    --dataset_config_name None \
    --k $K \
    --shard_name $SHARD_NAME
echo "done piqa"

# python evaluation/get_retrieved_dataset_from_disk_for_eval_seen_in_training_only.py \
#     --dataset_name race \
#     --dataset_config_name high \
#     --k $K \
#     --shard_name $SHARD_NAME
# echo "done race_high"

# python evaluation/get_retrieved_dataset_from_disk_for_eval_seen_in_training_only.py \
#     --dataset_name race \
#     --dataset_config_name middle \
#     --k $K \
#     --shard_name $SHARD_NAME
# echo "done race_middle"

# python evaluation/get_retrieved_dataset_from_disk_for_eval_seen_in_training_only.py \
#     --dataset_name super_glue \
#     --dataset_config_name boolq \
#     --k $K \
#     --shard_name $SHARD_NAME
# echo "done boolq"

# python evaluation/get_retrieved_dataset_from_disk_for_eval_seen_in_training_only.py \
#     --dataset_name super_glue \
#     --dataset_config_name multirc \
#     --k $K \
#     --shard_name $SHARD_NAME
# echo "done multirc"


