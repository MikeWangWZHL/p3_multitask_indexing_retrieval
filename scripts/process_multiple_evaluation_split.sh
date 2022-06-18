K=$1

python evaluation/get_retrieved_dataset_from_disk.py \
    --dataset_name openbookqa \
    --dataset_config_name main \
    --k $K
echo "done openbookqa_main"

python evaluation/get_retrieved_dataset_from_disk.py \
    --dataset_name piqa \
    --dataset_config_name None \
    --k $K
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


