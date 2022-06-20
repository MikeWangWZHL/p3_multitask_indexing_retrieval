K=$1
MAX_LENGTH=974 # n_token = 50
OUTPUT_DIR="/cephfs/user/mikeeewang/summer_22/workspace/data/p3_finetuning/max_length_${MAX_LENGTH}"
SHARD_NAMES="mulcqa_mixture_k-1_6-10_n-2"
INTERMEDIATE_DIR="/cephfs/user/mikeeewang/summer_22/code/p3_retrieval_and_processing/evaluation/retrieved_dataset_train_validation/mulcqa_mixture_k-1_6-10_n-2"
# SHARD_NAMES="p3_subset_6_6_multichoice_qa_new"
# INTERMEDIATE_DIR="/cephfs/user/mikeeewang/summer_22/code/p3_retrieval_and_processing/evaluation/retrieved_dataset_train_validation/p3_subset_6_6_multichoice_qa_new"



python evaluation/get_retrieved_dataset_from_disk.py \
    --dataset_name sciq \
    --dataset_config_name None \
    --k $K \
    --max_length $MAX_LENGTH \
    --output_dir $OUTPUT_DIR \
    --intermediate_dir ${INTERMEDIATE_DIR} \
    --shard_names ${SHARD_NAMES}
echo "done sciq"

python evaluation/get_retrieved_dataset_from_disk.py \
    --dataset_name dream \
    --dataset_config_name None \
    --k $K \
    --max_length $MAX_LENGTH \
    --output_dir $OUTPUT_DIR \
    --intermediate_dir ${INTERMEDIATE_DIR} \
    --shard_names ${SHARD_NAMES}
echo "done dream"

python evaluation/get_retrieved_dataset_from_disk.py \
    --dataset_name cos_e \
    --dataset_config_name v1.11 \
    --k $K \
    --max_length $MAX_LENGTH \
    --output_dir $OUTPUT_DIR \
    --intermediate_dir ${INTERMEDIATE_DIR} \
    --shard_names ${SHARD_NAMES}
echo "done cos_e"

python evaluation/get_retrieved_dataset_from_disk.py \
    --dataset_name cosmos_qa \
    --dataset_config_name None \
    --k $K \
    --max_length $MAX_LENGTH \
    --output_dir $OUTPUT_DIR \
    --intermediate_dir ${INTERMEDIATE_DIR} \
    --shard_names ${SHARD_NAMES}
echo "done cosmos_qa"

python evaluation/get_retrieved_dataset_from_disk.py \
    --dataset_name qasc \
    --dataset_config_name None \
    --k $K \
    --max_length $MAX_LENGTH \
    --output_dir $OUTPUT_DIR \
    --intermediate_dir ${INTERMEDIATE_DIR} \
    --shard_names ${SHARD_NAMES}
echo "done qasc"

python evaluation/get_retrieved_dataset_from_disk.py \
    --dataset_name quartz \
    --dataset_config_name None \
    --k $K \
    --max_length $MAX_LENGTH \
    --output_dir $OUTPUT_DIR \
    --intermediate_dir ${INTERMEDIATE_DIR} \
    --shard_names ${SHARD_NAMES}
echo "done quartz"

python evaluation/get_retrieved_dataset_from_disk.py \
    --dataset_name social_i_qa \
    --dataset_config_name None \
    --k $K \
    --max_length $MAX_LENGTH \
    --output_dir $OUTPUT_DIR \
    --intermediate_dir ${INTERMEDIATE_DIR} \
    --shard_names ${SHARD_NAMES}
echo "done social_i_qa"

python evaluation/get_retrieved_dataset_from_disk.py \
    --dataset_name wiqa \
    --dataset_config_name None \
    --k $K \
    --max_length $MAX_LENGTH \
    --output_dir $OUTPUT_DIR \
    --intermediate_dir ${INTERMEDIATE_DIR} \
    --shard_names ${SHARD_NAMES}
echo "done wiqa"





# python evaluation/get_retrieved_dataset_from_disk.py \
#     --dataset_name quail \
#     --dataset_config_name None \
#     --k $K
# echo "done quail"