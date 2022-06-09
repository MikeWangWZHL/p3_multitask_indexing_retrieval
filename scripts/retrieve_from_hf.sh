export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

SHARD_NAMES="p3_subset_6_6_multichoice_qa_new"
# SHARD_NAMES="p3_subset_6_6_multichoice_qa"
# SHARD_NAMES="p3_subset_6_3-part-2 p3_subset_6_1"
# SHARD_NAMES="p3_subset_6_3-part-1"

# DATASET_NAME="super_glue"
# DATASET_CONFIG_NAME="wic"
# DATASET_NAME="cos_e"
# DATASET_CONFIG_NAME="v1.11"
DATASET_NAME=$1
DATASET_CONFIG_NAME=$2

# TEMPLAETE_NAME="exam"

### retrieve train split as well ###
OUTPUT_ROOT="/cephfs/user/mikeeewang/summer_22/code/t-zero/evaluation/retrieved_dataset_train_validation"
python evaluation/get_retrieved_dataset_from_hf.py \
    --shard_names $SHARD_NAMES \
    --dataset_name $DATASET_NAME \
    --dataset_config_name $DATASET_CONFIG_NAME \
    --output_retrieved_dataset_root $OUTPUT_ROOT \
    --eval_all_templates \
    --retrieve_train \
    --use_faiss_gpu
    # --template_name $TEMPLAETE_NAME \


# ### retrieve only validation split ###
# OUTPUT_ROOT = "/cephfs/user/mikeeewang/summer_22/code/t-zero/evaluation/retrieved_dataset_validation"
# python evaluation/get_retrieved_dataset.py \
#     --shard_names $SHARD_NAMES \
#     --dataset_name $DATASET_NAME \
#     --dataset_config_name $DATASET_CONFIG_NAME \
#     --output_retrieved_dataset_root $OUTPUT_ROOT \
#     --eval_all_templates \
#     --use_faiss_gpu
#     # --retrieve_train \
#     # --template_name $TEMPLAETE_NAME \

