export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
MODEL_NAME="bigscience/T0_3B"
OUTPUT_DIR="./output/T0_3B"
# DATASET_NAME="super_glue"
# DATASET_CONFIG_NAME="wic"
# DATASET_NAME="openbookqa"
# DATASET_CONFIG_NAME="main"
DATASET_NAME="piqa"
DATASET_CONFIG_NAME="none"
# TEMPLAETE_NAME="after_reading"
PROMPT_MODE="original"

### eval all templates ###
# accelerate launch evaluation/run_eval_retrieve_online.py \
#     --dataset_name $DATASET_NAME \
#     --dataset_config_name $DATASET_CONFIG_NAME \
#     --model_name_or_path $MODEL_NAME \
#     --output_dir $OUTPUT_DIR \
#     --parallelize \
#     --eval_all_templates \
#     --template_name $TEMPLAETE_NAME \
#     --prompt_mode $PROMPT_MODE

python evaluation/run_eval_retrieve_online.py \
    --dataset_name $DATASET_NAME \
    --dataset_config_name $DATASET_CONFIG_NAME \
    --model_name_or_path $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --parallelize \
    --eval_all_templates \
    --prompt_mode $PROMPT_MODE
    # --template_name $TEMPLAETE_NAME \