export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
MODEL_NAME="bigscience/T0"
OUTPUT_DIR="./output/T0_11B"
DATASET_NAME="super_glue"
DATASET_CONFIG_NAME="wic"
TEMPLAETE_NAME="GPT-3-prompt"
PROMPT_MODE="original"

### eval all templates ###
python evaluation/run_eval.py \
    --dataset_name $DATASET_NAME \
    --dataset_config_name $DATASET_CONFIG_NAME \
    --template_name $TEMPLAETE_NAME \
    --model_name_or_path $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --parallelize \
    --eval_all_templates \
    --prompt_mode $PROMPT_MODE