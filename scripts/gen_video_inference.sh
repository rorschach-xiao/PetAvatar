LORA_PATH=$1
PROMPT_FILE=$2
OUTPUT_DIR=$3
STEP=$4
GUIDANCE_SCALE=$5
LORA_ALPHA=$6

echo "==>> Starting generating videos..."

# generate config file
python src/animation/config_generator.py --lora_weight_path ${LORA_PATH} --prompt_file_path ${PROMPT_FILE} --step ${STEP} --guidance_scale ${GUIDANCE_SCALE} --lora_alpha ${LORA_ALPHA}
mv ./5-lora.yaml ./src/animation/AnimateDiff/configs/prompts/

# generate video
cd ./src/animation/AnimateDiff
CUR_PATH=`pwd`
python -m scripts.animate --exp_config configs/prompts/5-lora.yaml --H 1024 --W 1024 --L 16 --xformers --output_dir ${CUR_PATH}/../../../sdxl_video/${OUTPUT_DIR}


echo "==>> Video generation completed!"