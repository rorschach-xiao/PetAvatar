LORA_PATH=$1
PROMPT_FILE=$2
OUTPUT_DIR=$3
STEP=$4
GUIDANCE_SCALE=$5
LORA_ALPHA=$6
SD_BASE=$7

echo "==>> Starting generating videos..."

# generate config file
if [[ "$SD_BASE" == "sd15" ]]
then
  if [[ "${LORA_PATH}" != "None" ]]
  then
    conda run -n animatediff python src/animation/config_generator.py --sd_base ${SD_BASE} --lora_weight_path ${LORA_PATH} --prompt_file_path ${PROMPT_FILE} --step ${STEP} --guidance_scale ${GUIDANCE_SCALE} --lora_alpha ${LORA_ALPHA}
    mv ./5-lora.yaml src/animation/AnimateDiff_sd15/configs/prompts/

    # copy lora weight
    cp ${LORA_PATH} src/animation/AnimateDiff_sd15/models/DreamBooth_LoRA/lora.safetensors

    # generate video
    cd ./src/animation/AnimateDiff_sd15
    CUR_PATH=`pwd`
    conda run -n animatediff python -m scripts.animate --config configs/prompts/5-lora.yaml --output_dir ${CUR_PATH}/../../../sd15_video/${OUTPUT_DIR}

    # remove the temporary lora weight
    rm ./models/DreamBooth_LoRA/lora.safetensors

  else
    conda run -n animatediff python src/animation/config_generator.py --sd_base ${SD_BASE} --prompt_file_path ${PROMPT_FILE} --step ${STEP} --guidance_scale ${GUIDANCE_SCALE}
    mv ./5-lora.yaml src/animation/AnimateDiff_sd15/configs/prompts/

    # generate video
    cd ./src/animation/AnimateDiff_sd15
    CUR_PATH=`pwd`
    conda run -n animatediff python -m scripts.animate --config configs/prompts/5-lora.yaml --output_dir ${CUR_PATH}/../../../sd15_video/${OUTPUT_DIR}
  fi

else 
  if [[ "${LORA_PATH}" != "None" ]]
  then
    conda run -n animatediff_xl python src/animation/config_generator.py --sd_base ${SD_BASE} --lora_weight_path ${LORA_PATH} --prompt_file_path ${PROMPT_FILE} --step ${STEP} --guidance_scale ${GUIDANCE_SCALE} --lora_alpha ${LORA_ALPHA}
    mv ./5-lora.yaml src/animation/AnimateDiff_sdxl/configs/prompts/

    # copy lora weight
    cp ${LORA_PATH} src/animation/AnimateDiff_sdxl/models/DreamBooth_LoRA/lora.safetensors

    # generate video
    cd ./src/animation/AnimateDiff_sdxl
    CUR_PATH=`pwd`
    conda run -n animatediff_xl python -m scripts.animate --exp_config configs/prompts/5-lora.yaml --H 1024 --W 1024 --L 16 --xformers --output_dir ${CUR_PATH}/../../../sdxl_video/${OUTPUT_DIR}

    # remove the temporary lora weight
    rm ./models/DreamBooth_LoRA/lora.safetensors

  else
    conda run -n animatediff_xl python src/animation/config_generator.py --sd_base ${SD_BASE} --prompt_file_path ${PROMPT_FILE} --step ${STEP} --guidance_scale ${GUIDANCE_SCALE}
    mv ./5-lora.yaml src/animation/AnimateDiff_sdxl/configs/prompts/

    # generate video
    cd ./src/animation/AnimateDiff_sdxl
    CUR_PATH=`pwd`
    conda run -n animatediff_xl python -m scripts.animate --exp_config configs/prompts/5-lora.yaml --H 1024 --W 1024 --L 16 --xformers --output_dir ${CUR_PATH}/../../../sdxl_video/${OUTPUT_DIR}

  fi
fi

echo "==>> Video generation completed!"