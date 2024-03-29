SD_BASE=$1
LORA_PATH=$2
PROMPT_FILE=$3

if [[ "$SD_BASE" != "sd15" && "$SD_BASE" != "sdxl" ]]; then
  echo "Sorry! we only support sd base model sd15 and sdxl now"
  exit 1
fi

echo "==>> Starting generating images..."

if [[ "$SD_BASE" == "sd15" ]]
then
  python src/customization/inference_sd15.py \
    --prompt_file=${PROMPT_FILE} \
    --lora_weights_dir="${LORA_PATH}" \
    --output_dir="./sd15_output" \
    
else
  python src/customization/inference_sdxl.py \
    --prompt_file=${PROMPT_FILE} \
    --lora_weights_dir="${LORA_PATH}" \
    --output_dir="./sdxl_output" \

fi

