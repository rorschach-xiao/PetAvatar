INPUT_DATA_DIR=$1
MODEL_OUTPUT_DIR=$2
SPECIES=$3
BREED=$4

# only support dog and cat species
if [[ "$SPECIES" != "dog" && "$SPECIES" != "cat" ]]; then
  echo "Sorry! we only support dog and cat species now"
  exit 1
fi


#!/usr/bin/env bash
## with prior preservation
accelerate launch src/customization/train_dreambooth_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --instance_data_dir=${INPUT_DATA_DIR} \
  --class_data_dir=${SPECIES} \
  --output_dir=${MODEL_OUTPUT_DIR} \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of TOK ${BREED} ${SPECIES}" \
  --class_prompt="a photo of ${SPECIES}" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --checkpointing_steps=100 \
  --seed="0"

## without prior preservation
# accelerate launch src/customization/train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
#   --instance_data_dir=${INPUT_DATA_DIR} \
#   --output_dir=${MODEL_OUTPUT_DIR} \
#   --instance_prompt="a photo of TOK ${BREED} ${SPECIES}" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=1e-4 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=500 \
#   --checkpointing_steps=100 \
#   --seed="0"

echo "==>> Training completed! Now you can run the inference script to generate images."