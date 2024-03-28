INPUT_DATA_DIR=$1
MODEL_OUTPUT_DIR=$2
SPECIES=$4
BREED=$4


# data preprocessing
python src/customization/preprocessing_data.py \
  --input_dir=${INPUT_DATA_DIR} \
  --output_dir="${INPUT_DATA_DIR}_vis" \
  --species=$SPECIES \
  --breed=$BREED

#Dreambooth LoRA fine-tuning
cd src/customization
accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --dataset_name=${INPUT_DATA_DIR} \
  --output_dir=${MODEL_OUTPUT_DIR} \
  --caption_column="prompt"\
  --mixed_precision="fp16" \
  --instance_prompt="a photo of TOK ${BREED} ${SPECIES}" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=3 \
  --gradient_checkpointing \
  --learning_rate=1e-4 \
  --snr_gamma=5.0 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --max_train_steps=500 \
  --checkpointing_steps=717 \
  --seed="0"

echo "==>> Training completed! Now you can run the inference script to generate images."