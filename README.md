# PetAvatar

This project uses DreamBooth + LoRA to generate customized model for your own pets, and uses AnimateDiff to animate your pets.


> pet images from user
![user pet](./imgs/vis.png)

<!-- <style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style> -->


<div class="center">

|prompt| static result|animated result|
|:---:|:---:|:---:|
|`a photo of a TOK collie dog wearing sunglasses and driving a car, high resolution, realistic, 4k, high quality`|![](imgs/output_a_photo_of_a_TOK_collie_dog_wearing_sunglasses_and_driving_a_car,_high_quality,_realistic,_4k,_high_resolution.png?width=760&height=760)|![](imgs/2024-03-31T15-38-31.gif)|
|`a photo of a TOK collie dog playing in the snowfield, mountain in background, high resolution, realistic, 4k, high quality`|![](imgs/output_a_photo_of_a_TOK_collie_dog_playing_in_the_snowfield,_mountain_in_background,_high_quality,_realistic,_4k,_high_resolution.png?width=760&height=760)|![](imgs/2024-03-31T15-48-14.gif)|
|`a photo of a TOK collie dog swimming in the sea, setting sun in background, orange sky, high resolution, realistic, 4k, high quality`|![](imgs/output_a_photo_of_a_TOK_collie_dog_swimming_in_the_sea,_setting_sun_in_background,_orange_sky,_high_quality,_realistic,_4k,_high_resolution.png?width=760&height=760)|![](imgs/2024-03-31T15-57-55.gif)|
|`a photo of a TOK collie dog standing in the Colosseum in Rome, high resolution, realistic, 4k, high quality`|![](imgs/output_a_photo_of_a_TOK_collie_dog_standing_in_the_Colosseum_in_Rome,_high_quality,_realistic,_4k,_high_resolution.png?width=760&height=760)|![](imgs/2024-03-31T16-07-38.gif)|

</div>

## Preparations
### Setup repo and create environment

```shell
git clone --recurse-submodules https://github.com/rorschach-xiao/PetAvatar.git
git submodule update --recursive --init

cd PetAvatar

conda env create -f environment.yaml
conda activate diffuser
```

### Setup animatediff submodule
#### prepare animatediff-sd15
- go to src/animation/AnimateDiff-sd15
- follow [readme](src/animation/AnimateDiff_sd15/README.md) to create a conda environment named `animatediff`, also put stable-diffsuion-sd15 repo/motion-module into corresponding folders

#### prepare animatediff-sdxl
- go to src/animation/AnimateDiff-sdxl
- follow [readme](src/animation/AnimateDiff_sdxl/README.md) to create a conda environment named `animatediff_xl`, also put stable-diffsuion-sdxl repo/motion-module into corresponding folders


## Usage
### prepare your data
put the 3-5 images of same pet in `PetAvatar/${YOUR PET NAME}`, make sure they are in `jpeg` format. The folder structure looks like

```
- PetAvatar
  |- ${YOUR PET NAME}
    |- *.jpeg
    |- *.jpeg
    |- *.jpeg
    |- ...
```

run the following scripts to generate captions for your pet images:

```shell
python src/customization/preprocessing_data.py --img_dir ${YOUR PET NAME} --vis_dir ${YOUR PET_NAME}_vis --breed ${BREED} --species ${SPECIES}

e.g.
python src/customization/preprocessing_data.py --img_dir gulu --vis_dir gulu_vis --breed collie --species dog
```

### train your own dreambooth-lora model
SD15 based model will yield bad results sometimes, SDXL based model is recommended.

```shell
# train sd15 based dreambooth-lora
conda activate diffuser
bash scripts/train_dreambooth_lora_sd15.sh ${INPUT_DATA_DIR} ${MODEL_OUTPUT_DIR} ${SPECIES} ${BREED}
e.g. 
bash scripts/train_dreambooth_lora_sd15.sh gulu lora_weights_sd15/gulu dog collie

# train sdxl based dreambooth-lora
conda activate diffuser
bash scripts/train_dreambooth_lora_sdxl.sh ${INPUT_DATA_DIR} ${MODEL_OUTPUT_DIR} ${SPECIES} ${BREED}
e.g. 
bash scripts/train_dreambooth_lora_sdxl.sh gulu lora_weights_sdxl/gulu dog collie

```

### generate customized images
Generate costomized images using the fine-tuned lora model in previous step. You can change the prompts in ./prompts/prompts.txt to obtain different outputs. The results can be found under `./sd15_output/${OUTPUT_dir}` or `./sdxl_output/${OUTPUT_dir}`

```shell
# if your lora model is based on SD15
bash scripts/gen_image_inference.sh sd15 ${LORA_PATH} ${PROMPT_FILE}
e.g.
bash scripts/gen_image_inference.sh sd15 ./lora_weights_sd15/gulu/pytorch_lora_weights.safetensors ./prompts/prompts.txt

# if your lora model is based on SDXL
bash scripts/gen_image_inference.sh sdxl ${LORA_PATH} ${PROMPT_FILE}
e.g.
bash scripts/gen_image_inference.sh sdxl ./lora_weights_sdxl/gulu/pytorch_lora_weights.safetensors ./prompts/prompts.txt

```

### generate customized videos
Generate costomized videos using the fine-tuned lora model in previous step. You can change the prompts in ./prompts/prompts.txt to obtain different outputs. The results can be found under `./sd15_video/${OUTPUT_dir}` or `./sdxl_video/${OUTPUT_dir}`

```shell
# if your lora model is based on SD15
conda activate animatediff
bash scripts/gen_video_inference.sh ${LORA_PATH} ${PROMPT_FILE} ${OUTPUT_DIR} ${STEP} ${GUIDANCE_SCALE} ${LORA_ALPHA} sd15

e.g.
bash scripts/gen_image_inference.sh ./lora_weights_sd15/gulu/pytorch_lora_weights.safetensors ./prompts/prompts.txt gulu 50 8.5 0.85 sd15

# if your lora model is based on SDXL
conda activate animatediff_xl
bash scripts/gen_video_inference.sh ${LORA_PATH} ${PROMPT_FILE} ${OUTPUT_DIR} ${STEP} ${GUIDANCE_SCALE} ${LORA_ALPHA} sdxl
e.g.
bash scripts/gen_image_inference.sh ./lora_weights_sd15/gulu/pytorch_lora_weights.safetensors ./prompts/prompts.txt gulu 100 9.5 0.85 sdxl
```

## Limitations

- SD15 baesd model requires high quality(clean background, multi views) pet images to produce satifying results.
- animatediff-sd15 can product good motion, while it results in low quality object sometimes; animatediff-sdxl can produce high qualtiy object, while its motion is not very satisfying.


## Citations
```
@article{guo2023animatediff,
  title={AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning},
  author={Guo, Yuwei and Yang, Ceyuan and Rao, Anyi and Liang, Zhengyang and Wang, Yaohui and Qiao, Yu and Agrawala, Maneesh and Lin, Dahua and Dai, Bo},
  journal={International Conference on Learning Representations},
  year={2024}
}

@article{ruiz2022dreambooth,
  title={DreamBooth: Fine Tuning Text-to-image Diffusion Models for Subject-Driven Generation},
  author={Ruiz, Nataniel and Li, Yuanzhen and Jampani, Varun and Pritch, Yael and Rubinstein, Michael and Aberman, Kfir},
  booktitle={arXiv preprint arxiv:2208.12242},
  year={2022}
}

@article{DBLP:journals/corr/abs-2106-09685,
  author       = {Edward J. Hu and
                  Yelong Shen and
                  Phillip Wallis and
                  Zeyuan Allen{-}Zhu and
                  Yuanzhi Li and
                  Shean Wang and
                  Weizhu Chen},
  title        = {LoRA: Low-Rank Adaptation of Large Language Models},
  journal      = {CoRR},
  volume       = {abs/2106.09685},
  year         = {2021},
  url          = {https://arxiv.org/abs/2106.09685},
}
```




