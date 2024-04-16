import gradio as gr
import os
from PIL import Image, ImageOps
import shutil
import subprocess


def resize_and_pad_images(ori_images):
    images = []
    target_size = (512, 512)

    # 读取文件夹下的所有图片
    for img in ori_images:
        # 保持比例缩放
        img.thumbnail(target_size, Image.LANCZOS)

        # 添加padding
        padding_color = (255, 255, 255)  # 白色背景，你可以根据需要更改
        padded_img = ImageOps.pad(
            img, target_size, color=padding_color, centering=(0.5, 0.5)
        )

        images.append(padded_img)

    # 将所有图片拼接成一张大图
    if images:
        # 计算拼接后的总宽度和高度
        total_width = target_size[0] * len(images)
        total_height = target_size[1]
        concat_image = Image.new("RGB", (total_width, total_height))

        x_offset = 0
        for img in images:
            concat_image.paste(img, (x_offset, 0))
            x_offset += target_size[0]

        return concat_image
    return None


def save_images(pet_name, images):
    directory = f"{pet_name}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    concat_imgs = []
    for i, file_info in enumerate(images):
        img = Image.open(file_info.name)
        img.save(os.path.join(directory, f"{pet_name}_{i}.jpeg"))
        concat_imgs.append(img)

    vis_imgs = resize_and_pad_images(concat_imgs)
    return vis_imgs


def train_model(pet_name, breed, species, model_type):
    model_output_dir = f"lora_weights_{model_type}/{pet_name}"
    # subprocess.run(["conda", "activate", "diffuser"], shell=True)
    subprocess.run(
        [
            "python",
            "src/customization/preprocessing_data.py",
            "--img_dir",
            pet_name,
            "--vis_dir",
            f"{pet_name}_vis",
            "--breed",
            breed,
            "--species",
            species,
        ]
    )
    subprocess.run(
        [
            "bash",
            f"scripts/train_dreambooth_lora_{model_type}.sh",
            pet_name,
            model_output_dir,
            species,
            breed,
        ]
        # shell=True,
    )
    # Cleanup
    shutil.rmtree(pet_name)
    shutil.rmtree(f"{pet_name}_vis")
    return "Model trained successfully!"


def generate_images(model_type, pet_name, custom_prompt=None):
    lora_path = (
        f"./lora_weights_{model_type}/{pet_name}/pytorch_lora_weights.safetensors"
    )
    prompt_file = "./prompts/cat-prompts.txt" if not custom_prompt else custom_prompt
    output_dir = f"./{model_type}_output/{pet_name}"
    subprocess.run(["conda", "activate", "diffuser"], shell=True)
    subprocess.run(
        ["bash", f"scripts/gen_image_inference.sh", model_type, lora_path, prompt_file],
        shell=True,
    )
    return f"Images generated in {output_dir}"


def generate_videos(
    model_type, pet_name, step, guidance_scale, lora_alpha, custom_prompt=None
):
    lora_path = (
        f"./lora_weights_{model_type}/{pet_name}/pytorch_lora_weights.safetensors"
    )
    prompt_file = "./prompts/cat-prompts.txt" if not custom_prompt else custom_prompt
    output_dir = f"./{model_type}_video/{pet_name}"

    subprocess.run(
        [
            "bash",
            f"scripts/gen_video_inference.sh",
            lora_path,
            prompt_file,
            output_dir,
            str(step),
            str(guidance_scale),
            str(lora_alpha),
            model_type,
        ],
        shell=True,
    )
    return f"Videos generated in {output_dir}"


with gr.Blocks() as app:
    with gr.Tab("Model Traing"):
        image_input = gr.Files(label="Upload Images", file_types=["image"])
        # image_input = gr.Image(label="Upload Image", type="file", multiple=True)
        with gr.Row():
            pet_name_input = gr.Textbox(label="Pet Name")
            breed_input = gr.Textbox(label="Breed")
            species_input = gr.Textbox(label="Species")
            model_type_input = gr.Radio(
                label="Select Model Type", choices=["sd15", "sdxl"]
            )
        with gr.Row():
            upload_btn = gr.Button("Upload Images")
            clear_btn = gr.Button("Clear")
        image_vis_output = gr.Image(type="pil", label="Uploaded Image Visualization")

        train_btn = gr.Button("Train Model")
        progress_bar = gr.Number(label="Training Progress", value=0, visible=False)
        upload_btn.click(
            save_images,
            inputs=[pet_name_input, image_input],
            outputs=[image_vis_output],
        )
        clear_btn.click(lambda: image_input.clear())
        train_btn.click(
            train_model,
            inputs=[pet_name_input, breed_input, species_input, model_type_input],
            outputs=[progress_bar],
        )
    with gr.Tab("Image Generation"):
        custom_prompt_input = gr.Textbox(label="Custom Prompt (Optional)")
        generate_img_btn = gr.Button("Generate Images")
        generate_img_btn.click(
            generate_images,
            inputs=[model_type_input, pet_name_input, custom_prompt_input],
            outputs=[],
        )

    with gr.Tab("Video Generation"):
        custom_prompt_input = gr.Textbox(label="Custom Prompt (Optional)")
        generate_vid_btn = gr.Button("Generate Videos")
        generate_vid_btn.click(
            generate_videos,
            inputs=[
                model_type_input,
                pet_name_input,
                gr.Slider(50, 100, step=1),
                gr.Slider(7.0, 10.0, step=0.1),
                gr.Slider(0.5, 1.0, step=0.05),
                custom_prompt_input,
            ],
            outputs=[],
        )
    # with gr.Row():
    #     with gr.Column():
    #         pet_name_input = gr.Textbox(label="Pet Name")
    #         breed_input = gr.Textbox(label="Breed")
    #         species_input = gr.Textbox(label="Species")
    #         image_input = gr.File(
    #             label="Upload Images", file_types=["image"], file_count=(3, 5)
    #         )
    #         upload_btn = gr.Button("Upload Images")
    #     with gr.Column():
    #         model_type_input = gr.Radio(
    #             label="Select Model Type", choices=["sd15", "sdxl"]
    #         )
    #         train_btn = gr.Button("Train Model")
    #         progress_bar = gr.Number(label="Training Progress", value=0, visible=False)

    # custom_prompt_input = gr.Textbox(label="Custom Prompt (Optional)")
    # generate_img_btn = gr.Button("Generate Images")
    # generate_vid_btn = gr.Button("Generate Videos")


app.launch()
