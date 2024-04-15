import gradio as gr
import os
from PIL import Image
import shutil
import subprocess


def save_images(pet_name, images):
    directory = f"PetAvatar/{pet_name}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i, file_info in enumerate(images):
        with open(os.path.join(directory, f"{pet_name}_{i}.jpeg"), "wb") as f:
            f.write(file_info["content"])
    return "Images saved successfully!"


def train_model(pet_name, breed, species, model_type):
    input_data_dir = f"PetAvatar/{pet_name}"
    model_output_dir = f"lora_weights_{model_type}/{pet_name}"
    subprocess.run(["conda", "activate", "diffuser"], shell=True)
    subprocess.run(
        [
            "bash",
            f"scripts/train_dreambooth_lora_{model_type}.sh",
            input_data_dir,
            model_output_dir,
            species,
            breed,
        ],
        shell=True,
    )
    # Cleanup
    shutil.rmtree(input_data_dir)
    return "Model trained successfully!"


def generate_images(model_type, pet_name, custom_prompt=None):
    lora_path = (
        f"./lora_weights_{model_type}/{pet_name}/pytorch_lora_weights.safetensors"
    )
    prompt_file = "./prompts/cat-prompts.txt" if not custom_prompt else custom_prompt
    output_dir = f"./{model_type}_output/{pet_name}"
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
    with gr.Row():
        with gr.Column():
            pet_name_input = gr.Textbox(label="Pet Name")
            breed_input = gr.Textbox(label="Breed")
            species_input = gr.Textbox(label="Species")
            image_input = gr.File(
                label="Upload Images", file_types=["image"], file_count=(3, 5)
            )
            upload_btn = gr.Button("Upload Images")
        with gr.Column():
            model_type_input = gr.Radio(
                label="Select Model Type", choices=["sd15", "sdxl"]
            )
            train_btn = gr.Button("Train Model")
            progress_bar = gr.Number(label="Training Progress", value=0, visible=False)

    custom_prompt_input = gr.Textbox(label="Custom Prompt (Optional)")
    generate_img_btn = gr.Button("Generate Images")
    generate_vid_btn = gr.Button("Generate Videos")

    upload_btn.click(save_images, inputs=[pet_name_input, image_input], outputs=[])
    train_btn.click(
        train_model,
        inputs=[pet_name_input, breed_input, species_input, model_type_input],
        outputs=[progress_bar],
    )
    generate_img_btn.click(
        generate_images,
        inputs=[model_type_input, pet_name_input, custom_prompt_input],
        outputs=[],
    )
    generate_vid_btn.click(
        generate_videos,
        inputs=[
            model_type_input,
            pet_name_input,
            gr.Slider(50, 100),
            gr.Slider(7.0, 10.0),
            gr.Slider(0.5, 1.0),
            custom_prompt_input,
        ],
        outputs=[],
    )

app.launch()
