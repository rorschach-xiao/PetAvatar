import argparse
import os

from diffusers import DiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_file', type=str, 
                        help='text file containing input prompts for image generation.')
    parser.add_argument('--lora_weights_dir', type=str, 
                        help='lora weights directory.', default='')
    parser.add_argument('--output_dir', type=str,  
                        help='output directory', default='sd15_output/')
    
    return parser.parse_args()

def main(args):
    pipe = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        use_safetensors=True
    )
    if args.lora_weights_dir != '':
        repo_id = args.lora_weights_dir
        assert os.path.exists(repo_id), f"{repo_id} does not exist."
        pipe.load_lora_weights(repo_id)
    _ = pipe.to("cuda")
    savedir = os.path.join(args.output_dir, os.path.dirname(args.lora_weights_dir).split("/")[-1])
    os.makedirs(savedir, exist_ok = True)
    with open(args.prompt_file, "r") as f:
        for prompt in f.readlines():
            prompt = prompt.replace("\n", "")
            print('==>> Generating image for prompt: ', prompt)
            image = pipe(prompt=prompt, num_inference_steps=25).images[0]
            output_path = os.path.join(savedir, 
                                f"output_{prompt}.png")
            image.save(output_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)



