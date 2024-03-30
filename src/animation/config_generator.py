from omegaconf import OmegaConf
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sd_base', type=str, 
                        help='sd15 or sdxl', default='sdxl', choices=['sd15', 'sdxl'])
    parser.add_argument('--lora_weight_path', type=str, 
                        help='lora weights path', default='')
    parser.add_argument('--prompt_file_path', type=str,   
                        help='text file that contain input prompts', default='')
    parser.add_argument('--step', type=int,   
                        help='diffusion step', default=100)
    parser.add_argument('--guidance_scale', type=float, 
                        help='guidance scale', default=7.5)
    parser.add_argument('--lora_alpha', type=float,
                        help='lora alpha', default=0.85)
    return parser.parse_args()


def main(args):
    prompts = []
    with open(args.prompt_file_path, "r") as f:
        for prompt in f.readlines():
            prompts.append(prompt.replace("\n", ""))
    if args.sd_base == "sd15":
        conf_dict = {"domain_lora_scale": 1.0,
                     "adapter_lora_path": "models/Motion_Module/v3_sd15_adapter.ckpt",
                     "lora_model_path": "models/DreamBooth_LoRA/lora.safetensors",
                     "inference_config": "configs/inference/inference-v3.yaml",
                     "motion_module": "models/Motion_Module/v3_sd15_mm.ckpt",
                     "seed": -1,
                     "guidance_scale": args.guidance_scale,
                     "step": args.step,
                     "lora_alpha": args.lora_alpha, 
                     "prompt": prompts,
                     "n_prompt": ["bad quality,worst quality" for _ in range(len(prompts))]}
        conf_dict = [conf_dict]
    elif args.sd_base == "sdxl":
        conf_dict = {"lora_path": "models/DreamBooth_LoRA/lora.safetensors",
                    "motion_module_path": "models/Motion_Module/mm_sdxl_v10_beta.ckpt",
                    "seed": -1,
                    "guidance_scale": args.guidance_scale,
                    "step": args.step,
                    "lora_alpha": args.lora_alpha, 
                    "prompt": prompts,
                    "n_prompt": ["bad quality,worst quality" for _ in range(len(prompts))]}
    
    
    conf = OmegaConf.create(conf_dict)
    OmegaConf.save(conf, "./5-lora.yaml")


if __name__ == '__main__':
    args = parse_args()
    main(args)