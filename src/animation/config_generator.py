from omegaconf import OmegaConf
import argparse
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
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
    conf_dict = {"lora_model_path": args.lora_weight_path,
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