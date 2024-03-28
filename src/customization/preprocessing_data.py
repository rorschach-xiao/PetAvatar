import argparse
import os
import gc
import glob

from transformers import AutoProcessor, BlipForConditionalGeneration
import torch

from PIL import Image
import json

# visulization utility
def image_grid(imgs, rows, cols, resize=256):

    if resize is not None:
        imgs = [img.resize((resize, resize)) for img in imgs]
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, 
                        help='fine-tuned images directory.')
    parser.add_argument('--vis_dir', type=str, 
                        help='visulization of fine-tuned images')
    parser.add_argument('--species', type=str, 
                        help='species of fine-tuned animals, only support dog and cat now', 
                        choices=['dog','cat'], default='dog')
    parser.add_argument('--breed', type=str, 
                        help='breed of fine-tuned species', default='border collie')
    return parser.parse_args()
    
def main(local_dir, vis_dir, species, breed):
    '''
    This function is used to generate the metadata.jsonl file for the fine-tuned images and the visulization of the fine-tuned images.
    '''
   
    assert os.path.exists(local_dir), f"{local_dir} does not exist."
    os.makedirs(vis_dir, exist_ok = True)
    # generate caption
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # load the processor and the captioning model
    blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base",torch_dtype=torch.float16).to(device)
    # captioning utility
    def caption_images(input_image):
        inputs = blip_processor(images=input_image, return_tensors="pt").to(device, torch.float16)
        pixel_values = inputs.pixel_values
    
        generated_ids = blip_model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_caption
        
    imgs_and_paths = [(path,Image.open(path)) for path in glob.glob(f"{local_dir}*.jpeg")]
    imgs = [Image.open(path) for path in glob.glob(f"{local_dir}*.jpeg")]
    # visulization
    num_imgs_to_preview = 5
    vis_grid = image_grid(imgs[:num_imgs_to_preview], 1, num_imgs_to_preview)
    vis_grid.save(os.path.join(vis_dir,'vis.png'))
    
    caption_prefix = f"a photo of TOK {breed} {species}, "
    with open(f'{local_dir}metadata.jsonl', 'w') as outfile:
      for img in imgs_and_paths:
          caption = caption_prefix + caption_images(img[1]).split("\n")[0]
          entry = {"file_name":img[0].split("/")[-1], "prompt": caption}
          json.dump(entry, outfile)
          outfile.write('\n')
    
    
    # delete the BLIP pipelines and free up some memory
    del blip_processor, blip_model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    args = parse_args()
    main(args.img_dir, args.vis_dir, args.species, args.breed)





