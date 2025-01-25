import torch
from transformers import AutoTokenizer, AutoModel
from accelerate.test_utils.testing import get_backend
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from pdf_utils import media_path_to_pil_images
from transformers import TextIteratorStreamer
from threading import Thread
import os
import matplotlib.pyplot as plt
import math
import argparse
import sys
import warnings

'''
build_transform from
https://huggingface.co/OpenGVLab/InternVL2_5-8B-MPO
'''
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

'''
find_closest_aspect_ratio from
https://huggingface.co/OpenGVLab/InternVL2_5-8B-MPO
'''
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

'''
dynamic_preprocess from
https://huggingface.co/OpenGVLab/InternVL2_5-8B-MPO
'''
def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_page_chunks(media_path, input_size=448, max_num=12, num_pages=None):
    
    images = media_path_to_pil_images(media_path, num_pages=num_pages)
    pixel_values = [] # will store (n pages x pdf image size)
    for image in images:
        image = image.convert('RGB')
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        single_img_pixel_values = [transform(image) for image in images]
        single_img_pixel_values = torch.stack(single_img_pixel_values)
        pixel_values.append(single_img_pixel_values)
        
    return pixel_values

def main(args):
    
    print(f"Using Pytorch: {torch.__version__}")
    device, _, _ = get_backend()
    print(f"Device: {device}")

    model_id = "OpenGVLab/InternVL2_5-8B-MPO"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        # low_cpu_mem_usage=True,
        # use_flash_attn=True,
        trust_remote_code=True,
        device_map='auto'
        ).eval()

    print('\n')
    # set the max number of tiles in `max_num`
    print(f"Loading {args.pdf}...")
    pdf_pixel_values = load_page_chunks(args.pdf, max_num=12, num_pages=args.pages) 
    
    if not args.pages:
        args.pages = len(pdf_pixel_values)
    print(f'Loaded {args.pages} page(s)')

    num_patches_list = [page.size(0) for page in pdf_pixel_values]
    pdf_pixel_tensors = torch.cat([page.to(torch.bfloat16).to(device) for page in pdf_pixel_values], dim=0)
    print(f"PDF Tensors Shape: {pdf_pixel_tensors.shape}")
    template = ''
    for i in range(args.pages):
        template += f'Image-{i}: <image>\n'

    prompt = template + args.prompt
    print('\n\n')
    print(f'Generated prompt:\n{prompt}')

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=None)

    ## Shared container for response & history for post-thread work
    response_container = {}

    # TODO is this even necessary to thread? I think no since text streamer is already async
    def threaded_chat():
        response, history = model.chat(
            tokenizer=tokenizer,
            pixel_values=pdf_pixel_tensors,
            question=prompt,
            history=None,
            return_history=True,
            generation_config=dict(max_new_tokens=1024, do_sample=False, streamer=streamer),
            num_patches_list=num_patches_list
        )
        response_container["response"] = response
        response_container["history"] = history

    # Launch inference in a separate thread
    thread = Thread(target=threaded_chat)
    thread.start()

    # Stream generated tokens
    generated_text = ""
    for new_text in streamer:
        print(new_text, end='', flush=True)  # Print in real-time
        generated_text += new_text

    thread.join()  # Wait for the model to finish

    # Get the full response and history
    response = response_container.get("response", generated_text)  # Fallback to what the streamer saw
    history = response_container.get("history", [])

    return 0
 

if __name__ == '__main__':

    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser(description="A script that takes named arguments.")
    
    parser.add_argument("--pdf", type=str, required=True, help="PDF Path")
    parser.add_argument("--prompt", type=str, required=False,default="Please describe the image shortly.", help="Text Prompt")
    parser.add_argument("--pages", type=int, required=False,default=None, help="Number of pages to read from pdf")
    
    args = parser.parse_args()
    main(args)
    