import argparse
import os
import sys

import imageio
import cv2
import numpy as np
import torch
import pdb
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig

from .model.evf_sam2_video import EvfSam2Model


def parse_args(args):
    parser = argparse.ArgumentParser(description="Robo-EVF infer")
    parser.add_argument("--sam_version_tokenizer", default="YxZhang/evf-sam2")
    parser.add_argument("--sam_version", required=True)
    parser.add_argument("--image_size", default=224, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--sam_load_in_8bit", action="store_true", default=False)
    parser.add_argument("--sam_load_in_4bit", action="store_true", default=False)
    parser.add_argument("--video_path", type=str, default="4_high_res.mp4")
    parser.add_argument("--prompt", type=str, default="[robot] robot.")
    parser.add_argument("--anchor_frequency", type=int, default=4, help="The interval of anchor frames for SAM2.")
    parser.add_argument("--save_fp", type=str, default="output.mp4")
    
    return parser.parse_args(args)


def sam_preprocess(
    x: np.ndarray,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024) -> torch.Tensor:
    assert img_size==1024, \
        "both SAM and Effi-SAM receive images of size 1024^2, don't change this setting unless you're sure that your employed model works well with another size."
    x = torch.from_numpy(x).permute(2,0,1).contiguous()
    x = F.interpolate(x.unsqueeze(0), (img_size, img_size), mode="bilinear", align_corners=False).squeeze(0)
    x = (x - pixel_mean) / pixel_std
    resize_shape = None
    
    return x, resize_shape

def beit3_preprocess(x: np.ndarray, img_size=224) -> torch.Tensor:
    beit_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC, antialias=None), 
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return beit_preprocess(x)


def init_video_models(sam_version_tokenizer, sam_version, device="cuda"):
    """Get the tokenizer and SAM model."""
    tokenizer = AutoTokenizer.from_pretrained(
        sam_version_tokenizer,
        padding_side="right",
        use_fast=False,
    )
    torch_dtype = torch.float32
    kwargs = {"torch_dtype": torch_dtype}

    model = EvfSam2Model.from_pretrained(
        sam_version, low_cpu_mem_usage=True, **kwargs
    )

    state_dict = model.state_dict()
    model = model.to_empty(device=device)
    for key, value in state_dict.items():
        if value.device.type != 'meta':
            model.state_dict()[key].copy_(value)
        else:
            print(f"Skipping meta parameter {key} for meta device {value.device}")

    # model = model.to(device)
    model.eval()

    return tokenizer, model


def sam_video_infer(image_size, video,  tokenizer, model, prompt, anchor_frequency=8):
    # clarify IO
    # if not os.path.exists(image_path):
    #     print("File not found in {}".format(image_path))
    #     exit()

    # video could be:
    # (1) a video file path with mp4 format
    # (2) a directory containing (only) frames images
    # (3) a list of frames images, with ndarray format and 0~255 range. 

    # preprocess: get a list of images in numpy format
    image_np_list = []
    if isinstance(video, str) and video.endswith(".mp4"):
        videos = cv2.VideoCapture(video)
        while True:
            ret, frame = videos.read()
            if not ret:
                break
            image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            image_np_list.append(image_np)
    elif isinstance(video, str):
        frame_names = [
            p
            for p in os.listdir(video)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
        ]
        frame_names.sort()
        for image_fp in frame_names:
            image_np = cv2.imread(image_fp)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_np_list.append(image_np)
    else:
        image_np_list = video
    
    # get a list of anchors, scattered evenly
    num_frames = len(image_np_list)
    beit_video_anchor = []
    if anchor_frequency == 0:
        image_np = image_np_list[0]
        image_beit = beit3_preprocess(image_np, image_size).to(dtype=model.dtype, device=model.device)
        beit_video_anchor.append((image_beit.unsqueeze(0), 0))
    else:
        for i in range(0, num_frames, anchor_frequency):
            image_np = image_np_list[i]
            image_beit = beit3_preprocess(image_np, image_size).to(dtype=model.dtype, device=model.device)
            beit_video_anchor.append((image_beit.unsqueeze(0), i))
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device=model.device)
    # pdb.set_trace()
    # infer
    output = model.inference(
        video,
        beit_video_anchor,
        input_ids,
        # resize_list=[resize_shape],
        # original_size_list=original_size_list,
    )
    n_frame_dict = len(output.keys())
    pred_mask = [output[i][1][0] for i in range(n_frame_dict)]
    pred_mask = np.stack(pred_mask, axis=0)
    return pred_mask


def save_visualization(image_np, pred_mask, save_fp):
    masked_image = np.copy(image_np)
    masked_image[~pred_mask] = 255 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    axes[1].imshow(pred_mask, cmap='gray')
    axes[1].set_title("Pred Mask")
    axes[1].axis('off')
    axes[2].imshow(masked_image)
    axes[2].set_title("Masked Image")
    axes[2].axis('off')
    plt.savefig(save_fp, dpi=150)


def save_video(save_fp, masks):
    video_writer = imageio.get_writer(save_fp, fps=30)
    for frame in masks:
        video_writer.append_data(np.uint8(frame) * 255)
    video_writer.close()


def main(args):
    args = parse_args(args)
    # use float16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer, model = init_video_models(args, args.sam_version_tokenizer, args.sam_version, device='cuda')

    video_path = args.video_path 
    prompt = "robot"
    video_path = "/root/project/robo_engine/diffusion_policy/data/realworld_fold_towel_env1/fold_towel_env1_videos/15_high_res.mp4"
    pred_mask = sam_video_infer(args, video_path, tokenizer, model, args.anchor_frequency, prompt)
    save_video(args.save_fp, pred_mask)


    # save_visualization(image_np, pred_mask, "vis.png")
    


if __name__ == "__main__":
    main(sys.argv[1:])

# Need wire, Not need base ? 
# Annotate the wrist camera setting 

# CUDA_VISIBLE_DEVICES=0 python infer_video.py --sam_version /cephfs/shared/yuanchengbo/roboaug/seg_ckpt/EVF-finetuning-ckpts/robot0108003103 --anchor_frequency 8

# CUDA_VISIBLE_DEVICES=0 python infer_video.py --sam_version YxZhang/evf-sam2 --prompt robot --anchor_frequency 8

# prompt: [robot] robot.
# prompt: [object] object1. object2. object3.
# prompt: [task] robot. object1. object2.