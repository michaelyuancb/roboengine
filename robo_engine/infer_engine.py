import argparse
import torch 
import os
import json
import pdb
import yaml
import hydra
import spacy
import random
import time
import numpy as np
from pathlib import Path
from box import Box
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2.functional import resize
from huggingface_hub import hf_hub_download, snapshot_download

from robo_engine.robo_sam.dataset_roboseg import extract_noun_phrases_with_adjectives
from robo_engine.robo_aug_diffusion.infer import get_sd_pipeline, infer_image, infer_image_batch
from robo_engine.robo_sam.infer import sam_infer
from robo_engine.robo_sam.infer_video import init_video_models, sam_video_infer
from robo_engine.utils.utils import image_crop_resize

try:
    from utils.stable_free_diffusion import StableFreeDiffusion
except:
    print("StableFreeDiffusion Not Found. <background> augmentation mode will not work.")
try:
    from utils.stable_impainting import StableImpainting
    from sam2_repo.build_sam import build_sam2_hf
    from sam2_repo.automatic_mask_generator import SAM2AutomaticMaskGenerator
except:
    print("SAM2 / StableInpainting Repo Not Found. <inpainting> augmentation mode will not work.")


def fix_sam_fp():
    # FIXME: Any better Solution to set the hydra config path ?
    # script_directory = Path(__file__).resolve().parent
    # current_working_directory = Path.cwd()
    # relative_config_path = Path(os.path.relpath(script_directory, current_working_directory))
    # print("Script Directory: ", script_directory)
    # print("Current Working Directory: ", current_working_directory)
    # print("Relative Config Path: ", relative_config_path)
    # hydra.initialize(config_path=str(relative_config_path), job_name="fix_sam_fp")
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=".", job_name="fix_sam_fp")


class RoboEngineRobotSegmentation:

    def __init__(self, 
                 sam_versoin="michaelyuanqwq/roboengine-sam",
                 sam_tokenizer_version="YxZhang/evf-sam2-multitask",
                 image_size=224, 
                 device='cuda'
                 ):     # "robo_sam", "robo_sam_video"
        self.image_size = image_size
        fix_sam_fp()
        self.robo_video_tokenizer, self.robo_video_sam = init_video_models(sam_tokenizer_version, sam_versoin, device)
        print(f"Robo Engine Robot Segmentation Initialized Successfully.")

    def gen_image(self, image_np, prompt="robot", preset_mask=None):
        mask = np.zeros(image_np.shape[:2])
        if preset_mask is not None:
            mask = mask + preset_mask
        mask = sam_video_infer(self.image_size, [image_np], self.robo_video_tokenizer, self.robo_video_sam, prompt)
        mask = (mask > 0).astype(np.float32)[0]
        return mask

    def gen_video(self, image_np_list, prompt="robot", anchor_frequency=8, preset_masks=None):
        masks = np.zeros((1, 1, 1))
        if preset_masks is not None:
            masks = masks + preset_masks
        vmask = sam_video_infer(self.image_size, image_np_list, self.robo_video_tokenizer, self.robo_video_sam, prompt, anchor_frequency=anchor_frequency)
        masks = masks + vmask
        masks = (masks > 0).astype(np.float32)
        return masks         

from enum import Enum

class SegmentationCue(Enum):
    points: str = "points"
    prompts: str = "prompts"
    

class RoboEngineObjectSegmentation:

    def __init__(self, 
                 sam_versoin="YxZhang/evf-sam2-multitask",
                 sam_tokenizer_version="YxZhang/evf-sam2-multitask",
                 image_size=224, 
                 device='cuda',
                 hf_sam2_version_for_points_based_segmentation="facebook/sam2-hiera-large",
                 segmentation_cue=SegmentationCue.points):    # "grounding_sam", "evf_sam", "evf_sam_video"
        self.nlp = spacy.load("en_core_web_sm")
        self.image_size = image_size
        self.segmentation_cue = segmentation_cue
        fix_sam_fp()

        if segmentation_cue == SegmentationCue.points:
            from sam2.sam2_video_predictor import SAM2VideoPredictor
            self.predictor = SAM2VideoPredictor.from_pretrained(hf_sam2_version_for_points_based_segmentation) 
        else:
            self.obj_video_tokenizer, self.obj_video_sam = init_video_models(sam_tokenizer_version, sam_versoin, device)
        print(f"Robo Engine Object Segmentation Initialized Successfully.")

    def gen_image(self, image_np, instruction=None, preset_mask=None, verbose=False):
        """Generate mask on all objects detected from instruction.
        1. detect a list of nouns
        2. for each noun/object, infer the mask with the object guidance.
        """
        mask = np.zeros(image_np.shape[:2])
        if instruction is not None:
            obj_instruction = extract_noun_phrases_with_adjectives(instruction, self.nlp)
            if len(obj_instruction) == 0:
                raise ValueError(f"No noun of objects get from the instruction [{instruction}].")
            if verbose is True:
                print("#"*60)
                print("Instruction: ", instruction)
                print("Object Instruction: ", obj_instruction)
        if preset_mask is not None:
            mask = mask + preset_mask
        if instruction is None:
            raise ValueError("Instruction is required when obj_seg.evf_sam is True. Give the instruction or set obj_seg.evf_sam to False.")
        for obj_name in obj_instruction:
            mask_obj = sam_video_infer(self.image_size, [image_np], self.obj_video_tokenizer, self.obj_video_sam, "[semantic] " + obj_name)
            mask = mask + mask_obj
        mask = (mask > 0).astype(np.float32)[0]
        return mask

    def gen_video(self, image_np_list, instruction=None, anchor_frequency=8, preset_masks=None, verbose=False):
        masks = np.zeros((1, 1, 1))
        if instruction is not None:
            obj_instruction = extract_noun_phrases_with_adjectives(instruction, self.nlp)
        if preset_masks is not None:
            masks = masks + preset_masks
        if instruction is None:
            raise ValueError("Instruction is required when obj_seg.evf_sam is True. Give the instruction or set obj_seg.evf_sam to False.")
        if len(obj_instruction) == 0:
            raise ValueError(f"No noun of objects get from the instruction [{instruction}].")
        for obj_name in obj_instruction:
            vmask_obj = sam_video_infer(self.image_size, image_np_list, self.obj_video_tokenizer, self.obj_video_sam, "[semantic] " + obj_name, anchor_frequency=anchor_frequency)
            masks = masks + vmask_obj
        masks = (masks > 0).astype(np.float32)
        if verbose is True:
            print("#"*60)
            print("Instruction: ", instruction)
            print("Object Instruction: ", obj_instruction)
        return masks
    
    def gen_video_with_point_labels(self, image_np_list, labels):
        masks = np.zeros((1, 1, 1))
        # TODO: get video_dir from input argument
        from robo_engine.robo_sam.model.segment_anything_2.sam2.sam2_video_inference import get_points_and_labels, propagate_labels_through_videos_with_predictor
        video_dir = Path("/datadrive/andyw/sam2/notebooks/videos/bedroom")
        labels = get_points_and_labels()
        sam2_results = propagate_labels_through_videos_with_predictor(video_dir, labels, self.predictor)

        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in sam2_results:
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        n_frame_dict = len(video_segments.keys())
        pred_mask = [video_segments[i][1][0] for i in range(n_frame_dict)]
        vmask_obj = np.stack(pred_mask, axis=0)
        masks = masks + vmask_obj
        masks = (masks > 0).astype(np.float32)
        return masks



class RoboEngineAugmentation:

    def __init__(self,
                 aug_method,
                 prompt_list_fp="infer_prompt_list.json",
                 engine_controlnet_version="michaelyuanqwq/roboengine-bg-diffusion",
                 engine_sd_base_version="stabilityai/stable-diffusion-2-inpainting",
                 imagenet_trainset_base=None,
                 device='cuda'
                ):
        self.device = device
        # engine, background, inpainting, imagenet, texture, black
        self.aug_method = aug_method
        if "engine" == aug_method:
            self.bg_sd_pipeline = get_sd_pipeline(engine_sd_base_version, engine_controlnet_version, device)
        elif "inpainting" == aug_method:
            self.inpainting_sd = StableImpainting(device)
            fix_sam_fp()
            self.sam2 = build_sam2_hf("facebook/sam2.1-hiera-large")
            self.sam2_mask_generator = SAM2AutomaticMaskGenerator(self.sam2, pred_iou_thresh=0.75, stability_score_thresh=0.9)
        elif "background" == aug_method:
            self.free_sd = StableFreeDiffusion(device)
        elif "imagenet" == aug_method:
            assert imagenet_trainset_base is not None
            base_fp = imagenet_trainset_base
            subset_fp_list = os.listdir(base_fp)
            self.imagenet_pool = [] 
            for subset_fp in tqdm(subset_fp_list, "Loading ImageNet Pool Subset"):
                sfp = os.path.join(base_fp, subset_fp)
                file_list = os.listdir(sfp)
                self.imagenet_pool = self.imagenet_pool + [os.path.join(sfp, file) for file in file_list]
        elif "texture" == aug_method:
            background_root = snapshot_download(repo_id="eugeneteoh/mil_data", repo_type="dataset", allow_patterns="*.png")
            self.texture_pool = sorted(Path(background_root).glob("**/*.png"))
        elif "black" == aug_method:
            pass
        else:
            raise ValueError(f"Augmentation Method [{aug_method}] is not supported.")
        this_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(this_dir, prompt_list_fp), "r") as file:
            self.prompt_list = json.load(file)
        print(f"Robo Engine Augmentation [{aug_method}] Initialized Successfully.")


    def gen_image(self, image_np, mask_np, tabletop=False, prompt=None, seed=None, 
                  num_inference_steps=5,
                  cond_scale=None, 
                  inpainting_num_aug=5,
                  inpainting_min_area_size_ratio=0.025,
                  verbose=False,):

        if verbose is True:
            st = time.time()
        if prompt is None:
            prompt = random.choice(self.prompt_list) + "."
        if seed is None:
            seed = random.randint(0, 1000000)
        if tabletop is True:
            prompt = prompt + " On the table."

        if self.aug_method == "engine":
            if cond_scale is None:
                cond_scale = 1.0
            aug_image = infer_image(Image.fromarray(image_np), mask_np, prompt, self.bg_sd_pipeline, cond_scale=cond_scale, num_images_per_prompt=1, num_inference_steps=num_inference_steps, seed=seed)    
            aug_image = np.array(aug_image[0])
        elif self.aug_method == "inpainting":
            if cond_scale is None:  
                cond_scale = 0.1
            masks = self.sam2_mask_generator.generate(image_np)
            inpainting_min_area_size = int(inpainting_min_area_size_ratio * (image_np.shape[0] * image_np.shape[1] - mask_np.sum()))
            seg_candidates = [mask for mask in masks if (((mask['segmentation'] * mask_np).sum() == 0) and (mask['segmentation'].sum() > inpainting_min_area_size))]
            if len(seg_candidates) == 0:
                seg_candidates = [mask for mask in masks if ((mask['segmentation'] * mask_np).sum() == 0)]
            if len(seg_candidates) == 0:
                aug_image = image_np
                return aug_image
            if len(seg_candidates) > inpainting_num_aug:
                select_idx_list = random.sample(range(len(seg_candidates)), inpainting_num_aug)
            else:
                select_idx_list = range(len(seg_candidates))
            mask_aug_sum = seg_candidates[select_idx_list[0]]['segmentation']
            for select_idx in select_idx_list:
                mask_aug = (seg_candidates[select_idx]['segmentation'] > 0)
                mask_aug_sum = mask_aug_sum + mask_aug
            mask_aug_sum = (mask_aug_sum > 0).astype(np.float32)
            aug_image = self.inpainting_sd.gen(image_np, mask_aug_sum, prompt="objects in " + prompt, num_inference_steps=num_inference_steps, cond_scale=cond_scale)
            aug_image = np.array(aug_image)
        elif self.aug_method == "background":
            aug_image = self.free_sd.gen(image_np, prompt, num_inference_steps=num_inference_steps)
            aug_image = np.array(aug_image) * (1 - mask_np)[:,:,None] + image_np * mask_np[:,:,None]
            aug_image = aug_image.astype(np.uint8)
        elif self.aug_method == "imagenet":
            bg_path = random.choice(self.imagenet_pool)
            background = Image.open(bg_path)
            background = image_crop_resize(background, image_np.shape)
            background = np.array(background)
            if background.ndim == 2:
                background = background[:,:,None]
            aug_image = image_np * mask_np[:,:,None] + background * (1 - mask_np[:,:,None])
            aug_image = aug_image.astype(np.uint8)
        elif self.aug_method == "texture":
            bg_path = random.choice(self.texture_pool)
            h, w = image_np.shape[:2]
            background = Image.open(bg_path).resize((w, h), Image.Resampling.LANCZOS)
            background = np.array(background)
            if background.ndim == 2:
                background = background[:,:,None]
            aug_image = image_np * mask_np[:,:,None] + background * (1 - mask_np[:,:,None])
            aug_image = aug_image.astype(np.uint8)
        elif self.aug_method == "black":
            background = np.zeros_like(image_np)
            aug_image = image_np * mask_np[:,:,None] + background * (1 - mask_np[:,:,None])
            aug_image = aug_image.astype(np.uint8)
        else:
            raise ValueError(f"Augmentation Method [{self.aug_method}] is not supported.")
            
        if verbose is True:
            print("#"*60)
            print("Prompt: ", prompt)
            print("Seed: ", seed)
            print("Conditional Scale: ", cond_scale)
            print("Time Elapsed: ", time.time() - st)
        return aug_image


    def gen_image_batch(self, image_np_list, mask_np_list, batch_size=16, tabletop=False, prompt=None, seed=None, 
                        num_inference_steps=5,
                        cond_scale=None, 
                        inpainting_num_aug=5,
                        inpainting_min_area_size_ratio=0.025,
                        verbose=False,):

        if verbose is True:
            st = time.time()
        n_batch = len(image_np_list) // batch_size
        batch_lr = [(i*batch_size, (i+1)*batch_size) for i in range(n_batch)]
        if len(image_np_list) % batch_size > 0:
            batch_lr.append((n_batch*batch_size, len(image_np_list)))
            n_batch = n_batch + 1
        aug_image_list = []
        if verbose: iter_bar = tqdm(range(n_batch), desc="Generating Augmented Images")
        else: iter_bar = range(n_batch)
        for i in iter_bar:
            l, r = batch_lr[i]
            image_np_batch = image_np_list[l:r]
            mask_np_batch = mask_np_list[l:r]
            if prompt is None:
                prompt_list = [random.choice(self.prompt_list) + "." for _ in range(len(image_np_batch))]
            else:
                prompt_list = [prompt for _ in range(len(image_np_batch))]
            if seed is None:
                seed = random.randint(0, 1000000)
            if tabletop is True:
                prompt_list = [prompt + " On the table." for prompt in prompt_list]
            if self.aug_method == "engine":
                if cond_scale is None:
                    cond_scale = 1.0
                image_batch = [Image.fromarray(image_np) for image_np in image_np_batch]
                aug_images = infer_image_batch(image_batch, mask_np_batch, prompt_list, self.bg_sd_pipeline, cond_scale=cond_scale, num_images_per_prompt=1, num_inference_steps=num_inference_steps, seed=seed)
                aug_image_list = aug_image_list + [np.array(aug_image).astype(np.uint8) for aug_image in aug_images]
            elif self.aug_method == "inpainting":
                if cond_scale is None:  
                    cond_scale = 0.1
                mask_aug_sum_list =[]   
                masks_batch = self.sam2_mask_generator.generate_batch(image_np_batch)
                for (image_np, mask_np, masks_gen) in zip(image_np_batch, mask_np_batch, masks_batch):
                    inpainting_min_area_size = int(inpainting_min_area_size_ratio * (image_np.shape[0] * image_np.shape[1] - mask_np.sum()))
                    seg_candidates = [mask for mask in masks_gen if (((mask['segmentation'] * mask_np).sum() == 0) and (mask['segmentation'].sum() > inpainting_min_area_size))]
                    len_seg_candidates = len(seg_candidates)
                    if len_seg_candidates == 0:
                        seg_candidates = [mask for mask in masks_gen if ((mask['segmentation'] * mask_np).sum() == 0)]
                    if len(seg_candidates) == 0:
                        mask_aug_sum_list.append(mask_np)
                        continue
                    if len(seg_candidates) > inpainting_num_aug:
                        select_idx_list = random.sample(range(len(seg_candidates)), inpainting_num_aug)
                    else:
                        select_idx_list = range(len(seg_candidates))   
                    mask_aug_sum = seg_candidates[select_idx_list[0]]['segmentation'] 
                    for select_idx in select_idx_list[1:]:
                        mask_aug = (seg_candidates[select_idx]['segmentation'] > 0)
                        mask_aug_sum = mask_aug_sum + mask_aug
                    mask_aug_sum = (mask_aug_sum > 0).astype(np.float32)
                    mask_aug_sum_list.append(mask_aug_sum)
                prompt_list = ["objects in "+prompt for prompt in prompt_list]    
                aug_images = self.inpainting_sd.gen_batch(image_np_batch, mask_aug_sum_list, prompt_list= prompt_list, num_inference_steps=num_inference_steps, cond_scale=cond_scale)
                aug_image_list= aug_image_list+[np.array(aug_image).astype(np.uint8) for aug_image in aug_images]
            elif self.aug_method == "background":
                aug_images = self.free_sd.gen_batch(image_np_batch, prompt_list, num_inference_steps=num_inference_steps)
                aug_images = [aug_images[i] * (1 - mask_np_batch[i])[:,:,None] + image_np_batch[i] * mask_np_batch[i][:,:,None] for i in range(len(prompt_list))]
                aug_images = [aug_image.astype(np.uint8) for aug_image in aug_images]
                aug_image_list = aug_image_list + aug_images 
            else:
                raise ValueError(f"Video Augmentation [{self.aug_method}] is not supported, please use gen_image() function.")
                
        if verbose is True:
            print("#"*60)
            print("Prompt: ", prompt_list)
            print("Seed: ", seed)
            print("Conditional Scale: ", cond_scale)
            print("Time Elapsed: ", time.time() - st)

        return aug_image_list
