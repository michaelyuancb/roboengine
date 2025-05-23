import imageio
import numpy as np
from PIL import Image
from robo_engine.infer_engine import RoboEngineRobotSegmentation
from robo_engine.infer_engine import RoboEngineObjectSegmentation
from robo_engine.infer_engine import RoboEngineAugmentation

from PIL import Image
import cv2
import numpy as np

def pil_images_to_video(pil_images, output_path, fps=30, codec='mp4v'):
    # GPT code
    if not pil_images:
        raise ValueError("The list of images is empty.")

    # Convert the first image to determine frame size
    first_frame = cv2.cvtColor(np.array(pil_images[0]), cv2.COLOR_RGB2BGR)
    height, width, layers = first_frame.shape
    frame_size = (width, height)

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    for img in pil_images:
        # Convert PIL image to OpenCV format
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # Resize frame if necessary
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, frame_size)
        video_writer.write(frame)

    video_writer.release()


def get_target_resolution(original_width, original_height, target_width=960):
    """For aspect ratio aware resizing.
    Compute target resolution based on the long edge: width.
    """
    # meta = reader.get_meta_data()
    # original_width, original_height = meta['size']

    scale_factor = target_width / original_width
    target_height = int(original_height * scale_factor)
    return (target_width, target_height)

def write_video(image_np, video_fp, fps):
    value_max = np.max(image_np) 
    if value_max <= 1.0 + 1e-5:
        image_vis_scale = 255.0
    else:
        image_vis_scale = 1.0
    writer = imageio.get_writer(video_fp, fps=fps, format="ffmpeg")
    for image in image_np:
        writer.append_data((image * image_vis_scale).astype(np.uint8))
    writer.close()
    
def denormalize_image_to_256(image_np):
    """denormalize to uint8 type for RoboEngine to process."""
    value_max = np.max(image_np) 
    if value_max <= 1.0 + 1e-5:
        image_vis_scale = 255.0
    else:
        image_vis_scale = 1.0
    return (image_np * image_vis_scale).astype(np.uint8)

def augment_image_util(image_np: np.array, prompt_image, engine_robo_seg, engine_obj_seg, engine_bg_aug, save_augmentation=False):
    # image_np = np.array(Image.open(image_fp))
    # image_np = np.array(Image.open(image_fp))
    print("image read, shape:", image_np.shape)

    # =============================== Segmentation ===============================
    mask_robot = engine_robo_seg.gen_image(image_np=image_np)
    mask_obj = engine_obj_seg.gen_image(image_np=image_np, instruction=prompt_image, verbose=True)
    mask = ((mask_robot + mask_obj) > 0).astype(np.float32)
    if save_augmentation:
        Image.fromarray((mask_robot*255).astype(np.uint8)).save('image_mask_robot.png')
        Image.fromarray((mask_obj*255).astype(np.uint8)).save('image_mask_obj.png')
        Image.fromarray((mask*255).astype(np.uint8)).save('image_mask.png')

    # =============================== Augmentation ===============================
    aug_image = engine_bg_aug.gen_image(image_np, mask, tabletop=True, verbose=True)
    return Image.fromarray(aug_image)


def augment_video_util(video_fp, prompt_video, engine_robo_seg, engine_obj_seg, engine_bg_aug, video_anchor_frequency=8,
                       resolution=(960, 544), save_augmentation=False, save_path="video_aug_result_engine.mp4"):
    video = imageio.get_reader(video_fp, size=resolution)
    # TODO: aspect ratio keeping resizing
    # TODO: add info on source resolution and dest resolution
    fps = video.get_meta_data()['fps']
    image_np_list = [frame for frame in video]
    print("video read, num frames:", len(image_np_list))

    # =============================== Segmentation ===============================
    if engine_obj_seg.segmentation_cue == "points":
        # TODO: interface with frontend to get video and labels
        obj_masks = engine_obj_seg.gen_video_with_point_labels(image_np_list=image_np_list, labels=None) 
    elif engine_obj_seg.segmentation_cue == "prompts":
        obj_masks = engine_obj_seg.gen_video(image_np_list=image_np_list, instruction=prompt_video, anchor_frequency=video_anchor_frequency) 
    robo_masks = engine_robo_seg.gen_video(image_np_list=image_np_list, anchor_frequency=video_anchor_frequency)

    masks = ((robo_masks + obj_masks) > 0).astype(np.float32)
    if save_augmentation:
        write_video(robo_masks, 'video_mask_robot.mp4', fps)
        write_video(obj_masks, 'video_mask_obj.mp4', fps)
        write_video(masks, 'video_mask.mp4', fps)

    # =============================== Augmentation ===============================
    masks_np_list = [mask for mask in masks]
    if engine_bg_aug.aug_method in ['engine', 'background', 'inpainting']:
        aug_images = engine_bg_aug.gen_image_batch(image_np_list, masks_np_list, batch_size=1, num_inference_steps=5, tabletop=True, verbose=True)
    elif engine_bg_aug.aug_method in ['texture', 'imagenet', "black"]:
        aug_images = []
        for image_np, mask in zip(image_np_list, masks_np_list):
            aug_images.append(engine_bg_aug.gen_image(image_np, mask, tabletop=True, verbose=False))
        aug_images = np.array(aug_images)
    else:
        raise ValueError(f"Invalid augmentation method: {engine_bg_aug}")
    write_video(aug_images, save_path, fps)
    return aug_images


class RoboEngineAugmentor:
    def __init__(self, aug_method='engine', segmentation_cue="points"):
        """Object for augment camera obs.
        
        aug_method: support both Engine and texture.
        """
        self.engine_robo_seg = RoboEngineRobotSegmentation()
        self.engine_obj_seg = RoboEngineObjectSegmentation(segmentation_cue="points")
        self.engine_bg_aug = RoboEngineAugmentation(aug_method=aug_method)

    def augment_image(self, image_org_fp: np.ndarray, prompt_image: str):
        """Args:
       
       image_org_fp: np.numpy, CHW 
        
        """
        import torch
        if isinstance(image_org_fp, torch.Tensor):
            image_org_fp = image_org_fp.numpy()

            
        augmented_image = augment_image_util(image_org_fp, prompt_image, self.engine_robo_seg, self.engine_obj_seg, self.engine_bg_aug)
        return augmented_image

    def aug_video(self, video_orig_fp, prompt_video, source_resolution=None, target_width=None, save_path="./video_aug_result_engine.mp4"):
        assert source_resolution[0] >= source_resolution[1], "source resolution should be passed as WH (width is expected to be  longer)"
        target_resolution = get_target_resolution(source_resolution[0], source_resolution[1], target_width)
        print("Target resolution:", target_resolution)
        augmented_video = augment_video_util(video_orig_fp, prompt_video, self.engine_robo_seg, self.engine_obj_seg, self.engine_bg_aug, resolution=target_resolution,
                                             save_path=save_path, save_augmentation=True)
        return augmented_video
    

if __name__ == "__main__":
    video_id = 120 
    prompt_video = "place mouse on the mouse pad"
    assert len(prompt_video.split(" ")) > 1  # must has separate words to get object name
    # https://huggingface.co/datasets/Fanqi-Lin/GoPro-Raw-Videos/tree/main/pick_place_mouse/env_1/object_1
    data_path = f"/datadrive/andyw/Data-Scaling-Laws/data/GoPro-Raw-Videos/pick_place_mouse/env_1/object_1/{video_id}.mp4"
    re_aug = RoboEngineAugmentor(aug_method="engine")
    re_aug.aug_video(data_path, prompt_video, source_resolution=(2704, 2028), target_width=960, save_path=f"/datadrive/andyw/roboengine/artifacts/video_aug_result_engine_{video_id}.mp4")
    # data_path = "/datadrive/andyw/Data-Scaling-Laws/data/Processed-Task-Dataset/arrange_mouse/data/camera0_rgb"
    # dataset = datasets.ImageFolder(root=data_path)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)


    