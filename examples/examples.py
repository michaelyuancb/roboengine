import pdb
import os
import imageio
import numpy as np
from PIL import Image
from robo_engine.infer_engine import RoboEngineRobotSegmentation
from robo_engine.infer_engine import RoboEngineObjectSegmentation
from robo_engine.infer_engine import RoboEngineAugmentation


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


def image_example(image_fp, prompt_image, engine_robo_seg, engine_obj_seg, engine_bg_aug):
    image_np = np.array(Image.open(image_fp))
    print("image read, shape:", image_np.shape)

    # =============================== Segmentation ===============================
    mask_robot = engine_robo_seg.gen_image(image_np=image_np)
    mask_obj = engine_obj_seg.gen_image(image_np=image_np, instruction=prompt_image, verbose=True)

    mask = ((mask_robot + mask_obj) > 0).astype(np.float32)
    Image.fromarray((mask_robot*255).astype(np.uint8)).save('image_mask_robot.png')
    Image.fromarray((mask_obj*255).astype(np.uint8)).save('image_mask_obj.png')
    Image.fromarray((mask*255).astype(np.uint8)).save('image_mask.png')

    # =============================== Augmentation ===============================
    aug_image = engine_bg_aug.gen_image(image_np, mask, tabletop=True, verbose=True)
    Image.fromarray(aug_image).save(f'image_aug_result_{engine_bg_aug.aug_method}.png')


def video_example(video_fp, prompt_video, engine_robo_seg, engine_obj_seg, engine_bg_aug, video_anchor_frequency=8):
    video = imageio.get_reader(video_fp)
    fps = video.get_meta_data()['fps']
    import ipdb; ipdb.set_trace()
    image_np_list = [frame for frame in video]
    print("video read, num frames:", len(image_np_list))

    # =============================== Segmentation ===============================
    robo_masks = engine_robo_seg.gen_video(image_np_list=image_np_list, anchor_frequency=video_anchor_frequency)
    obj_masks = engine_obj_seg.gen_video(image_np_list=image_np_list, instruction=prompt_video, anchor_frequency=video_anchor_frequency) 

    masks = ((robo_masks + obj_masks) > 0).astype(np.float32)
    write_video(robo_masks, 'video_mask_robot.mp4', fps)
    write_video(obj_masks, 'video_mask_obj.mp4', fps)
    write_video(masks, 'video_mask.mp4', fps)

    # =============================== Augmentation ===============================
    masks_np_list = [mask for mask in masks]
    if engine_bg_aug.aug_method in ['engine', 'background', 'inpainting']:
        aug_images = engine_bg_aug.gen_image_batch(image_np_list, masks_np_list, batch_size=8, num_inference_steps=5, tabletop=True, verbose=True)
    elif engine_bg_aug.aug_method in ['texture', 'imagenet', "black"]:
        aug_images = []
        for image_np, mask in zip(image_np_list, masks_np_list):
            aug_images.append(engine_bg_aug.gen_image(image_np, mask, tabletop=True, verbose=False))
        aug_images = np.array(aug_images)
    else:
        raise ValueError(f"Invalid augmentation method: {engine_bg_aug}")
    write_video(aug_images, f"video_aug_result_{engine_bg_aug.aug_method}.mp4", fps)
    


if __name__ == "__main__":

    engine_robo_seg = RoboEngineRobotSegmentation()
    engine_obj_seg = RoboEngineObjectSegmentation()
    engine_bg_aug = RoboEngineAugmentation(aug_method='engine')   # recommand: engine, texture

    image_org_fp = 'image_original.png'
    prompt_image = "fold the towel."
    video_org_fp = 'video_original.mp4'
    prompt_video = "fold the towel."

    # # Data-scaling data
    # prompt_video = "pick place mouse"
    # assert len(prompt_video.split(" ")) > 1  # must has separate words to get object name
    # # https://huggingface.co/datasets/Fanqi-Lin/GoPro-Raw-Videos/tree/main/pick_place_mouse/env_1/object_1
    # video_org_fp = f"/datadrive/andyw/Data-Scaling-Laws/data/GoPro-Raw-Videos/pick_place_mouse/env_1/object_1/1.mp4"
    
    # image_example(image_org_fp, prompt_image, engine_robo_seg, engine_obj_seg, engine_bg_aug)
    video_example(video_org_fp, prompt_video, engine_robo_seg, engine_obj_seg, engine_bg_aug)