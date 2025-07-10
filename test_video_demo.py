from sam2.build_sam import build_sam2_video_predictor
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
import re


def show_mask(mask, ax, obj_id=None, random_color=False, save_path=None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
    if save_path:
        plt.savefig(save_path)  # 保存整个绘图
    else:
        ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))



sam2_checkpoint = "checkpoints/sam2.1_hiera_large/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
device = torch.device("cuda")
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = "/data3/ljj/proj/vggt/examples/kitchen_4/images"

# 创建保存分割结果的目录
output_dir = f"./test_output/video/{video_dir.split('/')[-2]}0_video/"
os.makedirs(output_dir, exist_ok=True)
output_click_dir = os.path.join(output_dir, "clicks")
os.makedirs(output_click_dir, exist_ok=True)

frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png", ".bmp"]
]

def natural_sort_key(s):
    """用于自然排序的键函数，将字符串中的数字部分按数值排序"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

frame_names.sort(key=natural_sort_key)

# take a look the first video frame
frame_idx = 0
plt.figure(figsize=(9, 6))
# plt.title(f"frame {frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
plt.axis('off')  # 移除坐标轴


inference_state = predictor.init_state(video_path=video_dir)

predictor.reset_state(inference_state)

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# points = np.array([[1000, 650], [1100, 820]], dtype=np.float32)   ########## room2
# points = np.array([[1200, 500], [1200, 600]], dtype=np.float32) # room1
# points = np.array([[400, 200], [1000, 400]], dtype=np.float32) # room0
# points = np.array([[400, 200], [600, 400]], dtype=np.float32)  # llff_flower_4
points = np.array([[317, 78], [672, 354], [302, 292], [514, 139]], dtype=np.float32)  # kitchen_4 4points


# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1, 1, 1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
# plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
plt.axis('off')  # 移除坐标轴
save_path_click = os.path.join(output_click_dir, f"frame_{ann_frame_idx}_obj_{out_obj_ids[0]}_click.png")
show_points(points, labels, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0], save_path=save_path_click)


# video_segments contains the per-frame segmentation results
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# Ensure results for all frames are saved
for out_frame_idx in range(len(frame_names)):
    plt.figure(figsize=(9, 6))
    # plt.title(f"frame {out_frame_idx}")
    ax = plt.gca()
    ax.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    ax.axis('off')  # 移除坐标轴
    if out_frame_idx in video_segments:
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            save_path = os.path.join(output_dir, f"frame_{out_frame_idx}_obj_{out_obj_id}.png")
            show_mask(out_mask, ax, obj_id=out_obj_id, save_path=save_path)
    plt.close()