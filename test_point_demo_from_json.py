import torch
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import json

def show_mask(mask, ax, random_color=False, borders=True, custom_color=None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif custom_color is not None:
        color = np.array(custom_color + [0.6])  # 使用自定义颜色
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])  # 默认颜色
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, save_dir, point_coords=None, box_coords=None, input_labels=None, borders=True, base_name="image", custom_color=None):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)  # 确保绘制原始图像
        show_mask(mask, plt.gca(), borders=borders, custom_color=custom_color)  # 在原始图像上叠加 mask
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())  # 在原始图像上叠加 points
        if box_coords is not None:
            show_box(box_coords, plt.gca())  # 在原始图像上叠加 box
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')  # 隐藏坐标轴
        # 保存叠加了原始图像、mask 和 points 的图片，名称与原始图片名称对应
        plt.savefig(f"{save_dir}/{base_name}_mask_with_points_{i+1}.png")
        plt.close()

checkpoint = "checkpoints/sam2.1_hiera_large/sam2.1_hiera_large.pt"
model_cfg ="configs/sam2.1/sam2.1_hiera_l.yaml"
device = torch.device("cuda")
sam2_model = build_sam2(model_cfg, checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model, max_hole_area=1.0, max_sprinkle_area=1.0)

image_dir = "/data3/ljj/proj/vggt/examples/llff_flower"  # 修改为文件夹路径
save_dir = f"test_output/{image_dir.rsplit('/', 1)[-1]}_image_masks1"
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

# JSON 文件路径
json_path = os.path.join(image_dir, "points.json")

# 从 JSON 文件中读取数据
with open(json_path, "r") as f:
    data = json.load(f)

for item in data:
    image_path = os.path.join(image_dir, item["image_path"])  # 从 JSON 获取图片路径
    points = np.array(item["points"])  # 从 JSON 获取点
    input_label = np.ones(len(points), dtype=int)  # 假设所有点的标签为 1

    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=points,
            point_labels=input_label,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        mask_input = logits[np.argmax(scores), :, :]
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=input_label,
            mask_input=mask_input[None, :, :],
            multimask_output=False,
        )

    # 保存每张图片的结果
    base_name = os.path.splitext(os.path.basename(item["image_path"]))[0]  # 提取图片名称
    show_masks(image, masks, scores, save_dir, input_labels=input_label, borders=False, base_name=base_name, custom_color=[30/255, 255/255, 144/255], point_coords=points) 
