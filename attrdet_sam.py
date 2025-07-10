#!/usr/bin/env python3

import requests
import base64
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os  # 新增导入 os 模块
from dotenv import load_dotenv  # 新增导入

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

def show_masks(image, masks, scores, save_dir, point_coords=None, box_coords=None, input_labels=None, borders=True, custom_color=None):
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
        plt.savefig(save_dir)
        print(f"结果图像已保存到 {save_dir}")
        plt.close()

# 加载 .env 文件
load_dotenv()

# 配置
IMAGE_DIR = "/data3/ljj/proj/vggt/examples/llff_flower/images"
IMAGE_NAMES = ["000.png", "005.png", "010.png", "015.png"]  # 图片名称列表
API_URL = os.getenv("ATTRDET_API_URL")
SAVE_DIR = f"/data3/ljj/proj/sam2/test_output/attrdet_results/{IMAGE_DIR.split('/')[-2]}"
os.makedirs(SAVE_DIR, exist_ok=True)  # 确保保存目录存在

checkpoint = "checkpoints/sam2.1_hiera_large/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
device = torch.device("cuda")
sam2_model = build_sam2(model_cfg, checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model, max_hole_area=1.0, max_sprinkle_area=1.0)

# 遍历图片列表
for image_name in IMAGE_NAMES:
    IMAGE_PATH = os.path.join(IMAGE_DIR, image_name)
    OUTPUT_PATH = f"{SAVE_DIR}/{IMAGE_PATH.split('/')[-3]}_{image_name}"

    # 读取并编码图片
    with open(IMAGE_PATH, 'rb') as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')

    # 发送请求
    payload = {
        "model_id": "attrdet_convnext_vg_run016",
        "data": [image_base64],
        "src_type": "base64",
        "include_classes": ["橙色的植物"],  # 空 = 检测所有类别
        "threshold": 0.01,
        "translate_to_CN": True
    }

    response = requests.post(API_URL, json=payload)

    # 显示结果
    if response.status_code == 200:
        result = response.json()
        objects = result['objects'][0]
        print(f"检测到 {len(objects)} 个对象:")
        print(objects[0])  # 打印所有检测到的对象信息

        # 打开原始图像
        image = Image.open(IMAGE_PATH)
        draw = ImageDraw.Draw(image)
        image_rgb = np.array(image.convert("RGB"))

        # 绘制第一个对象的框
        obj = objects[0]
        xmin, ymin, xmax, ymax = obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']
        # draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

        # 设置 input_box
        input_box = np.array([xmin, ymin, xmax, ymax])

        predictor.set_image(image_rgb)
        # 调用 predictor.predict
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False
        )

        # 绘制并保存结果
        show_masks(image_rgb, masks, scores, save_dir=OUTPUT_PATH, borders=False)
    else:
        print(f"错误: {response.status_code}")