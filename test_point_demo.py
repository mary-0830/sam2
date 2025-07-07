import torch
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
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

def show_masks(image, masks, scores, save_dir, point_coords=None, box_coords=None, input_labels=None, borders=True, base_name="image"):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)  # 确保绘制原始图像
        show_mask(mask, plt.gca(), borders=borders)  # 在原始图像上叠加 mask
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

image_dir = "/data3/ljj/proj/vggt/examples/kitchen/images"  # 修改为文件夹路径
save_dir = "test_output/kitchen_image_masks"
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]  # 获取所有图片文件

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))

    input_point = np.array([[400, 200], [500, 250]])
    input_label = np.array([1, 1])

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        mask_input = logits[np.argmax(scores), :, :]
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            mask_input=mask_input[None, :, :],
            multimask_output=False,
        )

    # 保存每张图片的结果
    base_name = os.path.splitext(image_file)[0]  # 提取图片名称（如 00, 01）
    show_masks(image, masks, scores, save_dir, point_coords=input_point, input_labels=input_label, borders=True, base_name=base_name)
