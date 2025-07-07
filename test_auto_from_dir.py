from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(3)

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    # color_mask = np.array([1.0, 1.0, 0.0, 0.5])  # 固定为明黄色 (R=1.0, G=1.0, B=0.0, Alpha=0.5)
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)

sam2_checkpoint = "checkpoints/sam2.1_hiera_large/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
device = torch.device("cuda")

image_dir = "/data3/ljj/proj/vggt/examples/room/images/"
output_dir = f"test_output/{image_dir.rsplit('/', 3)[-3]}_automatic_masks"
os.makedirs(output_dir, exist_ok=True)

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2) # , pred_iou_thresh=0.9, use_m2m=True, stability_score_thresh=0.98, crop_n_layers=1

for image_name in os.listdir(image_dir):
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    image_path = os.path.join(image_dir, image_name)
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))

    masks = mask_generator.generate(image)
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks, borders=False)
    plt.axis('off')
    output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_automatic_masks.png")
    plt.savefig(output_path)
    plt.close()