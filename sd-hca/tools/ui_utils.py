import numpy as np
import cv2

def bbox_to_mask(bbox, h=64, w=64, img_w=1440, img_h=2560):
    """bbox: [x1,y1,x2,y2] 原图坐标 -> 64×64 binary mask"""
    x1, y1, x2, y2 = bbox
    mask = np.zeros((h, w), dtype=np.float32)
    x1, x2 = int(x1 * w / img_w), int(x2 * w / img_w)
    y1, y2 = int(y1 * h / img_h), int(y2 * h / img_h)
    mask[:, :] = 1.0
    mask[y1:y2, x1:x2] = 0.0
    return mask