from mmdet.apis import init_detector, inference_detector

import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


config = 'pytorch_D-RISE/mmdet_configs/configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'
checkpoint = 'faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'
device = 'cuda:0'

model = init_detector(config, checkpoint, device)

label_names = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
]





image = cv2.imread('dogs.png')
scale = 600 / min(image.shape[:2])
image = cv2.resize(image,
                   None,
                   fx=scale,
                   fy=scale,
                   interpolation=cv2.INTER_AREA)
plt.figure(figsize=(7, 7))
plt.imshow(image[:, :, ::-1])
plt.savefig('1.png')
print('1')




out = inference_detector(model, image)
res = image.copy()
for i, pred in enumerate(out):
    for *box, score in pred:
        if score < 0.5:
            break
        box = tuple(np.round(box).astype(int).tolist())
        print(i, label_names[i], box, score)
        cv2.rectangle(res, box[:2], box[2:], (0, 255, 0), 5)

plt.figure(figsize=(7, 7))
plt.imshow(res[:, :, ::-1])
plt.savefig('2.png')
print('2')





def generate_mask(image_size, grid_size, prob_thresh):
    image_w, image_h = image_size
    grid_w, grid_h = grid_size
    cell_w, cell_h = math.ceil(image_w / grid_w), math.ceil(image_h / grid_h)
    up_w, up_h = (grid_w + 1) * cell_w, (grid_h + 1) * cell_h

    mask = (np.random.uniform(0, 1, size=(grid_h, grid_w)) <
            prob_thresh).astype(np.float32)
    mask = cv2.resize(mask, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
    offset_w = np.random.randint(0, cell_w)
    offset_h = np.random.randint(0, cell_h)
    mask = mask[offset_h:offset_h + image_h, offset_w:offset_w + image_w]
    return mask




def mask_image(image, mask):
    masked = ((image.astype(np.float32) / 255 * np.dstack([mask] * 3)) *
              255).astype(np.uint8)
    return masked





np.random.seed(0)

image_h, image_w = image.shape[:2]

images = []
for _ in range(25):
    mask = generate_mask(image_size=(image_w, image_h),
                         grid_size=(16, 16),
                         prob_thresh=0.5)
    masked = mask_image(image, mask)
    out = inference_detector(model, masked)
    res = masked.copy()
    for pred in out:
        for *box, score in pred:
            if score < 0.5:
                break
            box = tuple(np.round(box).astype(int).tolist())
            cv2.rectangle(res, box[:2], box[2:], (0, 255, 0), 5)
    images.append(res)

fig = plt.figure(figsize=(15, 10))
axes = fig.subplots(5, 5)
for i in range(5):
    for j in range(5):
        axes[i][j].imshow(images[i * 5 + j][:, :, ::-1])
        axes[i][j].axis('off')
plt.tight_layout()
plt.savefig('3.png')
print('3')

  






def iou(box1, box2):
    box1 = np.asarray(box1)
    box2 = np.asarray(box2)
    tl = np.vstack([box1[:2], box2[:2]]).max(axis=0)
    br = np.vstack([box1[2:], box2[2:]]).min(axis=0)
    intersection = np.prod(br - tl) * np.all(tl < br).astype(float)
    area1 = np.prod(box1[2:] - box1[:2])
    area2 = np.prod(box2[2:] - box2[:2])
    return intersection / (area1 + area2 - intersection)
    










def generate_saliency_map(image,
                          target_class_index,
                          target_box,
                          prob_thresh=0.5,
                          grid_size=(16, 16),
                          n_masks=1000,
                          seed=0):
    np.random.seed(seed)
    image_h, image_w = image.shape[:2]
    res = np.zeros((image_h, image_w), dtype=np.float32)
    for _ in tqdm(range(n_masks)):
        mask = generate_mask(image_size=(image_w, image_h),
                             grid_size=grid_size,
                             prob_thresh=prob_thresh)
        masked = mask_image(image, mask)
        out = inference_detector(model, masked)
        pred = out[target_class_index]
        score = max([iou(target_box, box) * score for *box, score in pred],
                    default=0)
        res += mask * score
    return res















out = inference_detector(model, image)
res = image.copy()
counter = 0

for i, pred in enumerate(out):
    for *box, score in pred:
        if score < 0.5:
            break

        counter+=1
        box = tuple(np.round(box).astype(int).tolist())
        print(i, label_names[i], box, score)

        target_box = box
        saliency_map = generate_saliency_map(image,
                                            target_class_index=i,
                                            target_box=target_box,
                                            prob_thresh=0.5,
                                            grid_size=(16, 16),
                                            n_masks=200)

        image_with_bbox = image.copy()
        cv2.rectangle(image_with_bbox, tuple(target_box[:2]), tuple(target_box[2:]),
                      (0, 255, 0), 5)
        plt.figure(figsize=(7, 7))
        plt.imshow(image_with_bbox[:, :, ::-1])
        plt.imshow(saliency_map, cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.savefig('4{}.png'.format(counter))
        print('4{}'.format(counter))




