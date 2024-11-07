import albumentations as A
from transformers import AutoImageProcessor, DetrImageProcessor

from zindi_code import MODEL_NAME, W, H

IMAGE_SHAPE = 1333
TRAIN_TRANSFORM = A.Compose(
    [
        # A.LongestMaxSize(max_size=IMAGE_SHAPE, p=1),
        A.RandomCrop(H, W, pad_if_needed =True, p=0.1),
        A.RandomScale(0.2, p=0.4),
        A.Rotate(limit=179, p=0.3),
        # A.RandomRotate90(0.5),
        # A.Blur(blur_limit=3, p=0.2),
        A.GaussNoise(var_limit=(0.002, 0.01), p=0.2),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        # A.RandomBrightnessContrast(p=0.3),
    ],
    bbox_params=A.BboxParams(
        format="coco", label_fields=["category"], min_visibility=0.3, clip=True
    ),
)

EVAL_TRANSFORM = A.Compose(
    [
        A.NoOp(),
    ],
    bbox_params=A.BboxParams(
        format="coco",
        label_fields=["category"],
    ),
)

IMAGE_PROCESSOR: DetrImageProcessor = AutoImageProcessor.from_pretrained(
    MODEL_NAME,  # if (MODEL_NAME != "nielsr/yolov10n") else "hustvl/yolos-small",
    # revision="no_timm", use_fast=True
)
