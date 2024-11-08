import json
import os

import numpy as np
import pandas as pd
from datasets import Dataset
from PIL import Image
from sklearn.preprocessing import LabelEncoder

from zindi_code.transforms import (
    EVAL_TRANSFORM,
    IMAGE_PROCESSOR,
    TRAIN_TRANSFORM,
)
from functools import partial
import torch


def format_image_annotations_as_coco(image_id, categories, areas, bboxes):
    annotations = [
        {
            "image_id": image_id,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        }
        for category, area, bbox in zip(categories, areas, bboxes)
    ]

    return {
        "image_id": image_id,
        "annotations": annotations,
    }


def augment_and_transform_batch(
    examples, transform, image_processor, return_pixel_mask=True
):
    images = []
    annotations = []
    for image_id, image, objects in zip(
        examples["image_id"], examples["image"], examples["objects"]
    ):
        image = np.array(image.convert("RGB"))

        # apply augmentations
        output = transform(
            image=image, bboxes=objects["bbox"], category=objects["category"]
        )
        images.append(output["image"])

        # format annotations in COCO format
        formatted_annotations = format_image_annotations_as_coco(
            image_id, output["category"], objects["area"], output["bboxes"]
        )
        annotations.append(formatted_annotations)

    # Apply the image processor transformations: resizing, rescaling, normalization
    result = image_processor(
        images=images, annotations=annotations, return_tensors="pt"
    )

    if not return_pixel_mask:
        result.pop("pixel_mask", None)

    return result


def compute_area(bboxes):
    return [box[2] * box[3] for box in bboxes]


train_augmentation = partial(
    augment_and_transform_batch,
    transform=TRAIN_TRANSFORM,
    image_processor=IMAGE_PROCESSOR,
)

eval_augmentation = partial(
    augment_and_transform_batch,
    transform=EVAL_TRANSFORM,
    image_processor=IMAGE_PROCESSOR,
)


def train_val_split(train_ds: Dataset, test_size=0.1, seed=51):
    dataset = train_ds.train_test_split(test_size=test_size, seed=seed)
    train_set = dataset["train"].with_transform(train_augmentation)
    eval_set = dataset["test"].with_transform(eval_augmentation)
    return train_set, eval_set


def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data
