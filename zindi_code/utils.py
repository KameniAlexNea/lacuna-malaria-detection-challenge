import errno
import os
import random

import albumentations as A
import matplotlib.pyplot as plt
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import draw_bounding_boxes


# define the training transforms
def get_train_transform():
    return A.Compose(
        [
            A.Sequential(
                [
                    # A.Rotate(limit=75, p=0.4),
                    A.Flip(p=0.5),
                    A.Blur(blur_limit=3, p=0.5),
                    A.GaussNoise(var_limit=(0.002, 0.01), p=0.5),
                    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, p=0.5),
                    # A.Affine(scale=(0.5, 1.5), p=0.5),
                    # A.RandomSizedBBoxSafeCrop(RESIZE_TO_H, RESIZE_TO_W,  p=0.5, erosion_rate=0.2),
                    ToTensorV2(p=1.0),
                ]
            )
        ],
        bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
    )


# define the validation transforms
def get_valid_transform():
    return A.Compose(
        [
            ToTensorV2(p=1.0),
        ],
        bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
    )


def get_test_transform():
    return A.Compose(
        [
            ToTensorV2(p=1.0),
        ],
        bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
    )


def get_inference_transform():
    return A.Compose(
        [
            A.Sequential(
                [
                    ToTensorV2(p=1.0),
                ]
            )
        ]
    )


def Convert_bbox(bbox, x_max, y_max, value):
    # bbox format is [y_min, x_min, width, height]
    # the PASCAL VOC format is left, top, right, bottom (left < right, top < bottom)
    variables = bbox[1:-1].split(",")
    variables = [float(var) for var in variables]
    
    x_min, y_min, width, height = variables
    
    left = x_min
    right = min(x_min + width + 1, x_max) # y'a des boxs sous la forme [x,y,0,0], on va ref
    top = y_min
    bottom = min(y_min + height + 1, y_max)

    pascal_format = [int(left), int(top), int(right), int(bottom), value]

    return pascal_format


def apply_nms(orig_prediction, iou_thresh=0.7):
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction["boxes"], orig_prediction["scores"], iou_thresh)
    final_prediction = orig_prediction
    final_prediction["boxes"] = final_prediction["boxes"][keep]
    final_prediction["scores"] = final_prediction["scores"][keep]
    final_prediction["labels"] = final_prediction["labels"][keep]

    return final_prediction


def select_confident_predictions(orig_prediction, min_score=0.7):
    final_prediction = {}
    if isinstance(min_score, list):
        indices = torch.logical_or(torch.logical_and(orig_prediction["scores"] >= min_score[0], orig_prediction["labels"] == 0),
                   torch.logical_or(torch.logical_and(orig_prediction["scores"] >= min_score[1], orig_prediction["labels"] == 1),
                   torch.logical_and(orig_prediction["scores"] >= min_score[2], orig_prediction["labels"] == 2))).nonzero()
    else:
        indices = (orig_prediction["scores"] >= min_score).nonzero()

    final_prediction["boxes"] = orig_prediction["boxes"][indices].squeeze(1)
    final_prediction["scores"] = orig_prediction["scores"][indices].squeeze(1)
    final_prediction["labels"] = orig_prediction["labels"][indices].squeeze(1)

    return final_prediction


def collate_fn(batch):
    return tuple(zip(*batch))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

@torch.inference_mode()
def log_one_detection(
    model,
    epoch,
    valid_dataset,
    output_dir,
    iou_threshold,
    writer,
    device,
    classes,
    color_palette=None,
    nb_predictions=5,
):
    assert nb_predictions > 0 and nb_predictions <= len(valid_dataset)

    selected_indices = random.sample(list(range(len(valid_dataset))), nb_predictions)

    random_images = [valid_dataset[idx][0].to(device) for idx in selected_indices]
    image_ids = [valid_dataset[idx][1]["image_id"] for idx in selected_indices]

    model.eval()
    random_valid_outputs = model(random_images)

    if output_dir is not None:
        mkdir(output_dir)
        output_dir = os.path.join(output_dir, f"Epoch {epoch}")
        mkdir(output_dir)

    for i in range(nb_predictions):
        random_valid_output = random_valid_outputs[i]
        nms_prediction = apply_nms(random_valid_output, iou_thresh=iou_threshold)
        colors = (
            None
            if color_palette is None
            else [color_palette[f"{int(label)}"] for label in nms_prediction["labels"]]
        )
        transformed_image = draw_bounding_boxes(
            pil_to_tensor(to_pil_image(random_images[i]).convert("RGB")),
            nms_prediction["boxes"],
            labels=[
                str(classes[f"{int(nms_prediction['labels'][j])}"])
                + ":  "
                + str(round(nms_prediction["scores"][j].item(), 3))
                for j in range(len(nms_prediction["labels"]))
            ],
            width=2,
            colors=colors,
        )

        if output_dir is not None:
            fname = os.path.join(output_dir, f"epoch{epoch}-id{image_ids[i]}.png")
            plt.imsave(
                fname=fname, arr=transformed_image.permute(1, 2, 0).cpu().numpy(), format="png"
            )

    if writer is not None:
        writer.add_image(
            f"Epoch {epoch}: image {image_ids[-1]} orverlap threshold {iou_threshold}",
            transformed_image,
            global_step=epoch,
        )

    return transformed_image
