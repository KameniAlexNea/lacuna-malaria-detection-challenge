import argparse
import datetime
import glob as glob
import json
import os
import csv
import numpy as np
import torch

import zindi_code.utils as utils
from zindi_code.eval_utils import (
    evaluate_with_torchmetrics,
    readable_map_dict,
    readable_mar_dict,
)
from zindi_code import CLASSES_REVERSE, NUM_CLASS, RESIZE_TO_W
from zindi_code.dataset_class import get_dataloader
from zindi_code.models.fastercnn import create_model
import random

torch.manual_seed(24)
random.seed(24)
np.random.seed(24)

def evaluate_all_metrics(model, valid_loader, device, iou_thres=None, min_score=None):
    assert (iou_thres is None and min_score is None) or (iou_thres is not None and min_score is not None)
    
    coco_results = evaluate_with_torchmetrics(model, valid_loader, device=device)
    
    count_pred_labels = {label: [] for label in range(NUM_CLASS)}
    count_true_labels = {label: [] for label in range(NUM_CLASS)}
    image_names = []
    
    with torch.no_grad():
        model.eval()
        val_scores = []

        for images, targets in valid_loader:
            images = list(image.to(device) for image in images)
            targets = [
                {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                for t in targets
            ]
            loss_dict_reduced = model(images, targets)

            scores_reduced = [
                torch.mean(loss_dict_reduced[i]["scores"]).item()
                for i in range(len(loss_dict_reduced))
            ]
            score_value = np.mean(scores_reduced)
            val_scores.append(score_value)
            
            for i in range(len(loss_dict_reduced)):
                if isinstance(valid_loader.dataset, torch.utils.data.Subset):
                    image_names.append(valid_loader.dataset.dataset.all_images[targets[i]["image_id"]])
                else:
                    image_names.append(valid_loader.dataset.all_images[targets[i]["image_id"]])
                
                if min_score is not None:
                    loss_dict_reduced[i] = utils.apply_nms(loss_dict_reduced[i], iou_thresh=iou_thres)
                    loss_dict_reduced[i] = utils.select_confident_predictions(loss_dict_reduced[i], min_score=min_score)
            
                for label in range(NUM_CLASS):
                    nb_predicted_labels = torch.count_nonzero(loss_dict_reduced[i]["labels"] == label).item()
                    nb_true_labels = torch.count_nonzero(targets[i]["labels"] == label).item()
                    count_pred_labels[label].append(nb_predicted_labels)
                    count_true_labels[label].append(nb_true_labels)
    
    score_value = np.mean(val_scores)

    return coco_results, score_value, count_pred_labels, count_true_labels, image_names


def get_val_score(
    args,
    classes=CLASSES_REVERSE,
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    batch_size=48,
    num_workers=4,
):

    test_loader = get_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=4,
        adjust_brightness=args.adjust_brightness != 0
    )
    
    model = create_model(
        num_classes=NUM_CLASS, model_path=args.model_path)

    model.to(device)

    return evaluate_all_metrics(model, test_loader, device=device, iou_thres=args.nms_iou, min_score=args.min_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Faster RCNN finetuning")
    parser.add_argument(
        "--model_path",
        default="/workspace/logs/Faster-RCNN-V2_2024-06-06@09.24.09/checkpoint.pth",
        type=str,
        help="Path to model checkpoint.",
    )
    
    parser.add_argument(
        "--nms_iou", default=0.3, type=float, help="Non Maximum Suppression threshold for logging"
    )
    parser.add_argument(
        "--min_score",
        nargs='+',
        default=[0.4, 0.4, 0.4],
        type=float,
        help="Min score to select a box",
    )

    parser.add_argument(
        "--adjust_brightness",
        default=0,
        type=int,
        help="Bool to adjust test brightness or not",
    )
        
    print(RESIZE_TO_W)
    args = parser.parse_args()

    coco_results, score, count_pred_labels, count_true_labels, image_names = get_val_score(args, batch_size=8)

    for label in count_pred_labels:
        mean_pred = np.mean(count_pred_labels[label])
        mean_true = np.mean(count_true_labels[label])
        print(f"Label {CLASSES_REVERSE[str(label)]}:::: Mean Predicted count: {mean_pred} | Mean True count: {mean_true}")
            
    map_results = readable_map_dict(coco_results, precision=10)
    mar_results = readable_mar_dict(coco_results, precision=10)

    map_results.update(mar_results)

    date = datetime.datetime.today().strftime("_%Y-%m-%d@%H.%M.%S")
    filename = os.path.join(
        os.path.dirname(args.model_path),
        "test_scores" + date + ".json",
    )
    with open(filename, "w") as f:
        json.dump(map_results, f, indent=3)
        
    filename = os.path.join(
        os.path.dirname(args.model_path),
        "count_instances" + date + ".csv",
    )
    
    with open(filename, 'w') as f:
        csvwriter = csv.writer(f)
        
        csvwriter.writerow(["image_id", "True count", "Pred count"])
        
        for i in range(len(image_names)):
            
            for label in range(NUM_CLASS):
                img_name = image_names[i] + "_" + str(label+1)
                row = [img_name, count_true_labels[label][i], count_pred_labels[label][i]]
                csvwriter.writerow(row)
    
    