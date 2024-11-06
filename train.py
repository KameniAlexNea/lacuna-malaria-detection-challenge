import argparse
import datetime
import logging
import math
import os
import sys
import time
from pathlib import Path
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import traceback
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import zindi_code.utils as utils
from zindi_code.eval_utils import readable_map_dict, readable_mar_dict

from zindi_code import CLS_MAPPER, CLASSES_REVERSE, COLOR_PALETTE, NUM_CLASS

from zindi_code.dataset_class import get_dataloader
from zindi_code.eval import evaluate_all_metrics
from zindi_code.models.fastercnn import create_model

torch.manual_seed(24)
random.seed(24)
np.random.seed(24)

plt.style.use("ggplot")


def train_one_epoch(data_loader, model, optimizer, lr_scheduler, device, epoch, writer, batch_accum=1):
    model.train()

    for it, data in enumerate(tqdm(data_loader)):
        batch_idx = it
        it = len(data_loader) * epoch + it

        images, targets = data
        images = list(image.to(device) for image in images)

        targets = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
            for t in targets
        ]
        
        # for target in targets:
        #     if len(target["boxes"]) == 0:
        #         print(target)
        #         if isinstance(data_loader.dataset, torch.utils.data.Subset):
        #             print(data_loader.dataset.dataset.all_images[target["image_id"]])
        #         else:
        #             print(data_loader.dataset.all_images[target["image_id"]])
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # calcul de la loss sur le nombre de sorties
        model.eval()
        with torch.no_grad():
            pred_results = model(images)
        
            count_loss_labels = {label: [] for label in range(NUM_CLASS)}
            
            for i in range(len(pred_results)):
                pred_targets = pred_results[i]["labels"]
                
                for label in range(NUM_CLASS):
                    nb_predicted_labels = torch.count_nonzero(pred_targets == label).item()
                    nb_true_labels = torch.count_nonzero(targets[i]["labels"] == label).item()

                    count_loss_labels[label].append(abs(nb_predicted_labels - nb_true_labels))
            
            loss_labels = 0
            for label in count_loss_labels:
                loss_labels += np.mean(count_loss_labels[label])
            loss_labels /= NUM_CLASS
            
            losses += loss_labels     
                    
        model.train()    
        losses /= batch_accum
        losses.backward()
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = loss_dict
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)
        
        if (batch_idx + 1) % batch_accum == 0 or (batch_idx + 1) == len(data_loader):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        writer.add_scalar("Loss/Loss", losses_reduced, it)
        for k, v in loss_dict_reduced.items():
            writer.add_scalar(f"Loss/{k}", v, it)
        writer.add_scalar("Parameters/Learning-Rate", optimizer.param_groups[0]["lr"], it)
        writer.add_scalar("Parameters/Weight-decay", optimizer.param_groups[0]["weight_decay"], it)
        

def main(args, classes, color_palette, device):
    train_loader = get_dataloader(
        is_training=True,
        batch_size=args.batch_size,
        num_workers=8,
        nb_items=args.nb_items
    )
    valid_loader = get_dataloader(
        is_training=False,
        batch_size=args.batch_size,
        num_workers=8,
        nb_items=args.nb_items
    )

    # initialize the model and move to the computation device
    model = create_model(
        num_classes=NUM_CLASS, model_path=args.from_checkpoint, model_version=args.model_version,
        anchor_reduction=args.anchor_reduction, pretrained=True
    )

    model = model.to(device)

    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # define the optimizer
    optimizer = torch.optim.Adam(
        params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=len(train_loader) * args.epochs, pct_start=0.1
    )

    writer = SummaryWriter(log_dir=args.output_dir)
    with open(args.output_dir + "network_architecture.txt", "w") as f:
        print(model, "\n\n\n", optimizer, "\n\n\n", scheduler, file=f)
        print("Batch size: ", args.batch_size, " Batch accum: ", args.batch_accum, " LR: ", args.lr)

    best_small_diff = None
    actual_small_diff = float("inf")
    # start the training epochs
    for epoch in range(args.epochs):
        print(f"\nEPOCH {epoch+1} of {args.epochs}")
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            device,
            epoch,
            writer=writer,
            batch_accum=args.batch_accum
        )
        coco_results, val_score, count_pred_labels, count_true_labels, _ = evaluate_all_metrics(model, valid_loader, device=device)
        
        print("\nValidation score: ", val_score)
        writer.add_scalar("Val Box score", val_score, epoch)

        actual_small_diff = 0
        for label in count_pred_labels:
            mean_pred = np.mean(count_pred_labels[label])
            mean_true = np.mean(count_true_labels[label])
            actual_small_diff = max(actual_small_diff, abs(mean_pred - mean_true))
            print(f"Label {CLASSES_REVERSE[str(label)]}:::: Mean Predicted count: {mean_pred} | Mean True count: {mean_true}")
            writer.add_scalars(f"Label {CLASSES_REVERSE[str(label)]}/Mean Count true and predicted instances|img", {"Predicted count": mean_pred, 
                    "True count": mean_true}, epoch)
            
        checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": args,
                "epoch": epoch
            }
        
        # ============ checkpointing ... ========
        if (epoch % args.saveckp_freq == 0 and epoch >= 75) or epoch == args.epochs - 1:
            torch.save(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))

        if best_small_diff is None or best_small_diff > actual_small_diff :
            best_small_diff = actual_small_diff
            print("Best Mean difference: {} | Epoch {}".format(best_small_diff, epoch+1))
            torch.save(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

        writer.add_scalars(
            "Validation Metrics/Average Precision (@ bbox IoU with ground truth)",
            readable_map_dict(coco_results),
            epoch,
        )
        writer.add_scalars(
            "Validation Metrics/Average Recall (@ bbox IoU with ground truth)",
            readable_mar_dict(coco_results),
            epoch,
        )

        utils.log_one_detection(
            model,
            epoch,
            valid_loader.dataset,
            output_dir=os.path.join(args.output_dir, "imgs"),
            iou_threshold=args.nms_iou,
            writer=writer,
            device=device,
            classes=classes,
            color_palette=color_palette,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("View Detection (FasterRCNN)")

    parser.add_argument(
        "--model_version", default=2, type=int, help="Model version."
    )
    parser.add_argument(
        "--nb_trainable_layers", default=3, type=int, help="Number of trainable layers (0 to 5)."
    )
    parser.add_argument(
        "--custom_transform", default=1, type=int, help="Check if we apply custom transform or not"
    )
    parser.add_argument(
        "--saveckp_freq", default=10, type=int, help="Save checkpoint every x epochs."
    )

    # Training/Optimization parameters
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Per-GPU batch-size : number of distinct images loaded on one GPU.",
    )
    
    # Training/Optimization parameters
    parser.add_argument(
        "--batch_accum",
        default=4,
        type=int,
        help="Batch Accumulation",
    )
    
    parser.add_argument("--nb_items", default=0, type=int, help="Number of items in dataset.")
    parser.add_argument("--anchor_reduction", default=1, type=int, help="The division factor of anchor parameters (1,2,4)")

    parser.add_argument("--epochs", default=130, type=int, help="Number of epochs of training.")

    parser.add_argument(
        "--lr",
        default=0.001,
        type=float,
        help="""Learning rate at the end of
        linear warmup (highest LR used during training).""",
    )

    parser.add_argument("--seed", default=42, type=int, help="Random seed.")

    parser.add_argument(
        "--num_workers", default=0, type=int, help="Number of data loading workers per GPU."
    )

    parser.add_argument(
        "--output_dir",
        default="/workspace/logs/",
        type=str,
        help="Path to save logs and checkpoints.",
    )

    parser.add_argument(
        "--from_checkpoint",
        default=None,
        type=str,
        help="Checkpoint to load from",
    )

    parser.add_argument(
        "--nms_iou", default=0.6, type=float, help="Non Maximum Suppression threshold for logging"
    )

    args = parser.parse_args()
    args.output_dir = os.path.join(
        args.output_dir,
        "model" + "-V" + str(args.model_version) + datetime.datetime.today().strftime("_%Y-%m-%d@%H.%M.%S") + "-anchor_factor_" + str(args.anchor_reduction) + "-lr_" + str(args.lr) + "/",
    )
    
    print("\n" * 3, "Output logdir", args.output_dir, "\n" * 3)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(args.output_dir + "infos.txt", "w") as f:
        # print("Batch size: ", args.batch_size, " Batch accum: ", args.batch_accum, " Without zooming in/out", file=f)
        print(args, file=f)
        
    try:
        main(
            args,
            classes=CLASSES_REVERSE,
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            color_palette=COLOR_PALETTE,
        )
    except Exception as e:
        with open(args.output_dir + "error_log.txt", "w") as f:
            traceback.print_exc(file=f)

# /mnt/DISKE/EFA7/Zindi/code/logs/Faster-RCNN-V2_2024-04-22@14.31.40 intéressant