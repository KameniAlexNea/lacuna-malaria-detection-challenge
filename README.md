# README for Object Detection Model Training Script

## Overview

This repository contains a training script designed to fine-tune a pre-trained object detection model using the Hugging Face `transformers` library, with custom datasets prepared for a Zindi challenge. The script is built to facilitate model training and evaluation using a custom object detection model architecture (`ConditionalDetrForObjectDetection`), along with dataset augmentation and evaluation metrics. The Weights & Biases (W&B) integration is used for tracking experiments.

**For fair evaluation, use dataset in `train_validate` folder and don't train on it**

## Requirements

The script assumes that you have the following libraries installed:

- `transformers` (from Hugging Face)
- `torch` (PyTorch backend for model training)
- `wandb` (for tracking model training experiments)
- Custom dependencies such as `zindi_code`, which contains dataset loading, model configurations, and metric computations.

## Environment Variables

The script makes use of environment variables to control specific settings:

- `WANDB_PROJECT`: The name of the project in W&B.
- `WANDB_WATCH`: Controls W&B's behavior for logging (set to "none" to disable watching).
- `WANDB_NOTEBOOK_NAME`: Specifies the name of the notebook or script.
- `CUDA_DEVICE_ORDER` and `CUDA_VISIBLE_DEVICES`: Set the GPU device to be used for training.

## Code Structure

### Imports

The script imports key components from Hugging Face `transformers`, such as `AutoModelForObjectDetection`, `Trainer`, and `TrainingArguments`. It also imports the following custom modules from the `zindi_code` package:

- `MODEL_NAME`, `TRAIN_CSV`, `VAL_CSV`, `CLS_MAPPER`: Configuration and paths for the dataset and model.
- `load_dataset`, `transform_aug_ann`, `collate_fn`: Functions for loading the dataset and applying transformations.
- `IMAGE_PROCESSOR`: Custom preprocessor for the dataset.
- `compute_metrics`: Function to compute evaluation metrics.

### Dataset Preparation

- **Training Set**: Loaded from a CSV file specified by `TRAIN_CSV`. Augmentation transformations are applied to the training data using `transform_aug_ann`.
- **Evaluation Set**: Loaded from `VAL_CSV`. Different transformations are applied for evaluation purposes.

### Model Setup

- The object detection model is loaded using `AutoModelForObjectDetection` (from the Hugging Face Transformers library) and configured with the label mappings defined in `CLS_MAPPER`.
- The model type used in this script is `ConditionalDetrForObjectDetection`, a custom object detection model architecture.

### Training Arguments

Training parameters are passed using the `TrainingArguments` class. These parameters are parsed from the command line using the `HfArgumentParser`.

### Trainer

A `Trainer` instance is created to handle training and evaluation. The trainer is configured with:

- The model
- Training and evaluation datasets
- Custom collate function (`collate_fn`) for data loading
- Image processor (`IMAGE_PROCESSOR`) for input transformation
- Metrics computation (`compute_metrics`)
- Early stopping callback to avoid overfitting

### Model Training and Evaluation

The training process involves evaluating the model before and after training to track performance. The model is fine-tuned using the training dataset, and its performance is evaluated on the validation dataset.

## Running the Script

To run the script, set up the required environment variables and execute it in a suitable Python environment.

```bash
export WANDB_PROJECT="lacuna_zindi_challenge"
export CUDA_VISIBLE_DEVICES="1"
python train_hf.py \
    --output_dir logs \
    --run_name cond-detr-50 \
    --auto_find_batch_size \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 200 \
    --do_eval \
    --do_train \
    --fp16 \
    --learning_rate 1e-4 \
    --weight_decay 1e-6 \
    --save_total_limit 3 \
    --remove_unused_columns false \
    --push_to_hub false \
    --eval_strategy epoch \
    --save_strategy epoch \
    --report_to wandb \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --data_seed 41 \
    --save_safetensors \
    --save_only_model \
    --metric_for_best_model loss \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --dataloader_num_workers 8 \
    --gradient_accumulation_steps 2
```

Make sure to replace any paths or custom dependencies based on your specific setup.

## Customization

- **Model**: You can switch to other object detection models by replacing `ConditionalDetrForObjectDetection` with your desired architecture.
- **Hyperparameters**: Modify the training arguments in the script or pass them as command-line arguments.
- **Metrics**: Adjust or add new metrics by modifying the `compute_metrics` function.

## Dependencies

This script depends on the following:

- Python 3.x
- `transformers` library by Hugging Face
- `torch` for model training on the GPU
- `wandb` for logging (optional)

Make sure to install these dependencies before running the script. You may need additional dependencies if using custom modules like `zindi_code`.
