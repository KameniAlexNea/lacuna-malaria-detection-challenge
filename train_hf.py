import os


os.environ["WANDB_PROJECT"] = "lacuna_zindi_challenge"
# os.environ["WANDB_LOG_MODEL"] = "true"
os.environ["WANDB_WATCH"] = "none"
os.environ["WANDB_NOTEBOOK_NAME"] = "train_hf"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import (
    AutoModelForObjectDetection,
    # DetrForObjectDetection,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    ConditionalDetrForObjectDetection
)

from zindi_code import MODEL_NAME, TRAIN_CSV, VAL_CSV, CLS_MAPPER
from zindi_code.dataset import (
    collate_fn,
    train_augmentation,
    eval_augmentation
)
from zindi_code.dataset_loader import load_dataset
from zindi_code.transforms import IMAGE_PROCESSOR
from zindi_code.metrics_hug import eval_compute_metrics_fn


# from zindi_code.hf_alex.model import DetrForObjectDetection

train_set = load_dataset(TRAIN_CSV, nan_frac=0).with_transform(train_augmentation)
eval_set = load_dataset(VAL_CSV, False).with_transform(eval_augmentation)

print("Dataset Shape", len(train_set), len(eval_set))
examples = train_set[0]
print(examples)

print("Start Training : ", os.getpid(), MODEL_NAME)

label2id = CLS_MAPPER
id2label = {j: i for i, j in label2id.items()}
model: ConditionalDetrForObjectDetection = AutoModelForObjectDetection.from_pretrained(
    MODEL_NAME,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
    # revision="main",
)


(training_args,) = HfArgumentParser(TrainingArguments).parse_args_into_dataclasses()


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_set,
    eval_dataset=eval_set,
    processing_class=IMAGE_PROCESSOR,
    compute_metrics=eval_compute_metrics_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
)

trainer.evaluate()
trainer.train()
trainer.evaluate()
