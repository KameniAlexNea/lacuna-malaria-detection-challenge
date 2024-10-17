# %%
import os

os.environ["WANDB_PROJECT"] = "lacuna_zindi_challenge"
os.environ["WANDB_LOG_MODEL"] = "false"
os.environ["WANDB_WATCH"] = "none"
os.environ["WANDB_NOTEBOOK_NAME"] = "train_hf"

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, datasets, ops
from torchvision.transforms import v2 as transforms
import lightning as L

from lightning.pytorch.loggers import WandbLogger

torch.set_float32_matmul_precision('medium') # | 'high'

# %%
from lightning.pytorch.utilities.types import EVAL_DATALOADERS

batch_size = 16

class FacesData(L.LightningDataModule):
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		transforms.Resize(size=(800,), max_size=1333),
	])

	@staticmethod
	def convert_inputs(imgs, annot, device, small_thr=0.001):
		"""Conver dataset item to accepted target struture."""
		images, targets = [], []
		for img, annot in zip(imgs, annot):
			bbox = annot['bbox']
			small = (bbox[:, 2] * bbox[:, 3]) <= (img.size[1] * img.size[0] * small_thr)
			boxes = ops.box_convert(bbox[~small], in_fmt='xywh', out_fmt='xyxy')
			output_dict = FacesData.transform({"image": img, "boxes": boxes})
			images.append(output_dict['image'].to(device))
			targets.append({
				'boxes': output_dict['boxes'].to(device),
				'labels': torch.ones(len(boxes), dtype=int, device=device)
			})
		return images, targets
	
	@staticmethod
	def _collate_fn(batch):
		"""Define a collate function to handle batches."""
		return tuple(zip(*batch))

	def train_dataloader(self):
		train_dataset = datasets.WIDERFace(root='./data', split='train', download=True)
		return DataLoader(
			train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=self._collate_fn
		)
	
	def val_dataloader(self):
		train_dataset = datasets.WIDERFace(root='./data', split='val', download=True)
		return DataLoader(
			train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=self._collate_fn
		)


# %%
data = FacesData()

train = data.train_dataloader()

# %%
example = next(iter(train))

# %%
example

# %%
images, targets = FacesData.convert_inputs(example[0], example[1], device="cuda")

# %%
targets

# %%
images

# %%
model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")

# %%
model = model.cuda()
with torch.no_grad():
    preds = model(images, targets)

# %%
preds

# %%
model.eval()

with torch.no_grad():
    preds = model(images)

# %%
preds

# %%
# Use a pretrained Faster R-CNN model from torchvision and modify it
class FaceDetectionModel(L.LightningModule):
	def __init__(self):
		super().__init__()
		self.model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")

	def forward(self, images, targets=None):
		return self.model(images, targets)

	def training_step(self, batch, batch_idx):
		imgs, annot = batch
		images, targets = FacesData.convert_inputs(imgs, annot, device=self.device)
		loss_dict = self.model(images, targets)
		losses = sum(loss for loss in loss_dict.values())
		self.log("loss", losses)
		self.log_dict(loss_dict)
		return losses
	
	# def validation_step(self, batch, batch_idx):
	# 	imgs, annot = batch
	# 	images, targets = FacesData.convert_inputs(imgs, annot, device=self.device)
	# 	loss_dict = self.model(images, targets)
	# 	losses = sum(loss for loss in loss_dict.values())
	# 	self.log("loss", losses)
	# 	self.log_dict(loss_dict)
	# 	return losses

	def configure_optimizers(self):
		return optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)


# %%
data = FacesData()
model = FaceDetectionModel()

wandb_logger = WandbLogger(log_model="none")

trainer = L.Trainer(
    max_epochs=5, precision="16-mixed", log_every_n_steps=50, logger=wandb_logger
)
trainer.fit(model, data)

# %%
model

# %%



