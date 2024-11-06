import glob as glob
import os
import csv
import random
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from zindi_code import CLS_MAPPER

from zindi_code import (
    CLS_MAPPER,
    DATA_FOLDER,
    RESIZE_TO_W,
    RESIZE_TO_H
)
from zindi_code.utils import (
    get_test_transform,
    get_train_transform,
    get_valid_transform,
    get_inference_transform,
    mkdir,
    collate_fn
)

def load_and_format(path: str):
    data = pd.read_csv(path)
    image_id = data["Image_ID"]

    data["image_id"] = image_id
    data["id"] = data.index.astype(int).values
    data["category_id"] = data["class"].apply(
        lambda x: CLS_MAPPER.get(x, len(CLS_MAPPER))
    )
    data["bbox"] = data[["xmin", "ymin", "xmax", "ymax"]].apply(
        lambda x: [x["xmin"], x["ymin"], x["xmax"], x["ymax"]],
        axis=1,
    )

    # data.loc[data["Image_ID"] == "NEG", ["bbox", "category_id"]] = np.nan
    return data[["image_id", "bbox", "category_id", "id"]]


# the dataset class
class CustomDataset(Dataset):
    def __init__(
        self,
        width,
        height,
        classes,
        training=True,
        transforms=None,
        save_picture=False,
        resize_img=True,
        keep_empty_boxes=True
    ):
        self.transforms = transforms
        self.height = height
        self.width = width
        self.classes = classes
        self.should_save_picture = save_picture
        self.training = training
        self.resize_img = resize_img
        self.keep_empty_boxes = keep_empty_boxes
        
        if self.training:
            file_name = "TrainDataset"
        else :
            file_name = "ValDataset"
        
        data = load_and_format(os.path.join("/workspace", DATA_FOLDER, file_name + ".csv"))
        
        print("Number of rows without boxes: ", (data["category_id"] == len(CLS_MAPPER)).sum())
        if not self.keep_empty_boxes:
            data = data[data["category_id"] != len(CLS_MAPPER)]
            
        # print(self.data.head())
        data = data.groupby("image_id").agg({
            'bbox': list,
            'category_id': list
        }).reset_index()
        
        self.all_images = data["image_id"]
        self.bboxes = data["bbox"]
        self.category_id = data["category_id"]
        self.bad_images = 0
            
    def load_and_process_one_image(self, idx):
        """Load and process one image and eventually
        resizing it."""
        image_name = self.all_images[idx]
        image_path = os.path.join("/workspace",DATA_FOLDER, "images", image_name)
        image = cv2.imread(image_path)

        if self.resize_img:
            image_resized = cv2.resize(image, (self.width, self.height))
        else:
            image_resized = image
            
        image_resized = image_resized / 255.0
        return image_resized.astype(np.float32), (image.shape[1], image.shape[0]), image_name

    def load_and_process_one_annotation(self, idx, image_width, image_height):

        boxes = self.bboxes[idx]
        labels = self.category_id[idx]

        if len(labels) == 1 and labels[0] == len(CLS_MAPPER):
            # means there are no boxes
            boxes = None
            labels = None
        else:
            # the format of bbox is ymin, xmin, ymax, xmax. I should change it to xmin, ymin, ...
            
            if self.resize_img:
                for i in range(len(boxes)):
                    boxes[i][0] = (boxes[i][0] / image_width) * self.width
                    boxes[i][2] = (boxes[i][2] / image_width) * self.width
                    boxes[i][1] = (boxes[i][1] / image_height) * self.height
                    boxes[i][3] = (boxes[i][3] / image_height) * self.height
            
            
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        return boxes, labels

    def save_picture(self, image_resized, image_name):
        save_path = os.path.join("/workspace", DATA_FOLDER, "input_transformed")
        
        mkdir(save_path)

        image_resized = image_resized.squeeze(0).permute(1, 2, 0).numpy()
        
        im = Image.fromarray((image_resized * 255).astype(np.uint8))
        # im = im.convert("L")
        im.save(os.path.join(save_path, f"{image_name}"))

    def __getitem__(self, idx):
        image_resized, image_shape, image_name = self.load_and_process_one_image(idx)
        image_height = image_shape[1]
        image_width = image_shape[0]

        boxes, labels_classes = self.load_and_process_one_annotation(
            idx, image_width, image_height
        )
        
        image_id = idx
        
        if boxes is None:
            target = {}
            target["boxes"] = torch.empty((0,4))
            target["labels"] = torch.empty((0,), dtype=torch.long) # torch.tensor([], dtype= torch.int64)
            target["area"] = torch.tensor([])
            target["iscrowd"] = torch.tensor([])
            target["image_id"] = image_id
            
            if not self.transforms:
                self.transforms = get_test_transform()

            sample = self.transforms(image=image_resized, bboxes=np.array([]), labels=np.array([]))
            image_resized = sample["image"]

        else:
            # area of the bounding boxes
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # no crowd instances
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
            
            # prepare the final `target` dictionary
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels_classes
            target["area"] = area
            target["iscrowd"] = iscrowd
            target["image_id"] = image_id

            if not self.transforms:
                self.transforms = get_test_transform()
            
            try:
                sample = self.transforms(image=image_resized, bboxes=target["boxes"], labels=labels_classes.numpy())
            except:
                self.bad_images += 1
                return self.__getitem__(random.choice(range(len(self))))
            
            image_resized = sample["image"]
            target["boxes"] = torch.Tensor(sample["bboxes"])
            if len(sample["bboxes"]) == 0:
                target["boxes"] = torch.empty((0,4))
                
        if self.should_save_picture:
            self.save_picture(image_resized, image_name)

        # assert len(target['boxes']) == len(target['labels'])
        
        return image_resized, target
    

    def __len__(self):
        return len(self.all_images)


# prepare the final datasets and data loaders
def create_dataset(is_training=True, save_picture=False, resize_img=False):
    transform = get_train_transform() if is_training else get_valid_transform()

    dataset = CustomDataset(
        RESIZE_TO_W,
        RESIZE_TO_H,
        CLS_MAPPER,
        training=is_training,
        transforms=transform,
        save_picture=save_picture,
        resize_img=resize_img
    )

    return dataset

def create_inference_dataset(save_picture=False, resize_img=True):
    dataset = CustomDataset(
        None,
        RESIZE_TO_W,
        RESIZE_TO_H,
        CLS_MAPPER,
        get_inference_transform(),
        save_picture=save_picture,
        inference_mode=True,
        resize_img=resize_img
    )

    return dataset
    
def get_dataloader(
    batch_size,
    num_workers,
    is_training,
    nb_items=None,
    save_picture=False,
    inference_mode=False
):
    if inference_mode:
        dataset = create_inference_dataset(save_picture=save_picture, resize_img=(batch_size != 1))
    else:
        dataset = create_dataset(
                is_training=is_training,
                save_picture=save_picture, 
                resize_img=(batch_size != 1)
            )
    
    if nb_items is not None and nb_items > 0:
        dataset = torch.utils.data.Subset(dataset, range(nb_items))
    
    if inference_mode:
        print(f"Number of test samples: {len(dataset)}")
    else:
        print(f"Number of {'training' if is_training else 'validation'} samples: {len(dataset)}")
        
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    

def get_mean_var_train_dataset():
    """Dataset statistics computed on train
    - Mean: , Std: 
    """

    train_loader = get_dataloader(
        train=True,
        batch_size=5000,
        num_workers=4
    )
    
    means = []
    stds = []
    for data in train_loader:
        images = torch.stack(data[0])
        means.append(torch.mean(images, dim=(2,3,0)).numpy())
        stds.append(torch.std(images, dim=(2,3,0)).numpy())

    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)
    
    print("Mean: {}, Std: {}".format(mean, std))
    return mean, std


def get_mean_var_dataset_dimensions():
    """On all the dataset
        Height || Mean: 2350, Std: 929
        Width || Mean: 3303, Std: 1013
    """
    image_paths = glob.glob(os.path.join("/workspace", DATA_FOLDER, "images/*.jpg"))
    
    heights = []
    widths = []
    
    for i in tqdm(range(len(image_paths))):
        path = image_paths[i]
        image = cv2.imread(path)
        heights.append(image.shape[0])
        widths.append(image.shape[1])
        
    heightMean = np.mean(heights)
    heightStd = np.std(heights)
    
    widthMean = np.mean(widths)
    widthStd = np.std(widths)
    
    print("Height || Mean: {}, Std: {}".format(heightMean, heightStd))
    print("Width || Mean: {}, Std: {}".format(widthMean, widthStd))
    
    return widthMean, widthStd

def main():
    # test training and validation mode
    dataset = CustomDataset(
        RESIZE_TO_W,
        RESIZE_TO_H,
        CLS_MAPPER,
        training=True,
        transforms=None, # get_train_transform(),
        save_picture=True,
        resize_img=False
    )
    
    print("dataset directory :", DATA_FOLDER)
    print(f"Number of images: {len(dataset)}")

    NUM_SAMPLES_TO_VISUALIZE = np.random.choice(len(dataset) - 1, 300)
    
    # for i in NUM_SAMPLES_TO_VISUALIZE:
    #     image, target = dataset[i]
        
    for i in range(len(dataset)):
        image, target = dataset[i]
        
def get_mean_var_area_dataset():
    """
    Box side || Mean: 72.79011998358712, Std: 48.400166336253896
    Box to Img ratio (%)|| Mean: 0.1234954372048378, Std: 0.10944553464651108
    """
    # test training and validation mode
    dataset = CustomDataset(
        RESIZE_TO_W,
        RESIZE_TO_H,
        CLS_MAPPER,
        training=True,
        transforms=None,
        save_picture=False,
        resize_img=False
    )
    
    box_img_ratio = []
    boxes_sides = []
    
    SAMPLES_TO_VISUALIZE = np.random.choice(len(dataset) - 1, 1000)
    for i in tqdm(range(len(SAMPLES_TO_VISUALIZE))):

        image, target = dataset[SAMPLES_TO_VISUALIZE[i]]
            
        if len(target["labels"]) > 0:
            img_area = image.shape[1]*image.shape[2]
            
            for area in target["area"]:
                boxes_sides.append(np.math.sqrt(area))
                box_img_ratio.append(area*100/img_area)
    
    print("Nb bad images: ", dataset.bad_images)
    box_sides_mean = np.mean(boxes_sides)
    box_sides_std = np.std(boxes_sides)
    
    box_img_ratio_mean = np.mean(box_img_ratio)
    box_img_ratio_std = np.std(box_img_ratio)
    
    print("Box side || Mean: {}, Std: {}".format(box_sides_mean, box_sides_std))
    print("Box to Img ratio (%)|| Mean: {}, Std: {}".format(box_img_ratio_mean, box_img_ratio_std))


if __name__ == "__main__":
    # get_mean_var_area_dataset()
    # get_mean_var_dataset_dimensions()
    main()