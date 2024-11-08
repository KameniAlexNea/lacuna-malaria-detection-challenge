import json
import os

import numpy as np
import pandas as pd
from datasets import Dataset
from PIL import Image
from sklearn.preprocessing import LabelEncoder

from zindi_code import IMAGE_FOLDER, TRAIN_CSV, CLS_MAPPER


def _check_nan(category_id):
    # category_id of NEG is len(CLS_MAPPER) => avoid it
    return category_id != len(CLS_MAPPER)


def _check_df_nan(train):
    return train["category_id"] == len(CLS_MAPPER)


def dropna_from_df(data: pd.DataFrame, frac: bool = 1.0):
    na_checker = _check_df_nan(data)
    if not frac:
        return data[~na_checker]
    data_wna = data[~na_checker]
    data_na = data[na_checker]

    return pd.concat(
        [
            data_wna,
            data_na.sample(random_state=41, frac=frac),
        ]
    )


def _load_and_format(path: str):
    data = pd.read_csv(path)
    image_id = data["Image_ID"]

    data["image_id"] = image_id
    data["id"] = data.index.astype(int).values
    data["category_id"] = data["class"].apply(
        lambda x: CLS_MAPPER.get(x, len(CLS_MAPPER))
    )
    data["bbox"] = data[["xmin", "ymin", "xmax", "ymax"]].apply(
        lambda x: [x["xmin"], x["ymin"], x["xmax"] - x["xmin"], x["ymax"] - x["ymin"]],
        axis=1,
    )

    return data[["image_id", "bbox", "category_id", "id"]]


def load_pd_dataframe(data_pth: str, training: bool = False, frac: bool = 1.0):
    train = _load_and_format(data_pth)
    if training:  # drop empty images in training
        print(_check_df_nan(train).sum())
        train = dropna_from_df(train, frac)
        print(_check_df_nan(train).sum())
    train["image_id_int"] = LabelEncoder().fit_transform(train["image_id"])
    return train


def _create_bbox_data(bbox, category, id):
    return {
        "id": id,
        "area": bbox[2] * bbox[3],
        "bbox": bbox,
        "category": category,  # convert 1,2,3 => 0,1,2
    }


def convert_list_to_dict(objects):
    results = {
        "id": [i["id"] for i in objects if i["bbox"][2] and i["bbox"][3]],
        "area": [i["area"] for i in objects if i["bbox"][2] and i["bbox"][3]],
        "bbox": [i["bbox"] for i in objects if i["bbox"][2] and i["bbox"][3]],
        "category": [i["category"] for i in objects if i["bbox"][2] and i["bbox"][3]],
    }
    return results


def create_annotation_img(data: pd.DataFrame):
    image_id = data["image_id"].values[0]
    image_id_int = data["image_id_int"].values[0]
    # (_, bbox, category_id, id, _)
    objects = convert_list_to_dict(
        [
            _create_bbox_data(raw["bbox"], raw["category_id"], raw["id"])
            for _, raw in data.iterrows()
            if _check_nan(raw["category_id"])
        ]
    )

    path = os.path.join(IMAGE_FOLDER, image_id)
    img = Image.open(path)
    width, height = img.size
    return {
        "image_id": image_id_int,
        "image": img,
        "width": width,
        "height": height,
        "objects": objects,
    }


def load_dataset(data_pth=TRAIN_CSV, training: bool = True, nan_frac: bool = 0.0):
    train = load_pd_dataframe(data_pth, training, nan_frac)
    train_features = (
        train.groupby("image_id")[train.columns].apply(create_annotation_img).tolist()
    )
    train_ds = Dataset.from_list(train_features)
    return train_ds
