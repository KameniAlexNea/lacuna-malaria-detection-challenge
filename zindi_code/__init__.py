import os

DATA_FOLDER = "zindi_data"
TRAIN_CSV = os.path.join(DATA_FOLDER, "TrainDataset.csv")
VAL_CSV = os.path.join(DATA_FOLDER, "ValDataset.csv")
# TEST_CSV = os.path.join(DATA_FOLDER, "Test.csv")
IMAGE_FOLDER = os.path.join(DATA_FOLDER, "images/")

NUM_CLASS = 2
MODEL_NAME = "microsoft/conditional-detr-resnet-50" # "zongzhuofan/co-detr-vit-large-coco" # "microsoft/conditional-detr-resnet-50"  # hustvl/yolos-small "jozhang97/deta-resnet-50-24-epochs" "hustvl/yolos-base" # "jozhang97/deta-resnet-50" # "facebook/detr-resnet-50"

NMS_THR = 0.8
CLS_THR = 0.25

CLS_MAPPER = {"Trophozoite": 0, "WBC": 1}
