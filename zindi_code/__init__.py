import os

DATA_FOLDER = "zindi_data"
TRAIN_CSV = os.path.join(DATA_FOLDER, "TrainDataset.csv")
VAL_CSV = os.path.join(DATA_FOLDER, "TestDataset.csv")
# TEST_CSV = os.path.join(DATA_FOLDER, "Test.csv")
IMAGE_FOLDER = os.path.join(DATA_FOLDER, "images/")


MODEL_NAME = "microsoft/conditional-detr-resnet-50" # "zongzhuofan/co-detr-vit-large-coco" # "microsoft/conditional-detr-resnet-50"  # hustvl/yolos-small "jozhang97/deta-resnet-50-24-epochs" "hustvl/yolos-base" # "jozhang97/deta-resnet-50" # "facebook/detr-resnet-50"

NMS_THR = 0.8
CLS_THR = 0.25

CLS_MAPPER = {"Trophozoite": 0, "WBC": 1}
NUM_CLASS = len(CLS_MAPPER)


CLASSES_REVERSE = {0: "Trophozoite", 1: "WBC"}
COLOR_PALETTE = {"0": (255, 0, 0), "1": (0, 255, 0)}

RESIZE_TO_W = 800
RESIZE_TO_H = 800

W = 1333
H = 800
