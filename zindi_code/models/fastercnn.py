import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator


def create_model(num_classes, model_path=None, pretrained=True, model_version=2, anchor_reduction=1):
    loading_local_weights = isinstance(model_path, str)
    
    if loading_local_weights:
        checkpoint = torch.load(model_path)["model"]
        args = torch.load(model_path)["args"]

        try:
            model_version = args.model_version
        except:
            if "roi_heads.box_head.fc7.bias" in checkpoint:
                model_version = 1
            else:
                model_version = 2
        
    print("Model version: ", model_version)
    # load Faster RCNN pre-trained model
    if model_version == 1:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=None if (loading_local_weights or not pretrained) else torchvision.models.detection.faster_rcnn.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
    else:
        # load Faster RCNN V2
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            pretrained=(not loading_local_weights and pretrained)
        )
        
    model.transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(
        min_size=800,
        max_size=1333,
        image_mean=[0.485, 0.456, 0.406],  # [0.485, 0.456, 0.406]
        image_std=[0.229, 0.224, 0.225],
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.backbone.body.layer1.append(torch.nn.Dropout(p=0.2))

    if anchor_reduction > 1:
        anchor_generator = AnchorGenerator(
            sizes=((32//anchor_reduction,), (64//anchor_reduction,), (128//anchor_reduction,), (256//anchor_reduction,), (512//anchor_reduction,)),  # Tailles d'ancres plus petites que celles par défaut (32, 64, 128, ...)
            aspect_ratios=((0.5, 1.0, 2.0),)*5,  # Ratios des ancres (peut être ajusté si nécessaire)
        )
        model.rpn.anchor_generator = anchor_generator
        
    if model_path is not None:
        checkpoint = torch.load(model_path)["model"]
        model.load_state_dict(checkpoint)

    return model

if __name__ == "__main__":
    model = create_model(4)

    print(model)
