import time
from pprint import pprint

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def readable_map_dict(coco_results, precision=5):
    return {
        "map | all": round(coco_results["map"].item(), precision),
        "map_50 | all": round(coco_results["map_50"].item(), precision),
        "map_75 | all": round(coco_results["map_75"].item(), precision),
        "map | small": round(coco_results["map_small"].item(), precision),
        "map | medium": round(coco_results["map_medium"].item(), precision),
        "map | large": round(coco_results["map_large"].item(), precision),
        # "map | perclass": coco_results["map_per_class"].item(),
    }


def readable_mar_dict(coco_results, precision=5):
    return {
        "mar_1 | all": round(coco_results["mar_1"].item(), precision),
        "mar_10 | all": round(coco_results["mar_10"].item(), precision),
        "mar_100 | all": round(coco_results["mar_100"].item(), precision),
        "mar | small": round(coco_results["mar_small"].item(), precision),
        "mar | medium": round(coco_results["mar_medium"].item(), precision),
        "mar | large": round(coco_results["mar_large"].item(), precision),
        # "mar | 100 | perclass": coco_results["mar_100_per_class"].item(),
    }

@torch.inference_mode()
def evaluate_with_torchmetrics(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision()

    total_time = 0
    for data in data_loader:
        images = data[0]
        targets = data[1]
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
        total_time += time.time() - model_time
                
        metric.update(outputs, list(target for target in targets))

    print(f"Total time: {total_time} for {len(data_loader)} batchs")

    results = metric.compute()

    pprint(readable_map_dict(results))
    pprint(readable_mar_dict(results))

    return results
