import torch
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import nms
from transformers import EvalPrediction
from transformers import trainer_pt_utils

from zindi_code import NUM_CLASS, NMS_THR # , CLS_THR


def nms_index(bboxes, scores, thr=0.6):
    return nms(bboxes, scores, thr)

# https://discuss.huggingface.co/t/possible-fix-for-trainer-evaluation-with-object-detection/72307
# .venv/lib/python3.11/site-packages/transformers/trainer_pt_utils.py:112
def compute_metrics(eval_pred: EvalPrediction):
    """
    Compute detection metrics
    Thanks hf hub discussion: https://discuss.huggingface.co/t/add-metrics-to-object-detection-example/31213/5
    """
    # detr : losses, scores, pred_boxes, last_hidden_state, encoder_last_hidden_state
    _, scores, pred_boxes, *_ = eval_pred.predictions

    # scores shape: (number of samples, number of detected anchors, num_classes + 1) last class is the no-object class
    # pred_boxes shape: (number of samples, number of detected anchors, 4)
    # https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/detr-resnet50/README.md
    predictions = []

    def _extract_bbox(score, box):
        # Extract the bounding boxes, labels, and scores from the model's output
        pred_scores = torch.from_numpy(score).softmax(dim=-1)[
            ..., :NUM_CLASS
        ]  # Exclude the no-object class => take n classes supported for uniformity btw detr and deta
        pred_boxes = torch.from_numpy(box)

        pred_labels = torch.argmax(pred_scores, dim=-1)

        # Get the scores corresponding to the predicted labels
        pred_scores_for_labels = torch.gather(
            pred_scores, 1, pred_labels.unsqueeze(-1)
        ).squeeze(-1)
        # index = nms_index(pred_boxes, pred_scores_for_labels, thr=NMS_THR)
        return {
            "boxes": pred_boxes, # [index],
            "scores": pred_scores_for_labels, # [index],
            "labels": pred_labels # [index],
        }

    predictions = [_extract_bbox(score, box) for score, box in zip(scores, pred_boxes)]
    target = [
        {
            "boxes": torch.from_numpy(raw["boxes"]),
            "labels": torch.from_numpy(raw["class_labels"]),
        }
        for raw in eval_pred.label_ids
    ]
    mAP = MeanAveragePrecision(box_format="xywh", class_metrics=True)
    mAP.update(preds=predictions, target=target)
    results = mAP.compute()
    results = {
        k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in results.items()
    }
    return results
