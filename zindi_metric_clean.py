import os
import sys
import json
import math
import shutil
import random
import string
import numpy as np
import pandas as pd

MINOVERLAP = 0.5  # Default minimum overlap (IoU threshold)
columns = ["Image_ID", "class", "confidence", "ymin", "xmin", "ymax", "xmax"]

def setup_temp_directory():
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    temp_dir = os.path.join("temp_files", random_string)
    os.makedirs(temp_dir)
    return temp_dir

def parse_file_content_to_dict(file_path):
    df = pd.read_csv(file_path)[columns]
    df['boxes'] = df[['class', 'ymin', 'xmin', 'ymax', 'xmax']].values.tolist()
    return df.groupby('Image_ID')['boxes'].apply(list).to_dict()

def check_required_columns(df, required_columns):
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

def calculate_overlap(bb, bbgt):
    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
    iw, ih = bi[2] - bi[0] + 1, bi[3] - bi[1] + 1
    if iw > 0 and ih > 0:
        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
        return iw * ih / ua
    return 0

def voc_ap(rec, prec):
    mrec, mpre = [0] + rec + [1], [0] + prec + [0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i_list = [i for i in range(1, len(mrec)) if mrec[i] != mrec[i - 1]]
    ap = sum((mrec[i] - mrec[i - 1]) * mpre[i] for i in i_list)
    return ap

def log_average_miss_rate(prec, rec, num_images):
    if not len(prec): return 0, 1, 0
    fppi, mr = 1 - prec, 1 - rec
    fppi_tmp, mr_tmp = np.insert(fppi, 0, -1.0), np.insert(mr, 0, 1.0)
    ref = np.logspace(-2.0, 0.0, num=9)
    for i, ref_i in enumerate(ref):
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))
    return lamr

def compute_mAP(reference_dict, submission_dict):
    sum_AP, count_true_positives = 0.0, {}
    gt_classes = sorted(set(class_name for boxes in reference_dict.values() for class_name, *_ in boxes))
    
    for class_name in gt_classes:
        tp, fp, count_true_positives[class_name] = [], [], 0
        for image_id, detections in submission_dict.items():
            gt_boxes = [box for box in reference_dict.get(image_id, []) if box[0] == class_name]
            dr_boxes = [box for box in detections if box[0] == class_name]
            dr_boxes.sort(key=lambda x: -float(x[1]))  # Sort by confidence
            for _, xmin, ymin, xmax, ymax in dr_boxes:
                ovmax, match = -1, None
                for gt in gt_boxes:
                    iou = calculate_overlap([xmin, ymin, xmax, ymax], gt[1:])
                    if iou > ovmax: ovmax, match = iou, gt
                if ovmax >= MINOVERLAP and match not in tp:
                    tp.append(1)
                    count_true_positives[class_name] += 1
                else:
                    fp.append(1)

        prec, rec = np.cumsum(tp) / (np.cumsum(tp) + np.cumsum(fp)), np.cumsum(tp) / count_true_positives[class_name]
        sum_AP += voc_ap(rec.tolist(), prec.tolist())

    return sum_AP / len(gt_classes)

def main(reference_path, submission_path):
    temp_dir = setup_temp_directory()
    reference_dict = parse_file_content_to_dict(reference_path)
    submission_dict = parse_file_content_to_dict(submission_path)
    
    check_required_columns(pd.read_csv(submission_path), ["Image_ID", "class", "confidence", "ymin", "xmin", "ymax", "xmax"])
    
    mAP = compute_mAP(reference_dict, submission_dict)
    print(f"Mean Average Precision (mAP): {mAP:.2%}")
    
    shutil.rmtree(temp_dir)  # Clean up temporary files

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.argv.append("zindi_data/ValDataset.csv")
        sys.argv.append("zindi_data/validation/prediction_cond-detr-50_THR1.000_IOU0.800_ID348_P.csv")
        print("Usage: python script.py <reference_path> <submission_path>")
        # sys.exit(1)
    main(sys.argv[1], sys.argv[2])
