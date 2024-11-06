"""
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
"""

import json
import os
import shutil
import sys
import math
import random
import string
import numpy as np
import pandas as pd

MINOVERLAP = 0.5  # default value (defined in the PASCAL VOC2012 challenge)
reference_path = sys.argv[1]
submission_path = sys.argv[2]
random_string = "".join(random.choices(string.ascii_letters + string.digits, k=10))
temp_files_path = "temp_files/" + random_string
# make sure that the cwd() is the location of the python script (so that every path makes sense)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(temp_files_path)  # Make a folder under temp_files for working data
GT_PATH = os.path.join(os.getcwd(), "input", "ground-truth")
DR_PATH = os.path.join(os.getcwd(), "input", "detection-results")


def log_average_miss_rate(prec, rec, num_images):
    """
    log-average miss rate:
        Calculated by averaging miss rates at 9 evenly spaced FPPI points
        between 10e-2 and 10e0, in log-space.
    output:
            lamr | log-average miss rate
            mr | miss rate
            fppi | false positives per image
    references:
        [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
           State of the Art." Pattern Analysis and Machine Intelligence, IEEE
           Transactions on 34.4 (2012): 743 - 761.
    """
    # if there were no detections of that class
    if prec.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi
    fppi = 1 - prec
    mr = 1 - rec
    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)
    ref = np.logspace(
        -2.0, 0.0, num=9
    )  # Use 9 evenly spaced reference points in log-space
    for i, ref_i in enumerate(ref):
        # np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]
    # log(0) is undefined, so we use the np.maximum(1e-10, ref)
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))
    return lamr, mr, fppi


"""
 throw error and exit
"""


def error(msg):
    print("error: %s" % msg)
    quit()


"""
 check if the number is a float between 0.0 and 1.0
"""


def is_float_between_0_and_1(value):
    try:
        val = float(value)
        if val > 0.0 and val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False


"""
 Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.
"""


def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap, mrec, mpre


"""
 Convert the lines of a file to a list
"""


def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


try:
    """
    ground-truth
        Load each of the ground-truth files into a temporary ".json" file.
        Create a list of all the class names present in the ground-truth (gt_classes).
    """
    gt_counter_per_class = {}
    counter_images_per_class = {}
    ####__******* PANDAS
    reference = pd.read_csv(reference_path)
    reference_check = reference["Image_ID"].to_list()
    reference["boxes"] = reference[
        ["class", "ymin", "xmin", "ymax", "xmax"]
    ].values.tolist()
    reference_dict = reference.groupby("Image_ID")["boxes"].apply(list).to_dict()
    #### read sample submission
    sample_submission = pd.read_csv(submission_path)
    submission_list = list(sample_submission.columns)
    expected_list = ["Image_ID", "class", "confidence", "ymin", "xmin", "ymax", "xmax"]
    not_in_list = set(expected_list) - set(submission_list)
    for column in not_in_list:
        error_msg = "Missing columns {} on the submission file".format(column)
        error(error_msg)
    sample_submission_check = sample_submission["Image_ID"].to_list()
    current_submission = []
    # check the image id boxes in submission are the same with those in reference
    for image_id in reference_check:
        if image_id in sample_submission_check:
            current_submission.append(image_id)
        else:
            error_msg = "Error. Image ID {} not found in submission file:".format(
                image_id
            )
            error(error_msg)
    sample_submission = sample_submission[
        sample_submission["Image_ID"].isin(current_submission)
    ]
    sample_submission["boxes"] = sample_submission[
        ["class", "confidence", "ymin", "xmin", "ymax", "xmax"]
    ].values.tolist()
    sample_submission_dict = (
        sample_submission.groupby("Image_ID")["boxes"].apply(list).to_dict()
    )
    gt_files = []
    for k, v in reference_dict.items():
        file_id = k
        lines_list = [" ".join([str(item) for item in box]) for box in v]
        ### pandas
        # create ground-truth dictionary
        bounding_boxes = []
        is_difficult = False
        already_seen_classes = []
        for line in lines_list:
            try:
                if "difficult" in line:
                    class_name, left, top, right, bottom, _difficult = line.split()
                    is_difficult = True
                else:
                    class_name, left, top, right, bottom = line.split()
            except ValueError:
                error_msg = "Error: Image  " + k + " in the wrong format.\n"
                error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
                error_msg += " Received: " + line
                error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
                error(error_msg)
            # check if class is in the ignore list, if yes skip
            bbox = left + " " + top + " " + right + " " + bottom
            if is_difficult:
                bounding_boxes.append(
                    {
                        "class_name": class_name,
                        "bbox": bbox,
                        "used": False,
                        "difficult": True,
                    }
                )
                is_difficult = False
            else:
                bounding_boxes.append(
                    {"class_name": class_name, "bbox": bbox, "used": False}
                )
                # count that object
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    gt_counter_per_class[class_name] = 1
                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        # if class didn't exist yet
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)
        # dump bounding_boxes into a ".json" file
        new_temp_file = temp_files_path + "/" + file_id + "_ground_truth.json"
        gt_files.append(new_temp_file)
        with open(new_temp_file, "w") as outfile:
            json.dump(bounding_boxes, outfile)
    gt_classes = list(gt_counter_per_class.keys())
    gt_classes = sorted(gt_classes)  # let's sort the classes alphabetically
    n_classes = len(gt_classes)
    """
  detection-results
      Load each of the detection-results files into a temporary ".json" file.
  """
    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for k, v in sample_submission_dict.items():
            file_id = k
            if class_index == 0:
                if not k in reference_dict.keys():
                    error_msg = (
                        "Error. Image ID {} not found in reference file:".format(k)
                    )
                    error(error_msg)
            lines = [" ".join([str(item) for item in box]) for box in v]
            for line in lines:
                try:
                    tmp_class_name, confidence, left, top, right, bottom = line.split()
                except ValueError:
                    error_msg = "Error: Image " + k + " in the wrong format.\n"
                    error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                    error_msg += " Received: " + line
                    error(error_msg)
                if tmp_class_name == class_name:
                    # print("match")
                    bbox = left + " " + top + " " + right + " " + bottom
                    bounding_boxes.append(
                        {"confidence": confidence, "file_id": file_id, "bbox": bbox}
                    )
        # sort detection-results by decreasing confidence
        bounding_boxes.sort(key=lambda x: float(x["confidence"]), reverse=True)
        with open(temp_files_path + "/" + class_name + "_dr.json", "w") as outfile:
            json.dump(bounding_boxes, outfile)
    """
    Calculate the AP for each class
    """
    sum_AP = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}
    count_true_positives = {}
    # open file to store the output
    # with open(output_files_path + "/output.txt", 'w') as output_file:
    #     output_file.write("# AP and precision/recall per class\n")
    for class_index, class_name in enumerate(gt_classes):
        count_true_positives[class_name] = 0
        """
          Load detection-results of that class
      """
        dr_file = temp_files_path + "/" + class_name + "_dr.json"
        dr_data = json.load(open(dr_file))
        """
          Assign detection-results to ground-truth objects
      """
        nd = len(dr_data)
        tp = [0] * nd  # creates an array of zeros of size nd
        fp = [0] * nd
        for idx, detection in enumerate(dr_data):
            file_id = detection["file_id"]
            gt_file = (
                temp_files_path + "/" + file_id + "_ground_truth.json"
            )  # open ground-truth with that file_id
            ground_truth_data = json.load(open(gt_file))
            ovmax = -1
            gt_match = -1
            bb = [
                float(x) for x in detection["bbox"].split()
            ]  # load detected object bounding-box
            for obj in ground_truth_data:
                # look for a class_name match
                if obj["class_name"] == class_name:
                    bbgt = [float(x) for x in obj["bbox"].split()]
                    bi = [
                        max(bb[0], bbgt[0]),
                        max(bb[1], bbgt[1]),
                        min(bb[2], bbgt[2]),
                        min(bb[3], bbgt[3]),
                    ]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (
                            (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1)
                            + (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1)
                            - iw * ih
                        )
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj
            # set minimum overlap
            min_overlap = MINOVERLAP
            # if specific_iou_flagged:
            #     if class_name in specific_iou_classes:
            #         index = specific_iou_classes.index(class_name)
            #         min_overlap = float(iou_list[index])
            if ovmax >= min_overlap:
                if "difficult" not in gt_match:
                    if not bool(gt_match["used"]):
                        tp[idx] = 1  # true positive
                        gt_match["used"] = True
                        count_true_positives[class_name] += 1
                        # update the ".json" file
                        with open(gt_file, "w") as f:
                            f.write(json.dumps(ground_truth_data))
                    else:
                        # false positive (multiple detection)
                        fp[idx] = 1
            else:
                # false positive
                fp[idx] = 1
                if ovmax > 0:
                    status = "INSUFFICIENT OVERLAP"
        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        sum_AP += ap
        text = (
            "{0:.2f}%".format(ap * 100) + " = " + class_name + " AP "
        )  # class_name + " AP = {0:.2f}%".format(ap*100)
        rounded_prec = ["%.2f" % elem for elem in prec]
        rounded_rec = ["%.2f" % elem for elem in rec]
        ap_dictionary[class_name] = ap
        n_images = counter_images_per_class[class_name]
        lamr, mr, fppi = log_average_miss_rate(np.array(prec), np.array(rec), n_images)
        lamr_dictionary[class_name] = lamr
        mAP = sum_AP / n_classes
        text = "mAP = {0:.2f}%".format(mAP * 100)
    print(mAP)
finally:
    # remove the temp_files directory
    shutil.rmtree(temp_files_path)
