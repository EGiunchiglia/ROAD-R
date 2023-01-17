from constants import *
import numpy as np
import subprocess
import time
import argparse
import os

from data import VideoDataset
from torchvision import transforms
import data.transforms as vtf
from modules import utils
import torch.utils.data as data_utils

from data import custum_collate
from modules.utils import get_individual_labels
import torch

import modules.evaluation as evaluate
import json


def process_output_line(line):
    split_line = (line.replace(' ', '')).split(',')
    if len(split_line) > 1:
        assert len(split_line) == 49 or len(split_line) == 50
        videoname = split_line[0]
        img_name = split_line[1]
        bboxes = [float(b) for b in split_line[2:6]]
        agentness = float(split_line[6])
        preds =  [float(p) for p in split_line[7:48]]  #[float(split_line[6])]#
        return videoname, img_name, bboxes, agentness, preds
    return None, None, None, None, None



def main():
    # Parse input arguments    
    parser = argparse.ArgumentParser(description='Post-processing output')
    parser.add_argument('--model', type=str, help='model type')
    parser.add_argument('--data_split', default='test', type=str, help='data split.')
    parser.add_argument('--th', type=float, help='threshold at which the model is evaluated- If out >= th then label = 1, otherwise = 0')
    parser.add_argument('--iou_th', type=float, default=0.5, help="IOU threshold")
    parser.add_argument('--file_path', type=str, help="Output file on which to compute the mAP")
    parser.add_argument('--test_json_path', type=str, default="/home/user/road/road_test_v1.0.json", help="Path to the json file for the test split")

    args = parser.parse_args()

    map_dataset_path = args.file_path.split('/')[0]+"/"+args.file_path.split('/')[1]
    os.makedirs(f"classes_mAP@{args.iou_th:.1f}/{map_dataset_path}")
    print(map_dataset_path)


    args.DATASET = 'road'
    args.SUBSETS = ['test']
    args.MODE = 'test'
    args.BATCH_SIZE = 4
    args.MIN_SEQ_STEP, args.MAX_SEQ_STEP = 1, 1
    args.MEANS =[0.485, 0.456, 0.406]
    args.STDS = [0.229, 0.224, 0.225]
    args.MIN_SIZE = 512
    args.SEQ_LEN = 8
    args.NUM_WORKERS=8
    args.CONF_THRESH = 0.025
    args.GEN_CONF_THRESH = 0.025
    args.NMS_THRESH = 0.5
    args.TOPK = 10
    args.GEN_TOPK = 100
    args.GEN_NMS = 0.5
    args.MAX_SIZE = int(args.MIN_SIZE*1.35)
    args.ANCHOR_TYPE = 'RETINA'
    args.skip_beggning = 0
    args.skip_ending = 0
    args.num_label_types = 3
    if args.data_split == 'test':
        if args.model == 'I3D':
            args.skip_beggning = 2
            args.skip_ending = 2
        if args.model != 'C2D':
           args.skip_beggning = 2

    #TO BE CHANAGED WHEN DEALING WITH TEST SET
    val_transform = transforms.Compose([ 
                        vtf.ResizeClip(args.MIN_SIZE, args.MAX_SIZE),
                        vtf.ToTensorStack(),
                        vtf.Normalize(mean=args.MEANS,std=args.STDS)])
    skip_step = args.SEQ_LEN - args.skip_beggning
    val_dataset = VideoDataset(args, train=False, transform=val_transform, skip_step=skip_step, full_test=True)
    
    label_types = ['agent', 'action', 'loc']
    detections = {'agent':{}, 'action':{}, 'loc':{}}
    results = {'agent':{}, 'action':{}, 'loc':{}}

    with open(args.file_path, 'r') as rf:

        #if reading an outputs_corrected then we need to read the header 
        if 'outputs_corrected' in args.file_path:
            line = rf.readline()

        line = rf.readline() 
        o_videoname, o_img_name, o_bboxes, o_agentness, o_preds = process_output_line(line)

        while True:

            decoded_boxes_frame, confidence = [], []
            old_videoname, old_img = o_videoname, o_img_name
            
            while o_videoname == old_videoname and o_img_name == old_img:        
                # Read an output line                        
                assert o_videoname == old_videoname and o_img_name == old_img, f"{o_img_name}, {old_img:}, {o_videoname}, {old_videoname}"
                decoded_boxes_frame.append(o_bboxes)
                confidence.append(o_preds)

                # send the check ahead of one line
                line = rf.readline()
                if "video_name" in line:
                    line  = rf.readline()
                o_videoname, o_img_name, o_bboxes, o_agentness, o_preds = process_output_line(line)

            decoded_boxes_frame = torch.tensor(decoded_boxes_frame)
            confidence = torch.tensor(confidence)


            cc = 0 
            for i,nlt in enumerate(label_types):
                num_c = val_dataset.num_classes_list[i]
                detections[nlt][old_videoname+old_img[:-4]] = []
                
                for cl_ind in range(num_c):
                    
                    scores = confidence[:,cc].clone().squeeze()
                    cls_dets = utils.filter_detections(args, scores, decoded_boxes_frame)
                    detections[nlt][old_videoname+old_img[:-4]].append(cls_dets)
                    cc += 1

            if not line:
                break


    anno_file = args.test_json_path #'/users-2/eleonora/3D-RetinaNet/road_test_v1.0.json'
    with open(anno_file, 'r') as fff:
            final_annots = json.load(fff)


    for nlt, label_type in enumerate(label_types):
        # Get ground truth for frames
        ap_all = []
        ap_strs = []
        re_all = []
        sap = 0.0
        gt_frames = evaluate.get_gt_frames(final_annots, args.SUBSETS[0], label_type, 'road')
        classes = final_annots[label_type+'_labels']

        print(classes)

        for cl_id, class_name in enumerate(classes):
            print("class id", cl_id, class_name)
            class_gts = evaluate.get_gt_class_frames(gt_frames, cl_id)

            frame_ids = [f for f in class_gts.keys()]
            class_dets = evaluate.get_det_class_frames(detections[label_type], cl_id, frame_ids, 'road') 

            class_ap, num_postives, count, recall = evaluate.compute_class_ap(class_dets, class_gts, evaluate.compute_iou_dict, args.iou_th)

            recall = recall*100
            sap += class_ap
            ap_all.append(class_ap)
            re_all.append(recall)
            ap_str = class_name + ' : ' + str(num_postives) + \
                ' : ' + str(count) + ' : ' + str(class_ap) +\
                ' : ' + str(recall)
            ap_strs.append(ap_str)

            mAP = sap/len(classes)
        mean_recall = np.mean(np.asarray(re_all))
        ap_strs.append('\nMean AP:: {:0.2f} mean Recall {:0.2f}'.format(mAP,mean_recall))
        results[label_type] = {'mAP':mAP, 'ap_all':ap_all, 'ap_strs':ap_strs, 'recalls':re_all, 'mR':mean_recall}

        with open(f"classes_mAP@{args.iou_th:.1f}/{args.file_path}", "a") as wf:

            wf.write(f"Label type: {label_type}, IOU threshold: {args.iou_th} \n")

            wf.write(f"mAP: {mAP}\n")
            wf.write(f"ap_all: {ap_all}\n")
            wf.write(f"ap_strs: {ap_strs}\n")

            wf.write("\n\n")



if __name__ == "__main__":
    main()