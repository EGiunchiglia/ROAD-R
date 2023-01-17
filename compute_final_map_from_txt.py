import numpy as np
import argparse

def compute_map(path):

    with open(path, 'r') as rf:    
        precs = []
        for i, line in enumerate(rf.readlines()):
            
            if "ap_all" in line:
                broken_line = line.replace('[', ',').replace(']\n', '').split(',')
                assert "ap_all" in broken_line[0]
                precs_temp = [float(x) for x in broken_line[1:]]
                precs += precs_temp
         
        print(len(precs))   
        assert len(precs) == 41
        mAP = np.mean(precs)
        return mAP


if __name__ == "__main__":

    # Parse input arguments    
    parser = argparse.ArgumentParser(description='Compute overall fmAP')
    parser.add_argument('--model', type=str, help='model type')
    parser.add_argument('--logic', type=str, help="logic used")
    parser.add_argument('--wgt_log',type=float, help="weight associated to logic")
    parser.add_argument('--post_proc', type=str, help='type of post_processing used')
    parser.add_argument('--th', type=float, help='threshold theta used for post-processing')
    parser.add_argument('--iou', type=float, default="0.5", help='IOU at which we want to compute the final mAP')
    parser.add_argument('--direct_path', type=str, help='path to file containing the f-mAP scores for each label')
    args = parser.parse_args()

    if args.direct_path is not None:
        precision = compute_map(args.direct_path)
        print("final mAP", precision)
        exit()

    th = args.th

    path = f"./classes_mAP@{args.iou}/outputs_corrected_{args.post_proc}/{args.model}_logic_{args.logic}_w_{args.wgt_log:.1f}_ROAD_test/th_{th:.1f}.txt"
    precision = compute_map(path)
    print(f"POST-PROC: {args.post_proc}. Model:{args.model}. Th: {th:.1f}. Final mAP: {precision}")

