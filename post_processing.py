from constants import *
import numpy as np
import subprocess
import time
import argparse
import os



HARD_VALUE = 2**(NUM_LABELS+1)



################
#This function reads the mAP obtained by a model for each class 
# <-- the mAP is then used for post-processing purposes
#################
def read_maps(model, logic, wgt_logic, iou):

    maps = []
    with open(f"classes_mAP@{iou}/outputs/log-lo_ROAD_R_predictions_{model}_logic-{logic}-{wgt_logic}.txt") as rf:
        for line in rf.readlines():
            if "ap_all" in line:
                clean_line = line.replace('ap_all: [', '').replace(']\n', '')
                break_line = clean_line.split(',')
                
                maps += break_line
        
    maps = [float(m) for m in maps]
    
    return maps



####################
# Take the file with the constraints written in Prolog style and convert
# them in WDIMACS. Example of file in WDIMACS format:
# p wcnf 3 4
# 10 1 -2 0
# 3 -1 2 -3 0
# 0.8 -3 2 0
# 5.6 1 3 0
# Note: first numbers in each line are the weights
# Note: each line must end by zero and the vaiables must be NON-ZERO integers --> the integers associated to the labels are shifted by one 
####################
def create_WDIMACS_constraints(in_path, out_path):

    with open(in_path, 'r') as f:
        with open(out_path, 'w') as wf:

            # Write the first lines of the WDIMACS file
            # c at the beginning of the line denotes a comment
            # first line must be: 
            # p wcnf <num_variables> <num_clauses> <top>
            # Note1: if a clause if assigned the weight <top> then it is considered HARD
            # Note2: since we will add the outputs as cluases num_clauses = num_constraints + num_labels
            # Note3: in order to ensure that the constraints are considered HARD we need it to be larger that the sum of the weights of the falsified clauses in an optimal solution (top being the sum of the weights of all soft clauses plus 1 will always suffice)
            wf.write("c \nc constraints in WDIMACS form \nc \n")
            wf.write(f"p wcnf {NUM_LABELS} {NUM_REQ+NUM_LABELS} {HARD_VALUE}\n")

            for i, line in enumerate(f.readlines()):
                
                split_line = line.split()
                assert split_line[0]=='0.0' and split_line[2]==':-'
                # In order to have the clauses we need to keep each head as it is and invert each literal in the body          
                #Start line with <top> weight 
                wf.write(f"{HARD_VALUE} ")
                # Write the head 
                head = split_line[1]
                if head[0]=='n':
                    wf.write(f"-{int(head[1:])+1} ")
                else:
                    wf.write(f"{int(head)+1} ")
                # Write the body
                for b in split_line[3:]:
                    if b[0]=='n':
                        wf.write(f"{int(b[1:])+1} ")
                    else: 
                        wf.write(f"-{int(b)+1} ")
                # End line
                wf.write("0 \n")


###########
# Write temporary file with requirements (hard constraints) and output (soft constraints)
# Note: hard constraints have wegith 50.0, soft constraints have weight depending on the confidence of the model in the prediction
###########
def write_temp_file(preds, threshold, temp_file_path, wdimacs_path, post_proc, maps=None):

    assert len(preds) == NUM_LABELS
    assert post_proc == "basic" or post_proc == "map_based" or post_proc == "map_times_pred_based"

    with open(wdimacs_path, 'r') as f:
        with open(temp_file_path, 'w') as wf:
            # Write the hard constraints
            for i, line in enumerate(f.readlines()):
                wf.write(line)
            # Write the soft constraints (i.e., the predictions)
            if post_proc == "basic":
                for i,p in enumerate(preds): 
                    if p > threshold:
                        wf.write(f"1 {i+1} 0\n")
                    else:
                        wf.write(f"1 -{i+1} 0\n")
            elif post_proc == "map_based":
                assert maps != None
                # Write the soft constraints (i.e., the predictions)
                for i,(p,m) in enumerate(zip(preds,maps)): 
                    if p > threshold:
                        wf.write(f"{m} {i+1} 0\n")
                    else:
                        wf.write(f"{m} -{i+1} 0\n")
            elif post_proc == "map_times_pred_based":
                assert maps != None
                # Write the soft constraints (i.e., the predictions)
                for i,(p,m) in enumerate(zip(preds,maps)): 
                    if p > threshold:
                        wf.write(f"{p*m} {i+1} 0\n")
                    else:
                        wf.write(f"{(1-p)*m} -{i+1} 0\n")  
            else:
                print("Post-processing inserted is not valid")
                exit(1)

            


###########
# Call the solver MaxHs
# The function returns:
# 1. cost: integer representing how many predictions needed to be flipped 
# 2. assignement: list contaning the new assignment for each variable, for each element n in the list -n indicates negative literal and n positive.
#    Example: ['-1', '-2', '3', '-4', '-5', '-6', '-7', '-8', '-9', '-10', '-11', '-12', '-13', '14', '15', '-16', '-17', '-18', '-19', '-20', '-21', '22', '-23', '-24', '-25', '-26', '-27', '-28', '-29', '-30', '-31', '-32', '33', '-34', '-35', '-36', '-37', '38', '39', '-40', '-41']
# 3. time_elapsed: time taken by the solver to find solution (in seconds)
###########
def call_solver(temp_file_path):
    
    start = time.time()
    output = subprocess.run(["./MaxHS/build/release/bin/maxhs", "-printSoln", temp_file_path], capture_output=True)
    end = time.time()
    time_elapsed = end-start
    solver_output = output.stdout.decode("utf-8").splitlines()

    cost=-1
    for line in solver_output:
        if line[0] == 'o':
            cost = line.split()[1]
        elif line[0] =='v':
            assignment = line.split()[1:]
            assert len(assignment)==41#, "length assignment (%d)" % len(assignment)

    return float(cost), assignment, time_elapsed    




###############
# Take a prediction and return its negated
###############
def invert_pred(pred, th):
    # Since the predictions that get changed are not given by the model, we assign to them the lowest confidence possible (i.e., threhsold plus/minus epsilon)
    epsilon = 1e-3
    if pred == th:
        return th
    elif pred < th:
        new_pred = th + epsilon
    else:
        new_pred = th - epsilon
    return new_pred





####################
# This function takes as input the predictions, the assignment as returned by the solver and the threshold
# and computes new predictions that are guaranteed to be compliant with the constraints.
####################
def compute_corrected_output(preds, new_assgn, th, prop_negation = False):
    
    corrected_preds = []
    for pred, assgn in zip(preds, new_assgn):
        assgn = int(assgn)
        # if pred and assignment are coherent do nothing
        if (pred > th and assgn > 0) or (pred < th and assgn < 0):
            corrected_preds.append(pred)
        else:
            new_pred = invert_pred(pred, th)
            corrected_preds.append(new_pred)
    return corrected_preds


if __name__ == "__main__":

    # Parse input arguments    
    parser = argparse.ArgumentParser(description='Post-processing output on the ground of the confidence of the model')
    parser.add_argument('--model', type=str, help='model type')
    parser.add_argument('--data_split', type=str, help='data split.')
    parser.add_argument('--th', type=float, help='threshold at which the model is evaluated- If out >= th then label = 1, otherwise = 0')
    parser.add_argument('--post_proc', type=str, help="Post-processing to be used")
    parser.add_argument('--logic', type=str, default="None", help="Logic type in the loss used for training" )
    parser.add_argument('--wgt_log', type=float, default=0.0, help="Weight associated to the logic loss" )
    parser.add_argument('--iou', type=float, default=0.5, help="IOU value used to calculate the mAP" )


    args = parser.parse_args()

    wdimacs_constraints_path = f"constraints/WDIMACSconstraints_{args.post_proc}.txt"
    wdimacs_temp_constraints_path = f"constraints/WDIMACSconstraints_{args.post_proc}_temp/"


    if not os.path.exists(wdimacs_constraints_path):
        create_WDIMACS_constraints(CONSTRAINTS_PATH, wdimacs_constraints_path)

    with open(f"./outputs/log-lo_ROAD_R_predictions_{args.model}_logic-{args.logic}-{args.wgt_log}.txt", 'r') as rf:
        
        #If it doesn't exists, build folder 
        if not os.path.exists(f"./outputs_corrected_{args.post_proc}/{args.model}_logic_{args.logic}_w_{args.wgt_log}_ROAD_{args.data_split}"):
            os.makedirs(f"./outputs_corrected_{args.post_proc}/{args.model}_logic_{args.logic}_w_{args.wgt_log}_ROAD_{args.data_split}")
        
        with open(f"./outputs_corrected_{args.post_proc}/{args.model}_logic_{args.logic}_w_{args.wgt_log}_ROAD_{args.data_split}/th_{args.th}.txt", 'w') as wf:
            
            # Write first line
            wf.write(f"video_name, img_id, bbox1, bbox2, bbox3, bbox4, {', '.join([str(x+1) for x in np.arange(NUM_LABELS)])}, cost, time\n")
            
            if not os.path.exists(wdimacs_temp_constraints_path+args.model+"_ROAD_"+args.data_split+"_"+args.logic+"_"+str(args.wgt_log)):
                os.makedirs(wdimacs_temp_constraints_path+args.model+"_ROAD_"+args.data_split+"_"+args.logic+"_"+str(args.wgt_log))

            # For accuracy based post-processing
            maps=None
            if args.post_proc == "map_times_pred_based" or args.post_proc =="map_based":
                maps = read_maps(args.model, args.logic, args.wgt_log, args.iou)

            for i, l in enumerate(rf.readlines()):

                split_line = l.split(',')

                # Parse each line
                video_name = split_line[0]
                img_id = split_line[1]
                bbox = split_line[2:6]
                agentness = split_line[6]
                preds = np.array(split_line[7:-1], dtype=float)

                if all(i <= args.th for i in preds):  
                    corrected_preds = preds
                    cost = 0
                    time_elaps = 0   
                else:
                    # Write temporary file containing hard and soft constraints
                    write_temp_file(preds, args.th, wdimacs_temp_constraints_path+args.model+"_ROAD_"+args.data_split+"_"+args.logic+"_"+str(args.wgt_log)+"/th_"+str(args.th)+".txt", wdimacs_constraints_path, args.post_proc,maps=maps)
                    # call the solver on the temporary file
                    cost, new_assgn, time_elaps = call_solver(wdimacs_temp_constraints_path+args.model+"_ROAD_"+args.data_split+"_"+args.logic+"_"+str(args.wgt_log)+"/th_"+str(args.th)+".txt")            
                    # compute the new predictions
                    corrected_preds = compute_corrected_output(preds, new_assgn, args.th)
                

                # Write on file
                wf.write(f"{video_name}, {img_id}, {', '.join([str(b) for b in bbox])}, {agentness}, {', '.join([str(x) for x in corrected_preds])}, {cost}, {time_elaps}\n")
                print(f"Lines done: {i}", end='\r')
