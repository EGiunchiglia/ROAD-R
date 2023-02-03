# ROAD-R 

Dataset and code for the paper "[ROAD-R: The Autonomous Driving Dataset with Logical Requirements](https://learn-to-race.org/workshop-ai4ad-ijcai2022/assets/papers/paper_10.pdf)".
![](https://github.com/EGiunchiglia/ROAD-R/blob/main/extras/short_clip.gif)

## Table of Contents
- <a href='#dep'>Annotated requirements</a>
- <a href='#dep'>ROAD</a>
- <a href='#dep'>Dependencies and data preparation</a>
- <a href='#training'>Training</a>
- <a href='#testing'>Testing</a>
- <a href='#prostprocessing'>Post-processing</a>


## Annotated requirements

The annotated requirements can be found in the [requirements folder](requirements) in two formats:

1. [requirements_dimacs.txt](requirements/requirements_dimacs.txt) contains the requirements written in dimacs format. Here, each label is represented as a number.

2. [requirements_readable.txt](requirements/requirements_readable.txt) contains the requirements written in a human understable format. 

The natural language explanation of each requirement can be found in the appendix of the paper.


## ROAD 

The ROAD dataset is available at: https://github.com/gurkirt/road-dataset.


## Dependencies and data preparation
For the dataset preparation and packages required to train the models, please see the [Requirements](https://github.com/gurkirt/3D-RetinaNet#requirements) section from 3D-RetinaNet for ROAD.  

To download the pretrained weights, please see the end of the [Performance](https://github.com/gurkirt/3D-RetinaNet#performance) section from 3D-RetinaNet for ROAD.  

## Training

Let `/home/user/kinetics-pt/` be the path to the directory containing the pretrained weights. 
And suppose the dataset is in `/home/user/road/`.

Example train commands (to be run from the root of this repository) are provided below.

```
# to train a model with logic-based loss using Product t-norm and weight 10
# similarly for the other t-norms (Gödel and Lukasiewicz) -> change the LOGIC parameter at the end of the command
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py /home/user/ /home/user/  /home/user/kinetics-pt/ --MODE=train --ARCH=resnet50 --MODEL_TYPE=I3D --DATASET=road --TRAIN_SUBSETS=train_1 --VAL_SUBSETS=val_1 --SEQ_LEN=8 --TEST_SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.0041 --NUM_WORKERS=8 --req_loss_weight=10.0 --LOGIC=Product

# to train a model without logic-based loss (the baselines in the paper): 
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py /home/user/ /home/user/  /home/user/kinetics-pt/ --MODE=train --ARCH=resnet50 --MODEL_TYPE=I3D --DATASET=road --TRAIN_SUBSETS=train_1 --VAL_SUBSETS=val_1 --SEQ_LEN=8 --TEST_SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.0041 --NUM_WORKERS=8 --req_loss_weight=0.0
```

The second argument, `/home/user/`, specifies where the checkpoints and logs will be stored. 
In this case, these files will be found in 
`/home/user/road/road/log-lo_cache_logic_<LOGIC>_<req_loss_weight>/<experiment-name>/`.

The models will run for 30 epochs by default, but this can be changed via the `MAX_EPOCHS` parameter.


## Testing 
Below is an example command to test a model.

```
CUDA_VISIBLE_DEVICES=1 python main.py /home/user/ /home/user/  /home/user/kinetics-pt/ --MODE=gen_dets --ARCH=resnet50 --MODEL_TYPE=I3D --DATASET=road --TRAIN_SUBSETS=train_1 --VAL_SUBSETS=test --SEQ_LEN=8 --TEST_SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.0041 --NUM_WORKERS=8 --req_loss_weight=10.0 --LOGIC=Product 
```

This command will generate a file containing the detected boxes at the following location:
`/home/user/road/road/log-lo_cache_logic_<LOGIC>_<req_loss_weight>/<experiment-name>/detections-30-08-50_test/log-lo_ROAD_R_predictions_I3D_logic-Product-10.0.txt`.

Then, to get the f-mAP scores for each of the 41 labels, run:
```
python compute_classes_mAP.py --model I3D --data_split test --file_path outputs/log-lo_ROAD_R_predictions_I3D_logic-Product-10.0.txt --iou_th 0.5 --data_root /home/user/ 
```
The output will be saved in `classes_mAP/outputs/lo_ROAD_R_predictions_I3D_logic-Product-10.0.txt`.

Lastly, we compute the overall f-mAP (over all 41 classes) using `compute_final_map_from_txt.py`:
```
python compute_final_map_from_txt.py --direct_path classes_mAP/outputs/lo_ROAD_R_predictions_I3D_logic-Product-10.0.txt
```


## Post-processing
Assume we have the file containing detected boxes (from the previous step) copied into the `postprocessing/outputs` directory.
And assume we have the f-mAP scores (computed using steps similar to those described earlier) for the validation set.

We `cd` into the `postprocessing` directory in this repository.
Then we can run the `post_processing_general_logic.py` script by specifying:
- the `model` type (e.g. I3D)
- the data partition used for the evaluation `data_split` (e.g. test)
- the threshold `th` at which the model will be evaluated 
- the `logic` type (e.g. Product)
- the weight `wgt_log` associated with the logic-based loss (e.g. 10)
- and finally, the post-processing method `post_proc` (e.g. `map_times_pred_based`)

For instance, running
```
python post_processing_general_logic.py --logic Product --wgt_log 10.0 --post_proc "map_times_pred_based" --model I3D --data_split test --th 0.3
```
will produce an output file stored at `outputs_corrected_map_times_pred_based/I3D_logic_Product_w_10.0_ROAD_test/th_0.3.txt`.


An example of how to run this script with all thresholds between 0.1 and 0.9 (with step 0.1) is provided in 
`postprocessing_script.sh`. 

Next, we compute the mAP for each of the 3 classes (agent, action, location) for each of the 9 chosen thresholds, specifying the IOU. We will thus run, for example:
```
python compute_classes_mAP.py --model I3D --data_split test --file_path outputs_corrected_map_times_pred_based/I3D_logic_Product_w_10.0_ROAD_test/th_0.3.txt --iou_th 0.5 --data_root /home/user/ 
```
Depending on the chosen IOU value, the mAP values for the example above will be stored at `classes_mAP@<IOU_value>/outputs_corrected_map_times_pred_based/I3D_logic_Product_w_10.0_ROAD_test/th_0.3.txt`.

Lastly, we compute the overall mAP (over all 41 classes) using `compute_final_map_from_txt.py`, which simply computes the average f-mAP over the 41 classes. We can specify for which model we want to compute the final f-mAP by using the following input parameters:
- the `model` type (e.g. I3D)
- the threshold `th` at which the post-processing was conducted 
- the `logic` type that was used during training (e.g. Product)
- the weight `wgt_log` associated with the logic-based loss (e.g. 10.0)
- and finally, the post-processing method `post_proc` (e.g. `map_times_pred_based`)

An example of how to run it is given below: 
```
python compute_final_map_from_txt.py --model I3D --th 0.3 --logic Product --wgt_log 10.0 --post_proc map_times_pred_based 
```


## Reference
```
@article{giunchiglia2022jml,
    title     = {ROAD-R: The Autonomous Driving Dataset with Logical Requirements},
    author    = {Eleonora Giunchiglia and
                  Mihaela Catalina Stoian and 
                  Salman Khan and 
                  Fabio Cuzzolin and 
                  Thomas Lukasiewicz},
    journal = {Machine Learning},
    year = {2022}
}
