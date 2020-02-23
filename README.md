# Guiding Program Synthesis by Learning to Generate Examples
Larissa Laich, [Pavol Bielik](https://www.sri.inf.ethz.ch/people/pavol), [Martin Vechev](https://www.sri.inf.ethz.ch/people/martin)  

This repository contains our code, models and dataset for the paper [Guiding Program Synthesis by Learning to Generate Examples (ICLR'20)](https://openreview.net/pdf?id=BJl07ySKvS).

**Table of Contents**<br>
1.  [Setup](#1-setup) <br>
2.  [Oracle Evaluation](#2-oracle-evaluation) <br>
3.  [Oracle Ablation Study](#3-oracle-ablation-study) <br>
4.  [Training the oracle](#4-training-the-oracle) <br>
5.  [Synthesis Evaluation](#5-synthesis-evaluation) <br> 
6.  [How to generate datasets](#6-how-to-generate-datasets) <br>
    a. Ds+ <br>
    b. Ds <br>
    c. Ablation dataset<br>
7.  [Project Structure](#7-project-structure) <br>
8.  [References](#8-references) <br>

# (1) Setup

Prepare the virtual environment. Tested with python 3.6.8.
```bash
python3.6 -m venv venv
```
Activate the virtual environment.
```bash
source venv/bin/activate
```

Install this project and dependencies in editable state
```bash
pip install -e .
```


## Unzip trained models and datasets 
The trained models are saved in saved_modelsPaper.tar.gz. This folder contains the 8 models described in the paper:
MLP, CNN, RNN and RNN+CNN trained on ds and ds+.<br>
To run evaluations with these models, unzip them as well as the datasets.<br>
In dataset/data you will find the datasets used to train the models in the paper ds and dsplus as well as the ablation dataset.

Run 
```bash
./guidesyn/extractData.sh
```
This will need ~2.5G of disk space (792M for the models, 1.2G for ds, 343M dsplus, 413M ablation dataset)


# (2) Oracle Evaluation
Evaluate the pairwise accuracy and the 1 vs. many accuracy of the 8 different oracles on ds+ test dataset. 
(Corresponds to Table 5 in the paper).

To execute on a GPU pass the gpu id, otherwise pass -1.
```bash
cd guidesyn
python evaluate_oracles.py --gpus [gpu_id] 2>&1 | tee ./results/oracle_evaluation.txt
```

Expected outcome:
With a GPU the evalution for all 8 models should take ~15min.
The results are the following (for the evaluation on ds+ test):

Pairwise accuracy in %:


| Trained on / Model | MLP |  CNN | RNN | RNN+CNN |
| ------ | ------ | ------ | ------ | ------ |
| ds  | 84.4 [91.6] | 80.0 | 78.5 [80.0] | 83.5 |
| ds+ |  90.5 [97.6] | 84.3 | 95.6 | 96.8 | 

Accuracy One vs. many in %:

| Trained on / Model | MLP |  CNN | RNN | RNN+CNN |
| ------ | ------ | ------ | ------ | ------ |
| ds  | 44.5 [73.0] | 36.7 | 36.3 [38.3]  | 55.8 |
| ds+ | 54.6 [87.3] | 51.2 | 80.6 [81.0] | 85.9 | 


# (3) Oracle Ablation Study
Evaluate the pairwise accuracy of the 8 different oracles on ablation dataset (generated from the ds+ test dataset). 
(Corresponds to Table 6 & 7 in the paper).
```bash
cd guidesyn
python run_ablation.py --gpus [gpu_id] 2>&1 | tee ./results/ablation_study.txt
```
Expected execution time: With a GPU the evalution for all 8 models should take ~17min.

Pairwise Accuracy for models trained on ds:

| Dataset / Model       | MLP  |  CNN |  RNN  | RNN+CNN |
| ------ | ------       | ------ | ------ | ------ |
| Aspect ratio 1.0      | 85.4 | 54.2 | 65.6  | 70.8  |
| Horiz. centering-view | 97.5 | 86.7 | 88.0  | 89.5  |
| Horiz. centering-views| 96.3 | 86.6 | 87.8  | 89.4  |
| Vert. centering-view  | 91.4 | 76.0 | 85.9  | 78.6  |
| Vert. centering-views | 93.0 | 86.6 | 84.9  | 89.6  |
| Inside screen         | 100.0| 97.7 | 96.2  | 99.5  |
| Intersections         | 97.0 | 90.6 | 93.3  | 96.7  |
| Horizontal alignment  | 91.0 | 83.1 | 87.8  | 88.1  |
| Vertical alignment    | 96.4 | 90.7 | 85.6  | 97.3  |
| Horizontal margins    | 92.6 | 89.3 | 89.1  | 91.1  |
| Vertical margins      | 91.4 | 83.1 | 83.0  | 88.0  |
| Specific aspect ratio | 100.0| 100.0| 61.1  | 44.4  |
| Same dimensions       | 95.3 | 86.4 | 94.9  | 90.2  |

Pairwise Accuracy for models trained on ds+:

| Dataset / Model       | MLP  |  CNN |  RNN  | RNN+CNN |
| ------ | ------       | ------ | ------ | ------ |
| Aspect ratio 1.0      | 96.9 | 63.5 | 92.7  | 99.0  |
| Horiz. centering-view | 100.0| 89.0 | 96.6  | 98.2  |
| Horiz. centering-views| 99.7 | 89.0 | 96.5  | 98.0  |
| Vert. centering-view  | 98.5 | 83.1 | 97.4  | 96.8  |
| Vert. centering-views | 98.2 | 89.3 | 97.1  | 97.7  |
| Inside screen         | 100.0| 98.0 | 98.8  | 99.0  |
| Intersections         | 99.5 | 93.3 | 97.3  | 98.2  |
| Horizontal alignment  | 99.6 | 85.5 | 95.7  | 96.9  |
| Vertical alignment    | 99.7 | 95.0 | 98.0  | 98.6  |
| Horizontal margins    | 98.5 | 91.1 | 97.2  | 98.2  |
| Vertical margins      | 97.1 | 88.0 | 97.6  | 98.0  |
| Specific aspect ratio | 100.0| 61.1 | 100.0 | 100.0 |
| Same dimensions       | 100.0| 85.8 | 97.0  | 96.6  |


# (4) Training the oracle
```bash
cd guidesyn
python train.py --exp models-acc --model_path ./core/saved_models/ --should_cache 1 --gpus 0,1,2 | tee train.txt
```

Per default, the models will train the following epochs: <br>
MLP: 4 (ds, ~25min), 14 (ds+, ~28min) <br>
RNN: 120 (ds, ~5h), 70 (ds+, ~90min) <br>
CNN: 45 (ds, ~21h), 60 (ds+, ~9h) <br>
RNN+CNN (Ensemble): 50 (ds, ~32h), 80 (ds+, ~17h) <br>

The trained models are taken from the following epochs (selecting by maximum value on validation dataset): <br>
MLP: 2 (ds), 6 (ds+) <br>
RNN: 88 (ds), 57 (ds+) <br>
CNN: 11 (ds), 33 (ds+) <br>
RNN+CNN (Ensemble): 15 (ds), 56 (ds+) <br>

## Execute the trained models on the test set
The trained models are stored in the provided model\_path. To evaluate on the test set, select the models (/epochs) you want to evalute on and copy them into another folder (removing the \_epoch\_number which is the last part of the filename). 
Afterwards, run the oracle evaluation and specify the model\_path to the folder you just created.

# (5) Synthesis Evaluation
The results of the end-to-end synthesis experiment are in Table 1 of the paper. 
To replicate the results it is necessary to run the InferUI synthesizer which is not part of this work.
If you are interested in running these experiments, please reach out to [Pavol Bielik](https://www.sri.inf.ethz.ch/people/pavol) for more information.

# (6) How to generate datasets

## Ds
Download "UI Screenshots and View Hierarchies" from "http://interactionmining.org/rico" in guidesyn. <br>
Run 
```bash
cd guidesyn
python dataset/scripts/dudsds/data_generation.py
```
to generate the datasets containing the following steps: <br>
1.) generate_ds_du generates the ds and du dataset by taking the views from the RICO dataset and generate negative samples by modifying views (according to shifts derived from ds+) <br>
2.) split_test_validate_train splits the excisting ds+ dataset (generated with the program synthesizer) in test, train and validation <br>
3.) copy_dsplus_to_dsdu_folders copies the ds+ dataset to the corresponding ds folders. <br>

The naming schema of the files is [app\_id]-[canididate\_id]\_[label].txt.

For example, for `app_id=21046` the dataset contains following files:

```
# Positive Sample
21046_1.txt

# Negative Samples
21046_0_0.txt  
21046_1_0.txt  
21046_2_0.txt  
21046_3_0.txt  
21046_4_0.txt
```

Each of these files contains a list of views (one per line) in the format `[x, y, widht, height]`.


## Ds+
The resulting dataset is stored in dsplus. We split the dataset in train (579 < app\_id <= 2298), test (328 < app\_id <= 579) and validation (app\_id <= 328). This can be done with splitTestValidateTrain.
The naming schema of the files is [app\_id]-[device\_id]-[number\_of\_views]-[canididate\_id]\_[label].txt, while files named like [app\_id]-[number\_of\_views]-original.txt contain the views on the reference device.

Each line in these files contains 1 view ([x, y, widht, height]).

Note that generating this dataset requires the code of the InferUI synthesizer used to produce the negative examples.
For more information, please reach out to [Pavol Bielik](https://www.sri.inf.ethz.ch/people/pavol).

## Ablation dataset
```bash
python dataset/scripts/create_ablation_dataset.py
```
The ablation dataset contains the different ablation categories. Each category folder contains the corresponding sample files.
Each line in these files contains 1 view ([x, y, widht, height]).


# (7) Project Structure
```
Pipeline
├── core/
│   ├── features/           # features used in MLP, RNN, CNN models
│   ├── model/              # the MLP, RNN and CNN pytorch models
│   ├── server/             # HTTP server used to serve the trained models
│   ├── saved_models/       # containes the trained models
│   └── results/            # (created during execution) contains logs 
│   ├── custom_data_reader.py # classes used to load and cache the datasets
│   ├── arguments.py        # configs
│   ├── model_helper.py     # helper methods to train & evaluate
│   ├── run_helper.py       # load, save and train models
│   ├── view.py             # model of an (android) view
├── dataset/
│   ├── data/               # contains raw datasets (view positions)
│   │   ├── ablation_dataset.tar.gz/
│   │   ├── ds.tar.gz
│   │   └── dsplus.tar.gz
│   ├── scripts/            # scripts to generate datasets
│   │   ├── create_ablation_dataset.py
│   │   ├── dudsds+         # scripts to generate D_U and D_S datasets and split DS+ dataset
├── cache/                  # (created during training) contains cached datasets 
├── train.py                # main file used to train the models
├── evaluate_oracles.py     # evaluates performance of trained oracles
├── run_ablation.py     # evaluates performance of trained oracles on ablation dataset
```

# (8) References

Please cite the following work when using this repository.

```
@inproceedings{laich2020guiding,
title={Guiding Program Synthesis by Learning to Generate Examples},
author={Larissa Laich and Pavol Bielik and Martin Vechev},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=BJl07ySKvS}
}
```

