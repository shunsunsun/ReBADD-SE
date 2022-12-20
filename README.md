# ReBADD_SE
Multi-objective Molecular Optimisation using SELFIES Fragment and Off-Policy Self-critical SELFIES Training


## Install
```
conda env create -f environment.yml
```

----
# Task Descriptions
- TASK1: ReBADD-SE for GSK3b, JNK3, QED, and SA (frag-level)
- TASK3: ReBADD-SE for BCL2, BCLXL, and BCLW (frag-level)
- TASK4: ReBADD-SE for BCL2, BCLXL, and BCLW (char-level)
- TASK5: ReBADD-SE for BCL2, BCLXL, and BCLW (frag-level & original SCST)
- TASK7: SELFIES Collapse Analaysis between ReBADD-SE (frag, char-level) and GA+D

----
# Notebook Descriptions

## 0_preprocess_data.ipynb
- Read the training data
- Preprocess the data for model training
- The preprocessed data are stored in the 'processed_data' directory

## 1_pretraining.ipynb
- Read the training data
- The generator learns the grammar rules of SELFIES

## 2_optimize+{objectives}.ipynb
- (Important!) Please check first the 'ReBADD_config.py' in which a reward function have to be defined appropriately
- Load the pretrained generator

## 3_checkpoints+{objectives}.ipynb
- Load the checkpoints stored during optimization
- Sample molecules for each checkpoint

## 4_calculate_properties.ipynb
- For each checkpoint, load the sampled molecules
- Evaluate their property scores

## 5_evaluate_checkpoints.ipynb
- Calculate metrics (e.g. success rate)
- Find the best checkpoint

----
# Note
- For more detail information about this repository, please contact via email:
```
mathcombio@yonsei.ac.kr
```