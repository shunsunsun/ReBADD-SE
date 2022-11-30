# ReBADD_SE
Multi-objective Molecular Optimization via Off-Policy Self-critical SELFIES Training


## Install
```
conda env create -f environment.yml
```

----
# Task Descriptions
- TASK1: ReBADD-SE for GSK3b, JNK3, QED, and SA
- TASK2: ReBADD-SE for GSK3b, JNK3, QED, and SA (wo SELFIES fragment)
- TASK3: ReBADD-SE for BCL2, BCLXL, and BCLW
- TASK4: ReBADD-SE for BCL2, BCLXL, and BCLW (wo SELFIES fragment)
- TASK5: Ablation Study of ReBADD-SE (BCL2, BCLXL, and BCLW)


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
