from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import pickle
from module.RNN import *
from module.DNN import *
from module.helpers import *

def main():

    ### 1. Data load
    interactions_file_path = "./data/davis_data.tsv"  

    print(f"{interactions_file_path} is load ...")
    interactions_data = pd.read_csv(interactions_file_path, sep = "\t")
    train_data, val_data, test_data = split_data(interactions_data, frac = [0.7,0.1,0.2], seed = 0)
    print()
    
    #USE_CUDA = True # default
    USE_CUDA = False # for window
    GPU_NUM = 2
 
    if USE_CUDA:
        device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device)
        print ('Current cuda device ', torch.cuda.current_device()) # check
    print()

    ### 3. Model load
    with open("../../data/pretraining/Sequence_voca.pkl", "rb") as f:
        Protein_voca = pickle.load(f)

    with open("../../data/pretraining/SMILES_voca.pkl", "rb") as f:
        SMILES_voca = pickle.load(f)
    
    model_path = "./model/train_davis.pth"  
    
    regressor = load_checkpoint_eval(model_path, USE_CUDA, device) 
    
def split_data(data, frac, seed):
    
    train_frac, val_frac, test_frac = frac
    test = data.sample(frac = test_frac, replace = False, random_state = seed)
    
    train_val = data[~data.index.isin(test.index)]
    val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = seed)
    train = train_val[~train_val.index.isin(val.index)]
    
    print(f"Train samples: {train.shape[0]}")
    print(f"Validation samples: {val.shape[0]}")
    print(f"Test samples: {test.shape[0]}")
    print()
    
    train_p_seq_lengths = [len(i) for i in train.iloc[:, 0].values]
    val_p_seq_lengths = [len(i) for i in val.iloc[:, 0].values]
    test_p_seq_lengths = [len(i) for i in test.iloc[:, 0].values]

    print(f"Train max sequence lengths: {np.max(train_p_seq_lengths)}")
    print(f"Val max sequence lengths: {np.max(val_p_seq_lengths)}")
    print(f"Test max sequence lengths: {np.max(test_p_seq_lengths)}")
    print()
    
    train_c_seq_lengths = [len(i) for i in train.iloc[:, 1].values]
    val_c_seq_lengths = [len(i) for i in val.iloc[:, 1].values]
    test_c_seq_lengths = [len(i) for i in test.iloc[:, 1].values]

    print(f"Train max SMILES lengths: {np.max(train_c_seq_lengths)}")
    print(f"Val max SMILES lengths: {np.max(val_c_seq_lengths)}")
    print(f"Test max SMILES lengths: {np.max(test_c_seq_lengths)}")
    print()
    
    return train.reset_index(drop = True), val.reset_index(drop = True), test.reset_index(drop = True)
    #return train, val, test

if __name__ == "__main__":
    main()    