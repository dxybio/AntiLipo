from torch.utils.data import DataLoader, Subset
from dataset import Peptide_Dataset, collate_fn_peptide_dataset, collate_fn_peptide_mask_dataset, ComplexFeatureDataset, collate_fn_affinity_dataset
import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
from ALP_classify import ALP_Transformer, predict_classify_epoch
from ALP_regress import Affinity_SelfAttention, predict_regress_epoch
import os
import pandas as pd
from feature_protein import process_total



target_names = [
    "CE-P19835",
    "CE-P30122",
    "FAS-P49327",
    "FAS-TE-P49327",
    "HMG-CoAR-P04035",
    "HMG-CoAR-P09610",
    "PPL-P00591"
]
    
def predict_classify_kfold(sequences):
    task_name='temp'
    task_df_path = f'task/{task_name}/{task_name}.csv'
    os.makedirs(f"task/{task_name}", exist_ok=True)
    df = pd.DataFrame({'PeptideSequence':sequences})
    df.to_csv(task_df_path, sep='\t', index=False)
    task_df = pd.read_csv(task_df_path, sep='\t')
    task_dir = f'task/{task_name}/'
    os.makedirs(task_dir, exist_ok=True)
    process_total(df_path=task_df_path, type='peptide', save_dir=task_dir, name_column='PeptideSequence', sequence_column='PeptideSequence',)       
    batch_size = 16
    all_test_probs = []
    dataset = Peptide_Dataset(csv_path=task_df_path, predict=True, padded=True,
                              peptide_esm_dir=f'{task_dir}/peptide/esm2_t6_8M_UR50D', 
                              peptide_phychem_dir=f'{task_dir}/peptide/phychem',
                              peptide_column='PeptideSequence',
                              )
    device = 'cpu'
    for fold in range(5):
        predict_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_peptide_mask_dataset)
        model = ALP_Transformer(320, 42).to(device)
        model.load_state_dict(torch.load(f'pretrained/ALP_classify/Transformer/fold{fold+1}_best_model.pth', map_location=device))
        test_probs = predict_classify_epoch(model, predict_loader, device)
        all_test_probs.append(test_probs)
    all_test_probs = np.mean(np.array(all_test_probs), axis=0)
    task_df['PredictedProb'] = all_test_probs.round(4)
    task_df.to_csv(f'{task_dir}/kfold_results.csv', sep='\t', index=False)
    affinity_csv = f"{task_dir}/affinity_input.csv"
    with open(affinity_csv, 'w') as f:
        f.write(f"PeptideSequence\tTargetStandardName\n")
        for index, row in task_df.iterrows():
            
            for target_name in target_names:
                f.write(f"{row['PeptideSequence']}\t{target_name}\n")
    batch_size = 16
    all_test_probs = []
    dataset =  ComplexFeatureDataset(csv_path=affinity_csv, predict=True, padded=True,
                              peptide_esm_dir=f'{task_dir}/peptide/esm2_t6_8M_UR50D', 
                              peptide_phychem_dir=f'{task_dir}/peptide/phychem',
                              peptide_column='PeptideSequence',
                              )
    device = 'cpu'
    for fold in range(5):
        predict_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_affinity_dataset)
        model = Affinity_SelfAttention(320, 42, 320, 42).to(device)
        model.load_state_dict(torch.load(f'pretrained/ALP_regress/SelfAttention/fold{fold+1}_best_model.pth', map_location=device))
        test_probs = predict_regress_epoch(model, predict_loader, device)
        all_test_probs.append(test_probs)
    all_test_probs = np.mean(np.array(all_test_probs), axis=0)
    all_test_probs = 10 ** all_test_probs
    all_test_probs = all_test_probs.reshape(-1, len(target_names))
    for i, target_name in  enumerate(target_names):
        task_df[target_name] = all_test_probs[:, i].round(4)
    task_df.to_csv(f'{task_dir}/kfold_results.csv', sep='\t', index=False)
    print(task_df)

predict_classify_kfold(['FLF', 'YK'])