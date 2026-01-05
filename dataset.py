import torch
import os
import torch
import dgl
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from feature_protein import load_all_embeddings
from feature_complex import load_all_complex_heterograph

class Peptide_Dataset(Dataset):
    def __init__(self, 
                 padded=True,
                 predict=False, 
                 csv_path='dataset/ALP/ALP_classify.csv', 
                 peptide_esm_dir='dataset/public/peptide/esm2_t6_8M_UR50D', 
                 peptide_phychem_dir='dataset/public/peptide/phychem',
                 peptide_column='PeptideSequence',
                 ):
        self.df = pd.read_csv(csv_path, sep='\t')
        self.pep_esm_dict = load_all_embeddings(csv_path, peptide_column, peptide_esm_dir)
        self.pep_phychem_dict = load_all_embeddings(csv_path, peptide_column, peptide_phychem_dir)
        self.peptide_column = peptide_column
        self.padded = padded
        self.max_len = 23
        self.predict = predict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.predict: label =  torch.tensor([0.0], dtype=torch.float32)
        else: label = torch.tensor([row['Label']], dtype=torch.float32)
        pep_feat_esm = torch.from_numpy(self.pep_esm_dict[row[self.peptide_column]]).float()
        pep_feat_phychem = torch.from_numpy(self.pep_phychem_dict[row[self.peptide_column]]).float()
        mask_pep = torch.zeros(self.max_len)
        if not self.padded:
            return pep_feat_esm, pep_feat_phychem, label
        else:
            seq_len = self.pep_esm_dict[row[self.peptide_column]].shape[0]
            mask_pep[:seq_len] = 1
            padded_esm_feat = torch.zeros((self.max_len, pep_feat_esm.shape[1]))
            padded_phychem_feat = torch.zeros((self.max_len, pep_feat_phychem.shape[1]))
            padded_esm_feat[:seq_len, :] = pep_feat_esm
            padded_phychem_feat[:seq_len, :] = pep_feat_phychem
            return padded_esm_feat, padded_phychem_feat, mask_pep, label
    
def collate_fn_peptide_dataset(batch):
    pep_feats_esm, pep_feats_phychem, labels = zip(*batch)
    pep_feats_esm = torch.stack(pep_feats_esm, dim=0)
    pep_feats_phychem = torch.stack(pep_feats_phychem, dim=0)
    labels = torch.stack(labels, dim=0)
    return pep_feats_esm, pep_feats_phychem, labels 

def collate_fn_peptide_mask_dataset(batch):
    pep_feats_esm, pep_feats_phychem, mask_peps, labels = zip(*batch)
    pep_feats_esm = torch.stack(pep_feats_esm, dim=0)
    pep_feats_phychem = torch.stack(pep_feats_phychem, dim=0)
    mask_peps = torch.stack(mask_peps, dim=0)   
    labels = torch.stack(labels, dim=0)
    return pep_feats_esm, pep_feats_phychem, mask_peps, labels 



class ComplexFeatureDataset(Dataset):
    def __init__(self, 
                 padded=True,
                 predict=False,
                 csv_path='dataset/ALP/ALP_regress.csv', 
                 peptide_esm_dir='dataset/public/peptide/esm2_t6_8M_UR50D', 
                 peptide_phychem_dir='dataset/public/peptide/phychem',
                 peptide_column='PeptideSequence',
                 protein_column='TargetStandardName',
                 protein_esm_dir='dataset/public/protein/esm2_t6_8M_UR50D',
                 protein_phychem_dir='dataset/public/protein/phychem',
                 ):
        self.df = pd.read_csv(csv_path, sep='\t')
        if not predict:
            self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        self.pep_esm_dict = load_all_embeddings(csv_path, peptide_column, peptide_esm_dir)
        self.pep_phychem_dict = load_all_embeddings(csv_path, peptide_column, peptide_phychem_dir)
        self.prot_esm_dict = load_all_embeddings(csv_path, protein_column, protein_esm_dir)
        self.prot_phychem_dict = load_all_embeddings(csv_path, protein_column, protein_phychem_dir)
        self.peptide_column = peptide_column
        self.protein_column = protein_column
        self.padded = padded
        self.max_pep_len = 23
        self.max_prot_len = 877
        self.predict = predict


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.predict: label = torch.tensor([0.0], dtype=torch.float32)
        else: label = torch.tensor([row['pIC50']], dtype=torch.float32)
        pep_feat_esm = torch.from_numpy(self.pep_esm_dict[row[self.peptide_column]]).float()
        pep_feat_phychem = torch.from_numpy(self.pep_phychem_dict[row[self.peptide_column]]).float()
        prot_feat_esm = torch.from_numpy(self.prot_esm_dict[row[self.protein_column]]).float()
        prot_feat_phychem = torch.from_numpy(self.prot_phychem_dict[row[self.protein_column]]).float()
        mask_pep = torch.zeros(self.max_pep_len)
        mask_prot = torch.zeros(self.max_prot_len)
        if not self.padded:
            return pep_feat_esm, pep_feat_phychem, prot_feat_esm, prot_feat_phychem, mask_pep, mask_prot, label
        else:
            pep_seq_len = pep_feat_esm.shape[0]
            mask_pep[:pep_seq_len] = 1
            prot_seq_len = prot_feat_esm.shape[0]
            mask_prot[:prot_seq_len] = 1
            pep_padded_esm_feat = torch.zeros((self.max_pep_len, pep_feat_esm.shape[1]))
            pep_padded_esm_feat[:pep_seq_len, :] = pep_feat_esm
            pep_padded_phychem_feat = torch.zeros((self.max_pep_len, pep_feat_phychem.shape[1]))
            pep_padded_phychem_feat[:pep_seq_len, :] = pep_feat_phychem
            prot_padded_esm_feat = torch.zeros((self.max_prot_len, prot_feat_esm.shape[1]))
            prot_padded_esm_feat[:prot_seq_len, :] = prot_feat_esm
            prot_padded_phychem_feat = torch.zeros((self.max_prot_len, prot_feat_phychem.shape[1]))
            prot_padded_phychem_feat[:prot_seq_len, :] = prot_feat_phychem
            return pep_padded_esm_feat, pep_padded_phychem_feat, prot_padded_esm_feat, prot_padded_phychem_feat, mask_pep, mask_prot, label


def collate_fn_affinity_dataset(batch):
    pep_esms, pep_phychems, prot_esm, prot_phychems, mask_pep, mask_prot, label = zip(*batch)
    pep_esms = torch.stack(pep_esms, dim=0)
    pep_phychems = torch.stack(pep_phychems, dim=0) 
    prot_feats = torch.stack(prot_esm, dim=0)
    prot_phychems = torch.stack(prot_phychems, dim=0)
    mask_peps = torch.stack(mask_pep, dim=0)   
    mask_prots = torch.stack(mask_prot, dim=0)   
    affinity_labels = torch.stack(label, dim=0) 
    return pep_esms, pep_phychems, prot_feats, prot_phychems, mask_peps, mask_prots, affinity_labels



class ComplexHeteroGraphDataset(Dataset):
    def __init__(self, csv_path='dataset/ALP/ALP_regress.csv', 
                 complex_hetero_graph_dir='dataset/public/complex/heterograph', 
                 protein_column='TargetStandardName',
                 peptide_column='PeptideSequence',
                 ):
        self.df = pd.read_csv(csv_path, sep='\t')
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        self.hg_dict = load_all_complex_heterograph(csv_path, complex_hetero_graph_dir)
        self.protein_column = protein_column
        self.peptide_column = peptide_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        affinity_label = torch.tensor([row['pIC50']], dtype=torch.float32)
        hetero_graph = self.hg_dict[f"{row[self.peptide_column]}_{row[self.protein_column]}"]
        return hetero_graph, affinity_label
    

def collate_fn_complex_hg_affinity_dataset(batch):
    hgs, affinity_labels = zip(*batch)
    hgs = dgl.batch(hgs)
    affinity_labels = torch.stack(affinity_labels, dim=0)
    return hgs, affinity_labels


