
from torch.utils.data import Dataset
import os
import random
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
from Bio.PDB import PDBParser, PPBuilder
parent_dir = os.path.abspath(os.path.dirname(__file__))
parent_parent_dir =  os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


aa_mass = {
    "A": 89.0935, "R": 174.2017, "N": 132.1184, "D": 133.1032,
    "C": 121.1590, "E": 147.1299, "Q": 146.1451, "G": 75.0669,
    "H": 155.1552, "I": 131.1736, "L": 131.1736, "K": 146.1882,
    "M": 149.2124, "F": 165.1900, "P": 115.1310, "S": 105.0930,
    "T": 119.1197, "W": 204.2262, "Y": 181.1894, "V": 117.1469
}

def peptide_mw(seq):
    s = seq.replace(" ", "").upper()
    mw = sum(aa_mass[a] for a in s) - (len(s)-1)*18.015
    return mw

def to_umol_L(value, unit, mw):
    if unit == "μmol/L":
        return value
    if unit == "mg/mL":
        return value * 1000000 / mw
    if unit == "mmol/L":
        return value * 1000
    raise ValueError(unit)

def pdb_to_seq(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("p", pdb_file)
    ppb = PPBuilder()
    peptides = ppb.build_peptides(structure)
    if len(peptides) == 0:
        return ""
    seq = str(peptides[0].get_sequence())
    return seq

def process_regress(task_name='ALP', dataset_dir='dataset'):
    task_dir = f"{parent_dir}/{dataset_dir}/{task_name}"
    df = pd.read_csv(f"{task_dir}/{task_name}.csv", sep='\t')
    df = df[df['ActiveType']=='IC50']
    df = df.drop(df[df['PeptideSequence'] == 'Pyr-GF'].index)
    df['ActiveValue'] = pd.to_numeric(df['ActiveValue'])
    df['MW'] = df['PeptideSequence'].apply(peptide_mw)
    df['ActiveValueStandard'] = df.apply(lambda x: to_umol_L(x['ActiveValue'], x['ActiveUnit'], x['MW']), axis=1)
    df['pIC50'] = np.log10(df['ActiveValueStandard'])
    df.dropna(subset=['TargetUniportID'], inplace=True)
    df['TargetStandardName'] = df['TargetName']+'-'+df['TargetUniportID']
    df['TargetSequence'] = df['TargetStandardName'].apply(lambda x: pdb_to_seq(f'dataset/public/protein/pdbs/{x}.pdb'))
    df.to_csv(f"{task_dir}/{task_name}_regress.csv", sep='\t', index=False)

def random_peptide(length):
    aa = "ACDEFGHIKLMNPQRSTVWY"
    return ''.join(random.choice(aa) for _ in range(length))

def process_classify(task_name='ALP', dataset_dir='dataset'):
    task_dir = f"{parent_dir}/{dataset_dir}/{task_name}"
    df = pd.read_csv(f"{task_dir}/{task_name}.csv", sep='\t')
    df['seq_length'] = df['PeptideSequence'].apply(len)
    max_len = df['Length'].max()
    print(max_len)
    df = df.drop(df[df['PeptideSequence'] == 'Pyr-GF'].index)
    df = df.drop_duplicates(subset=['PeptideSequence'])
    df['RandomPeptide'] = df['PeptideSequence'].apply(lambda x: random_peptide(len(x)))
    df_orig = pd.DataFrame({"Peptide": df["PeptideSequence"],"Label": 1})
    df_rand = pd.DataFrame({"Peptide": df["RandomPeptide"],"Label": 0 })
    df_new = pd.concat([df_orig, df_rand], ignore_index=True)
    df_new.to_csv(f"{task_dir}/{task_name}_classify.csv", sep="\t", index=False)


if __name__ ==  "__main__":
    process_classify()
    process_regress()