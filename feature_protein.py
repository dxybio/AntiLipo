
import os 
import torch
import esm
import pandas as pd
import collections
import numpy as np
import os
import shutil
from Bio.PDB import PDBParser

from PeptideBuilder import Geometry
import PeptideBuilder
from Bio.PDB import PDBIO
from rdkit import Chem
from rdkit.Chem import AllChem

DSSP = './tools/dssp'



def build_peptide_rdkit(seq, out_pdb):
    structure = PeptideBuilder.initialize_res(seq[0])
    for aa in seq[1:]:
        structure = PeptideBuilder.add_residue(structure, aa)
    io = PDBIO()
    io.set_structure(structure)
    temp_pdb = "temp_raw.pdb"
    io.save(temp_pdb)
    mol = Chem.MolFromPDBFile(temp_pdb, removeHs=False)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.UFFOptimizeMolecule(mol)
    Chem.MolToPDBFile(mol, out_pdb)
    print("Saved:", out_pdb)
    os.remove(temp_pdb)



ESM_MODELS_INFO = {
    "esm2_t6_8M_UR50D": {"layers": 6, "hidden_size": 320},
    "esm2_t12_35M_UR50D": {"layers": 12, "hidden_size": 480},
    "esm2_t30_150M_UR50D": {"layers": 30, "hidden_size": 640},
    "esm2_t33_650M_UR50D": {"layers": 33, "hidden_size": 1280},
    "esm2_t36_3B_UR50D": {"layers": 36, "hidden_size": 2560},
    "esm2_t48_15B_UR50D": {"layers": 48, "hidden_size": 5120}, 
    "esm1v_t33_650M_UR90S_1": {"layers": 33, "hidden_size": 1280}, 
}


def esm_embeddings(tuple_list, esm_model_name='esm2_t6_8M_UR50D', save_dir='single_mut/WT'):
    os.makedirs(save_dir, exist_ok=True)
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(f'tools/esm/{esm_model_name}.pt')
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    batch_labels, batch_strs, batch_tokens = batch_converter(tuple_list)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1) 
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=False)
        token_representations = results["representations"][model.num_layers]
    for i, tokens_len in enumerate(batch_lens):
        aa_emb = token_representations[i, 1:tokens_len-1].cpu().numpy()
        np.save(f'{save_dir}/{batch_labels[i]}.npy', aa_emb)

def generate_all_esm_embeddings(df_path, name_column, sequence_column, esm_model_name, batch_size=50, temp_output_dir='output', save_dir='single_mut/WT'):
    df = pd.read_csv(df_path, sep='\t')
    seq_ls = list(df[sequence_column])
    name_ls = list(df[name_column])
    tuple_ls = []
    for index in range(len(seq_ls)):
        if not os.path.exists(f'{save_dir}/{name_ls[index]}.npy'):
            tuple_ls.append((name_ls[index], seq_ls[index]))
    tuple_ls = list(set(tuple_ls))
    os.makedirs(temp_output_dir, exist_ok=True)
    tasks = list(range(0, len(tuple_ls), batch_size))
    for i in range(len(tasks)):
        print(f'Processing task {i+1}/{len(tasks)}')
        if i != (len(tasks) - 1):
            esm_embeddings(tuple_ls[tasks[i]:tasks[i+1]], esm_model_name, save_dir)
        else:
            esm_embeddings(tuple_ls[tasks[i]:], esm_model_name, save_dir)
    shutil.rmtree(temp_output_dir)


def load_all_embeddings(df_path, name_column, save_dir='single_mut/WT'):
    df = pd.read_csv(df_path, sep='\t')
    name_ls = df[name_column].unique().tolist()
    feature_embeddings = {}
    for name in name_ls:
        
        emb = np.load(f'{save_dir}/{name}.npy')
        feature_embeddings[name] = emb
    return feature_embeddings


def process_pdb_for_dssp(input_path, output_path):
    with open(input_path) as f:
        lines = f.readlines()
    cryst1_lines = [line for line in lines if line.startswith("CRYST1")]
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("MODEL"):
            start_idx = i
            break
    with open(output_path, "w") as out:
        for line in cryst1_lines:
            out.write(line)
        for line in lines[start_idx:]:
            out.write(line)

def generate_all_pep_pdb(df_path='', 
                    save_pdb_dir='', 
                    sequence_column='PeptideSequence'):
    os.makedirs(save_pdb_dir, exist_ok=True)
    df=pd.read_csv(df_path,sep='\t')
    seqs = df[sequence_column].unique()
    for seq in seqs:
        if not os.path.exists(f"{save_pdb_dir}/{seq}.pdb"):
            build_peptide_rdkit(seq, f"{save_pdb_dir}/{seq}.pdb")


def process_all_pdb_for_dssp(df_path='', 
                            input_pdb_dir='', 
                            save_pdb_dir='', 
                            protein_column='Protein_UniProtID'):
    os.makedirs(save_pdb_dir, exist_ok=True)
    df=pd.read_csv(df_path,sep='\t')
    uniprot_ids = df[protein_column].unique()
    for uniprot_id in uniprot_ids:
        if not os.path.exists(f"{save_pdb_dir}/{uniprot_id}.pdb"):
            process_pdb_for_dssp(f"{input_pdb_dir}/{uniprot_id}.pdb", f"{save_pdb_dir}/{uniprot_id}.pdb")


AA_LIST = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]
SS_LIST = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']
map_aa_number = {aa: i for i, aa in enumerate(AA_LIST)}
map_ss_dssp = {ss: i for i, ss in enumerate(SS_LIST)}
map_aa_phy = {
    'C': 0,
    'D': 1, 'E': 1,
    'R': 2, 'K': 2,
    'H': 3, 'N': 3, 'Q': 3, 'W': 3,
    'Y': 4, 'M': 4, 'T': 4, 'S': 4,
    'I': 5, 'L': 5, 'F': 5, 'P': 5,
    'A': 6, 'G': 6, 'V': 6,
}
map_aa_chem = {
    'A': 0,'V':0,'L':0,'I':0,'M':0,
    'F':0,'W':0,'P':0,'G':0,'C':0,
    'S':1,'T':1,'N':1,'Q':1,'Y':1,
    'D':2,'E':2,
    'K':3,'R':3,'H':3
}
SANDER_MAX_ASA = {
    'A': 106, 'R': 248, 'N': 157, 'D': 163, 'C': 135, 'Q': 198, 'E': 194,
    'G': 84,  'H': 184, 'I': 169, 'L': 164, 'K': 205, 'M': 188, 'F': 197,
    'P': 136, 'S': 130, 'T': 142, 'W': 227, 'Y': 222, 'V': 142
}

def one_hot(idx, dim):
    v = np.zeros(dim, dtype=np.float32)
    if 0 <= idx < dim:
        v[idx] = 1.0
    return v

def get_bfactors(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)
    bfactors = []
    for model in structure:
        for chain in model:
            for res in chain:
                try:
                    bfactors.append(res['CA'].get_bfactor())
                except KeyError:
                    bfactors.append(0.0)
    return bfactors

def cal_protein_phychem_feat(dssp_rst_path, pdb_path):
    # dim 42
    bfactors = get_bfactors(pdb_path)
    with open(dssp_rst_path, 'r') as fr:
        dssp_data = fr.readlines()
    seq_feature = []
    for idx,i in enumerate(range(25, len(dssp_data))):
        line = dssp_data[i]
        aa = line[13].upper() if line[13] != '!' else 'X'
        ss = line[16] if line[16] in SS_LIST else ' '
        rasa = int(line[35:38].strip())/SANDER_MAX_ASA[aa]
        rasa = rasa if rasa <1 else 1.
        f_aa_number = one_hot(map_aa_number.get(aa, 21), 21)
        f_aa_chem  = one_hot(map_aa_chem.get(aa, 0), 4)
        f_aa_phy = one_hot(map_aa_phy.get(aa, 6), 7)
        f_ss_dssp  = one_hot(map_ss_dssp.get(ss, 7), 8)
        bfactor = bfactors[idx]/100
        feature = np.concatenate([f_aa_number, f_aa_chem, f_aa_phy, f_ss_dssp, np.array([rasa, bfactor])])
        seq_feature.append(feature)
    seq_feature = np.array(seq_feature, dtype=np.float32)
    return seq_feature

def generate_all_protein_phychem(df_path='dataset/feature/PDBbind/Ki/random/all.csv', 
                              dssp_dir="dataset/feature/protein/dssp", 
                              dssp_pdb_dir="dataset/feature/protein/pdbs_noheader", 
                              phychem_dir="dataset/feature/protein/phychem",
                              protein_column='Protein_UniProtID'):
    df = pd.read_csv(df_path, sep="\t")
    uniprot_ids = df[protein_column].unique()
    os.makedirs(dssp_dir, exist_ok=True)
    for uniprot_id in uniprot_ids:
        if os.path.exists(f"{dssp_dir}/{uniprot_id}.dssp"):
            continue
        os.system(f"{DSSP} -i {dssp_pdb_dir}/{uniprot_id}.pdb -o {dssp_dir}/{uniprot_id}.dssp")
    os.makedirs(phychem_dir, exist_ok=True)
    for uniprot_id in uniprot_ids:
        print(uniprot_id)
        if os.path.exists(f"{phychem_dir}/{uniprot_id}.npy"):
            continue
        phychem_feat = cal_protein_phychem_feat(f'{dssp_dir}/{uniprot_id}.dssp', f"{dssp_pdb_dir}/{uniprot_id}.pdb")
        np.save(f'{phychem_dir}/{uniprot_id}.npy', phychem_feat)



def delete_feature(uniprot_id='A0A0F7UUA6', 
                dssp_dir="dataset/public/peptide/dssp", 
                dssp_pdb_dir="dataset/public/peptide/dssp_pdbs",
                ):
    os.remove(f"{dssp_pdb_dir}/{uniprot_id}.pdb")
    os.remove(f"{dssp_dir}/{uniprot_id}.dssp")


def extract_ca_coordinates(structure, chain_id):
    coords = []
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for res in chain:
                    if "CA" in res:
                        coords.append(res["CA"].coord)
        break
    return np.array(coords)

def build_edges_by_distance(coords, cutoff=8.0):
    N = coords.shape[0]
    edges_src = []
    edges_dst = []

    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(coords[i] - coords[j])
            if d <= cutoff:
                edges_src += [i, j]
                edges_dst += [j, i]
    return edges_src, edges_dst

def process_total(df_path='dataset/ALP/ALP_regress.csv', 
                  type='protein', 
                  save_dir='dataset/public',
                  name_column='TargetStandardName', 
                  sequence_column='TargetSequence',):
    esm_dir=f'{save_dir}/{type}/esm2_t6_8M_UR50D'
    pdb_dir=f'{save_dir}/{type}/pdbs'
    dssp_pdb_dir=f'{save_dir}/{type}/dssp_pdbs'
    dssp_dir=f'{save_dir}/{type}/dssp'
    phychem_dir=f"{save_dir}/{type}/phychem" 
    if type == 'peptide':
        generate_all_pep_pdb(df_path=df_path, 
                            save_pdb_dir=pdb_dir, 
                            sequence_column=sequence_column)
    generate_all_esm_embeddings(df_path, 
                                name_column, 
                                sequence_column, 
                                esm_model_name='esm2_t6_8M_UR50D', 
                                batch_size=50, 
                                temp_output_dir='temp_esm_embeddings', 
                                save_dir=esm_dir)
    process_all_pdb_for_dssp(df_path=df_path, 
                            input_pdb_dir=pdb_dir, 
                            save_pdb_dir=dssp_pdb_dir, 
                            protein_column=name_column)
    generate_all_protein_phychem(df_path=df_path, 
                                dssp_dir=dssp_dir, 
                                dssp_pdb_dir=dssp_pdb_dir, 
                                phychem_dir=phychem_dir,
                                protein_column=name_column)





if __name__ ==  "__main__":
    process_total(df_path='dataset/ALP/ALP_regress.csv', type='peptide', name_column='PeptideSequence',  sequence_column='PeptideSequence',)
    process_total(df_path='dataset/ALP/ALP_regress.csv', type='protein', name_column='TargetStandardName',  sequence_column='TargetSequence',)
    process_total(df_path='dataset/ALP/ALP_classify.csv', type='peptide', name_column='PeptideSequence',  sequence_column='PeptideSequence',)