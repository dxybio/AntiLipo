import dgl
import torch
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from feature_protein import load_all_embeddings
import os

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

def build_complex_heterograph(pdb_path, pep_esm, pep_phychem, prot_esm, prot_phychem, cutoff=8.0, save_path=None):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)
    
    pep_coords = extract_ca_coordinates(structure, chain_id="B")
    pro_coords = extract_ca_coordinates(structure, chain_id="A")
    pep_len = pep_coords.shape[0]
    pro_len = pro_coords.shape[0]
    # ---------- 1) peptide-peptide edges by distance ----------
    pep_src, pep_dst = build_edges_by_distance(pep_coords, cutoff=cutoff)
    # ---------- 2) peptide-protein interaction edges ----------
    pep_pro_src = []
    pep_pro_dst = []
    selected_pro_res = set()
    for i in range(pep_len):
        for j in range(pro_len):
            d = np.linalg.norm(pep_coords[i] - pro_coords[j])
            if d <= cutoff:
                pep_pro_src.append(i)
                pep_pro_dst.append(j)
                selected_pro_res.add(j)

    selected_pro_res = sorted(list(selected_pro_res))

    # ---------- 3) build protein subgraph ----------
    pro_old_to_new = {old: idx for idx, old in enumerate(selected_pro_res)}
    pro_coords_sel = pro_coords[selected_pro_res]

    pep_pro_dst_mapped = [pro_old_to_new[x] for x in pep_pro_dst]

    pro_src, pro_dst = build_edges_by_distance(pro_coords_sel, cutoff=cutoff)

    # ---------- 4) build heterograph ----------
    graph = dgl.heterograph({
        ("peptide", "pep-pep", "peptide"): (pep_src, pep_dst),
        ("protein", "pro-pro", "protein"): (pro_src, pro_dst),
        ("peptide", "pep-pro", "protein"): (pep_pro_src, pep_pro_dst_mapped),
        ("protein", "pro-pep", "peptide"): (pep_pro_dst_mapped, pep_pro_src),
    })
    print(pep_esm.shape, pep_phychem.shape, prot_esm.shape, prot_phychem.shape)
    graph.nodes["peptide"].data["esm"] = torch.from_numpy(pep_esm).float()
    graph.nodes["peptide"].data["phychem"] = torch.from_numpy(pep_phychem).float()
    graph.nodes["protein"].data["esm"] = torch.from_numpy(prot_esm)[selected_pro_res].float()
    graph.nodes["protein"].data["phychem"] = torch.from_numpy(prot_phychem)[selected_pro_res].float()
    dgl.save_graphs(save_path, graph)
    return graph



def generate_all_complex_heterograph(df_path='dataset/ALP/ALP_regress.csv', pdb_dir='dataset/public/complex/pdbs', save_dir='dataset/public/complex/heterograph',):
    df = pd.read_csv(df_path, sep='\t')
    peptide_column = 'PeptideSequence'
    protein_column = 'TargetStandardName'
    peptide_esm_dir = 'dataset/public/peptide/esm2_t6_8M_UR50D'
    peptide_phychem_dir = 'dataset/public/peptide/phychem'
    protein_esm_dir = 'dataset/public/protein/esm2_t6_8M_UR50D'
    protein_phychem_dir = 'dataset/public/protein/phychem'  
    pep_esm_dict = load_all_embeddings(df_path, peptide_column, peptide_esm_dir)
    pep_phychem_dict = load_all_embeddings(df_path, peptide_column, peptide_phychem_dir)
    prot_esm_dict = load_all_embeddings(df_path, protein_column, protein_esm_dir)
    prot_phychem_dict = load_all_embeddings(df_path, protein_column, protein_phychem_dir)
    for index,row in df.iterrows():
        pdb_path = f"{pdb_dir}/{row['PeptideSequence']}_{row['TargetName']}-{row['TargetUniportID']}.pdb"
        save_path=f"{save_dir}/{row['PeptideSequence']}_{row['TargetStandardName']}.dgl"
        if os.path.exists(pdb_path):
            print(f"Warning: {pdb_path} not exists!")
            continue
        build_complex_heterograph(pdb_path, pep_esm_dict[row['PeptideSequence']], pep_phychem_dict[row['PeptideSequence']], 
                                  prot_esm_dict[row['TargetStandardName']], prot_phychem_dict[row['TargetStandardName']], 
                                  cutoff=8.0, save_path=save_path)


def load_all_complex_heterograph(df_path='dataset/ALP/ALP_regress.csv', heterograph_dir='dataset/public/complex/heterograph'):
    df = pd.read_csv(df_path, sep='\t')
    print(df.columns)
    graphs_dict = {}
    for index,row in df.iterrows():
        graph_path = f"{heterograph_dir}/{row['PeptideSequence']}_{row['TargetStandardName']}.dgl"
        graph = dgl.load_graphs(graph_path)[0][0]
        graphs_dict[f"{row['PeptideSequence']}_{row['TargetStandardName']}"] = graph
    return graphs_dict

if __name__ ==  "__main__":
    generate_all_complex_heterograph()
    load_all_complex_heterograph()