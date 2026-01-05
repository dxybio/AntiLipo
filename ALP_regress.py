
from ALP_classify import ALP_MLP
from dataset import ComplexHeteroGraphDataset, ComplexFeatureDataset, collate_fn_complex_hg_affinity_dataset, collate_fn_affinity_dataset
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, r2_score
import torch.nn as nn
import torch.nn.functional as F
from models import MLP,Seq1DCNN,TransformerBlock
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import HeteroGraphConv, GraphConv
import pandas as pd
import os

class HeteroGNN(nn.Module):
    def __init__(self, feat_dim1, feat_dim2, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.pep_feat1_mlp = MLP(feat_dim1, hidden_dim, hidden_dim)
        self.pep_feat2_mlp = MLP(feat_dim2, hidden_dim, hidden_dim)
        self.prot_feat1_mlp = MLP(feat_dim1, hidden_dim, hidden_dim)
        self.prot_feat2_mlp = MLP(feat_dim2, hidden_dim, hidden_dim)
        self.pep_feat_mlp = MLP(hidden_dim*2, hidden_dim, hidden_dim)
        self.prot_feat_mlp = MLP(hidden_dim*2, hidden_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            convs = {
                ('peptide', 'pep-pep', 'peptide'): GraphConv(hidden_dim, hidden_dim, norm='both', weight=True, bias=True),
                ('protein', 'pro-pro', 'protein'): GraphConv(hidden_dim, hidden_dim, norm='both', weight=True, bias=True),
                ('peptide', 'pep-pro', 'protein'): GraphConv(hidden_dim, hidden_dim, norm='both', weight=True, bias=True),
                ('protein', 'pro-pep', 'peptide'): GraphConv(hidden_dim, hidden_dim, norm='both', weight=True, bias=True),
            }
            self.layers.append(HeteroGraphConv(convs, aggregate='sum'))
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.readout_mlp = MLP(hidden_dim*2, hidden_dim, 1)

    def forward(self, g):
        pep_esm = self.pep_feat1_mlp(g.nodes['peptide'].data['esm'])
        pep_phy = self.pep_feat2_mlp(g.nodes['peptide'].data['phychem'])
        pro_esm = self.prot_feat1_mlp(g.nodes['protein'].data['esm'])
        pro_phy = self.prot_feat2_mlp(g.nodes['protein'].data['phychem'])
        h_pep = self.pep_feat_mlp(torch.cat([pep_esm, pep_phy], dim=-1))    
        h_prot = self.prot_feat_mlp(torch.cat([pro_esm, pro_phy], dim=-1))
        h = {
            'peptide': h_pep,
            'protein': h_prot
        }
        for layer in self.layers:
            h = layer(g, h)
            for ntype in h:
                h[ntype] = self.act(h[ntype])
                h[ntype] = self.dropout(h[ntype])

        g.nodes['peptide'].data['h'] = h['peptide']
        g.nodes['protein'].data['h'] = h['protein']

        pep_mean = dgl.mean_nodes(g, 'h', ntype='peptide')
        pro_mean = dgl.mean_nodes(g, 'h', ntype='protein')

        out = torch.cat([pep_mean, pro_mean], dim=-1)
        logits = self.readout_mlp(out).squeeze(-1)
        return logits

def train_epoch_hg(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for hgs, labels in loader:
        hgs = hgs.to(device)
        labels = labels.to(device).squeeze()
        optimizer.zero_grad()
        logits = model(hgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)

def evaluate_epoch_hg(model, loader, criterion, device):   
    model.eval()
    total_loss = 0
    all_logits, all_labels = [], []
    with torch.no_grad():
        for hgs, labels in loader:
            hgs = hgs.to(device)
            labels = labels.to(device).squeeze()
            logits = model(hgs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    return total_loss / len(loader.dataset), all_logits, all_labels.numpy()


def train_HeteroGNN_regress_kfold(model_name='HeteroGNN'):
    batch_size = 15
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    result_dir = f'pretrained/ALP_regress/{model_name}'
    os.makedirs(result_dir, exist_ok=True)
    dataset = ComplexHeteroGraphDataset()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_best_test_probs, all_test_labels = [], []
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        train_dataset, test_dataset = Subset(dataset, train_idx), Subset(dataset, test_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_complex_hg_affinity_dataset, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_complex_hg_affinity_dataset, drop_last=False)
        model = HeteroGNN(320, 42).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        criterion = nn.MSELoss()
        patience, no_improve, best_loss, best_state = 10, 0, float('inf'), None
        best_test_probs = None
        for epoch in range(200):
            train_loss = train_epoch_hg(model, train_loader, optimizer, criterion, device)
            test_loss, test_probs, test_labels = evaluate_epoch_hg(model, test_loader, criterion, device)
            r2 = r2_score(test_labels, test_probs)
            print(f"Fold {fold+1}, Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test R2: {r2:.4f}")
            if test_loss < best_loss: 
                best_loss, no_improve, best_state = test_loss, 0, model.state_dict()
                best_test_probs = test_probs
                torch.save(best_state, f'{result_dir}/fold{fold+1}_best_model.pth')
            else: 
                no_improve += 1
                if no_improve >= patience: 
                    print(f"Fold {fold+1}, Early stopping at epoch {epoch+1}"); 
                    break
        all_best_test_probs.extend(best_test_probs)
        all_test_labels.extend(test_labels)
    df = pd.DataFrame({'TrueLabel': all_test_labels, 'PredictedProb': all_best_test_probs}).round(4)
    df.to_csv(f'{result_dir}/fold_results.csv', sep='\t', index=False)
    all_best_test_probs = np.array(all_best_test_probs)
    r2 = r2_score(all_test_labels, all_best_test_probs)
    print(r2)



class Affinity_MLP(nn.Module):
    def __init__(self, pep_dim1, pep_dim2, prot_dim1, prot_dim2, hidden_dim=128):
        super().__init__()
        self.pep_feat1_mlp = MLP(pep_dim1, hidden_dim, hidden_dim)
        self.pep_feat2_mlp = MLP(pep_dim2, hidden_dim, hidden_dim)
        self.pep_feat_mlp = MLP(hidden_dim*2, hidden_dim, hidden_dim)
        self.prot_feat1_mlp = MLP(prot_dim1, hidden_dim, hidden_dim)
        self.prot_feat2_mlp = MLP(prot_dim2, hidden_dim, hidden_dim)
        self.prot_feat_mlp = MLP(hidden_dim*2, hidden_dim, hidden_dim)
        self.feat_mlp = MLP(hidden_dim*2, hidden_dim, hidden_dim)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, pep_esms, pep_phychems, prot_feats, prot_phychems, mask_peps, mask_prots):
        h_pep = self.pep_feat_mlp(torch.cat([self.pep_feat1_mlp(pep_esms), self.pep_feat2_mlp(pep_phychems)], dim=-1)).mean(dim=1)
        h_prot = self.prot_feat_mlp(torch.cat([self.prot_feat1_mlp(prot_feats), self.prot_feat2_mlp(prot_phychems)], dim=-1)).mean(dim=1)
        h = torch.cat([h_pep, h_prot], dim=-1)
        h = self.feat_mlp(h)
        logits = self.predictor(h)
        return logits.squeeze(1)


class Affinity_CNN(nn.Module):
    def __init__(self, pep_dim1, pep_dim2, prot_dim1, prot_dim2, hidden_dim=128):
        super().__init__()
        self.pep_feat1_mlp = MLP(pep_dim1, hidden_dim, hidden_dim)
        self.pep_feat2_mlp = MLP(pep_dim2, hidden_dim, hidden_dim)
        self.pep_feat_mlp = MLP(hidden_dim*2, hidden_dim, hidden_dim)
        self.prot_feat1_mlp = MLP(prot_dim1, hidden_dim, hidden_dim)
        self.prot_feat2_mlp = MLP(prot_dim2, hidden_dim, hidden_dim)
        self.prot_feat_mlp = MLP(hidden_dim*2, hidden_dim, hidden_dim)
        self.feat_mlp = MLP(hidden_dim*2, hidden_dim, hidden_dim)
        self.prot_cnn = Seq1DCNN(hidden_dim, hidden_dim, kernel_size=7, num_layers=2, dropout=0.1)
        self.pep_cnn = Seq1DCNN(hidden_dim, hidden_dim, kernel_size=3, num_layers=2, dropout=0.1)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, pep_esms, pep_phychems, prot_feats, prot_phychems, mask_peps, mask_prots):
        h_pep = self.pep_feat_mlp(torch.cat([self.pep_feat1_mlp(pep_esms), self.pep_feat2_mlp(pep_phychems)], dim=-1))
        h_pep=self.pep_cnn(h_pep).mean(dim=1)
        h_prot = self.prot_feat_mlp(torch.cat([self.prot_feat1_mlp(prot_feats), self.prot_feat2_mlp(prot_phychems)], dim=-1))
        h_prot = self.prot_cnn(h_prot).mean(dim=1)
        h = torch.cat([h_pep, h_prot], dim=-1)
        h = self.feat_mlp(h)
        logits = self.predictor(h)
        return logits.squeeze(1)


class Affinity_SelfAttention(nn.Module):
    def __init__(self, pep_dim1, pep_dim2, prot_dim1, prot_dim2, hidden_dim=128):
        super().__init__()
        self.pep_feat1_mlp = MLP(pep_dim1, hidden_dim, hidden_dim)
        self.pep_feat2_mlp = MLP(pep_dim2, hidden_dim, hidden_dim)
        self.pep_feat_mlp = MLP(hidden_dim*2, hidden_dim, hidden_dim)
        self.prot_feat1_mlp = MLP(prot_dim1, hidden_dim, hidden_dim)
        self.prot_feat2_mlp = MLP(prot_dim2, hidden_dim, hidden_dim)
        self.prot_feat_mlp = MLP(hidden_dim*2, hidden_dim, hidden_dim)
        self.feat_mlp = MLP(hidden_dim*2, hidden_dim, hidden_dim)
        self.pep_transformer = TransformerBlock(hidden_dim, num_heads=4, dropout=0.1)
        self.prot_transformer = TransformerBlock(hidden_dim, num_heads=4, dropout=0.1)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, pep_esms, pep_phychems, prot_feats, prot_phychems, mask_peps, mask_prots):
        h_pep = self.pep_feat_mlp(torch.cat([self.pep_feat1_mlp(pep_esms), self.pep_feat2_mlp(pep_phychems)], dim=-1))
        h_pep = self.pep_transformer(h_pep, h_pep, h_pep, k_mask=mask_peps.bool()).mean(dim=1)
        h_prot = self.prot_feat_mlp(torch.cat([self.prot_feat1_mlp(prot_feats), self.prot_feat2_mlp(prot_phychems)], dim=-1))
        h_prot = self.prot_transformer(h_prot, h_prot, h_prot, k_mask=mask_prots.bool()).mean(dim=1)
        h = torch.cat([h_pep, h_prot], dim=-1)
        h = self.feat_mlp(h)
        logits = self.predictor(h)
        return logits.squeeze(1)



class Affinity_CrossAttention(nn.Module):
    def __init__(self, pep_dim1, pep_dim2, prot_dim1, prot_dim2, hidden_dim=128):
        super().__init__()
        self.pep_feat1_mlp = MLP(pep_dim1, hidden_dim, hidden_dim)
        self.pep_feat2_mlp = MLP(pep_dim2, hidden_dim, hidden_dim)
        self.pep_feat_mlp = MLP(hidden_dim*2, hidden_dim, hidden_dim)
        self.prot_feat1_mlp = MLP(prot_dim1, hidden_dim, hidden_dim)
        self.prot_feat2_mlp = MLP(prot_dim2, hidden_dim, hidden_dim)
        self.prot_feat_mlp = MLP(hidden_dim*2, hidden_dim, hidden_dim)
        self.feat_mlp = MLP(hidden_dim*2, hidden_dim, hidden_dim)
        self.pep_transformer = TransformerBlock(hidden_dim, num_heads=4, dropout=0.1)
        self.prot_transformer = TransformerBlock(hidden_dim, num_heads=4, dropout=0.1)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, pep_esms, pep_phychems, prot_feats, prot_phychems, mask_peps, mask_prots):
        h_pep = self.pep_feat_mlp(torch.cat([self.pep_feat1_mlp(pep_esms), self.pep_feat2_mlp(pep_phychems)], dim=-1))
        h_prot = self.prot_feat_mlp(torch.cat([self.prot_feat1_mlp(prot_feats), self.prot_feat2_mlp(prot_phychems)], dim=-1))
        h_pep_new = self.pep_transformer(h_pep, h_prot, h_prot, k_mask=mask_prots.bool()).mean(dim=1)
        h_prot_new = self.prot_transformer(h_prot, h_pep, h_pep, k_mask=mask_peps.bool()).mean(dim=1)
        h = torch.cat([h_pep_new, h_prot_new], dim=-1)
        h = self.feat_mlp(h)
        logits = self.predictor(h)
        return logits.squeeze(1)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for pep_esms, pep_phychems, prot_feats, prot_phychems, mask_peps, mask_prots, labels in loader:
        pep_esms, pep_phychems, prot_feats, prot_phychems, mask_peps, mask_prots, labels = pep_esms.to(device), pep_phychems.to(device), prot_feats.to(device), prot_phychems.to(device), mask_peps.to(device), mask_prots.to(device), labels.to(device).squeeze()    
        optimizer.zero_grad()
        logits = model(pep_esms, pep_phychems, prot_feats, prot_phychems, mask_peps, mask_prots) 
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)

def evaluate_epoch(model, loader, criterion, device):   
    model.eval()
    total_loss = 0
    all_logits, all_labels = [], []
    with torch.no_grad():
        for pep_esms, pep_phychems, prot_feats, prot_phychems, mask_peps, mask_prots, labels in loader:
            pep_esms, pep_phychems, prot_feats, prot_phychems, mask_peps, mask_prots = pep_esms.to(device), pep_phychems.to(device), prot_feats.to(device), prot_phychems.to(device), mask_peps.to(device), mask_prots.to(device)    
            labels = labels.to(device).squeeze()
            logits = model(pep_esms, pep_phychems, prot_feats, prot_phychems, mask_peps, mask_prots) 
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    return total_loss / len(loader.dataset), all_logits, all_labels.numpy()

def predict_regress_epoch(model, loader, device):   
    model.eval()
    all_logits = []
    with torch.no_grad():
        for pep_esms, pep_phychems, prot_feats, prot_phychems, mask_peps, mask_prots, labels in loader:
            pep_esms, pep_phychems, prot_feats, prot_phychems, mask_peps, mask_prots = pep_esms.to(device), pep_phychems.to(device), prot_feats.to(device), prot_phychems.to(device), mask_peps.to(device), mask_prots.to(device)    
            labels = labels.to(device).squeeze()
            logits = model(pep_esms, pep_phychems, prot_feats, prot_phychems, mask_peps, mask_prots) 
            all_logits.append(logits.cpu())
    all_logits = torch.cat(all_logits)
    return all_logits

def train_regress_kfold(model_name='MLP'):
    batch_size = 15
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    result_dir = f'pretrained/ALP_regress/{model_name}'
    os.makedirs(result_dir, exist_ok=True)
    dataset = ComplexFeatureDataset()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_best_test_probs, all_test_labels = [], []
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        train_dataset, test_dataset = Subset(dataset, train_idx), Subset(dataset, test_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_affinity_dataset, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_affinity_dataset, drop_last=False)
        if model_name=='MLP':
            model = Affinity_MLP(320, 42, 320, 42).to(device)
        elif model_name=='CNN':
            model = Affinity_CNN(320, 42, 320, 42).to(device)
        elif model_name=='SelfAttention':
            model = Affinity_SelfAttention(320, 42, 320, 42).to(device)
        elif model_name=='CrossAttention':
            model = Affinity_CrossAttention(320, 42, 320, 42).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        criterion = nn.MSELoss()
        patience, no_improve, best_loss, best_state = 10, 0, float('inf'), None
        best_test_probs = None
        for epoch in range(200):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            test_loss, test_probs, test_labels = evaluate_epoch(model, test_loader, criterion, device)
            r2 = r2_score(test_labels, test_probs)
            print(f"Fold {fold+1}, Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test R2: {r2:.4f}")
            if test_loss < best_loss: 
                best_loss, no_improve, best_state = test_loss, 0, model.state_dict()
                best_test_probs = test_probs
                torch.save(best_state, f'{result_dir}/fold{fold+1}_best_model.pth')
            else: 
                no_improve += 1
                if no_improve >= patience: 
                    print(f"Fold {fold+1}, Early stopping at epoch {epoch+1}"); 
                    break
        all_best_test_probs.extend(best_test_probs)
        all_test_labels.extend(test_labels)
    df = pd.DataFrame({'TrueLabel': all_test_labels, 'PredictedProb': all_best_test_probs}).round(4)
    df.to_csv(f'{result_dir}/fold_results.csv', sep='\t', index=False)
    all_best_test_probs = np.array(all_best_test_probs)
    r2 = r2_score(all_test_labels, all_best_test_probs)
    print(r2)



if __name__ == '__main__':
    train_regress_kfold(model_name='MLP')
    train_regress_kfold(model_name='CNN')
    train_regress_kfold(model_name='SelfAttention')
    train_regress_kfold(model_name='CrossAttention')
    train_HeteroGNN_regress_kfold(model_name='HeteroGNN')