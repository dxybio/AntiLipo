from torch.utils.data import DataLoader, Subset
from dataset import Peptide_Dataset, collate_fn_peptide_dataset, collate_fn_peptide_mask_dataset
import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
from models import MLP,Seq1DCNN,TransformerBlock





def predict_classify_epoch(model, loader, device):   
    model.eval()
    all_logits = []
    with torch.no_grad():
        for pep_feat_esm, pep_feat_phychem, mask_pep, labels in loader:
            pep_feat_esm, pep_feat_phychem, mask_pep = pep_feat_esm.to(device), pep_feat_phychem.to(device), mask_pep.to(device)
            labels = labels.float().to(device).squeeze()
            logits = model(pep_feat_esm, pep_feat_phychem, mask_pep)
            all_logits.append(logits.cpu())
    all_logits = torch.cat(all_logits)
    probs = torch.sigmoid(all_logits).numpy()
    return  probs




def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for pep_feat_esm, pep_feat_phychem, mask_pep, labels in loader:
        pep_feat_esm = pep_feat_esm.to(device)
        pep_feat_phychem = pep_feat_phychem.to(device)
        mask_pep = mask_pep.to(device)
        labels = labels.float().to(device).squeeze()
        optimizer.zero_grad()
        logits = model(pep_feat_esm, pep_feat_phychem, mask_pep)
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
        for pep_feat_esm, pep_feat_phychem, mask_pep, labels in loader:
            pep_feat_esm, pep_feat_phychem, mask_pep = pep_feat_esm.to(device), pep_feat_phychem.to(device), mask_pep.to(device)
            labels = labels.float().to(device).squeeze()
            logits = model(pep_feat_esm, pep_feat_phychem, mask_pep)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    probs = torch.sigmoid(all_logits).numpy()
    preds = (probs > 0.5).astype(int)
    return total_loss / len(loader.dataset), probs, preds, all_labels.numpy()






class ALP_MLP(nn.Module):
    def __init__(self, feat_dim1, feat_dim2, hidden_dim=128):
        super().__init__()
        self.feat1_mlp = MLP(feat_dim1, hidden_dim, hidden_dim)
        self.feat2_mlp = MLP(feat_dim2, hidden_dim, hidden_dim)
        self.feat_mlp = MLP(hidden_dim*2, hidden_dim, hidden_dim)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, pep_feat1, pep_feat2, mask_pep):
        h1 = self.feat1_mlp(pep_feat1)
        h2 = self.feat2_mlp(pep_feat2)
        h = torch.cat([h1, h2], dim=-1)
        h = self.feat_mlp(h).mean(dim=1)
        logits = self.predictor(h)
        return logits.squeeze(1)
    

class ALP_CNN(nn.Module):
    def __init__(self, feat_dim1, feat_dim2, hidden_dim=128):
        super().__init__()
        self.feat1_mlp = MLP(feat_dim1, hidden_dim, hidden_dim)
        self.feat2_mlp = MLP(feat_dim2, hidden_dim, hidden_dim)
        self.feat_mlp = MLP(hidden_dim*2, hidden_dim, hidden_dim)
        self.cnn = Seq1DCNN(hidden_dim, hidden_dim, kernel_size=3, num_layers=2, dropout=0.1)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, pep_feat1, pep_feat2, mask_pep):
        h1 = self.feat1_mlp(pep_feat1)
        h2 = self.feat2_mlp(pep_feat2)
        h = torch.cat([h1, h2], dim=-1)
        h = self.feat_mlp(h)
        h = self.cnn(h).mean(dim=1)
        logits = self.predictor(h)
        return logits.squeeze(1)
    
class ALP_Transformer(nn.Module):
    def __init__(self, feat_dim1=320, feat_dim2=35, hidden_dim=128):
        super().__init__()
        self.feat1_mlp = MLP(feat_dim1, hidden_dim, hidden_dim)
        self.feat2_mlp = MLP(feat_dim2, hidden_dim, hidden_dim)
        self.feat_mlp = MLP(hidden_dim*2, hidden_dim, hidden_dim)
        self.transformer = TransformerBlock(hidden_dim, num_heads=4, dropout=0.1)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, pep_feat1, pep_feat2, mask_pep):
        h1 = self.feat1_mlp(pep_feat1)
        h2 = self.feat2_mlp(pep_feat2)
        h = torch.cat([h1, h2], dim=-1)
        h = self.feat_mlp(h)
        h = self.transformer(h, h, h, k_mask=mask_pep.bool())
        h = h.mean(dim=1)
        logits = self.predictor(h)
        return logits.squeeze(1)

import pandas as pd
import os
def train_classify_kfold(model_name='MLP'):
    batch_size = 32
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    result_dir = f'pretrained/ALP_classify/{model_name}'
    os.makedirs(result_dir, exist_ok=True)
    all_best_test_probs, all_test_labels = [], []
    dataset = Peptide_Dataset()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        train_dataset, test_dataset = Subset(dataset, train_idx), Subset(dataset, test_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_peptide_mask_dataset)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_peptide_mask_dataset)
        if model_name == 'MLP': model = ALP_MLP(320, 42).to(device)
        elif model_name == 'CNN': model = ALP_CNN(320, 42).to(device)
        elif model_name == 'Transformer': model = ALP_Transformer(320, 42).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        criterion = nn.BCEWithLogitsLoss()
        patience, no_improve, best_loss, best_state = 10, 0, float('inf'), None
        best_test_probs = None
        for epoch in range(200):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            test_loss, test_probs, test_preds, test_labels = evaluate_epoch(model, test_loader, criterion, device)
            acc = accuracy_score(test_labels, test_preds)
            print(f"Fold {fold+1}, Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test ACC: {acc:.4f}")
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
    all_best_test_preds = (all_best_test_probs > 0.5).astype(int)
    acc = accuracy_score(all_test_labels, all_best_test_preds)
    print(acc)


# train_classify_kfold('MLP')
# train_classify_kfold('CNN')
# train_classify_kfold('Transformer')