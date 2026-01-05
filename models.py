import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq1DCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, kernel_size=3, num_layers=2, dropout=0.1):
        super().__init__()
        layers=[]
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers += [
                nn.Conv1d(in_dim, hidden_dim, kernel_size, padding=(kernel_size-1)//2),
                nn.GELU(),
                nn.Dropout(dropout)
            ]
        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)
        h = self.cnn(x)
        h = h.transpose(1, 2)
        return h

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads=4,dropout=0.1):
        super().__init__()
        assert d_model%num_heads==0
        self.h=num_heads
        self.d=d_model//num_heads
        self.W_Q=nn.Linear(d_model,d_model)
        self.W_K=nn.Linear(d_model,d_model)
        self.W_V=nn.Linear(d_model,d_model)
        self.W_O=nn.Linear(d_model,d_model)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.ffn=nn.Sequential(
            nn.Linear(d_model,4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model,d_model)
        )
        self.dropout=nn.Dropout(dropout)

    def forward(self,q,k,v,k_mask=None):
        B,Lq,D=q.shape
        _,Lk,_=k.shape
        Q=self.W_Q(q)
        K=self.W_K(k)
        V=self.W_V(v)
        Q=Q.reshape(B,Lq,self.h,self.d).transpose(1,2)  # [B,h,Lq,d]
        K=K.reshape(B,Lk,self.h,self.d).transpose(1,2)  # [B,h,Lk,d]
        V=V.reshape(B,Lk,self.h,self.d).transpose(1,2)  # [B,h,Lk,d]
        attn_logits=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(self.d)
        if k_mask is not None:
            mask=k_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,Lk]
            attn_logits=attn_logits.masked_fill(~mask,float('-inf'))
        attn=F.softmax(attn_logits,dim=-1)
        attn=self.dropout(attn)
        out=torch.matmul(attn,V)  # [B,h,Lq,d]
        out=out.transpose(1,2).contiguous().reshape(B,Lq,self.h*self.d)
        out=self.W_O(out)
        q=q+self.dropout(out)
        q=self.norm1(q)
        ffn_out=self.ffn(q)
        q=q+self.dropout(ffn_out)
        q=self.norm2(q)
        return q

