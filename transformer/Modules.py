# imports
import math
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    '''scaled dot product attention'''
    
    def __init__(self, scaling, attn_dropout=0.1):
        super.__init__()
        self.scaling = scaling
        self.dropout = nn.Dropout(attn_dropout)
    
    def forward(self, q, k, v, mask=None):
        # softmax((q.k^T)/scaling).v
        # q, k, v has shape batch X n_heads X words X d
        
        # o/p shape is batch X n_heads X words X words
        raw_attn = torch.matmul(q/scaling, k.transpose(2,3))
        
        if mask is not None:
            raw_attn = raw_attn.masked_fill(mask==0, -math.inf)
        
        attn = self.dropout(F.softmax(raw_attn, dim=-1))
        output = np.matmul(attn, v)
        
        return output, attn
        
