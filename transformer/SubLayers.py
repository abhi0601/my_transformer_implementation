# imports
import torch.nn as nn
import torch.nn.functional as F

from Modules import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    '''Multi head attention Module'''
    
    def __init__(self, n_heads=8, d_model=512, d_k=64, d_v=64, dropout=0.1):
        super().__init__()
        
        self.n_heads=n_heads
        self.d_k = d_k
        self.d_v = d_v
        
        # initalize the query, key and value matrices
        self.w_q = nn.Linear(d_model, n_heads*d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_heads*d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_heads*d_v, bias=False)
        
        # scaled dot product attention
        self.attention = ScaledDotProductAttention(scaling=d_k ** 0.5)
        
        # initalize the matrix that multiplies with concatenated outputs
        self.fc = nn.Linear(n_heads*d_v, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, q, k, v, mask=None):
        # q, k, v are of shape bs X words X d_model
        
        batch_size = q.size[0]
        n_query = q.size[1]
        n_keys = k.size[1]
        n_values = v.size[1]
        

        residual_q = q
        
        # view differnt heads separately
        q = self.w_q(q).view(batch_size, n_query, self.n_heads, self.d_k)
        k = self.w_k(k).view(batch_size, n_key, self.n_heads, self.d_k)
        v = self.w_v(v).view(batch_size, n_value, self.n_heads, self.d_v)
        
        # Now transpose to bring in format batch X n_heads X words X d
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        # z has dimension batch X n_heads X words X d_v
        z, attn = self.attention(q, k, v, mask=mask)
        
        '''
        concatenate all attention outputs in Z, 
        make new dimesion as batch X words X (n_heads*d_v),
        then pass through fc layer and make dim batch X words X d_model
        
        ''' 
        # try replacing n_query with n_keys
        z = self.dropout(self.fc(z.tranpose(1,2).contguous().view(batch_size, n_query, -1)))
        
        # residual connection
        z += residual_q
        
        z = self.layer_norm(z)
        
        return z, attn    
    
    
class PositionWiseFeedForward(nn.Module):
    '''A two layer feed forward NN '''
    
    def __init__(self, input_dim=512, hidden_dim=2048, dropout=0.1):
        super().__init__()
        
        # position wise
        self.w_1 = nn.Linear(input_dim, hidden_dim)
        self.w_2 = nn.Linear(hidden_dim, input_dim)

        self.layer_norm = nn.LayerNorm(input_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x += self.dropout(self.w_2(F.relu(self.w_1(x))))
        
        x = self.layer_norm(x)
        
        return x
