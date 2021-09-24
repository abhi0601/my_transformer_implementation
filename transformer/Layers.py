# imports
import torch.nn as nn

from SubLayers import MultiHeadAttention, PositionWiseFeedForward


class EncoderLayer(nn.Module):
    '''The encoder layer composed of self attention and pointwise FF NN'''
    
    def __init__(self, d_model, hidden_dim, n_heads, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout)
        self.pw_ffnn = PositionWiseFeedForward(d_model, hidden_dim, dropout)
        
    def forward(self, enc_input, self_attn_mask=None):
        enc_output, enc_self_attn = self.self_attn(enc_input, enc_input, enc_input,
                                                   self_attn_mask)
        enc_output = self.pw_ffnn(enc_output)
        
        return enc_output, enc_self_attn
        

class DecoderLayer(nn.Module):
    '''The decoder layer having masked self attn, enc-dec-attn, position wise ffnn'''
    
    def __init__(self, d_model, hidden_dim, n_heads, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout)
        self.enc_dec_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout)
        self.pw_ffnn = PositionWiseFeedForward(d_model, hidden_dim, dropout)
        
    def forward(self, dec_input, enc_output, self_attn_mask=None, enc_dec_attn_mask=None):
        dec_ouput, dec_self_attn = self_attn(dec_input, dec_input, dec_input,
                                             mask=self_attn_mask)
        dec_output, enc_dec_attn = self.enc_dec_attn(dec_output, enc_output, enc_output,
                                                     mask=enc_dec_attn_mask)
        dec_output = self.pw_ffnn(dec_output)
        
        return dec_output, dec_self_attn, enc_dec_attn
