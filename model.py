from typing import Sequence, Union
import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * self.d_model ** -0.5
    
class PositionEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len).unsqueeze(1)
        div = math.exp(-(torch.arange(0, d_model, 2) / d_model) * math.log(10000.0))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        pe = self.pe.unsqueeze(0)

        self.register_buffer('pe', self.pe)

    def forward(self, x):
        return self.pe[:, x.shape[1], :].requires_grad_(False)
    
class LayerNormalization(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones([1]))
        self.bias = nn.Parameter(torch.ones([1]))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std - self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_model, d_ff)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(nn.Modeul):
    def __init__(self, d_model: int, h: int, dropout: Union[None, float]):
        self.h = h
        self.d_k = d_model // h
        self.w_k = nn.Linear(d_model, d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, k, q, v, mask: Union[None, torch.Tensor]):
        def get_attention_scores():
            logits = query @ key.transpose(-1, -2) / self.dk ** 0.5
            if mask is not None:
                logits.fill_masked_(mask == 0, -1e9)
            attention_scores = logits.softmax(dim = -1)
            if self.dropout is not None:
                attention_scores = self.dropout(attention_scores)
            return attention_scores @ value, attention_scores
        key = self.w_k(k)
        query = self.w_q(q)
        value = self.w_v(v)

        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k)

        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)

        x, self.attention_scores = get_attention_scores()
        return self.w_o(x.transpose(1, 2).continous().view(x.shape[0], x.shape[1]))

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super.__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttentionBlock, ffwd: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.ffwd = ffwd
        self.residual_1 = ResidualConnection(dropout)
        self.residual_2 = ResidualConnection(dropout)

    def forward(self, x, mask):
        x = self.residual_1(x, lambda x: self.self_attention(x, x, x, mask))
        x = self.residual_2(x, self.ffwd)


class Encoder(nn.Module):
    def __init__(self, blocks: Sequence[EncoderBlock]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttentionBlock, cross_attention: MultiHeadAttentionBlock, ffwd: FeedForwardBlock, dropout: float):
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.ffwd = ffwd
        self.residual_1 = ResidualConnection(dropout)
        self.residual_2 = ResidualConnection(dropout)
        self.residual_3 = ResidualConnection(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_1(x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residual_2(x, lambda x: self.cross_attention(encoder_output, encoder_output, x, src_mask))
        x = self.residual_3(x, self.ffwd)
        return x
    
class Decoder(nn.Module):
    def __init__(self, blocks: Sequence[DecoderBlock]):
        self.blocks = nn.ModuleList(blocks)
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for block in self.blocks:
            x = block(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.linear(x)
        return torch.log_softmax(x, dim=-1)
    
class Transformer(nn.Module):
    def __init__(self, src_embd: InputEmbedding, tgt_embd: InputEmbedding, src_pos: PositionEmbedding, tgt_pos: PositionEmbedding, encoder: Encoder, decoder: Decoder, proj: ProjectionLayer, dropout: float = 0.1):
        super().__init__()
        self.src_embd = src_embd
        self.tgt_embd = tgt_embd
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.encoder = encoder
        self.decoder = decoder
        self.proj = proj
        self.dropout = nn.Dropout(dropout)

    def encoder(self, x, src_mask):
        x = self.dropout(self.src_embd(x) + self.pos(x))
        return self.encoder(x, src_mask)
    
    def decoder(self, encoder_output, x, src_mask, tgt_mask):
        x = self.dropout(self.tgt_embd(x) + self.pos(x))
        return self.decoder(x, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.proj(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N = 6, h: int = 8, d_ff: int = 512 * 4, dropout: float = 0.1):
    src_embd = InputEmbedding(src_vocab_size, d_model)
    tgt_embd = InputEmbedding(tgt_vocab_size, d_model)
    src_pos = PositionEmbedding(src_seq_len, d_model)
    tgt_pos = PositionEmbedding(tgt_seq_len, d_model)
    
    encoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        ffwd = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_blocks.append(EncoderBlock(self_attention_block, ffwd, dropout))
    encoder = Encoder(encoder_blocks)

    decoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        ffwd = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_blocks.append(DecoderBlock(self_attention_block, cross_attention_block, ffwd, dropout))
    decoder = Decoder(decoder_blocks)

    proj = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(src_embd, tgt_embd, src_pos, tgt_pos, encoder, decoder, proj, dropout)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer