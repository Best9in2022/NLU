import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq import utils


class TransformerEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiHeadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.attention_dropout
        self.activation_dropout = args.activation_dropout

        self.fc1 = generate_linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = generate_linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, state, encoder_padding_mask):
        """Forward pass of a single Transformer Encoder Layer"""
        residual = state.clone()

        '''
        ___QUESTION-5-DESCRIBE-D-START___
        1.  What is the purpose of encoder_padding_mask? 
        We use a special pad symbol to make the input sequences have the same length, We do not need to compute 
        the attention score between other words and the pad symbol, so we need to use encoder_padding_mask. 
        '''
        state, _ = self.self_attn(query=state, key=state, value=state, key_padding_mask=encoder_padding_mask)
        '''
        ___QUESTION-5-DESCRIBE-D-END___
        '''

        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.self_attn_layer_norm(state)

        residual = state.clone()
        state = F.relu(self.fc1(state))
        state = F.dropout(state, p=self.activation_dropout, training=self.training)
        state = self.fc2(state)
        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.final_layer_norm(state)

        return state


class TransformerDecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout = args.attention_dropout
        self.activation_dropout = args.activation_dropout
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.self_attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_attn_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True
        )

        self.encoder_attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_attn_heads=args.decoder_attention_heads,
            kdim=args.encoder_embed_dim,
            vdim=args.encoder_embed_dim,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )

        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = generate_linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = generate_linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

    def forward(self,
                state,
                encoder_out=None,
                encoder_padding_mask=None,
                incremental_state=None,
                prev_self_attn_state=None,
                self_attn_mask=None,
                self_attn_padding_mask=None,
                need_attn=False,
                need_head_weights=False):
        """Forward pass of a single Transformer Decoder Layer"""

        # need_attn must be True if need_head_weights
        need_attn = True if need_head_weights else need_attn

        residual = state.clone()
        state, _ = self.self_attn(query=state,
                                  key=state,
                                  value=state,
                                  key_padding_mask=self_attn_padding_mask,
                                  need_weights=False,
                                  attn_mask=self_attn_mask)
        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.self_attn_layer_norm(state)

        residual = state.clone()
        '''
        ___QUESTION-5-DESCRIBE-E-START___
        1.  How does encoder attention differ from self attention? 
        The encoder attention used in the TransformerDecoderLayer is cross-attention where the query comes from 
        the former layer of the decoder and the key and value come from the encoder. In self attention, all of 
        the query, key and value are the same. 
        2.  What is the difference between key_padding_mask and attn_mask? 
        We use key_padding_mask because we use a special padding token to make the input sequences have the same 
        length, and we do not need to compute the attention score between the query and padding token. The attn_mask is 
        used in the decoder self attention layer, because we want to use the decoder to predict the next token, 
        so we avoid the attention mechenism to copy from the tokens after the current 
        token, it can only use the information of the words before the current and the current word. 
        3.  If you understand this difference, then why don't we need to give attn_mask here?
        Because here we want to compute the cross-attention, to let the decoder use the information from the encoder
        so that we can compute the conditional probability of the target sentence given the source sentence, so we 
        should not use attn_mask so that the decoder can better use the information from the encoder. 
        '''
        state, attn = self.encoder_attn(query=state,
                                        key=encoder_out,
                                        value=encoder_out,
                                        key_padding_mask=encoder_padding_mask,
                                        need_weights=need_attn or (not self.training and self.need_attn))
        '''
        ___QUESTION-5-DESCRIBE-E-END___
        '''

        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.encoder_attn_layer_norm(state)

        residual = state.clone()
        state = F.relu(self.fc1(state))
        state = F.dropout(state, p=self.activation_dropout, training=self.training)
        state = self.fc2(state)
        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.final_layer_norm(state)

        return state, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""
    def __init__(self,
                 embed_dim,
                 num_attn_heads,
                 kdim=None,
                 vdim=None,
                 dropout=0.,
                 self_attention=False,
                 encoder_decoder_attention=False):
        '''
        ___QUESTION-6-MULTIHEAD-ATTENTION-NOTE
        You shouldn't need to change the __init__ of this class for your attention implementation
        '''
        super().__init__()
        self.embed_dim = embed_dim
        self.k_embed_size = kdim if kdim else embed_dim
        self.v_embed_size = vdim if vdim else embed_dim

        self.num_heads = num_attn_heads
        self.attention_dropout = dropout
        self.head_embed_size = embed_dim // num_attn_heads  # this is d_k in the paper
        self.head_scaling = math.sqrt(self.head_embed_size)

        self.self_attention = self_attention
        self.enc_dec_attention = encoder_decoder_attention

        kv_same_dim = self.k_embed_size == embed_dim and self.v_embed_size == embed_dim
        assert self.head_embed_size * self.num_heads == self.embed_dim, "Embed dim must be divisible by num_heads!"
        assert not self.self_attention or kv_same_dim, "Self-attn requires query, key and value of equal size!"
        assert self.enc_dec_attention ^ self.self_attention, "One of self- or encoder- attention must be specified!"

        self.k_proj = nn.Linear(self.k_embed_size, embed_dim, bias=True)
        self.v_proj = nn.Linear(self.v_embed_size, embed_dim, bias=True)
        self.q_proj = nn.Linear(self.k_embed_size, embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        # Xavier initialisation
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self,
                query,
                key,
                value,
                key_padding_mask=None,
                attn_mask=None,
                need_weights=True):

        # Get size features
        tgt_time_steps, batch_size, embed_dim = query.size()
        assert self.embed_dim == embed_dim

        '''
        ___QUESTION-6-MULTIHEAD-ATTENTION-START
        Implement Multi-Head attention  according to Section 3.2.2 of https://arxiv.org/pdf/1706.03762.pdf.
        Note that you will have to handle edge cases for best model performance. Consider what behaviour should
        be expected if attn_mask or key_padding_mask are given?
        '''

        # attn is the output of MultiHead(Q,K,V) in Vaswani et al. 2017
        # attn must be size [tgt_time_steps, batch_size, embed_dim]
        # attn_weights is the combined output of h parallel heads of Attention(Q,K,V) in Vaswani et al. 2017
        # attn_weights must be size [num_heads, batch_size, tgt_time_steps, key.size(0)]
        # TODO: REPLACE THESE LINES WITH YOUR IMPLEMENTATION ------------------------ CUT
        # attn = torch.zeros(size=(tgt_time_steps, batch_size, embed_dim))
        # attn_weights = torch.zeros(size=(self.num_heads, batch_size, tgt_time_steps, -1)) if need_weights else None
        query = query.transpose(0, 1)  # query: [batch_size, q_len(tgt_time_steps), embed_dim]
        key = key.transpose(0, 1)  # key: [batch_size, k_len, embed_dim(k_embed_size)]
        value = value.transpose(0, 1)  # value:[batch_size, v_len, embed_dim(v_embed_size)]

        q_n = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_embed_size).transpose(1, 2)
        # q_n: [batch_size, num_heads, q_len(tgt_time_steps), head_embed_size]
        k_n = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_embed_size).transpose(1, 2)
        # k_n: [batch_size, num_heads, k_len, head_embed_size]
        v_n = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_embed_size).transpose(1, 2)
        # v_n: [batch_size, num_heads, v_len, head_embed_size]
        # score = torch.matmul(q_n, k_n.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_embed_size))
        score = torch.matmul(q_n, k_n.transpose(-1, -2)) / self.head_scaling
        # score: [batch_size, num_heads, q_len(tgt_time_steps), k_len(src_time_steps)]

        if key_padding_mask is not None:
            # key_padding_mask: [batch_size, src_time_steps]
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2) # key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).repeat(1, self.num_heads, tgt_time_steps, 1) ?
            # key_padding_mask: [batch_size, num_heads, tgt_time_steps, src_time_steps]
            score = score.masked_fill_(key_padding_mask, float('-inf')) # replace valus with '-inf' in places where key_padding_mask == True
        if attn_mask is not None:
            # attn_mask: [tgt_time_steps, tgt_time_steps]
            # attn_mask = attn_mask.unsqueeze(0).unsqueeze(1).repeat(batch_size, self.num_heads, 1, 1)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)
            # attn_mask: [batch_size, num_heads, tgt_time_steps, src_time_steps->tgt_time_steps]
            score = score + attn_mask

        normalized_score = F.softmax(score, dim=-1)
        # normalized_score: [batch_size, num_heads, q_len, k_len]
        attn_weights = torch.matmul(normalized_score, v_n)
        # attn_weights: [batch_size, num_heads, q_len, head_embed_size]
        attn = self.out_proj(attn_weights.transpose(1, 2).reshape(batch_size, -1, self.embed_dim))
        # attn: [batch_size, q_len, embed_dim]
        attn_weights = attn_weights.transpose(0, 1)
        # attn_weights: [num_heads, batch_size, q_len, head_embed_size]
        attn = attn.transpose(0, 1)
        # attn: [q_len, batch_size, embed_dim]

        # TODO: --------------------------------------------------------------------- CUT

        '''
        ___QUESTION-6-MULTIHEAD-ATTENTION-END
        '''

        return attn, attn_weights


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.weights = PositionalEmbedding.get_embedding(init_size, embed_dim, padding_idx)
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embed_dim, padding_idx=None):
        half_dim = embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embed_dim % 2 == 1:
            # Zero pad in specific mismatch case
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0.
        return emb

    def forward(self, inputs, incremental_state=None, timestep=None):
        batch_size, seq_len = inputs.size()
        max_pos = self.padding_idx + 1 + seq_len

        if self.weights is None or max_pos > self.weights.size(0):
            # Expand embeddings if required
            self.weights = PositionalEmbedding.get_embedding(max_pos, self.embed_dim, self.padding_idx)
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            #   Positional embed is identical for all tokens during single step decoding
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weights.index_select(index=self.padding_idx + pos, dim=0).unsqueeze(1).repeat(batch_size, 1, 1)

        # Replace non-padding symbols with position numbers from padding_idx+1 onwards.
        mask = inputs.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(inputs) * mask).long() + self.padding_idx

        # Lookup positional embeddings for each position and return in shape of input tensor w/o gradient
        return self.weights.index_select(0, positions.view(-1)).view(batch_size, seq_len, -1).detach()


def LayerNorm(normal_shape, eps=1e-5):
    return torch.nn.LayerNorm(normalized_shape=normal_shape, eps=eps, elementwise_affine=True)


def fill_with_neg_inf(t):
    return t.float().fill_(float('-inf')).type_as(t)


def generate_embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def generate_linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m
