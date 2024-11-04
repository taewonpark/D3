# Fast weight layers using custom kernels.
# Many code duplications to be refactored!
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.fast_fast_weight import fast_weight_delta
from utils.fast_transformers import fast_weight_sum


@torch.jit.script
def elu_p1(x):
    return F.elu(x, 1., False) + 1.


@torch.jit.script
def sum_norm(x):
    return x / x.sum(-1, keepdim=True)


@torch.jit.script
def sum_norm_eps(x):
    return x / (x.sum(-1, keepdim=True) + 1e-5)


@torch.jit.script
def elu_p1_sum_norm(x):
    y = F.elu(x, 1., False) + 1.
    return y / y.sum(-1, keepdim=True)


@torch.jit.script
def elu_p1_sum_norm_eps(x):
    y = F.elu(x, 1., False) + 1.
    return y / (y.sum(-1, keepdim=True) + 1e-5)


def get_uniform_keys(n_keys, dim, seed):
    """
    Generate random uniform keys (same initialization as nn.Linear).
    """
    rng = np.random.RandomState(seed)
    bound = 1 / math.sqrt(dim)
    keys = rng.uniform(-bound, bound, (n_keys, dim))
    return keys.astype(np.float32)


def get_normal_values(n_keys, dim, seed):
    """
    Generate random uniform keys (same initialization as nn.Linear).
    """
    rng = np.random.RandomState(seed)
    values = rng.normal(0, 1 / math.sqrt(dim), size=(n_keys, dim))
    return values.astype(np.float32)


class HashingSymbol(nn.Module):
    """
        Large Memory Layers with Product Keys (NeurIPS'19)
        https://arxiv.org/pdf/1907.05242.pdf
        https://github.com/facebookresearch/XLM/blob/main/PKM-layer.ipynb
    """
    def __init__(self, input_dim, mem_size, slot_size, role_num_slots, filler_num_slots, n_keys, top_k=8, model_type='r'):

        super().__init__()

        assert model_type == 'r' or model_type == 'rf'

        self.input_dim = input_dim
        self.mem_size = mem_size
        self.slot_size = slot_size
        self.role_num_slots = role_num_slots
        self.filler_num_slots = filler_num_slots
        self.num_slots = self.role_num_slots + self.filler_num_slots if model_type == 'rf' else self.role_num_slots
        self.n_keys = n_keys
        self.top_k = top_k
        self.model_type = model_type

        self.k_dim = int(self.slot_size // 2)
        self.v_dim = self.slot_size

        self.dropout = nn.Dropout(p=0.1)

        # initialize keys / values
        self.initialize_kv()

        # query network
        self.query_proj = nn.Linear(self.input_dim, (self.num_slots + self.role_num_slots) * self.k_dim, bias=True)
        self.norm_query = nn.LayerNorm(self.k_dim)

        self.mem_proj_layer = nn.Linear(self.v_dim, self.mem_size)

        # residual network
        self.residual_linear = nn.Linear(self.k_dim, self.v_dim)

    def initialize_kv(self):
        """
            keys: (n_keys, k_dim)
        """

        self.keys = nn.Parameter(torch.from_numpy(np.array([
            get_uniform_keys(self.n_keys, self.k_dim, seed=i)
            for i in range(self.num_slots)
        ])).view(self.num_slots, self.n_keys, self.k_dim))

        self.values = nn.Parameter(torch.from_numpy(np.array([
            get_normal_values(self.n_keys, self.v_dim, seed=i)
            for i in range(self.num_slots)
        ])).view(self.num_slots, self.n_keys, self.v_dim))


    def compute_query_key(self, query_encoding, query_decoding, key_encoding, key_decoding):
        
        # (bs, slots, k_dim)
        query = torch.cat((query_encoding, query_decoding), dim=1)
        # (slots, n_keys, k_dim)
        key = torch.cat((key_encoding, key_decoding), dim=0)
        key = key / torch.norm(key, dim=-1, keepdim=True)

        # (bs, slots, n_keys)
        scores = torch.einsum("bnd,nkd->bnk", query, key)
        top_k_indices = torch.topk(scores, self.top_k, dim=2).indices

        mask_ = torch.ones_like(scores).view(-1, self.n_keys)
        row_index = np.arange(mask_.size(0))
        row_index = np.repeat(row_index, self.top_k)
        mask_[row_index, top_k_indices.view(-1)] = 0.
        mask_ = mask_.view(scores.shape)

        scores = scores + (-1e6) * mask_
        scores = F.softmax(scores, dim=-1)

        return scores

    def compute_score_value(self, scores, value_encoding, value_decoding):

        # (slots, n_keys, v_dim)
        value = torch.cat((value_encoding, value_decoding), dim=0)

        # (bs, slots, v_dim)
        averaged_value = torch.einsum("bnk,nkd->bnd", scores, value)

        return averaged_value

    def forward(self, input):
        """
            input:  (batch, seq_len, input_dim)
            output: (batch, seq_len, num_slots, slot_size)
        """
        # input dimensions
        assert input.shape[-1] == self.input_dim
        prefix_shape = input.shape[:-1]
        bs = np.prod(prefix_shape)

        # (bs,slots*k_dim)
        query = self.query_proj(input.contiguous().view(-1, self.input_dim))    
        query = query.view(bs, self.num_slots + self.role_num_slots, self.k_dim)
        query = self.norm_query(query)
        query = self.dropout(query)

        query_encoding, query_decoding = torch.split(query, [self.num_slots, self.role_num_slots], dim=1)

        scores = self.compute_query_key(query_encoding, query_decoding, self.keys, self.keys[:self.role_num_slots])
        output = self.compute_score_value(scores, self.values, self.values[:self.role_num_slots])

        output = output + self.residual_linear(query)
        output = self.mem_proj_layer(output)
        output = output.view(prefix_shape + (-1, self.mem_size,))  # (...,v_dim) 

        if self.model_type == 'rf':
            return torch.split(output, [self.role_num_slots, self.filler_num_slots, self.role_num_slots], dim=2)
        else:
            return torch.split(output, [self.role_num_slots, self.role_num_slots], dim=2)
        

class LinearD3Layer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, layer_id=None, num_layer=None,
                 skip_attn_normalization=False, use_sum_norm=False,
                 slot_size=32, n_keys=128, module_type='rf',
                 ):
        super(LinearD3Layer, self).__init__()
        print(f"Using LinearD3Layer {layer_id} -")

        assert layer_id is not None
        assert num_layer is not None
        self.layer_id = layer_id
        self.num_layer = num_layer

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        assert module_type == "rf" or module_type == "r"

        # self.qkv_net = nn.Linear(d_model, n_head * 3 * d_head, bias=False)
        if module_type == "rf":
            self.v_net = None
        else:
            self.v_net = nn.Linear(d_model, n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.normalize_attn_scores = (not skip_attn_normalization)
        self.use_sum_norm = use_sum_norm
        self.eps = eps

        # Decomposition parameters.
        self.slot_size = slot_size
        self.n_keys = n_keys
        self.module_type = module_type

        self.hashingsymbol = HashingSymbol(
            input_dim=self.d_model,
            mem_size=self.d_head,
            slot_size=self.slot_size,
            role_num_slots=self.n_head,
            filler_num_slots=self.n_head,
            n_keys=self.n_keys,
            model_type=self.module_type,
        )

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        # multihead attention
        # shape h: (len, B, n_head * d_head)

        if self.pre_lnorm:
            # layer normalization
            h = self.layer_norm(h)

        slen, bsz, _ = h.size()

        if self.module_type == "rf":
            head_k, head_v, head_q = self.hashingsymbol(h)
        else:
            head_v = self.v_net(h)
            head_v = head_v.view(slen, bsz, self.n_head, -1)
            head_k, head_q = self.hashingsymbol(h)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)

        # TODO add dropout here?
        # transform q and k
        head_q = F.elu(head_q, 1., False) + 1.
        head_k = F.elu(head_k, 1., False) + 1.

        # normalize k and q, crucial for stable training.
        if self.use_sum_norm:
            head_k = head_k / head_k.sum(-1, keepdim=True)
            head_q = head_q / head_q.sum(-1, keepdim=True)

        if self.normalize_attn_scores:
            denominator_acc = torch.cumsum(head_k, dim=2)

        if mems is None:
            mem_fast_weights = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_k.device)
        else:
            assert carry_over_fast_weight
            mem_fast_weights, fast_denom = mems
            # bsz can be smaller for the last batch
            mem_fast_weights = mem_fast_weights[:bsz]
            if self.normalize_attn_scores:
                denominator_acc = denominator_acc + fast_denom[:bsz]

        if self.normalize_attn_scores:
            denominator = torch.einsum(
                'lbij,lbij->lbi', denominator_acc, head_q).unsqueeze(-1)

        layer_out = fast_weight_sum(
            head_q, head_k, head_v, mem_fast_weights)

        # shape (B, n_head, len, d_head)
        if self.normalize_attn_scores:
            layer_out = self.scale * layer_out / (denominator + self.eps)
        else:
            layer_out = self.scale * layer_out

        layer_out = layer_out.transpose(1, 2)

        layer_out = layer_out.reshape(
            bsz, slen, self.n_head * self.d_head)

        layer_out = layer_out.transpose(0, 1)

        # expect [qlen, B, n_head * d_head]

        # linear projection
        attn_out = self.o_net(layer_out)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        if carry_over_fast_weight:
            # last values of accumulator should be carried over.
            # clone is needed as backward modifies the data of fast weight
            if self.normalize_attn_scores:
                new_k_acc = denominator_acc[:, :, -1, :].unsqueeze(2).detach()
            else:
                new_k_acc = None
            new_mem = (mem_fast_weights.clone().detach(), new_k_acc)
            return output, new_mem

        return output