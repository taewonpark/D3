
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
from torch.nn import Dropout, Conv2d
from torch.nn.modules.utils import _pair

class GroupLinearLayer(nn.Module):
    """Modularized Linear Layer"""
    def __init__(self, num_blocks, din, dout, bias=True):
        super(GroupLinearLayer, self).__init__()

        self.bias=bias
        self.w = nn.Parameter(torch.Tensor(num_blocks, din, dout))
        self.b = nn.Parameter(torch.Tensor(1, num_blocks, dout))

        stdv = math.sqrt(6.0) / math.sqrt(din + dout)
        nn.init.uniform_(self.w, -stdv, stdv)
        nn.init.zeros_(self.b)

    def forward(self,x):
        # x - (bsz, num_blocks, din)
        x = x.permute(1,0,2)
        x = torch.bmm(x, self.w)
        x = x.permute(1,0,2)

        if self.bias:
            x = x + self.b

        return x

def get_positional(seq_len, dim):
    pe = torch.zeros(seq_len, dim)
    normalizer = 1. / (1. + math.exp(-1))
    for pos in range(seq_len):
        for i in range(0, dim, 2):
            pe[pos, i] = normalizer * math.sin(pos / (10000 ** ((2 * i)/dim)))
            pe[pos, i+1] = normalizer * math.cos(pos / (10000 ** ((2 * (i+1))/dim)))

    pe = pe.unsqueeze(0)
    return pe

class BasicModel(nn.Module):
    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        self.name=name

    def train_(self, input_img, input_qst, label):
        self.optimizer.zero_grad()
        output = self(input_img, input_qst)
        loss = F.nll_loss(output, label)
        loss.backward()
        self.optimizer.step()
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy, loss
        
    def test_(self, input_img, input_qst, label):
        output = self(input_img, input_qst)
        loss = F.nll_loss(output, label)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy, loss

    def save_model(self, epoch, name):
        torch.save(self.state_dict(), f'{name}/epoch_{epoch:02d}.pth')


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


class Codebook_Linear_Attention_D3_wF(nn.Module):
    def __init__(self, dim, nheads=4, slot_size=32, n_keys=128, top_k=8):
        super(Codebook_Linear_Attention_D3_wF, self).__init__()

        self.dim = dim
        self.nheads = nheads
        self.head_dim = dim // nheads
        self.slot_size = slot_size
        self.n_keys = n_keys
        self.top_k = top_k

        self.norm_before = True

        self.final = nn.Linear(dim, dim)

        self.res = nn.Sequential(
            nn.Linear(dim,2 * dim),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(2 * dim, dim),
            nn.Dropout(p=0.1)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Decomposition parameters.
        self.hashingsymbol = HashingSymbol(
            input_dim=self.dim,
            mem_size=self.head_dim,
            slot_size=self.slot_size,
            role_num_slots=self.nheads,
            filler_num_slots=self.nheads,
            n_keys=self.n_keys,
            top_k=self.top_k,
            model_type='rf',
        )

    def forward(self, x):
        bsz, n_read, _ = x.shape
        _, n_write, _ = x.shape

        res = x
        if self.norm_before:
            x = self.norm1(x)

        k, v, q = self.hashingsymbol(x)
        k = k.permute(0,2,3,1)
        v = v.permute(0,2,1,3)
        q = q.permute(0,2,1,3) / np.sqrt(self.head_dim)

        q = F.elu(q, 1., False) + 1.
        k = F.elu(k, 1., False) + 1.
        score = torch.matmul(q, k)

        eps = 1e-5
        denominator = torch.sum(score, dim=-1, keepdim=True) + eps
        score = score / denominator

        out = torch.matmul(score, v) # (bsz, nheads, n_read, att_dim)
        out = out.view(bsz, self.nheads, n_read, self.head_dim)

        out = out.permute(0, 2, 1, 3).reshape(bsz, n_read, self.dim)
        out = self.final(out)

        if not self.norm_before:
            out = self.norm1(res + out)
        else:
            out = res + out

        res = out

        if self.norm_before:
            out = self.norm2(out)
            out = res + self.res(out)
        else:
            out = self.norm2(res + self.res(out))

        return out


class Codebook_Linear_Attention_D3_woF(nn.Module):
    def __init__(self, dim, nheads=4, slot_size=32, n_keys=128, top_k=8):
        super(Codebook_Linear_Attention_D3_woF, self).__init__()

        self.dim = dim
        self.nheads = nheads
        self.head_dim = dim // nheads
        self.slot_size = slot_size
        self.n_keys = n_keys
        self.top_k = top_k

        self.norm_before = True

        self.value_net = nn.Linear(dim, dim)

        self.final = nn.Linear(dim, dim)

        self.res = nn.Sequential(
            nn.Linear(dim,2 * dim),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(2 * dim, dim),
            nn.Dropout(p=0.1)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Decomposition parameters.
        self.hashingsymbol = HashingSymbol(
            input_dim=self.dim,
            mem_size=self.head_dim,
            slot_size=self.slot_size,
            role_num_slots=self.nheads,
            filler_num_slots=self.nheads,
            n_keys=self.n_keys,
            top_k=self.top_k,
            model_type='r',
        )

    def forward(self, x):
        bsz, n_read, _ = x.shape
        _, n_write, _ = x.shape

        res = x
        if self.norm_before:
            x = self.norm1(x)

        k, q = self.hashingsymbol(x)
        k = k.permute(0,2,3,1)
        q = q.permute(0,2,1,3) / np.sqrt(self.head_dim)

        q = F.elu(q, 1., False) + 1.
        k = F.elu(k, 1., False) + 1.
        score = torch.matmul(q, k)

        v = self.value_net(x).reshape(bsz, n_write, self.nheads, self.head_dim)
        v = v.permute(0,2,1,3)

        eps = 1e-5
        denominator = torch.sum(score, dim=-1, keepdim=True) + eps
        score = score / denominator

        out = torch.matmul(score, v) # (bsz, nheads, n_read, att_dim)
        out = out.view(bsz, self.nheads, n_read, self.head_dim)

        out = out.permute(0, 2, 1, 3).reshape(bsz, n_read, self.dim)
        out = self.final(out)

        if not self.norm_before:
            out = self.norm1(res + out)
        else:
            out = res + out

        res = out

        if self.norm_before:
            out = self.norm2(out)
            out = res + self.res(out)
        else:
            out = self.norm2(res + self.res(out))

        return out


class Encoder(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, img_size, question_size, in_channels=3, hidden_size=256):
        super(Encoder, self).__init__()
        img_size = _pair(img_size)

        patch_size = (15, 15)
        self.n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)

        self.question_representation = nn.Linear(question_size, hidden_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.pe = get_positional(25, hidden_size).cuda()

    def forward(self, x, que):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        que = self.question_representation(que).unsqueeze(1).expand(-1, self.n_patches+1, -1)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = x + self.pe
        embeddings = torch.cat((cls_tokens, x), dim=1)
        embeddings = torch.cat((embeddings, que), dim=-1)
        return embeddings


class Model(BasicModel):
    def __init__(self, args):
        super(Model, self).__init__(args, 'Model')

        self.transformer_dim = args.transformer_dim
        self.code_size = args.code_size
        self.n_heads = args.n_heads
        self.iterations = args.iterations
        self.n_keys = args.n_keys

        self.encoder = Encoder((75, 75), 18, hidden_size = self.transformer_dim // 2)
        self.mapping = nn.Linear((self.transformer_dim // 2) * 2, args.transformer_dim)

        if args.model == 'D3_wF':
            self.transformer = Codebook_Linear_Attention_D3_wF(self.transformer_dim, self.n_heads, self.code_size, self.n_keys)
        elif args.model == 'D3_woF':
            self.transformer = Codebook_Linear_Attention_D3_woF(self.transformer_dim, self.n_heads, self.code_size, self.n_keys)


        self.final = nn.Linear(self.transformer_dim, 10)

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

    def forward(self, img, qst):
        x = self.encoder(img, qst)
        x = self.mapping(x)

        for _ in range(self.iterations):
            x = self.transformer(x)

        y = self.final(x[:,0,:])
        return F.log_softmax(y, dim=1)

if __name__=="__main__":
    image = torch.randn(1, 3, 75, 75)
    question = torch.randn(1, 18)
    embedding = Encoder(image.shape[-2:], question.shape[-1])
    out = embedding(image, question)
    print(out.shape)
