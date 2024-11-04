from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import MLP, LayerNorm, OptionalLayer

import math
import numpy as np

from itertools import combinations

AVAILABLE_ELEMENTS = ('e1', 'e2', 'r1', 'r2', 'r3')


class TprRnn(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(TprRnn, self).__init__()

        self.input_module = InputModule(config)
        self.update_module = UpdateModule(config=config)
        self.inference_module = InferenceModule(config=config)

        self.decomposition_module = Decomposition(
            input_size=config["symbol_size"],
            slot_size=config["slot_size"],
            binding_num_slots=5,
            reasoning_num_slots=4,
            n_keys=config['n_keys'],
            top_k=config['top_k'],
        )

        self.vocab_size = config['vocab_size']
        self.reconstruction_linear = nn.Linear(config["symbol_size"], self.vocab_size)
        self.recon_fn = nn.CrossEntropyLoss(reduction='none')
        self.recon_dropout = nn.Dropout(p=0.1)

    def forward(self, story: torch.Tensor, query: torch.Tensor, reconstruction=False, orthogonality=False, hsic=False):
        # story_embed: [b, s, w, e]
        # query_embed: [b, w, e]
        story_embed, query_embed, story_mask, query_mask, sentence_sum, query_sum = self.input_module(story, query)

        binding_slots = self.decomposition_module.binding_slot_attention(story_embed, sentence_sum, story_mask)
        TPR = self.update_module(binding_slots)

        query_embed = query_embed.unsqueeze(dim=1)
        query_mask = query_mask.unsqueeze(dim=1)
        query_sum = query_sum.unsqueeze(dim=1)
        reasoning_slots = self.decomposition_module.reasoning_slot_attention(query_embed, query_sum, query_mask)
        reasoning_slots = reasoning_slots.squeeze(dim=1)
        logits = self.inference_module(reasoning_slots, TPR)

        if reconstruction or orthogonality:
            auxiliary = {}

            if reconstruction:
                embed = torch.cat((story_embed, query_embed), dim=1)  # [b, s+1, w, e]
                recon_logits = self.reconstruction_linear(embed)
                recon_target = torch.cat((story, query.unsqueeze(dim=1)), dim=1)  # [b, s+1, w]

                recon_loss = self.recon_fn(recon_logits.permute(0, 3, 1, 2), recon_target)
                recon_loss = recon_loss #* recon_mask

                auxiliary['recon_loss'] = recon_loss.mean()

            return logits, auxiliary

        return logits


class InputModule(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(InputModule, self).__init__()
        self.word_embed = nn.Embedding(num_embeddings=config["vocab_size"],
                                       embedding_dim=config["symbol_size"])
        nn.init.uniform_(self.word_embed.weight, -config["init_limit"], config["init_limit"])
        # positional embeddings
        self.pos_embed = nn.Parameter(torch.ones(config["max_seq"], config["symbol_size"]))
        nn.init.ones_(self.pos_embed.data)
        self.pos_embed.data /= config["max_seq"]

    def forward(self, story: torch.Tensor, query: torch.Tensor):
        # Sentence embedding
        sentence_embed = self.word_embed(story)  # [b, s, w, e]
        sentence_embed = torch.einsum('bswe,we->bswe', sentence_embed, self.pos_embed[:sentence_embed.shape[2]])
        sentence_mask = (story != 0)
        sentence_sum = torch.einsum("bswe,bsw->bse", sentence_embed, sentence_mask.type(torch.float))

        # Query embedding
        query_embed = self.word_embed(query)  # [b, w, e]
        query_embed = torch.einsum('bwe,we->bwe', query_embed, self.pos_embed[:query_embed.shape[1]])
        query_mask = (query != 0)
        query_sum = torch.einsum("bwe,bw->be", query_embed, query_mask.type(torch.float))
        return sentence_embed, query_embed, sentence_mask, query_mask, sentence_sum, query_sum

def get_uniform_keys(n_keys, dim, seed):
    """
    Generate random uniform keys (same initialization as nn.Linear).
    """
    rng = np.random.RandomState(seed)
    bound = 1 / math.sqrt(dim)
    keys = rng.uniform(-bound, bound, (n_keys, dim))
    return keys.astype(np.float32)


class HashingSymbol(nn.Module):
    def __init__(self, input_dim, slot_size, binding_num_slots, reasoning_num_slots, n_keys, top_k=8):

        super().__init__()

        self.input_dim = input_dim
        self.slot_size = slot_size
        self.binding_num_slots = binding_num_slots
        self.reasoning_num_slots = reasoning_num_slots
        self.num_slots = binding_num_slots + reasoning_num_slots
        self.n_keys = n_keys
        self.top_k =top_k

        self.k_dim = int(self.slot_size // 2)
        self.v_dim = self.slot_size

        self.dropout = nn.Dropout(p=0.1)

        # initialize keys / values
        self.initialize_keys()

        self.binding_values = nn.ModuleList([nn.EmbeddingBag(self.n_keys, self.v_dim, mode='sum', sparse=False) for _ in range(self.binding_num_slots)])
        for i in range(self.binding_num_slots):
            nn.init.normal_(self.binding_values[i].weight, mean=0, std=self.v_dim ** -0.5)
        
        self.reasoning_values = nn.ModuleList([nn.EmbeddingBag(self.n_keys, self.v_dim, mode='sum', sparse=False) for _ in range(self.reasoning_num_slots)])
        for i in range(self.reasoning_num_slots):
            nn.init.normal_(self.reasoning_values[i].weight, mean=0, std=self.v_dim ** -0.5)

        # query network
        self.query_proj = nn.Linear(self.input_dim, self.k_dim, bias=True)

        # residual network
        self.residual_linear = nn.Linear(self.k_dim, self.v_dim)

    def initialize_keys(self):
        """
            keys: (n_keys, k_dim)
        """

        keys = nn.Parameter(torch.from_numpy(np.array([
            get_uniform_keys(self.n_keys, self.k_dim, seed=i)
            for i in range(self.num_slots)
        ])).view(self.num_slots, self.n_keys, self.k_dim))

        self.binding_keys = nn.Parameter(keys[:self.binding_num_slots])
        self.reasoning_keys = nn.Parameter(keys[self.binding_num_slots:])

    def _get_indices(self, query, subkeys):
        """
        Generate scores and indices for a specific slot.
        """
        assert query.dim() == 2 and query.size(1) == self.k_dim
        bs = query.size(0)
        top_k = self.top_k
        n_keys = self.n_keys

        subkeys = subkeys / torch.norm(subkeys, dim=-1, keepdim=True)

        # compute indices with associated scores
        scores = F.linear(query, subkeys, bias=None)                    # (bs,n_keys)
        scores, indices = scores.topk(top_k, dim=1)                     # (bs,top_k)

        assert scores.shape == indices.shape == (bs, top_k)
        return scores, indices

    def get_indices(self, query, flag):
        """
        Generate scores and indices.
        """
        assert query.dim() == 3 and query.size(2) == self.k_dim
        bs = len(query)

        if flag == 'binding':
            outputs = [self._get_indices(query[:, 0], self.binding_keys[i]) for i in range(self.binding_num_slots)]
        else:
            outputs = [self._get_indices(query[:, 0], self.reasoning_keys[i]) for i in range(self.reasoning_num_slots)]
        s = torch.cat([s.view(1, bs, self.top_k) for s, _ in outputs], 0)  # (slots,bs,top_k)
        i = torch.cat([i.view(1, bs, self.top_k) for _, i in outputs], 0)  # (slots,bs,top_k)
        return s, i

    def forward(self, input, flag):
        """
            input:  (batch, seq_len, input_dim)
            output: (batch, seq_len, num_slots, slot_size)
        """
        # input dimensions
        assert input.shape[-1] == self.input_dim
        assert flag == "binding" or flag == 'reasoning'
        prefix_shape = input.shape[:-1]
        bs = np.prod(prefix_shape)

        query = self.query_proj(input.contiguous().view(-1, self.input_dim))    # (bs,slots*k_dim)
        query = query.view(bs, 1, self.k_dim)                     # (bs,slots,k_dim)
        query = self.dropout(query)                                             # (bs,heads,k_dim)

        # retrieve indices and scores
        scores, indices = self.get_indices(query, flag)                               # (bs*slots,top_k)
        scores = F.softmax(scores.float(), dim=-1).type_as(scores)              # (bs*slots,top_k)

        # weighted sum of values
        if flag == 'binding':
            output = [self.binding_values[i](indices[i], per_sample_weights=scores[i]) for i in range(self.binding_num_slots)]
        else:
            output = [self.reasoning_values[i](indices[i], per_sample_weights=scores[i]) for i in range(self.reasoning_num_slots)]
        output = torch.stack(output, dim=1) # (bs,v_dim)

        # reshape output
        if len(prefix_shape) >= 2:
            output = output + self.residual_linear(query)
            output = output.view(prefix_shape + (-1, self.v_dim,))  # (...,v_dim)

        return output


class Decomposition(nn.Module):
    def __init__(self,  input_size, slot_size,
                        binding_num_slots,
                        reasoning_num_slots,
                        n_keys=128,
                        top_k=8,
                        epsilon=1e-6,
        ):
        super(Decomposition, self).__init__()
    
        self.input_size = input_size
        self.binding_num_slots = binding_num_slots
        self.reasoning_num_slots = reasoning_num_slots
        self.slot_size = slot_size
        self.n_keys = n_keys
        self.top_k = top_k
        self.epsilon = epsilon

        self.hashingsymbol = HashingSymbol(
            self.input_size,
            self.slot_size,
            self.binding_num_slots,
            self.reasoning_num_slots,
            n_keys=self.n_keys,
            top_k=self.top_k,
        )

    def forward(self, inputs):
        """
            inputs:         [seq_len, batch_size, word_len, input_size]
            inputs_mask:    [seq_len, batch_size, word_len]
        """
        
        return inputs
    
    def binding_slot_attention(self, sentence, sentence_sum, mask):

        batch_size, seq_len = sentence.shape[0], sentence.shape[1]

        initial_slots = self.hashingsymbol(sentence_sum, 'binding')
        initial_slots = initial_slots.view(batch_size, seq_len, -1, self.slot_size)

        return initial_slots
    
    def reasoning_slot_attention(self, sentence, sentence_sum, mask):

        batch_size, seq_len = sentence.shape[0], sentence.shape[1]
        
        initial_slots = self.hashingsymbol(sentence_sum, 'reasoning')
        initial_slots = initial_slots.view(batch_size, seq_len, -1, self.slot_size)

        return initial_slots


class UpdateModule(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(UpdateModule, self).__init__()
        self.role_size = config["role_size"]
        self.ent_size = config["entity_size"]
        self.hidden_size = config["hidden_size"]
        self.symbol_size = config["symbol_size"]

        self.slot_size = config["slot_size"]
        epsilon = 1e-6

        self.e1_linear = nn.Linear(self.slot_size, self.ent_size)
        self.e2_linear = nn.Linear(self.slot_size, self.ent_size)

        self.r1_linear = nn.Linear(self.slot_size, self.role_size)
        self.r2_linear = nn.Linear(self.slot_size, self.role_size)
        self.r3_linear = nn.Linear(self.slot_size, self.role_size)

        self.attention_mask = torch.tensor([1.-4*epsilon, epsilon, epsilon, epsilon, epsilon])

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        # sentence: [b, s, w, e]
        batch_size = slots.size(0)

        attention_mask = self.attention_mask.to(slots.device)
        
        e1 = torch.tanh(self.e1_linear(torch.einsum("bsne,n->bse", slots, torch.roll(attention_mask, 0))))
        e2 = torch.tanh(self.e2_linear(torch.einsum("bsne,n->bse", slots, torch.roll(attention_mask, 1))))

        r1 = torch.tanh(self.r1_linear(torch.einsum("bsne,n->bse", slots, torch.roll(attention_mask, 2))))
        r2 = torch.tanh(self.r2_linear(torch.einsum("bsne,n->bse", slots, torch.roll(attention_mask, 3))))
        r3 = torch.tanh(self.r3_linear(torch.einsum("bsne,n->bse", slots, torch.roll(attention_mask, 4))))

        partial_add_W = torch.einsum('bsr,bsf->bsrf', r1, e2)
        partial_add_B = torch.einsum('bsr,bsf->bsrf', r3, e1)

        inputs = (e1, r1, partial_add_W, e2, r2, partial_add_B, r3)

        # TPR-RNN steps
        TPR = torch.zeros(batch_size, self.ent_size, self.role_size, self.ent_size).to(slots.device)
        for x in zip(*[torch.unbind(t, dim=1) for t in inputs]):
            e1_i, r1_i, partial_add_W_i, e2_i, r2_i, partial_add_B_i, r3_i = x
            w_hat = torch.einsum('be,br,berf->bf', e1_i, r1_i, TPR)
            partial_remove_W = torch.einsum('br,bf->brf', r1_i, w_hat)

            m_hat = torch.einsum('be,br,berf->bf', e1_i, r2_i, TPR)
            partial_remove_M = torch.einsum('br,bf->brf', r2_i, m_hat)
            partial_add_M = torch.einsum('br,bf->brf', r2_i, w_hat)

            b_hat = torch.einsum('be,br,berf->bf', e2_i, r3_i, TPR)
            partial_remove_B = torch.einsum('br,bf->brf', r3_i, b_hat)

            # operations
            write_op = partial_add_W_i - partial_remove_W
            move_op = partial_add_M - partial_remove_M
            backlink_op = partial_add_B_i - partial_remove_B
            delta_F = torch.einsum('be,brf->berf', e1_i, write_op + move_op) + \
                      torch.einsum('be,brf->berf', e2_i, backlink_op)
            delta_F = torch.clamp(delta_F, -1., 1.)
            TPR = TPR + delta_F
        return TPR


class InferenceModule(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(InferenceModule, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.ent_size = config["entity_size"]
        self.role_size = config["role_size"]
        self.symbol_size = config["symbol_size"]
        # output embeddings
        self.Z = nn.Linear(config["entity_size"], config["vocab_size"])

        self.slot_size = config["slot_size"]
        epsilon = 1e-6

        self.attention_mask = torch.tensor([1.-3*epsilon, epsilon, epsilon, epsilon])

        self.e1_linear = nn.Linear(self.slot_size, self.ent_size)

        self.r1_linear = nn.Linear(self.slot_size, self.role_size)
        self.r2_linear = nn.Linear(self.slot_size, self.role_size)
        self.r3_linear = nn.Linear(self.slot_size, self.role_size)

        self.l1, self.l2, self.l3 = [OptionalLayer(LayerNorm(hidden_size=self.ent_size), active=config["LN"])
                                     for _ in range(3)]

    def forward(self, slots: torch.Tensor, TPR: torch.Tensor):
        
        attention_mask = self.attention_mask.to(slots.device)

        e1 = self.e1_linear(torch.einsum("bne,n->be", slots, torch.roll(attention_mask, 0)))

        r1 = self.r1_linear(torch.einsum("bne,n->be", slots, torch.roll(attention_mask, 1)))
        r2 = self.r2_linear(torch.einsum("bne,n->be", slots, torch.roll(attention_mask, 2)))
        r3 = self.r3_linear(torch.einsum("bne,n->be", slots, torch.roll(attention_mask, 3)))

        i1 = self.l1(torch.einsum('be,br,berf->bf', e1, r1, TPR))
        i2 = self.l2(torch.einsum('be,br,berf->bf', i1, r2, TPR))
        i3 = self.l3(torch.einsum('be,br,berf->bf', i2, r3, TPR))

        step_sum = i1 + i2 + i3
        logits = self.Z(step_sum)
        return logits
