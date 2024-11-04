import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

import numpy as np

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SequentialContext(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden=2):
        super(SequentialContext, self).__init__()
    
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden = num_hidden

        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_hidden, dropout=0)

    def init_hidden(self, batch_size, device):
        rnn_init_hidden = (
            torch.zeros(self.num_hidden, batch_size, self.hidden_size).to(device),
            torch.zeros(self.num_hidden, batch_size, self.hidden_size).to(device),
        )
        return rnn_init_hidden

    def forward(self, inputs):
        
        max_length, batch_size = inputs.size(0), inputs.size(1)

        prev_states = self.init_hidden(batch_size, inputs.device)
        output, _ = self.rnn(inputs, prev_states)

        return output


def get_uniform_keys(n_keys, dim, seed):
    """
    Generate random uniform keys (same initialization as nn.Linear).
    """
    rng = np.random.RandomState(seed)
    bound = 1 / math.sqrt(dim)
    keys = rng.uniform(-bound, bound, (n_keys, dim))
    return keys.astype(np.float32)


class HashingSymbol(nn.Module):
    def __init__(self, input_dim, mem_size, slot_size, binding_num_slots, reasoning_num_slots, n_keys, top_k=8):

        super().__init__()

        self.input_dim = input_dim
        self.mem_size = mem_size
        self.slot_size = slot_size
        self.binding_num_slots = binding_num_slots
        self.reasoning_num_slots = reasoning_num_slots
        self.num_slots = binding_num_slots + reasoning_num_slots
        self.n_keys = n_keys
        self.top_k = top_k

        self.k_dim = int(self.slot_size // 2)
        self.v_dim = self.slot_size

        self.dropout = nn.Dropout(p=0.1)

        # initialize keys / values
        self.initialize_keys()

        self.binding_values = nn.ModuleList([nn.EmbeddingBag(self.n_keys, self.v_dim, mode='sum', sparse=False) for _ in range(self.binding_num_slots)])
        for i in range(self.binding_num_slots):
            nn.init.normal_(self.binding_values[i].weight, mean=0, std=self.v_dim ** -0.5)

        # query network
        self.binding_query_proj = nn.Linear(self.input_dim, self.binding_num_slots * self.k_dim, bias=True)
        self.reasoning_query_proj = nn.Linear(self.input_dim, self.reasoning_num_slots * self.k_dim, bias=True)
        self.norm_query = nn.LayerNorm(self.k_dim)

        self.mem_proj_layer = nn.Linear(self.v_dim, self.mem_size)

        # residual network
        self.residual_linear = nn.Linear(self.k_dim, self.v_dim)

    def initialize_keys(self):
        """
            keys: (n_keys, k_dim)
        """

        keys = nn.Parameter(torch.from_numpy(np.array([
            get_uniform_keys(self.n_keys, self.k_dim, seed=i)
            for i in range(self.binding_num_slots)
        ])).view(self.binding_num_slots, self.n_keys, self.k_dim))

        self.binding_keys = nn.Parameter(keys[:self.binding_num_slots])

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
            outputs = [self._get_indices(query[:, i], self.binding_keys[i]) for i in range(self.binding_num_slots)]
        else:
            outputs = [self._get_indices(query[:, 0], self.binding_keys[0])]
            outputs += [self._get_indices(query[:, i], self.binding_keys[1]) for i in range(1, self.reasoning_num_slots)]
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

        # compute query
        if flag == 'binding':
            query = self.binding_query_proj(input.contiguous().view(-1, self.input_dim))    # (bs,slots*k_dim)
            query = query.view(bs, self.binding_num_slots, self.k_dim)                     # (bs,slots,k_dim)
            query = self.norm_query(query)
            query = self.dropout(query)                                             # (bs,heads,k_dim)
        else:
            query = self.reasoning_query_proj(input.contiguous().view(-1, self.input_dim))    # (bs,slots*k_dim)
            query = query.view(bs, self.reasoning_num_slots, self.k_dim)                     # (bs,slots,k_dim)
            query = self.norm_query(query)
            query = self.dropout(query)                                             # (bs,heads,k_dim)

        # retrieve indices and scores
        scores, indices = self.get_indices(query, flag)                               # (bs*slots,top_k)
        scores = F.softmax(scores.float(), dim=-1).type_as(scores)              # (bs*slots,top_k)

        # weighted sum of values
        if flag == 'binding':
            output = [self.binding_values[i](indices[i], per_sample_weights=scores[i]) for i in range(self.binding_num_slots)]
        else:
            output = [self.binding_values[0](indices[0], per_sample_weights=scores[0])]
            output += [self.binding_values[1](indices[i], per_sample_weights=scores[i]) for i in range(1, self.reasoning_num_slots)]
        output = torch.stack(output, dim=1) # (bs,v_dim)

        # reshape output
        if len(prefix_shape) >= 2:
            output = output + self.residual_linear(query)
            output = self.mem_proj_layer(output)
            output = output.view(prefix_shape + (-1, self.mem_size,))  # (...,v_dim)

        return output


class Decomposition(nn.Module):
    def __init__(self,  input_size, mem_size, slot_size,
                        binding_num_slots,
                        reasoning_num_slots,
                        n_keys=128,
                        top_k=8,
                        epsilon=1e-8,
        ):
        super(Decomposition, self).__init__()
    
        self.input_size = input_size
        self.mem_size = mem_size
        self.binding_num_slots = binding_num_slots
        self.reasoning_num_slots = reasoning_num_slots
        self.slot_size = slot_size
        self.n_keys = n_keys
        self.top_k = top_k
        self.epsilon = epsilon

        self.hashingsymbol = HashingSymbol(
            self.input_size,
            self.mem_size,
            self.slot_size,
            self.binding_num_slots,
            self.reasoning_num_slots,
            n_keys=self.n_keys,
            top_k=self.top_k,
        )

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, inputs):
        """
            inputs:   [batch_size, input_size]
        """
        seq_len, batch_size = inputs.shape[0], inputs.shape[1]

        return inputs, inputs.view(seq_len, batch_size, -1)
    
    def binding_slot_attention(self, inputs, inputs_cat):

        seq_len, batch_size = inputs.shape[0], inputs.shape[1]

        initial_slots = self.hashingsymbol(inputs_cat, 'binding')
        initial_slots = initial_slots.view(seq_len, batch_size, -1, self.mem_size)

        return initial_slots

    def reasoning_slot_attention(self, inputs, inputs_cat):

        seq_len, batch_size = inputs.shape[0], inputs.shape[1]

        initial_slots = self.hashingsymbol(inputs_cat, 'reasoning')
        initial_slots = initial_slots.view(seq_len, batch_size, -1, self.mem_size)

        return initial_slots


class AssociativeBinding(nn.Module):
    def __init__(self, input_size, hidden_size, mem_size):
        super(AssociativeBinding, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mem_size = mem_size

        self.linear_write_gate = nn.Linear(hidden_size, 1)
        nn.init.xavier_normal_(self.linear_write_gate.weight)

        epsilon = 1e-6
        self.attention_mask = torch.tensor([1.-2*epsilon, epsilon, epsilon])
    
    def prepare(self, slots):

        attention_mask = self.attention_mask.to(slots.device)

        role1 = torch.einsum("sbne,n->sbe", slots, torch.roll(attention_mask, 0))
        role2 = torch.einsum("sbne,n->sbe", slots, torch.roll(attention_mask, 1))
        filer = torch.einsum("sbne,n->sbe", slots, torch.roll(attention_mask, 2))

        role1 = torch.tanh(role1)
        role2 = torch.tanh(role2)
        filer = torch.tanh(filer)

        return role1, role2, filer

    def forward(self, memory_state, hidden_state, role1, role2, filer):
        """
            memory_state:   [batch_size, mem_size, mem_size, mem_size]
            role1:           [batch_size, input_size]            (decomposition)
            role2:           [batch_size, input_size]            (decomposition)
            filer:          [batch_size, input_size]            (decomposition)
            hidden_state:   [batch_size, hidden_size]           (sequential context) -> gate information
        """
        write_gate = self.linear_write_gate(hidden_state)
        write_gate = torch.sigmoid(write_gate + 1)

        role = torch.einsum("br,bt->brt", role1, role2)
        prev_info = torch.einsum("brt,brtf->bf", role, memory_state)
        cur_info = write_gate * (filer - prev_info)

        scale = 1. / self.mem_size
        new_memory_state = memory_state + torch.einsum("brt,bf->brtf", role, cur_info * scale)

        memory_norm = new_memory_state.view(new_memory_state.shape[0], -1).norm(dim=-1)
        memory_norm = torch.relu(memory_norm - 1) + 1
        new_memory_state = new_memory_state / memory_norm.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        return new_memory_state

class AssociativeReasoning(nn.Module):
    def __init__(self, input_size, mem_size, n_read):
        super(AssociativeReasoning, self).__init__()
    
        self.input_size = input_size
        self.mem_size = mem_size
        self.n_read = n_read

        self.ln = nn.LayerNorm(self.mem_size, elementwise_affine=False)

        epsilon = 1e-6
        self.attention_mask = torch.tensor([1.-(1+n_read)*epsilon] + [epsilon] * n_read)
    
    def prepare(self, slots):

        attention_mask = self.attention_mask.to(slots.device)

        unbinding1 = torch.einsum("sbne,n->sbe", slots, torch.roll(attention_mask, 0))
        unbinding1 = torch.tanh(unbinding1)
        
        unbinding2 = []
        for i in range(self.n_read):
            u = torch.einsum("sbne,n->sbe", slots, torch.roll(attention_mask, i+1))
            u = torch.tanh(u)
            unbinding2.append(u)
        unbinding2 = torch.stack(unbinding2, dim=1)

        return unbinding1, unbinding2

    def forward(self, memory_state, unbinding1, unbinding2):
        """
            memory_state:   [batch_size, mem_size, mem_size, mem_size]
            unbinding1:      [batch_size, input_size]
            unbinding2:      [batch_size, n_read, input_size]
        """

        unbinding = unbinding1

        for i in range(self.n_read):
            unbinding = torch.einsum("bsrv,bs,br->bv", memory_state, unbinding, unbinding2[i])
            unbinding = self.ln(unbinding)

        return unbinding


class Network(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, vocab_size,
        num_hidden=2,
        head_size=32,
        mem_size=32,
        n_read=1,
        n_keys=128,
        top_k=8,
        batch_first=True,
    ):
        super(Network, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.mem_size = mem_size
        self.batch_first = batch_first
        self.vocab_size = vocab_size
        self.n_read = n_read
        self.slot_size = head_size

        self.sequential_context = SequentialContext(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_hidden=num_hidden,
        )
        self.decomposition = Decomposition(
            input_size=self.hidden_size,
            mem_size=self.mem_size,
            binding_num_slots=3,
            reasoning_num_slots=1 + n_read,
            slot_size=self.slot_size,
            n_keys=n_keys,
            top_k=top_k,
        )
        self.binding = AssociativeBinding(
            input_size=self.decomposition.slot_size,
            hidden_size=self.hidden_size,
            mem_size=self.mem_size,
        )
        self.reasoning = AssociativeReasoning(
            input_size=self.decomposition.slot_size,
            mem_size=self.mem_size,
            n_read=n_read,
        )
        self.output_proj_linear = nn.Linear(self.mem_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(p=0.5)

        self.embedding = Embedding(vocab_size=vocab_size, embedding_size=input_size)
        self.reconstruction_linear = nn.Linear(input_size, vocab_size)

        self.recon_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)

    def init_memory_state(self, batch_size, device):
        
        memory_init_hidden = torch.zeros(batch_size, self.mem_size, self.mem_size, self.mem_size).to(device)
        
        return memory_init_hidden

    def forward(self, inputs, reconstruction=False):
        """
            inputs:   [batch_size, seq_len, input_size]
        """
        """
            self.sequential_context:        [inputs, prev_states, sequence_length]
            self.decomposition:             [inputs]
            self.binding:                   [memory_state, role, filer, hidden_state]
            self.reasoning:                 [memory_state, unbinding]
        """

        if self.batch_first:
            inputs = inputs.transpose(0, 1)

        seq_len, batch_size = inputs.size(0), inputs.size(1)
        inputs_embed = self.embedding(inputs)

        # Sequential Context
        sequential_context = self.sequential_context(inputs_embed)

        # Decomposition
        sequential_context, sequential_context_cat = self.decomposition(sequential_context)
        bindings = self.decomposition.binding_slot_attention(sequential_context, sequential_context_cat)
        reasonings = self.decomposition.reasoning_slot_attention(sequential_context, sequential_context_cat)

        memory_state = self.init_memory_state(batch_size, inputs.device)

        role1, role2, filer = self.binding.prepare(bindings)
        unbinding1, unbinding2 = self.reasoning.prepare(reasonings)

        outputs = []
        for t, context in enumerate(sequential_context_cat):

            # Association Binding
            memory_state = self.binding(memory_state, context, role1[t], role2[t], filer[t])

            # Association Reasoning
            output_t = self.reasoning(memory_state, unbinding1[t], unbinding2[t])

            outputs.append(output_t)

        outputs = torch.stack(outputs, dim=0)
        outputs = self.dropout(sequential_context_cat) + self.output_proj_linear(outputs)
        final_outputs = self.output_layer(outputs)

        if self.batch_first:
            final_outputs = final_outputs.transpose(0, 1)

        if reconstruction:
            loss = 0.

            if self.batch_first:
                inputs = inputs.transpose(0, 1)
                inputs_embed = inputs_embed.transpose(0, 1)
            
            reconstruction = self.reconstruction_linear(inputs_embed)
            reconstruction_loss = self.recon_fn(reconstruction.transpose(1, 2), inputs)
            loss += reconstruction_loss

            return final_outputs, loss

        return final_outputs


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(Embedding, self).__init__()
        self.word_embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        nn.init.xavier_uniform_(self.word_embed.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, input_sequence):
        return self.word_embed(input_sequence)
