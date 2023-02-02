import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np

class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(cheb_k*dim_in, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)
        
    def forward(self, x, node_embeddings):
        if len(node_embeddings.shape)==2:
            x_g = []
            node_num = node_embeddings.shape[0]
            supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
            support_set = [torch.eye(node_num).to(supports.device), supports]
            for k in range(2, self.cheb_k):
                support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2]) 
            for support in support_set:
                x_g.append(torch.einsum("nm,bmc->bnc", support, x))
            x_g = torch.cat(x_g, dim=-1) # B, N, cheb_k * dim_in
            x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias  # b, N, dim_out
        else:
            x_g = []
            node_num = node_embeddings.shape[1]
            supports = F.softmax(F.relu(torch.einsum('bnc,bmc->bnm', node_embeddings, node_embeddings)), dim=-1)
            support_set = [torch.eye(node_num).repeat(node_embeddings.shape[0], 1, 1).to(supports.device), supports]
            for k in range(2, self.cheb_k):
                support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
            for support in support_set:
                x_g.append(torch.einsum("bnm,bmc->bnc", support, x).contiguous())
            x_g = torch.cat(x_g, dim=-1) # B, N, cheb_k * dim_in
            x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias  # b, N, dim_out
        return x_gconv
    
class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k)
        self.update = AGCN(dim_in+self.hidden_dim, dim_out, cheb_k)

    def forward(self, x, state, node_embeddings):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
    
class ADCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(ADCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D), shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #return current_inputs, torch.stack(output_hidden, dim=0)
        return current_inputs, output_hidden
    
    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return init_states

class ADCRNN_STEP(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(ADCRNN_STEP, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, xt, init_state, node_embeddings):
        # xt: (B, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.dcrnn_cells[i](current_inputs, init_state[i], node_embeddings)
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden


class MegaCRN(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, horizon, rnn_units, num_layers=1, embed_dim=8, cheb_k=3,
                 ycov_dim=1, mem_num=10, mem_dim=32, memory_type=True, meta_type=True, cl_decay_steps=2000, use_curriculum_learning=True):
        super(MegaCRN, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.rnn_units = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.cheb_k = cheb_k
        self.memory_type = memory_type
        self.meta_type = meta_type
        self.ycov_dim = ycov_dim
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning
        
        # memory
        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.memory = self.construct_memory()
        if self.memory_type:
            self.decoder_dim = self.rnn_units + self.mem_dim
        else:
            self.decoder_dim = self.rnn_units
        # encoder
        self.encoder = ADCRNN(self.num_nodes, self.input_dim, self.rnn_units, self.cheb_k, self.num_layers)
        # deocoder
        self.decoder = ADCRNN_STEP(self.num_nodes, self.output_dim + self.ycov_dim, self.decoder_dim, self.cheb_k, self.num_layers)
        # output
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim, bias=True))
    
    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)     # (M, d)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.rnn_units, self.mem_dim), requires_grad=True)    # project to query
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict
    
    def query_memory(self, h_t:torch.Tensor):
        B = h_t.shape[0] # h_t = h_t.squeeze(1) # B, N, hidden
        query = torch.matmul(h_t, self.memory['Wq'])     # (B, N, d)
        att_score = torch.softmax(torch.matmul(query, self.memory['Memory'].t()), dim=-1)         # alpha: (B, N, M)
        proto = torch.matmul(att_score, self.memory['Memory'])     # (B, N, d)
        _, ind = torch.topk(att_score, k=2, dim=-1)
        pos = self.memory['Memory'][ind[:, :, 0]] # B, N, d
        neg = self.memory['Memory'][ind[:, :, 1]] # B, N, d
        emb = self.memory['Memory'] # N, d
        return emb, proto, query, pos, neg
            
    def forward(self, x, y_cov, labels=None, batches_seen=None):       
        init_state = self.encoder.init_hidden(x.shape[0])
        h_en, state_en = self.encoder(x, init_state, self.node_embeddings) # B, T, N, hidden      
        h_t = h_en[:, -1, :, :] # B, N, hidden (last state)        
        
        if self.memory_type == True:
            _node_embeddings, h_att, query, pos, neg = self.query_memory(h_t)
            h_t = torch.cat([h_t, h_att], dim=-1)
        else:
            _node_embeddings = None
        ht_list = [h_t]*self.num_layers
        
        go = torch.zeros((x.shape[0], self.num_nodes, self.output_dim), device=x.device)
        out = []
        for t in range(self.horizon):
            if self.meta_type == True:
                assert _node_embeddings is not None, 'meta graph (node embedding) must derive from memory...'
                h_de, ht_list = self.decoder(torch.cat([go, y_cov[:, t, ...]], dim=-1), ht_list, _node_embeddings)
            else:
                h_de, ht_list = self.decoder(torch.cat([go, y_cov[:, t, ...]], dim=-1), ht_list, self.node_embeddings)
            go = self.proj(h_de)
            out.append(go)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen):
                    go = labels[:, t, ...]
        output = torch.stack(out, dim=1)
        
        if self.memory_type == True:
            return output, h_att, query, pos, neg
        else:
            return output

def print_params(model):
    # print trainable params
    param_count = 0
    print('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            param_count += param.numel()
    print(f'In total: {param_count} trainable parameters.')
    return

def main():
    import sys
    import argparse
    from torchsummary import summary
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['METRLA', 'PEMSBAY'], default='METRLA', help='which dataset to run')
    parser.add_argument('--num_nodes', type=int, default=207, help='num_nodes')
    parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
    parser.add_argument('--horizon', type=int, default=12, help='output sequence length')
    parser.add_argument('--input_dim', type=int, default=1, help='number of input channel')
    parser.add_argument('--output_dim', type=int, default=1, help='number of output channel')
    parser.add_argument('--num_rnn_layers', type=int, default=1, help='number of rnn layers')
    parser.add_argument('--rnn_units', type=int, default=64, help='number of rnn units')
    parser.add_argument('--mem_num', type=int, default=207, help='number of meta-nodes/prototypes')
    parser.add_argument('--mem_dim', type=int, default=64, help='dimension of meta-nodes/prototypes')
    parser.add_argument("--memory", type=eval, choices=[True, False], default='True', help="whether to use memory: True or False")
    parser.add_argument("--meta", type=eval, choices=[True, False], default='True', help="whether to use meta-graph: True or False")
    parser.add_argument("--use_curriculum_learning", type=eval, choices=[True, False], default='True', help="use_curriculum_learning")
    parser.add_argument("--cl_decay_steps", type=int, default=2000, help="cl_decay_steps")
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
    args = parser.parse_args()
    device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
    model = MegaCRN(num_nodes=args.num_nodes, input_dim=args.input_dim, output_dim=args.output_dim, horizon=args.horizon, 
                    rnn_units=args.rnn_units, num_layers=args.num_rnn_layers, mem_num=args.mem_num, mem_dim=args.mem_dim, 
                    memory_type=args.memory, meta_type=args.meta, cl_decay_steps=args.cl_decay_steps, use_curriculum_learning=args.use_curriculum_learning).to(device)
    summary(model, [(args.seq_len, args.num_nodes, args.input_dim), (args.horizon, args.num_nodes, args.output_dim)], device=device)
    print_params(model)
    
if __name__ == '__main__':
    main()
