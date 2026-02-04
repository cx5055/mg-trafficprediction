import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torch_geometric.graphgym.optim import none_scheduler
class FC(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(FC, self).__init__()
        self.hyperGNN_dim = 16
        self.middle_dim = 2
        self.mlp=nn.Sequential(
                OrderedDict([('fc1', nn.Linear(dim_in, self.hyperGNN_dim)),
                             ('sigmoid1', nn.Sigmoid()),
                             ('fc2', nn.Linear(self.hyperGNN_dim, self.middle_dim)),
                             ('sigmoid2', nn.Sigmoid()),
                             ('fc3', nn.Linear(self.middle_dim, dim_out))]))
    def forward(self, x):
        ho = self.mlp(x)
        return ho
class FC_pop(nn.Module):
    def __init__(self, dim_in, dim_out, hyper_dim=16, middle_dim=2):
        super(FC_pop, self).__init__()
        self.hyperGNN_dim = hyper_dim
        self.middle_dim  = middle_dim

        self.mlp = nn.Sequential(OrderedDict([
            ('fc1',      nn.Linear(dim_in,      self.hyperGNN_dim)),
            ('sigmoid1', nn.Sigmoid()),
            ('fc2',      nn.Linear(self.hyperGNN_dim, self.middle_dim)),
            ('sigmoid2', nn.Sigmoid()),
            ('fc3',      nn.Linear(self.middle_dim,     dim_out)),
        ]))

    def forward(self, x):
        return self.mlp(x)
class GCN_cell(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(GCN_cell, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k*2+1, dim_in, dim_out))
        self.weights = nn.Parameter(torch.FloatTensor(cheb_k*2+1,dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        self.gcn = gcn(cheb_k)
    def forward(self, x, adj, node_embedding):
        x_g = self.gcn(x, adj)
        weights = torch.einsum('nd,dkio->nkio', node_embedding, self.weights_pool)
        bias = torch.matmul(node_embedding, self.bias_pool)
        x_g = x_g.permute(0, 2, 1, 3)
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias
        return x_gconv

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()
    def forward(self, x, A):
        x = torch.einsum("bnm,bmc->bnc", A,x)
        return x.contiguous()

class gcn(nn.Module):
    def __init__(self,k=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        self.k = k
    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.k + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2
        h = torch.stack(out, dim=1)
        return h
class GRU_Cell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim, use_dis=True, use_pop=True):
        super(GRU_Cell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.use_dis = use_dis
        self.use_pop = use_pop
        self.gate = GCN_cell(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, embed_dim, time_dim)
        self.update = GCN_cell(dim_in + self.hidden_dim, dim_out, cheb_k, embed_dim, time_dim)
        self.dual_embed = DualHeadEmbedding(
            input_dim=dim_in,
            hidden_dim=dim_out,
            dist_dim=node_num,
            pop_dim=6,
            embed_dim=embed_dim,
            use_dis=use_dis,
            use_pop=use_pop
        )
    def forward(self, x, state, dist_t=None, pop_t=None, static_emb=None):
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        emb1, emb2 = self.dual_embed(x, state, dist_t, pop_t)
        affinity = torch.matmul(emb1, emb2.transpose(2, 1)) - torch.matmul(emb2, emb1.transpose(2, 1))
        adj1 = F.softmax(F.relu(affinity),dim=-1)
        adj2 = F.softmax(F.relu(-affinity.transpose(-2, -1)),dim=-1)
        adj = [adj1, adj2]
        u_r = torch.sigmoid(self.gate(input_and_state, adj, static_emb))
        u, r = torch.split(u_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, u * state), dim=-1)
        hc = torch.tanh(self.update(candidate, adj, static_emb))
        h = r * state + (1 - r) * hc
        return h
    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
    def preprocessing(adj):
        num_nodes= adj.shape[-1]
        adj = adj +  torch.eye(num_nodes).to(adj.device)
        x= torch.unsqueeze(adj.sum(-1), -1)
        adj = adj / x
        return adj

class multi_pre_Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim, num_layers=1, use_dis=True, use_pop=True):
        super(multi_pre_Encoder, self).__init__()
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.use_dis = use_dis
        self.use_pop = use_pop

        self.multi_pre_cells = nn.ModuleList()
        self.multi_pre_cells.append(GRU_Cell(node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim, use_dis, use_pop))
        for _ in range(1, num_layers):
            self.multi_pre_cells.append(GRU_Cell(node_num, dim_out, dim_out, cheb_k, embed_dim, time_dim, use_dis, use_pop))

    def forward(self, x, dist, pop, init_state, static_emb):
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                x_t = current_inputs[:, t, :, :]
                dist_t = dist[:, t, :, :] if self.use_dis else None
                pop_t = pop.transpose(1, 2) if self.use_pop else None
                state = self.multi_pre_cells[i](x_t, state, dist_t, pop_t, static_emb)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.multi_pre_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)

class DualHeadEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, pop_dim, dist_dim, embed_dim, use_dis=True, use_pop=True):
        super(DualHeadEmbedding, self).__init__()
        self.use_dis = use_dis
        self.use_pop = use_pop
        self.tra_proj1 = FC(input_dim + hidden_dim, embed_dim)
        self.tra_proj2 = FC(input_dim + hidden_dim, embed_dim)
        if self.use_dis:
            self.dis_proj1 = FC(dist_dim + hidden_dim, embed_dim)
            self.dis_proj2 = FC(dist_dim + hidden_dim, embed_dim)
        if self.use_pop:
            self.pop_proj1 = FC_pop(pop_dim + hidden_dim, embed_dim, 64, 16)
            self.pop_proj2 = FC_pop(pop_dim + hidden_dim, embed_dim, 64, 16)

    def forward(self, x_t, h_prev, dist_t=None, pop_t=None):
        tra_input = torch.cat([x_t, h_prev], dim=-1)
        z_tra1 = self.tra_proj1(tra_input)
        z_tra2 = self.tra_proj2(tra_input)

        if self.use_dis and dist_t is not None:
            dis_input = torch.cat([dist_t, h_prev], dim=-1)
            z_dis1 = self.dis_proj1(dis_input)
            z_dis2 = self.dis_proj2(dis_input)
        else:
            z_dis1 = z_dis2 = 1.0

        if self.use_pop and pop_t is not None:
            pop_input = torch.cat([pop_t, h_prev], dim=-1)
            z_pop1 = self.pop_proj1(pop_input)
            z_pop2 = self.pop_proj2(pop_input)
        else:
            z_pop1 = z_pop2 = 1.0

        e1 = torch.tanh(z_tra1 * z_dis1 * z_pop1)
        e2 = torch.tanh(z_tra2 * z_dis2 * z_pop2)

        return e1, e2

class trapre_model(nn.Module):
    def __init__(self, args):
        super(trapre_model, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.embed_dim = args.embed_dim
        self.time_dim = args.time_dim
        self.cheb_k = args.cheb_k
        self.num_layers = args.num_layers
        self.use_dis = args.use_dis
        self.use_pop = args.use_pop
        #静态节点嵌�?
        self.node_embeddings = nn.Parameter(torch.empty(self.num_node, self.embed_dim))
        nn.init.xavier_uniform_(self.node_embeddings)
        self.encoder = multi_pre_Encoder(
            self.num_node, self.input_dim, self.hidden_dim, self.cheb_k,
            self.embed_dim, self.time_dim, self.num_layers,
            use_dis=self.use_dis, use_pop=self.use_pop
        )

        self.task_projs = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.output_dim) for _ in range(4)
        ])

    def forward(self, traffics, dist, pop):
        B = traffics[0].shape[0]
        outputs = []
        for traffic in traffics:
            init_state = self.encoder.init_hidden(B).to(traffic.device)
            state, _ = self.encoder(traffic, dist, pop, init_state,self.node_embeddings)
            last_state = state[:, -1, :, :]
            out = self.task_projs[len(outputs)](last_state)
            out = F.relu(out)
            outputs.append(out)
        return outputs




