
# -*- coding: utf-8 -*-
'''
Created on : Thursday 18 Jun, 2020 : 00:47:36
Last Modified : Sunday 22 Aug, 2021 : 23:52:53

@author       : Adapted by Rishabh Joshi from Original ASAP Pooling Code
Institute     : Carnegie Mellon University
'''
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
from torch_scatter import scatter_mean, scatter_max
import pdb
import math, pdb

import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_scatter import scatter_add, scatter_max

from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, softmax
from torch_geometric.nn.pool.topk_pool import topk

from torch_sparse import coalesce
from torch_sparse import transpose
from torch_sparse import spspmm

# torch.set_num_threads(1)
def StAS(index_A, value_A, index_S, value_S, device, N, kN):
    r"""StAS: a function which returns new edge weights for the pooled graph using the formula S^{T}AS"""

    index_A, value_A = coalesce(index_A, value_A, m=N, n=N)
    index_S, value_S = coalesce(index_S, value_S, m=N, n=kN)
    index_B, value_B = spspmm(index_A, value_A, index_S, value_S, N, N, kN)

    index_St, value_St = transpose(index_S, value_S, N, kN)
    index_B, value_B = coalesce(index_B, value_B, m=N, n=kN)
    index_E, value_E = spspmm(index_St.cpu(), value_St.cpu(), index_B.cpu(), value_B.cpu(), kN, N, kN)

    return index_E.to(device), value_E.to(device)

def graph_connectivity(device, perm, edge_index, edge_weight, score, ratio, batch, N):
    r"""graph_connectivity: is a function which internally calls StAS func to maintain graph connectivity"""
    kN = perm.size(0)
    perm2 = perm.view(-1, 1)
    
    # mask contains uint8 mask of edges which originate from perm (selected) nodes
    mask = (edge_index[0]==perm2).sum(0, dtype=torch.uint8)
    
    # create the S
    S0 = edge_index[1][mask].view(1, -1)
    S1 = edge_index[0][mask].view(1, -1)
    index_S = torch.cat([S0, S1], dim=0)
    value_S = score[mask].detach().squeeze()
    
    save_index_S = index_S.clone()
    save_value_S = value_S.clone()
    
    # relabel for pooling ie: make S [N x kN]
    n_idx = torch.zeros(N, dtype=torch.long)
    n_idx[perm] = torch.arange(perm.size(0))
    index_S[1] = n_idx[index_S[1]]

    # create A
    index_A = edge_index.clone()
    if edge_weight is None:
        value_A = value_S.new_ones(edge_index[0].size(0))
    else:
        value_A = edge_weight.clone()
    
    fill_value=1
    index_E, value_E = StAS(index_A, value_A, index_S, value_S, device, N, kN)
    index_E, value_E = remove_self_loops(edge_index=index_E, edge_attr=value_E)
    index_E, value_E = add_remaining_self_loops(edge_index=index_E, edge_weight=value_E, 
        fill_value=fill_value, num_nodes=kN)
    
    
    return index_E, value_E, save_index_S, save_value_S

class ASAP_Pooling(torch.nn.Module):

    def __init__(self, in_channels, ratio, dropout_att=0, negative_slope=0.2):
        super(ASAP_Pooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.negative_slope = negative_slope
        self.dropout_att = dropout_att
        self.lin_q = Linear(in_channels, in_channels)
        self.gat_att = Linear(2*in_channels, 1)
        self.gnn_score = LEConv(self.in_channels, 1) # gnn_score: uses LEConv to find cluster fitness scores
        self.gnn_intra_cluster = GCNConv(self.in_channels, self.in_channels) # gnn_intra_cluster: uses GCN to account for intra cluster properties, e.g., edge-weights
        self.reset_parameters()
        


    def reset_parameters(self):

        self.lin_q.reset_parameters()
        self.gat_att.reset_parameters()
        self.gnn_score.reset_parameters()
        self.gnn_intra_cluster.reset_parameters()
        
             
    def forward(self, x, edge_index, edge_weight=None, batch=None):
        x2 = x.clone(); edge_index2 = edge_index.clone(); batch2 = batch.clone()
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        # NxF
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        # Add Self Loops
        fill_value = 1
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        edge_index, edge_weight = add_remaining_self_loops(edge_index=edge_index, edge_weight=edge_weight, 
            fill_value=fill_value, num_nodes=num_nodes.sum())

        N = x.size(0) # total num of nodes in batch

        # ExF
        x_pool = self.gnn_intra_cluster(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x_pool_j = x_pool[edge_index[1]]
        x_j = x[edge_index[1]]
        
        #---Master query formation---
        # NxF
        X_q, _ = scatter_max(x_pool_j, edge_index[0], dim=0)
        # NxF
        M_q = self.lin_q(X_q)    
        # ExF
        M_q = M_q[edge_index[0].tolist()]

        score = self.gat_att(torch.cat((M_q, x_pool_j), dim=-1))
        score = F.leaky_relu(score, self.negative_slope)
        score = softmax(score, edge_index[0], num_nodes=num_nodes.sum())
        att_wts = score.clone()

        # Sample attention coefficients stochastically.
        score = F.dropout(score, p=self.dropout_att, training=self.training)
        # ExF
        v_j = x_j * score.view(-1, 1)
        #---Aggregation---
        # NxF
        out = scatter_add(v_j, edge_index[0], dim=0)
        
        #---Cluster Selection
        # Nx1
        fitness = torch.sigmoid(self.gnn_score(x=out, edge_index=edge_index)).view(-1)
        perm = topk(x=fitness, ratio=self.ratio, batch=batch)
        x = out[perm] * fitness[perm].view(-1, 1)
        
        #---Maintaining Graph Connectivity
        batch = batch[perm]
        edge_index, edge_weight, S_index, S_weight = graph_connectivity(
            device = x.device,
            perm=perm,
            edge_index=edge_index,
            edge_weight=edge_weight,
            score=score,
            ratio=self.ratio,
            batch=batch,
            N=N)
        
        return x, edge_index, edge_weight, batch, perm, S_index, S_weight, att_wts

    def __repr__(self):
        return '{}({}, ratio={})'.format(self.__class__.__name__, self.in_channels, self.ratio)

def readout(x, batch):
    x_mean = scatter_mean(x, batch, dim=0)
    x_max, _ = scatter_max(x, batch, dim=0) 
    return torch.cat((x_mean, x_max), dim=-1)

class ASAP_Pool(torch.nn.Module):
    '''
    Code Modified by Rishabh Joshi
    Original code from http://github.com/malllabiisc/ASAP
    '''
    def __init__(self, config, strat_or_da):
    #def __init__(self, dataset, num_layers, hidden, ratio=0.8, **kwargs):
        super(ASAP_Pool, self).__init__()
        ratio           = config.ratio			# 0.8
        if strat_or_da == 'da':
            node_features = len(config.da_lbl2id)
        else:
            node_features = len(config.negotiation_lbl2id)
        num_features = node_features
        #node_features   = config.node_feats		# num_strat
        hidden          = config.graph_hidden		# 64
        dropout_att     = config.graph_drop		# 0.0
        num_layers      = config.graph_layers		# 3
        self.graph_model= config.graph_model		# 'gcn'
        if type(ratio)!=list:
            ratio = [ratio for i in range(num_layers)]
        if not config.node_embed:
            self.embeddings         = torch.nn.Embedding(num_features, num_features, padding_idx=-1) # Embeddings for the strategies (num_features is num_strategies)
            self.embeddings.weight  = torch.nn.Parameter(torch.FloatTensor(np.diag(np.diag(np.ones((num_features, num_features))))))  # diag matrix of 1 hot
            node_features           = num_features
            self.embeddings.weight.requires_grad = True
            # TODO NO TRAIN
        else:
            self.embeddings = torch.nn.Embedding(num_features, node_features, padding_idx=-1) # Embeddings for the strategies (num_features is num_strategies)
        if self.graph_model == 'gcn':
            self.conv1 = GCNConv(node_features, hidden)
        elif self.graph_model == 'gat':
            self.conv1 = GATConv(node_features, hidden, heads = config.num_heads)
        else:
            raise NotImplementedError
        self.pool1 = ASAP_Pooling(in_channels=hidden*config.num_heads, ratio=ratio[0], dropout_att=dropout_att)
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if self.graph_model == 'gcn':
                self.convs.append(GCNConv(hidden, hidden))
            elif self.graph_model == 'gat':
                self.convs.append(GATConv(hidden, hidden, heads = config.num_heads))
            else:
                raise NotImplementedError
            self.pools.append(ASAP_Pooling(in_channels=hidden * config.num_heads, ratio=ratio[i], dropout_att=dropout_att))
        self.lin1 = Linear(2*hidden * config.num_heads, hidden) # 2*hidden due to readout layer
        self.lin2 = Linear(hidden, num_features - 1)     # projection layer -> -1 for <start>
        self.reset_parameters()
        self.strat_or_da = strat_or_da
        self.undirected = config.undirected
        self.self_loops = config.self_loops
        self.num_heads  = config.num_heads

        try:
            if 'only_hidden' in config:
                self.only_hidden = config.only_hidden
            else:
                self.only_hidden = False
        except:
            self.only_hidden = False

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def set_embeddings(self, np_embedding):
        #self.embeddings.weight.data.copy_(torch.from_numpy(np_embedding))
        assert np_embedding.shape == self.embeddings.weight.shape
        self.embeddings.weight = torch.nn.Parameter(torch.FloatTensor(np_embedding))

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.pool1.reset_parameters()
        for conv, pool in zip(self.convs, self.pools):
            conv.reset_parameters()
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    #def forward(self, data, return_extra = False):
    #    data = data['input_graph']
    #    x, edge_index, batch = data.x, data.edge_index, data.batch                  # x is num_graph x 1
    #    x = self.embeddings(x.squeeze(1))              # added                                 # x is num_graph x node_feats / 22
    #    x = F.relu(self.conv1(x, edge_index)); #import pdb; pdb.set_trace()                     # x: num_graph x 64, 2 x 21252 -> more dense, whwereas x graphs goes down
    #    x, edge_index, edge_weight, batch, perm, S_index, S_weight, att_wts = self.pool1(x=x, edge_index=edge_index, edge_weight=None, batch=batch)
    #    save_perm = perm.clone()
    #    xs = readout(x, batch)
    #    for conv, pool in zip(self.convs, self.pools):
    #        if self.graph_model == 'gcn':
    #            x = F.relu(conv(x=x, edge_index=edge_index, edge_weight=edge_weight))
    #        elif self.graph_model == 'gat':
    #            x = F.relu(conv(x=x, edge_index=edge_index))
    #        else: 
    #            raise NotImplementedError
    #        x, edge_index, edge_weight, batch, perm, _, _, _ = pool(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch) # IGNORING S OF FUTURE LAYERS
    #        xs += readout(x, batch)
    #    x = F.relu(self.lin1(xs))
    #    if self.only_hidden:
    #        return x
    #    x = F.dropout(x, p=0.0, training=self.training)
    #    x = self.lin2(x)
    #    #out = F.log_softmax(x, dim=-1)
    #    # x is logits
    #    # dont need mask here to calculate loss
    #    logitloss = self.logitcriterion(x, data.y)
    #    return logitloss, 0.0, x, 0.0, (S_index, S_weight, att_wts, save_perm)
    def forward(self, feats, utt_mask, return_extra = True):
        # CREATE GRAPH DATA HERE
        #pdb.set_trace()
        from torch_geometric.data import Batch
        num_conv=feats.shape[0] 
        data_list = []
        for i in range(num_conv):
            data_list += self.convert_strategyvec_to_graph(feats[i], utt_mask[i])
        #data_list = self.convert_strategyvec_to_graph(feats)
        batch_graph = Batch.from_data_list(data_list).to(feats.device)
        num_utt = feats.shape[1]
        num_strategies = feats.shape[2]
        x, edge_index, batch = batch_graph.x, batch_graph.edge_index, batch_graph.batch
        #x, edge_index, batch = data.x, data.edge_index, data.batch  # x is num_graph x 1
        #pdb.set_trace()
        # if torch.max(x) > self.num_strat:
        #     pdb.set_trace()#print (x.squeeze(1))
        x = self.embeddings(x.squeeze(1))  # added                                 # x is num_graph x node_feats / 22
        if self.graph_model == 'gcn':
            x = F.relu(self.conv1(x, edge_index));  # import pdb; pdb.set_trace()                     # x: num_graph x 64, 2 x 21252 -> more dense, whwereas x graphs goes down
        else:
            # THIS PART#################

            #pdb.set_trace()
            #x, gat_attn_wts = self.conv1(x, edge_index, return_attention_weights=True)
            x = self.conv1(x, edge_index)
            gat_attn_wts = self.conv1.attention_score
            x = F.relu(x)
        x, edge_index, edge_weight, batch, perm, S_index, S_weight, att_wts = self.pool1(x=x, edge_index=edge_index, edge_weight=None, batch=batch)
        save_perm = perm.clone()
        xs = readout(x, batch)
        for conv, pool in zip(self.convs, self.pools):
            if self.graph_model == 'gcn':
                x = F.relu(conv(x=x, edge_index=edge_index, edge_weight=edge_weight))
            elif self.graph_model == 'gat':
                x = F.relu(conv(x=x, edge_index=edge_index))
            else:
                raise NotImplementedError
            x, edge_index, edge_weight, batch, perm, _, _, _ = pool(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)  # IGNORING S OF FUTURE LAYERS
            xs += readout(x, batch)
        x = F.relu(self.lin1(xs))
        if self.only_hidden:
            return x
        x = F.dropout(x, p=0.0, training=self.training)
        logits = self.lin2(x)#.view(1, num_utt, -1)

        #outputs, _ = self.gru(feats, None)
        #logits = self.projection_layer(self.relu(outputs))
        #logits = logits[:, -1, :].view(1, -1)
        if return_extra:
            if self.graph_model == 'gat':
                return logits, batch_graph.y, (S_index, S_weight, att_wts, save_perm, batch_graph, gat_attn_wts)
            else:
                return logits, batch_graph.y, (S_index, S_weight, att_wts, save_perm, batch_graph, gat_attn_wts)
        return logits, batch_graph.y, None

    def convert_strategyvec_to_graph(self, strategies_vec, utt_maski):
        '''
        Takes a strategies vector and converts it to a list of torch_geometric.data Data items
        '''
        from torch_geometric.data import Data
        #pdb.set_trace()
        device = strategies_vec.device
        graph_data = []
        adj_x, adj_y = [], []
        # skip for time step 0
        # lower triangle useful
        total_rows = 0
        for i in range(len(strategies_vec)):
            #adj_y.append(np.array(strategies_vec[i + 1]))
            num_strategies_in_turn = int(torch.sum(strategies_vec[i]))
            new_matrix = np.zeros((total_rows + num_strategies_in_turn, total_rows + num_strategies_in_turn))#.to(device)
            new_strategies = np.zeros((total_rows + num_strategies_in_turn, 1))#, dtype=torch.long).to(device)
            if i != 0:
                new_matrix[: total_rows, : total_rows] = adj_x[i - 1]['matrix']  # copy prev matrix
                new_strategies[: total_rows] = adj_x[i - 1]['strategies']
            curr_row = total_rows
            ##stdinturn=0
            for stidx, sval in enumerate(strategies_vec[i]):
                if sval == 0: continue
                new_strategies[curr_row, 0] = stidx
                # new_strategies.append(stidx)
                new_matrix[curr_row, : total_rows] = 1  # connecting to all in lower half except self
                ##new_matrix[curr_row, total_rows + stdinturn] = 1
                ##stdinturn+=1
                curr_row += 1
            total_rows = curr_row
            #new_matrix = torch.LongTensor(new_matrix).to(device)
            #new_strategies = torch.LongTensor(new_strategies).to(device)
            adj_x.append({
                'matrix': new_matrix,
                'strategies': new_strategies
            })
            x = torch.LongTensor(new_strategies).to(device)  # (num_strategies, 1) for now. Later will do embedding lookup
            edge_index = self.get_edge_index_from_adj_matrix(torch.LongTensor(new_matrix).to(device))
            #y = torch.FloatTensor(np.array(strategies_vec[i + 1]).reshape(1, -1))
            #y = torch.FloatTensor(np.array(strategies_vec[i]).reshape(1, -1))
            try:
                y = strategies_vec[i+1, :-1].reshape(1, -1)				# -1 for start
            except:
                y = strategies_vec[0, :-1].reshape(1, -1)#None
            #y= None

            graph_data.append(Data(x=x, edge_index=edge_index, y=y))
            #if i+2 == len(strategies_vec) or utt_maski[i+2] == 0: break
            if i+1 == len(strategies_vec) or utt_maski[i+1] == 0: break
        return graph_data

    def get_edge_index_from_adj_matrix(self, adj_matrix):
        from torch_geometric.utils.sparse import dense_to_sparse
        from torch_geometric.utils.undirected import to_undirected
        from torch_geometric.utils.loop import add_self_loops
        edge_index, edge_value = dense_to_sparse(adj_matrix)
        undirected = self.undirected 
        self_loops = self.self_loops
        if edge_index.shape[1] != 0 and undirected:
            edge_index = to_undirected(edge_index)
        if edge_index.shape[1] != 0 and self_loops:
            edge_index, _ = add_self_loops(edge_index)
        return edge_index

    def __repr__(self):
        return self.__class__.__name__

import torch
from torch.nn import Parameter
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_scatter import scatter_add

from torch_geometric.nn.inits import uniform

class LEConv(torch.nn.Module):
    r"""Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels, out_channels, bias=True):
        super(LEConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin1 = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.lin2 = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, size=None):
        """"""
        num_nodes = x.shape[0]
        h = torch.matmul(x, self.weight)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),),
                                     dtype=x.dtype,
                                     device=edge_index.device)
        edge_index, edge_weight = remove_self_loops(edge_index=edge_index, edge_attr=edge_weight)
        deg = scatter_add(edge_weight, edge_index[0], dim=0, dim_size=num_nodes)  # + 1e-10

        h_j = edge_weight.view(-1, 1) * h[edge_index[1]]
        aggr_out = scatter_add(h_j, edge_index[0], dim=0, dim_size=num_nodes)
        out = (deg.view(-1, 1) * self.lin1(x) + aggr_out) + self.lin2(x)
        edge_index, edge_weight = add_self_loops(edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_nodes)
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

