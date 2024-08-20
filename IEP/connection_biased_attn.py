import torch.nn as nn
import torch, copy, os
from math import sqrt
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable as V
from torch_geometric.nn.models import GCN
class Norm(nn.Module):
    def __init__(self, effective_d, eps = 1e-6):
        super().__init__()
    
        self.size = effective_d
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
    
class FeedForward(nn.Module):
    def __init__(self, d_model, dim_ffn, dropout = 0.1):
        super().__init__() 
        
        self.linear_1 = nn.Linear(d_model, dim_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dim_ffn, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
    
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



def attention(q, k, v, d_k, use_q_bias, bias_query, bias_key, bias_value, mask=None, dropout=None):
    
    # edge_bias.shape [b, seq_len, seq_len, d]
    # q, k, v [b, nh, seq_len, d_k]
    b, nh, s, _ = q.shape
    
    # default score compute
    # scores = torch.matmul(q, k.transpose(-2, -1)) /  sqrt(d_k) # [b, nh, seq_len, seq_len]
    # custom edge biased attention score computation
    # [b, nh, seq_len, 1, d] x [b, nh, seq_len, d, seq_len] -> [b, nh, seq_len,1,seq_len]
    
    edge_biased_k = k.unsqueeze(dim=2) + bias_key.reshape(b,s,s,nh,d_k).transpose(1,3) # edge_biased_k: [b, nh, seq_len, seq_len, d]
    scores = torch.matmul(q.unsqueeze(dim=3), edge_biased_k.transpose(-2, -1)).squeeze() /  sqrt(d_k) # [b, nh, seq_len, seq_len]
    if mask is not None:
        mask = mask.unsqueeze(1).unsqueeze(-1)
    
    # NOTE: I updated the mask checking condition to 1; consistent with pytorch
    scores = scores.masked_fill_(mask == 1, -1e9) # TODO fix error
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    # default value compute
    # output = torch.matmul(scores, v)
    edge_biased_value = v.unsqueeze(dim=2) + bias_value.reshape(b,s,s,nh,d_k).transpose(1,3)
    output = torch.matmul(scores.unsqueeze(dim=3), edge_biased_value)

    return output.squeeze()


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, use_qb, edge_bias_q=None, edge_bias_k=None, edge_bias_v=None, mask=None):
        
        bs = q.size(0)
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k) # k = (W_k)(x)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k) # q = (W_q)(x)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k) # v = (W_v)(x)
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        
        scores = attention(q, k, v, self.d_k, use_qb, edge_bias_q, edge_bias_k, edge_bias_v, mask, self.dropout) # [b, nh, seq_len, d]
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output
class EncoderLayer(nn.Module):
    def __init__(self, d_model, dim_ffn, heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, dim_ffn)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, use_qb, edge_bias_q=None, edge_bias_k=None, edge_bias_v=None, mask=None):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2, use_qb, edge_bias_q, edge_bias_k, edge_bias_v, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
class Encoder(nn.Module):
    def __init__(self, d_model, dim_ffn, nl, heads):
        super().__init__()
        self.nl = nl
        self.layers = get_clones(EncoderLayer(d_model, dim_ffn, heads), nl)
        self.norm = Norm(d_model)
    def forward(self, x, use_qb, edge_bias_q=None, edge_bias_k=None, edge_bias_v=None, mask=None):
        
        for i in range(self.nl):
            x = self.layers[i](x, use_qb, edge_bias_q, edge_bias_k, edge_bias_v, mask)
        return self.norm(x)

class CBLiP(nn.Module):
    def __init__(self, args, rel_num, device, e_features_train=None, e_features_test=None, edge_idx_train=None, edge_idx_test=None, train_e_idx=0):
        super().__init__()
        self.rel_num = rel_num
        self.device = device
        self.handle_arg(args, rel_num, e_features_train, e_features_test, edge_idx_train, edge_idx_test, train_e_idx)       
        
        
        if args.pe == 'basic':
            self.type_encoding = nn.Parameter(torch.randn(3, self.embed_dim)) # 0 is OTHER, 1 is head, 2 is tail
        
        self.edge_bias_encoding_key = nn.Parameter(torch.randn(14, self.effective_d)) # 6 kinds of edge can exist
        self.edge_bias_encoding_value = nn.Parameter(torch.randn(14, self.effective_d)) # 6 kinds of edge can exist
        self.rel_encoding = nn.Parameter(torch.randn(self.rel_num, self.rel_embed_dim))
        
        self.target_rel_special_token = nn.Parameter(torch.randn(self.rel_embed_dim))
        
        
        self.rsc = args.rsc
        if args.up:
            if args.rsc:
                self.score_linear_1 = nn.Linear(self.rel_embed_dim * 2, int(self.rel_embed_dim))
                self.score_linear_2 = nn.Linear(int(self.rel_embed_dim), 1, bias=False)
            else:
                self.score_linear_1 = nn.Linear(self.rel_embed_dim, int(0.5*self.rel_embed_dim))
                self.score_linear_2 = nn.Linear(int(0.5*self.rel_embed_dim), 1, bias=False)
        else:
            if args.rsc:
                # concat context and target rel
                self.score_linear_1 = nn.Linear(self.effective_d+self.rel_embed_dim, int(self.effective_d))
                self.score_linear_2 = nn.Linear(int(self.effective_d), 1, bias=False)
            else:
                self.score_linear_1 = nn.Linear(self.effective_d, int(self.effective_d * 0.5))
                self.score_linear_2 = nn.Linear(int(0.5*self.effective_d), 1, bias=False)
            
        
        self.possible_values = torch.arange(self.m).to(self.device)
        self.transformer_encoder = Encoder(self.effective_d, self.dim_feedforward, self.nlayers, self.nheads)

        
    def handle_arg(self, args, rel_num, e_features_train, e_features_test, edge_idx_train, edge_idx_test, train_e_idx):
        self.embed_dim = args.d
        self.m = args.m
        self.k = args.k # how many eigenvectors to keep
        
        self.ffn = args.ffn # default 2 in this setup
        self.nheads = args.nheads
        self.nlayers = args.nlayers
        self.dropout = args.dr
        
        self.use_qb = args.qb
        self.up = args.up
        self.p2 = args.p2
        self.us = args.us
        self.max_path_len = args.max_path_len
        self.max_path_count = args.max_path_count
        self.dataset = args.dataset
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.sigmoid = nn.Sigmoid()
        
        
        self.nt = args.nt
        self.gcn = args.gcn
        
        self.get_entity_embed = self.omit_signature
        self.rel_embed_dim = self.embed_dim
        

        
        if args.agg == 'concat':
            self.aggregator = self.concat
            self.effective_d = 3 * self.rel_embed_dim
            if args.pe=='eig':
                # raise ValueError('Check with rel_embed_dim')
                self.eig_linear = nn.Linear(args.k, self.embed_dim, bias=False) 
                self.get_pe = self.get_pos_enc_eig
            else:
                self.get_pe = self.get_type_encoding
            
            
        elif args.agg == 'mean':
            # if we use eig or not, effective d stays the same which is d
            self.effective_d = self.rel_embed_dim 
            if args.pe=='eig':
                if self.k * 2 > self.embed_dim:
                    raise ValueError('Dimensions not supported.')
                self.aggregator = self.custom_sum
                self.get_pe = self.get_pos_enc_eig
            else:
                self.aggregator = self.mean
                self.get_pe = self.get_type_encoding
        elif args.agg == 'mean_lin':
            # never used without eig
            if not args.pe=='eig':
                raise ValueError('Operation not supported.')
            self.effective_d = self.rel_embed_dim 
            self.linear_0 = nn.Linear(self.k * 2, self.effective_d) # to convert the concatenated pe into effective d
            
            
            self.aggregator = self.mean_lin
            self.get_pe = self.get_pos_enc_eig
        elif args.agg == 'svd_agg':
            # never used without eig
            if not args.pe=='eig':
                raise ValueError('Operation not supported.')
            self.effective_d = self.rel_embed_dim 
            self.linear_0 = nn.Linear(self.k * 4, self.effective_d) # to convert the concatenated pe into effective d
            self.linear_1 = nn.Linear(self.effective_d, self.effective_d)
            
            self.aggregator = self.mean_lin
            self.get_pe = self.get_pos_enc_eig
        elif args.agg == 'w':
            # use weighted sum
            self.alpha = nn.Parameter(torch.tensor([0.5]))
            self.get_pe = self.get_pos_enc_eig
            self.aggregator = self.weighted_sum_triple
            self.effective_d = self.rel_embed_dim 
        else:
            raise ValueError('Aggregator not recognized or supported.')
        self.dim_feedforward = int(self.ffn * self.effective_d)
        if self.up:
            
            self.padding = nn.Parameter(torch.randn(self.rel_embed_dim))
            self.direction_vector = nn.Parameter(torch.randn(self.rel_embed_dim))
            if args.wsum:
                # use a weighted sum to combine path and con
                self.path_weight = nn.Parameter(torch.randn(1))
                self.wsum_lin_layer = nn.Linear(self.effective_d, self.rel_embed_dim, bias=False)
                self.con_path_agg = self.wsum
            else:
                # use a concatenation and linear transformation
                self.con_path_agg = self.cat_lin
                self.cat_lin_layer = nn.Linear(self.rel_embed_dim+self.effective_d, self.rel_embed_dim, bias=False)
            if args.uptr:
                # use pytorch's basic layer here
                ## transformer layers for combining context and path
                # nh, dim_feed, dr; set dropout to 0 here
                self.path_agg = self.path_agg_tr
                p_layer = nn.TransformerEncoderLayer(self.rel_embed_dim, 2, self.rel_embed_dim * 2, 0., batch_first=True) 
                self.p_encoder = nn.TransformerEncoder(p_layer, 2)
            
            elif args.upmlp:
                self.path_agg = self.path_agg_mlp
                self.sp_linear_1 = nn.Linear((self.max_path_count)*self.rel_embed_dim, self.rel_embed_dim)

            else:
                self.path_agg = self.path_agg_mean
            
            if args.p2:
                self.path_from_rel = self.mlp_2
                self.path_linear_1 = nn.Linear(self.rel_embed_dim * self.max_path_len, int(self.rel_embed_dim * self.max_path_len/2))
                self.path_linear_2 = nn.Linear(int(self.rel_embed_dim * self.max_path_len/2), self.rel_embed_dim) 
            else:
                self.path_from_rel = self.mlp_1
                self.path_linear = nn.Linear(self.rel_embed_dim * self.max_path_len, self.rel_embed_dim)
        else:
            self.con_path_agg = self.only_con
        
    def weighted_sum_triple(self, type_head, rel, type_tail):
        
        l=type_head.shape[2] * 2
        type_enc = torch.cat((type_head, type_tail), dim=2) * self.alpha + rel[:,:,:l] * (1-self.alpha)
        return torch.cat((type_enc, rel[:,:,l:]), dim=2)

    def concat(self, type_head, rel, type_tail):
        return torch.cat((type_head, rel, type_tail), dim=2) # [b, m, 3d]
    
    def mean(self, type_head, rel, type_tail):
        return (type_head + rel + type_tail)/3 # [b, m, d]
    
    def custom_sum(self, type_head, rel, type_tail):
        # k, d, k
        # d <= 2k
        rel[:,:,:2 * self.k] = rel[:,:,: 2 * self.k] + torch.cat((type_head, type_tail), dim=2)
        return rel # [b, m, d]
    def mean_lin(self, type_head, rel, type_tail):
        # concat the head and tail eig pe
        # pass through a linear transformation to d dimentional vector
        # mean and return
        return (rel + self.linear_0(torch.cat((type_head, type_tail), dim=2)))/2
    
    def get_pos_enc_eig(self, type_info, pos_enc):
        
        return self.eig_linear(pos_enc[:,0,:,:]), self.eig_linear(pos_enc[:,1,:,:])
         
    
    def get_type_encoding(self, type_info, pos_enc):
        return self.type_encoding[type_info[:,:,0]], self.type_encoding[type_info[:,:,2]]
    def only_con(self, context, path_info):
        return context
    def wsum(self, context, path_info):
        path_embed = self.compute_path_embed(path_info)
        # each path d, each context effective_d
        
        return self.path_weight * path_embed + (1-self.path_weight) * self.wsum_lin_layer(context)
    def cat_lin(self, context, path_info):
        # context [b, effective_d]
        # path [b, rel_embed]
        path_embed = self.compute_path_embed(path_info)
        return self.relu(self.cat_lin_layer(torch.cat((context, path_embed), dim=1))) # use relu as this output goes to another linear layer right after
    def path_agg_tr(self, path_feat):
        # path_feat [b, mpc, effective_d]
        out = self.p_encoder(path_feat) # [b, mpc, effective_d]
        out = out.mean(dim=1)
        return out
    def path_agg_mean(self, path_feat, n_paths):
        # path_feat [b, mpc, effective_d]
        # n_path is a 1D tensor of effective batch size
        # feat_dim = path_feat.shape[2]
        # possible_values = torch.arange(self.max_path_count).to(self.device)
        # reset_flag = n_paths.unsqueeze(1)<=possible_values.unsqueeze(0)
        # path_feat[reset_flag] = torch.zeros(feat_dim).to(self.device)
        return path_feat.mean(dim=1)
    def path_agg_mlp(self, path_feat, n_paths):
        # path_feat [b, mpc, effective_d]
        
        path_feat = path_feat.reshape(-1, self.max_path_count * self.rel_embed_dim)
        return self.self.leaky_relu(self.sp_linear_1(path_feat))

    def mlp_1(self, x):
        return self.path_linear(x)
    def mlp_2(self, x):
        return self.path_linear_2(self.leaky_relu(self.path_linear_1(x)))
    def omit_signature(self, pe_1, pe_2, pairs, mode):
        return pe_1, pe_2
    def use_signature_cluster(self, pe_1, pe_2, pairs, mode):
        # find the entities in this batch
        # use gradient descent to match with closest signature
        # use updated entity features in final token representation
        
        # ent_t = self.e_features[pairs]  # [(num_neg+1) * b, m, 2, d]            
        # ent_t = self.sig_linear_layer_2(self.leaky_relu(self.sig_linear_layer_1(ent_t)))  # when MLP is applied first
        # for t in range(self.T):
        #     ent_t = self.sig_model(ent_t)
        # if self.nt:
        #     return ent_t[:,:,0,:], ent_t[:,:,1,:]
        # return torch.cat((ent_t[:,:,0,:], pe_1), dim=-1), torch.cat((ent_t[:,:,1,:], pe_2), dim=-1)

        pair_h_embed = self.e_features_as_head[pairs[:,:,0]]
        pair_t_embed = self.e_features_as_tail[pairs[:,:,1]]
        pair_h_embed = self.sig_linear_layer_2_h(self.relu(self.sig_linear_layer_1_h(pair_h_embed)))
        pair_t_embed = self.sig_linear_layer_2_t(self.relu(self.sig_linear_layer_1_t(pair_t_embed)))
        # TODO: see if two separate cluster models help
        for t in range(self.T):
            pair_h_embed = self.sig_model_h(pair_h_embed)
            pair_t_embed = self.sig_model_t(pair_t_embed)

        return torch.cat((pair_h_embed, pe_1), dim=-1), torch.cat((pair_t_embed, pe_2), dim=-1)
        
    def use_signature_only(self, pe_1, pe_2, pairs, mode):
        # find the entities in this batch
        # use gradient descent to match with closest signature
        # use updated entity features in final token representation
        # pairs shape [(num_neg+1) * b, m, 2]
                   
        pair_h_embed = self.e_features_as_head[pairs[:,:,0]]
        pair_t_embed = self.e_features_as_tail[pairs[:,:,1]]
        pair_h_embed = self.sig_linear_layer_2_h(self.relu(self.sig_linear_layer_1_h(pair_h_embed)))
        pair_t_embed = self.sig_linear_layer_2_t(self.relu(self.sig_linear_layer_1_t(pair_t_embed)))

        return torch.cat((pair_h_embed, pe_1), dim=-1), torch.cat((pair_t_embed, pe_2), dim=-1)
        # ent_t = self.e_features[pairs]  # [(num_neg+1) * b, m, 2, d] 
        # ent_t = self.sig_linear_layer_2(self.leaky_relu(self.sig_linear_layer_1(ent_t)))  # when MLP is applied first
        # if self.nt:
        #     return ent_t[:,:,0,:], ent_t[:,:,1,:]
        
        # return torch.cat((ent_t[:,:,0,:], pe_1), dim=-1), torch.cat((ent_t[:,:,1,:], pe_2), dim=-1)
    
    def use_gcn_only(self, pe_1, pe_2, pairs, mode):
        if mode == 'train' or mode == 'valid':
            ent_embed_as_h = self.gcn_h(self.e_features_train[:, :self.rel_num], self.edge_idx_train)
            ent_embed_as_t = self.gcn_t(self.e_features_train[:, self.rel_num:], self.edge_idx_train)
            
        else:
            ent_embed_as_h = self.gcn_h(self.e_features_test[:, :self.rel_num], self.edge_idx_test)
            ent_embed_as_t = self.gcn_t(self.e_features_test[:, self.rel_num:], self.edge_idx_test)
            pairs -= self.train_e_idx # pairs can contain -1 values (padding for missing neighbors)
            
        # if self.nt:
        #     return pair_embed[:,:,0,:], pair_embed[:,:,1,:]
        return torch.cat((ent_embed_as_h[pairs[:,:,0]], pe_1), dim=-1), torch.cat((ent_embed_as_t[pairs[:,:,1]], pe_2), dim=-1)
        return torch.cat((pair_embed[:,:,0,:], pe_1), dim=-1), torch.cat((pair_embed[:,:,1,:], pe_2), dim=-1)
    
    def use_gcn_cluster(self, pe_1, pe_2, pairs, mode):
        if mode == 'train' or mode == 'valid':
            ent_embed_as_h = self.gcn_h(self.e_features_train[:, :self.rel_num], self.edge_idx_train)
            ent_embed_as_t = self.gcn_t(self.e_features_train[:, self.rel_num:], self.edge_idx_train)
            
        else:
            ent_embed_as_h = self.gcn_h(self.e_features_test[:, :self.rel_num], self.edge_idx_test)
            ent_embed_as_t = self.gcn_t(self.e_features_test[:, self.rel_num:], self.edge_idx_test)
            pairs -= self.train_e_idx # pairs can contain -1 values (padding for missing neighbors)
          
        h_embed = ent_embed_as_h[pairs[:,:,0]]
        t_embed = ent_embed_as_t[pairs[:,:,1]]
        for t in range(self.T):
            h_embed = self.sig_model_h(h_embed)
            t_embed = self.sig_model_t(t_embed)
        # if self.nt:
        #     return pair_embed[:,:,0,:], pair_embed[:,:,1,:]
        return torch.cat((h_embed, pe_1), dim=-1), torch.cat((t_embed, pe_2), dim=-1)
    
    def compute_path_embed(self, path_info):
        # concatenate relations in a path to preserve sequence
        # use an MLP to transform to specific feature dimension
        # combine multiple paths using preferred agg method
        
        paths, n_paths, n_rels = path_info[0], path_info[1], path_info[2]
        
        direction_flag = paths > self.rel_num        
        len_flag = paths==-1 # n_rels.unsqueeze(-1) <= torch.arange(self.max_path_len).cuda()

        effective_paths = paths % self.rel_num
        path_feat = self.rel_encoding[effective_paths] # features of all rels

        path_feat[direction_flag] += self.direction_vector # for reverse relations
        path_feat[len_flag] = self.padding # path_feat [effective_b, mpc, mpl, d]

        # concatenate the relation features in a path and transform to effective dim
        path_feat = self.path_from_rel(path_feat.reshape(path_feat.shape[0], path_feat.shape[1], -1)) # path_feat [effective_b, mpc, d] NOT effective d now! see result

        return self.path_agg(path_feat, n_paths) # [b, d]

    
    
    def forward(self, n, adj, type_info, pos_enc, path_info, pairs=[], mode='m'):

        edge_bias_query_embed = None
        

        rel_features = self.rel_encoding[type_info[:,:,1]] # [b,m,d]
        rel_features[:,0,:] = rel_features[:,0,:] + self.target_rel_special_token
        edge_bias_key_embed = self.edge_bias_encoding_key[adj.long()]
        edge_bias_value_embed = self.edge_bias_encoding_value[adj.long()]
        pe_1, pe_2 = self.get_pe(type_info, pos_enc) # pe shape [b*num_neg, m, d]
        e_1, e_2 = self.get_entity_embed(pe_1, pe_2, pairs, mode)

        src = self.aggregator(e_1, rel_features, e_2)

        mask = self.possible_values>=n.unsqueeze(-1) # [b,m]
        output = self.transformer_encoder(src, self.use_qb, edge_bias_query_embed,\
                edge_bias_key_embed, edge_bias_value_embed, mask) # [b,m,d] or [b,m,3d]
        y_embed = output[:,0, :] # Tensor: shape (BATCH_SIZE, EMBED_DIM * 3), using the whole target triple embedding instead of a CLS token
        
        con_path = self.con_path_agg(y_embed, path_info) # returns only con if not using path
        if self.rsc:
            con_path = torch.cat((con_path , rel_features[:,0,:]), dim=1) # cat
        score = self.sigmoid(self.score_linear_2(self.leaky_relu(self.score_linear_1(con_path))))
        # print(score)
        # score = self.score_linear_2(self.leaky_relu(self.score_linear_1(con_path)))

        ## use wsum
        # then con-path becomes rel_embed dim
        # then use target r a) mean, b) cat and then score
        

        ## use sig
        # the use <h,r,t, con_path> for scoring
        
        
        return torch.squeeze(score)





        
