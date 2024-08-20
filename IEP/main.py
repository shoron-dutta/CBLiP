# from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import torch, time, copy, random
from tqdm import tqdm
from torch import nn, tensor
import pprint, pickle, json
import os.path, sys
from datetime import timedelta
from preprocessing import InductiveData
from connection_biased_attn import CBLiP
from torch.nn.functional import pad
from sklearn.metrics import average_precision_score
import parser_
import numpy as np
# torch.manual_seed(20)
pp = pprint.PrettyPrinter(indent=4)

def json_print(data):
    # data is a json object
    print('\n\nBest model by\t|| hits@1\t|| loss\t\t|| MRR')
    print('test MR\t\t|| ', data['hits_1']['test_mr'],'\t|| ', data['loss']['test_mr'],'\t|| ',data['mrr']['test_mr'])
    print('test MRR\t|| ', data['hits_1']['test_mrr'],'\t|| ', data['loss']['test_mrr'],'\t|| ',data['mrr']['test_mrr'])
    print('test hits@1\t|| ', data['hits_1']['test_hits_1'],'\t|| ', data['loss']['test_hits_1'],'\t|| ',data['mrr']['test_hits_1'])
    print('test hits@3\t|| ', data['hits_1']['test_hits_3'],'\t|| ', data['loss']['test_hits_3'],'\t|| ',data['mrr']['test_hits_3'])
    print('test hits@10\t|| ', data['hits_1']['test_hits_10'],'\t|| ', data['loss']['test_hits_10'],'\t|| ',data['mrr']['test_hits_10'])
    print('test AUCPR\t|| ', data['hits_1']['test_auc'],'\t|| ', data['loss']['test_auc'],'\t|| ',data['mrr']['test_auc'])
    print('test loss\t|| ', data['hits_1']['test_loss'],'\t|| ', data['loss']['test_loss'],'\t|| ',data['mrr']['test_loss'])
    print('best epoch\t|| ', data['hits_1']['best_epoch'],'\t\t|| ', data['loss']['best_epoch'],'\t\t|| ',data['mrr']['best_epoch'])

    return

class CBLiPDataloader(Dataset):

    

    def __init__(self, args, mode, paths, edge_index, edge_type, triple_subgraphs_1, triple_subgraphs_2=None, triple_subgraphs_3=None, triple_subgraphs_4=None):  
        
        self.mode = mode
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.hop = args.hop        
        self.m = args.m # maximum triple in a combined neighborhood of a triple
        self.num_triples = edge_index.shape[1] # total number of triples in this mode      
        self.k = args.k
        self.pe = args.pe
        self.up = args.up
        self.us = args.us
        self.gcn = args.gcn
        self.reuse = args.reuse
        self.hneg = args.hneg
        if mode == 'test':
            self.skip_shuffle = True
        else:
            self.skip_shuffle = args.ss
        if self.mode == 'train' and self.hneg:
            self.create_hard_neg_samples()
        if self.pe == 'eig':
            self.get_pe = self.get_eig_pe
            self.label = 'eig_vectors'
        elif self.pe == 'svd':
            self.get_pe = self.get_svd_pe
            self.label = 'svd'
        else:
            self.get_pe = self.get_basic_pe
            self.label = 'type_info'
        # TODO the data shuffle does not impact the variables here; fix that
        # remove unique as it disrupts order, and see impact.
        if args.common:
            self.combined_ = triple_subgraphs_1
            self.shuffle_recreate = self.shuffle_common
        else:
            if self.hop >= 1:
                self.shuffle_recreate = self.shuffle_recreate_1
                self.subg_1 = triple_subgraphs_1
                self.subg_1_size = [triple_subgraphs_1[i].shape[0] for i in range(len(triple_subgraphs_1))]
                self.combined_ = [triple_subgraphs_1[i] for i in range(len(triple_subgraphs_1))]

                if self.hop >= 2:
                    self.shuffle_recreate = self.shuffle_recreate_2
                    self.subg_2 = triple_subgraphs_2
                    self.subg_2_size = [triple_subgraphs_2[i].shape[0] for i in range(len(triple_subgraphs_2))]
                    self.combined_ = [torch.cat((triple_subgraphs_1[i], triple_subgraphs_2[i]), dim=0) for i in range(len(triple_subgraphs_1))] 
                    
                    if self.hop == 3:
                        self.shuffle_recreate = self.shuffle_recreate_3
                        self.subg_3 = triple_subgraphs_3
                        self.subg_3_size = [triple_subgraphs_3[i].shape[0] for i in range(len(triple_subgraphs_3))]
                        self.combined_ = [torch.cat((triple_subgraphs_1[i], triple_subgraphs_2[i], triple_subgraphs_3[i]), dim=0) for i in range(len(triple_subgraphs_1))] 
                        if self.hop == 4:
                            self.shuffle_recreate = self.shuffle_recreate_4
                            self.subg_4 = triple_subgraphs_4
                            self.subg_4_size = [triple_subgraphs_4[i].shape[0] for i in range(len(triple_subgraphs_4))]
                            # self.combined_ = [torch.unique(torch.cat((triple_subgraphs_1[i], triple_subgraphs_2[i], triple_subgraphs_3[i], triple_subgraphs_4[i]), dim=0), dim=0) for i in range(len(triple_subgraphs_1))] 
                            self.combined_ = [torch.cat((triple_subgraphs_1[i], triple_subgraphs_2[i], triple_subgraphs_3[i], triple_subgraphs_4[i]), dim=0) for i in range(len(triple_subgraphs_1))] 
        
        if self.up:
            self.path_map = paths # key: pair of ht, value: list of paths
            self.max_path_len = args.max_path_len
            self.max_path_count = args.max_path_count
    
    def create_hard_neg_samples(self):
        filename_ = './data/' + args.dataset + '/train_neg_choices.pkl'
        
        if self.reuse and os.path.isfile(filename_):
            self.triple_specific_choices = pickle.load(open(filename_, 'rb'))
            return
        st_ = time.time()
        # create hard negative samples for training
        # for each relation, list the heads that appear [regardless of tail]
        # similarly, list the tails that appear
        # this will give better neg samples: <college-a, locatedin, place-b> would be corrupted by <college-x, locatedin, place-b>
        # otherwise, trivial neg samples could generate <award-y, locatedin, place-b>
        # we want to do this only for training


        ## repeating the choice/filter option from create_choice_entities method of Data class
        hr_pairs = torch.stack([self.edge_index[0,:], self.edge_type], dim=1) # [-, 2]
        tr_pairs = torch.stack([self.edge_index[1,:], self.edge_type], dim=1) # [-, 2]
        t_avoid=dict() # key: [head, rel], value: [list of true tails]
        h_avoid=dict() # key: [tair, rel], value: [list of true heads]
        
        for i in range(self.edge_type.shape[0]):
            if tuple(hr_pairs[i,:].tolist()) in t_avoid:
                t_avoid[tuple(hr_pairs[i,:].tolist())].append(self.edge_index[1,:][i].item())
            else:
                t_avoid[tuple(hr_pairs[i,:].tolist())] = [self.edge_index[1,:][i]]

            if tuple(tr_pairs[i,:].tolist()) in h_avoid:
                h_avoid[tuple(tr_pairs[i,:].tolist())].append(self.edge_index[0,:][i].item())
            else:
                h_avoid[tuple(tr_pairs[i,:].tolist())] = [self.edge_index[0,:][i].item()]

        rel_train = list(set(self.edge_type.tolist()))
        h_choice_for_r = dict() # entities that can hold head position for relation r
        t_choice_for_r = dict() # entities that can hold tail position for relation r
        for relation in rel_train:
            flag = self.edge_type==relation
            h_choice_for_r[relation] = list(set(self.edge_index[0][flag].tolist()))
            t_choice_for_r[relation] = list(set(self.edge_index[1][flag].tolist()))

        e_idx = self.edge_index.max()
        all_choices = set(range(e_idx+1)) # assign all possible choices if none found [for rels that only appear once etc.]
        self.triple_specific_choices = dict()
        
        for idx in tqdm(range(self.edge_type.shape[0])):
            # store triple specific h, t options
            # sref shuffle in getitem and select
            
            valid_h_choice = h_choice_for_r[self.edge_type[idx].item()]
            valid_h_choice = list(set(valid_h_choice) - set(h_avoid[tuple(tr_pairs[i,:].tolist())]) - {self.edge_index[0][idx].item()})
            if len(valid_h_choice) == 0:
                valid_h_choice = list(all_choices - {self.edge_index[0][idx].item()})
            
            ## repeat for tail
            valid_t_choice = t_choice_for_r[self.edge_type[idx].item()]
            valid_t_choice = list(set(valid_t_choice) - set(t_avoid[tuple(hr_pairs[i,:].tolist())]) - {self.edge_index[1][idx].item()})
            if len(valid_t_choice) == 0:
                valid_t_choice = list(all_choices - {self.edge_index[1][idx].item()})

            self.triple_specific_choices[idx] = (valid_h_choice, valid_t_choice)
        
        if self.reuse:
            pickle.dump(self.triple_specific_choices, open(filename_, 'wb'))
        print(f'Time needed to generate hard neg samples: {time.time()-st_} seconds.')
        
        return

    def shuffle_recreate_1(self):
        for i in (range(len(self.subg_1))):
            self.subg_1[i] = self.subg_1[i][torch.randperm(self.subg_1_size[i])]
        self.combined_ = [self.subg_1[i] for i in range(len(self.subg_1))]
        return
    def shuffle_recreate_2(self):
        for i in (range(len(self.subg_1))):
            self.subg_1[i] = self.subg_1[i][torch.randperm(self.subg_1_size[i])]
            self.subg_2[i] = self.subg_2[i][torch.randperm(self.subg_2_size[i])]
        self.combined_ = [torch.cat((self.subg_1[i], self.subg_2[i]), dim=0) for i in range(len(self.subg_1))] 
        return
    def shuffle_recreate_3(self):
        for i in (range(len(self.subg_1))):
            self.subg_1[i] = self.subg_1[i][torch.randperm(self.subg_1_size[i])]
            self.subg_2[i] = self.subg_2[i][torch.randperm(self.subg_2_size[i])]
            self.subg_3[i] = self.subg_3[i][torch.randperm(self.subg_3_size[i])]
        self.combined_ = [torch.cat((self.subg_1[i], self.subg_2[i], self.subg_3[i]), dim=0) for i in range(len(self.subg_1))] 
        return
    def shuffle_recreate_4(self):
        for i in (range(len(self.subg_1))):
            self.subg_1[i] = self.subg_1[i][torch.randperm(self.subg_1_size[i])]
            self.subg_2[i] = self.subg_2[i][torch.randperm(self.subg_2_size[i])]
            self.subg_3[i] = self.subg_3[i][torch.randperm(self.subg_3_size[i])]
            self.subg_4[i] = self.subg_4[i][torch.randperm(self.subg_4_size[i])]
        self.combined_ = [torch.cat((self.subg_1[i], self.subg_2[i], self.subg_3[i], self.subg_4[i]), dim=0) for i in range(len(self.subg_1))] 
        
        return
    def shuffle_common(self):
        ns = len(self.combined_)
        each_subg_len = [self.combined_[i].shape[0] for i in range(ns)]    
        self.combined_ = [self.combined_[i][torch.randperm(each_subg_len[i])] for i in range(ns)]
    
    def get_basic_pe(self, selected_triples, head, tail, n):
        type_info = torch.zeros((self.m,3), dtype=torch.long) # start type, rel id, end type; rel id is actual rel ids so that we can access embeddings from model param
        type_info[:n,1] = selected_triples[:,1]

        # NOTE: approach with one OTHER
        st_type = (selected_triples[:,0]==head).int()+(selected_triples[:,0]==tail).int()+(selected_triples[:,0]==tail).int()
        en_type = (selected_triples[:,2]==head).int()+(selected_triples[:,2]==tail).int()+(selected_triples[:,2]==tail).int()
        
        # type 0 means other, 1 means head, 2 means tail
        # print(f'type_info[0] ->: {type_info[0]}') # should be 1 and 2 in 0th and 2nd idx always
        st_type[st_type>2] = 2
        en_type[en_type>2] = 2

        type_info[:n,0] = st_type
        type_info[:n,2] = en_type
        return True, type_info
    
    def get_svd_pe(self, selected_triples, head, tail, n):
        flag = True
        og_unique_nodes = torch.cat((selected_triples[:,0],selected_triples[:,2]),dim=0).unique()
        # if not len(og_unique_nodes)<=self.m+2:
        #     print(len(og_unique_nodes), og_unique_nodes, selected_1.shape,'\n', selected_triples) # checking if m+2 is the max
        og_n = og_unique_nodes.numel()
        if og_n < self.k:
            flag = False
        min_k = min(self.k, og_n)

        e1_idx = (og_unique_nodes==selected_triples[:,0].unsqueeze(1)).nonzero()[:,1]
        e2_idx = (og_unique_nodes==selected_triples[:,2].unsqueeze(1)).nonzero()[:,1]
        
        adj = torch.zeros(og_n, og_n)
        adj[e1_idx, e2_idx]=1
        adj[e2_idx, e1_idx]=1

        # Computing the svd of adjacency with (self-loops)
        adj += torch.eye(og_n)
        
        u, _, vh = torch.linalg.svd(adj)
        triple_svd = torch.zeros(2, self.m, 2 * self.k)
        triple_svd[:, :n, :min_k*2] = torch.stack([torch.cat((u[e1_idx, :self.k], vh.T[e1_idx, :self.k]), dim=1), \
                                                torch.cat((u[e2_idx, :self.k], vh.T[e2_idx, :self.k]), dim=1)])

        return flag, triple_svd

    def get_eig_pe(self, selected_triples, head, tail, n):
        flag = True
        og_unique_nodes = torch.cat((selected_triples[:,0],selected_triples[:,2]),dim=0).unique()
        og_n = og_unique_nodes.numel()
        if og_n < self.k:
            flag = False
        min_k = min(self.k, og_n)

        e1_idx = (og_unique_nodes==selected_triples[:,0].unsqueeze(1)).nonzero()[:,1]
        e2_idx = (og_unique_nodes==selected_triples[:,2].unsqueeze(1)).nonzero()[:,1]
        
        adj = torch.zeros(og_n, og_n)
        adj[e1_idx, e2_idx]=1
        adj[e2_idx, e1_idx]=1
        
        degree_sqrt = torch.sqrt(torch.diag(adj.sum(dim=0)))
        # Compute laplacian eigenvectors: I - D^{-1/2}AD^{-1/2}
        laplacian = torch.eye(og_n) - torch.mm(torch.mm(torch.linalg.inv(degree_sqrt), adj), degree_sqrt)
        eig_vectors = torch.linalg.eigh(laplacian)[1] # eigenvalues are returned in ascending order
        # 0th slice in dim 0- laplacian eigenvectors for selected triples' head according to this subgraph
        # 1st slice in dim 0- laplacian eigenvectors for selected triples' tail according to this subgraph
        triple_eig_v = torch.zeros(2, self.m, self.k)
        triple_eig_v[:,:n, :min_k] = torch.stack([eig_vectors[e1_idx],eig_vectors[e2_idx]])[:,:,:self.k] # 2, m, k
        # print(triple_eig_v)
        return flag, triple_eig_v

    def __len__(self):
        return self.edge_index.shape[1]

    def __getitem__(self, idx):
        
        if idx == 0 and not self.skip_shuffle:
            self.shuffle_recreate()
        
            
        # construct subgraph for one triple
        head = self.edge_index[:,idx][0].item()
        tail = self.edge_index[:,idx][1].item()
        target_rel = self.edge_type[idx].item()
        target_triple = torch.tensor([head, target_rel, tail])
        
        
        selected_triples = torch.cat((target_triple.unsqueeze(0), self.combined_[idx]), dim=0)[:self.m] 
        # print(f'Inside getitem: target triple: {target_triple}, common neighbors: {self.combined_[idx]}')   
        n = selected_triples.shape[0] # <= self.m, number of triples in original graph, number of nodes in line graph

        # share no entities 0
        # r1-h == r2-t -> 1
        # r1-t == r2-h -> 2
        # r1-h == r2-h -> 4
        # r1-t == r2-t -> 5
        # ht and th -> 3 automatically

        edge_bias_adj = torch.zeros((n,n), dtype=torch.int)
        pattern = (selected_triples[:,0]==selected_triples[:,2].unsqueeze(dim=1)).int() # r1-h == r2-t -> 1
        edge_bias_adj += pattern
        pattern[pattern!=0]=2 # r1-t == r2-h -> 2
        edge_bias_adj += pattern.T
        pattern = (selected_triples[:,0]==selected_triples[:,0].unsqueeze(dim=1)).int()  # r1-h == r2-h -> 4
        pattern[pattern!=0]=4
        edge_bias_adj += pattern
        pattern = (selected_triples[:,2]==selected_triples[:,2].unsqueeze(dim=1)).int() # r1-t == r2-t -> 5
        pattern[pattern!=0]=5
        edge_bias_adj += pattern

        # edge_bias_adj[edge_bias_adj==0] = 13
        edge_bias_adj.fill_diagonal_(0)
        edge_bias_adj = pad(edge_bias_adj,(0,self.m-n,0,self.m-n),'constant',0)
        
        # store info on type of nodes

        # NOTE: approach with other-1, other-2
        # set head and tail to 0 and 1; other nodes start from 2
        # unique_nodes = [head, tail] + list(set(selected_triples[:,0].tolist() + selected_triples[:,2].tolist())-{head,tail})
        # type_info[:n,0] = torch.tensor([unique_nodes.index(i) for i in selected_triples[:,0]])
        # type_info[:n,2] = torch.tensor([unique_nodes.index(i) for i in selected_triples[:,2]])
        flag, positional_encoding = self.get_pe(selected_triples, head, tail, n)
        _, type_info = self.get_basic_pe(selected_triples, head, tail, n)

        # send choices for neg sampling
        
        res_dict = {'edge_bias': edge_bias_adj,
                    'n':n, 
                    'target_rel': target_rel, 
                    'head':head, 
                    'tail':tail,
                    'flag':flag,
                    self.label: positional_encoding,
                    'type_info': type_info
                    }
        if self.hneg:
            if self.mode == 'train':
                h_choices, t_choices = self.triple_specific_choices[idx]
                random.shuffle(h_choices)
                random.shuffle(t_choices)
                if len(h_choices) == 0:
                    h_choices.append(-1)
                if len(t_choices) == 0:
                    t_choices.append(-1)
            else:
                h_choices = [-2]
                t_choices = [-2]
            res_dict['neg_h'] = h_choices[0]
            res_dict['neg_t'] = t_choices[0]
        if self.us or self.gcn:
            pairs = selected_triples[:,[0,2]]
            pairs = pad(pairs, (0, 0, 0, self.m-selected_triples.shape[0]), 'constant', -1)
            res_dict['pairs'] = pairs
        if self.up:
            print('~~ getitem; inside self.up=True')
            selected_paths = self.path_map[(head, tail)]
            # no need to trim shorter paths here since we store path length specific files
            # selected_paths = [i for i in selected_paths if len(i)<=self.max_path_len] 
            random.shuffle(selected_paths)
            selected_paths = selected_paths[:self.max_path_count]
            n_paths = len(selected_paths)
            
            if n_paths > 0:
                n_rels = torch.tensor([len(path) for path in selected_paths]) # for each path, how many relations, max is 4/6 if hop=2/3
                n_rels = pad(n_rels, (0, self.max_path_count-n_paths), 'constant', -1) # use positive number if using lstm later [.gather() requires that]
                
                selected_paths = [torch.tensor(path) for path in selected_paths]
                
                selected_paths = [pad(z, (0, self.max_path_len-z.shape[0]), 'constant', -1) for z in selected_paths] # pad using the last relation in the sequence
                # so if a path is [a,b], it will become [a,b,-1,-1]
                selected_paths = pad(torch.stack(selected_paths),(0, 0, 0, self.max_path_count-n_paths), 'constant', -1)
                res_dict['no_path'] = False
            else:
                res_dict['no_path'] = True
                n_rels = torch.ones(self.max_path_count) * -1
                selected_paths = torch.ones(self.max_path_count, self.max_path_len)*-1 
            res_dict['paths'] = selected_paths.long() 
            res_dict['n_paths'] = n_paths  
            res_dict['n_rels'] = n_rels.long()
               
        return res_dict
        

        
    def create_dict(lst):
        res_dct = map(lambda i: (lst[i], i), range(len(lst)))
        return dict(res_dct)



def load_data_lg_hopwise(args, mode, edge_index, edge_type, paths, triple_subgraphs_1, triple_subgraphs_2=None, triple_subgraphs_3=None, triple_subgraphs_4=None):
    # the Dataloader calls getitem after each batch_size item if shuffle=False (e.g., 0, 128, 256, 384, 512, ..)
    if mode == 'train':
        b = args.batch_size
    else:
        b = args.b_eval
    lg_dataloader = DataLoader(CBLiPDataloader(args, mode, edge_index, edge_type, paths, triple_subgraphs_1, triple_subgraphs_2, triple_subgraphs_3, triple_subgraphs_4),
                                    batch_size=b,
                                    num_workers=0,
                                    pin_memory=True,
                                    shuffle=False) 
    return lg_dataloader

def combine_(args, data, mode_instance, num_neg, train_kg_e_idx=0):
    pe_var = {'basic':'type_info','eig':'eig_vectors', 'svd':'svd'}
    b = data['head'][data['flag']].shape[0]
    
    if args.hneg:
        corrupted_pos_enc, corrupted_n, all_labels, corrupted_edge_bias, corrupted_type_info, corrupted_path_info, corrupted_pairs = \
    mode_instance.corrupt_batch(data['head'][data['flag']], data['target_rel'][data['flag']], \
    data['tail'][data['flag']], num_neg, args, data['neg_h'][data['flag']], data['neg_t'][data['flag']])
    else:
        corrupted_pos_enc, corrupted_n, all_labels, corrupted_edge_bias, corrupted_type_info, corrupted_path_info, corrupted_pairs = \
    mode_instance.corrupt_batch(data['head'][data['flag']], data['target_rel'][data['flag']], \
    data['tail'][data['flag']], num_neg, args)
        
    all_pos_enc = torch.cat((data[pe_var[args.pe]][data['flag']], corrupted_pos_enc), dim=0).to(device)
    all_n = torch.cat((data['n'][data['flag']], corrupted_n), dim=0).to(device)
    all_edge_bias = torch.cat((data['edge_bias'][data['flag']].to(device), corrupted_edge_bias.to(device)))
    all_type_info = torch.cat((data['type_info'][data['flag']], corrupted_type_info), dim=0).to(device) # always send type info
    all_pairs = []
    if args.us or args.gcn:
        all_pairs = torch.cat((data['pairs'][data['flag']], corrupted_pairs), dim=0).long().to(device)
        all_pairs += train_kg_e_idx # we're modifying the entity indices here so it gets modified only if it's created inside this IF
    path_info = []
    if args.up:
        path_info.append(torch.cat((data['paths'][data['flag']], corrupted_path_info['paths']), dim=0).to(device))
        path_info.append(torch.cat((data['n_paths'][data['flag']], corrupted_path_info['n_paths']), dim=0).to(device))
        path_info.append(torch.cat((data['n_rels'][data['flag']], corrupted_path_info['n_rels']), dim=0).to(device))
    
    return b, all_pos_enc, all_n, all_edge_bias, all_type_info, path_info, all_pairs

# functions to train and evaluate

def  train(model, train_data, metrics, args, num_sample):

    model.train()  # turn on train mode
    hits_1_sum, hits_3_sum, hits_10_sum = 0, 0, 0
    rank_sum, rr_sum = 0., 0.
    total_loss, auc, num_sample = 0., 0., 0.
    # if not rp:
    num_sample = 0 # remove this if we're not skipping any data item
    for data in tqdm(train_data, desc='train', ncols = 100):
        optimizer.zero_grad()

        b, all_pos_enc, all_n, all_edge_bias, all_type_info, path_info, pairs = combine_(args, data, d_train, args.nn_train)
        num_sample += b # remove this we're not skipping any data item
        output = model(all_n, all_edge_bias, all_type_info, all_pos_enc, path_info, pairs, 'train')

        y = torch.ones(b * args.nn_train).to(device)
        true_score, corrupted_score = output[:b].repeat(args.nn_train), output[b:]
        loss = criterion(true_score, corrupted_score, y) # we want true triples to have higher scores
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        ## compute auc before reshaping output
        auc += average_precision_score(np.concatenate((np.ones(b), np.zeros(b)), axis=0), np.array(output[:b*2].tolist()))
        ## metrics computation
        output = output.reshape(args.nn_train+1, b).T
        score_idx = output.argsort(dim=1, descending=True)
        rank = 1 + (score_idx==0).nonzero()[:,1] # 0th position is true fact
        hits_1_sum += (rank==1).sum().item()
        hits_3_sum += (rank<=3).sum().item()
        hits_10_sum += (rank<=10).sum().item()
        rank_sum += rank.sum().item()
        rr_sum += (1/rank).sum().item()
        

    
    metrics['train']['mr'].append(rank_sum/num_sample)
    metrics['train']['mrr'].append(rr_sum/num_sample)
    metrics['train']['hits_1'].append(hits_1_sum/num_sample)
    metrics['train']['hits_3'].append(hits_3_sum/num_sample)
    metrics['train']['hits_10'].append(hits_10_sum/num_sample)
    metrics['train']['loss'].append(total_loss /len(train_data) )
    metrics['train']['auc'].append(auc/len(train_data))
    return total_loss/len(train_data)
    

def evaluate(model, eval_data, metrics, args, num_sample):
    
    model.eval()  # turn on evaluation mode
    hits_1_sum, hits_3_sum, hits_10_sum = 0, 0, 0
    rank_sum, rr_sum = 0., 0.
    total_loss, auc, num_sample = 0., 0., 0.
    
    with torch.no_grad():
        for data in tqdm(eval_data, desc='Valid', ncols = 80):
            b, all_pos_enc, all_n, all_edge_bias, all_type_info, path_info, pairs = combine_(args, data, d_train, args.nn_valid)
            num_sample += b
            output = model(all_n, all_edge_bias, all_type_info, all_pos_enc, path_info, pairs, 'valid') 
 
            y = torch.ones(b * args.nn_valid).to(device)
            true_score, corrupted_score = output[:b].repeat(args.nn_valid), output[b:]
            loss = criterion(true_score, corrupted_score, y)
            total_loss += loss.item()
            
            ## compute auc before reshaping output
            auc += average_precision_score(np.concatenate((np.ones(b), np.zeros(b)), axis=0), np.array(output[:b*2].tolist()))
            ## metrics computation
            output = output.reshape(args.nn_valid+1, b).T
            score_idx = output.argsort(dim=1, descending=True)
            rank = 1 + (score_idx==0).nonzero()[:,1] # 0th position is true fact
            hits_1_sum += (rank==1).sum().item()
            hits_3_sum += (rank<=3).sum().item()
            hits_10_sum += (rank<=10).sum().item()
            rank_sum += rank.sum().item()
            rr_sum += (1/rank).sum().item()
            

    metrics['eval']['mr'].append(rank_sum/num_sample)
    metrics['eval']['mrr'].append(rr_sum/num_sample)
    metrics['eval']['hits_1'].append(hits_1_sum/num_sample)
    metrics['eval']['hits_3'].append(hits_3_sum/num_sample)
    metrics['eval']['hits_10'].append(hits_10_sum/num_sample)
    metrics['eval']['loss'].append(total_loss/len(eval_data))
    metrics['eval']['auc'].append(auc/len(eval_data))
    return total_loss/len(eval_data), rr_sum/num_sample, hits_1_sum/num_sample


def evaluate_test(model, test_data, args, num_sample, epoch_, train_kg_e_idx):
    model.eval()  # turn on evaluation mode
    hits_1_sum, hits_3_sum, hits_10_sum = 0, 0, 0
    rank_sum, rr_sum = 0., 0.
    total_loss, auc, num_sample = 0., 0., 0.
    with torch.no_grad():
        for _, data in enumerate(test_data):
            b, all_pos_enc, all_n, all_edge_bias, all_type_info, path_info, pairs = combine_(args, data, d_test, args.nn_test, train_kg_e_idx)
            num_sample += b            
            output = model(all_n, all_edge_bias, all_type_info, all_pos_enc, path_info, pairs, 'test')
            # if not bce: ## margin based ranking loss    
            y = torch.ones(b * args.nn_test).to(device)
            true_score, corrupted_score = output[:b].repeat(args.nn_test), output[b:]
            loss = criterion(true_score, corrupted_score, y)
            total_loss += loss.item()
            ## compute auc before reshaping output
            auc += average_precision_score(np.concatenate((np.ones(b), np.zeros(b)), axis=0), np.array(output[:b*2].tolist()))
            ## metrics computation
            output = output.reshape(args.nn_test+1, b).T
            score_idx = output.argsort(dim=1, descending=True)
            rank = 1 + (score_idx==0).nonzero()[:,1] # 0th position is true fact
            hits_1_sum += (rank==1).sum().item()
            hits_3_sum += (rank<=3).sum().item()
            hits_10_sum += (rank<=10).sum().item()
            rank_sum += rank.sum().item()
            rr_sum += (1/rank).sum().item()
            
    
    
    metrics_dict = {}
    
    metrics_dict['test_mr'] = round(rank_sum/num_sample, 3)
    metrics_dict['test_mrr'] = round(rr_sum/num_sample, 3)
    metrics_dict['test_hits_1'] = round(hits_1_sum/num_sample, 3)
    metrics_dict['test_hits_3'] = round(hits_3_sum/num_sample, 3)
    metrics_dict['test_hits_10'] = round(hits_10_sum/num_sample, 3)
    metrics_dict['test_loss'] = round(total_loss / len(test_data), 3)
    metrics_dict['test_auc'] = round(auc / len(test_data), 3)
    metrics_dict['best_epoch'] = epoch_
    return metrics_dict

        
    

if __name__ == '__main__':

    st = time.time()
    args, device, suffix, filepath  = parser_.create_parser()
    
    # initialize metrics dictionary
    metrics = {'train':{}, 'eval':{}}
    for mode in ['train', 'eval']:
        metrics[mode]['mrr'] = []
        metrics[mode]['hits_1'] = []
        metrics[mode]['hits_3'] = []
        metrics[mode]['hits_10'] = []
        metrics[mode]['mr'] = []
        metrics[mode]['loss'] = []
        metrics[mode]['auc'] = []

    d_test = InductiveData(args.dataset+'_ind', '/test.txt', args)
    # call test set first and then train set
    d_train = InductiveData(args.dataset, '/valid.txt', args, d_test.id2edge)    
    
    if args.rp:
        # training can be done with 1 neg sample
        args.nn_valid = d_train.r_idx - 1
        args.nn_test = d_test.r_idx - 1
    else:
        if args.nn_test == -1:
            args.nn_test = d_test.e_idx - 1
        if args.nn_valid == -1:
            args.nn_valid = d_train.e_idx - 1
    
    print('Dataset\t\t', args.dataset)
    print('Train #E\t', d_train.e_idx)
    print('Train #R\t', d_train.r_idx)
    print('Train #tr\t', d_train.kg_num_triples)
    print('Valid #tr\t', d_train.task_num_triples)
    print('Test #E\t\t', d_test.e_idx)
    print('Test #R\t\t', d_test.r_idx)
    print('Test fact #tr\t', d_test.kg_num_triples)
    print('Test #tr\t',  d_test.task_num_triples)

    num_relation = d_train.r_idx
    if args.common:
        train_data = load_data_lg_hopwise(args, 'train', d_train.path_list['kg'], d_train.kg_edge_index, d_train.kg_edge_type, d_train.kg_common_neighbors)
        valid_data = load_data_lg_hopwise(args, 'valid', d_train.path_list['task'], d_train.task_edge_index, d_train.task_edge_type, d_train.task_common_neighbors)
        test_data = load_data_lg_hopwise(args, 'test', d_test.path_list['task'], d_test.task_edge_index, d_test.task_edge_type, d_test.task_common_neighbors)
    
    elif args.hop == 4:
        train_data = load_data_lg_hopwise(args, 'train', d_train.path_list['kg'], d_train.kg_edge_index, d_train.kg_edge_type, d_train.kg_triple_subgraphs_1, d_train.kg_triple_subgraphs_2, d_train.kg_triple_subgraphs_3, d_train.kg_triple_subgraphs_4)
        valid_data = load_data_lg_hopwise(args, 'valid', d_train.path_list['task'], d_train.task_edge_index, d_train.task_edge_type, d_train.task_triple_subgraphs_1, d_train.task_triple_subgraphs_2, d_train.task_triple_subgraphs_3, d_train.task_triple_subgraphs_4)
        test_data = load_data_lg_hopwise(args, 'test', d_test.path_list['task'], d_test.task_edge_index, d_test.task_edge_type, d_test.task_triple_subgraphs_1, d_test.task_triple_subgraphs_2, d_test.task_triple_subgraphs_3, d_test.task_triple_subgraphs_4)
    elif args.hop == 3:
        train_data = load_data_lg_hopwise(args, 'train', d_train.path_list['kg'], d_train.kg_edge_index, d_train.kg_edge_type, d_train.kg_triple_subgraphs_1, d_train.kg_triple_subgraphs_2, d_train.kg_triple_subgraphs_3)
        valid_data = load_data_lg_hopwise(args, 'valid', d_train.path_list['task'], d_train.task_edge_index, d_train.task_edge_type, d_train.task_triple_subgraphs_1, d_train.task_triple_subgraphs_2, d_train.task_triple_subgraphs_3)
        test_data = load_data_lg_hopwise(args, 'test', d_test.path_list['task'], d_test.task_edge_index, d_test.task_edge_type, d_test.task_triple_subgraphs_1, d_test.task_triple_subgraphs_2, d_test.task_triple_subgraphs_3)
    elif args.hop == 2:
        train_data = load_data_lg_hopwise(args, 'train', d_train.path_list['kg'], d_train.kg_edge_index, d_train.kg_edge_type, d_train.kg_triple_subgraphs_1, d_train.kg_triple_subgraphs_2)
        valid_data = load_data_lg_hopwise(args, 'valid', d_train.path_list['task'], d_train.task_edge_index, d_train.task_edge_type, d_train.task_triple_subgraphs_1, d_train.task_triple_subgraphs_2)
        test_data = load_data_lg_hopwise(args, 'test', d_test.path_list['task'], d_test.task_edge_index, d_test.task_edge_type, d_test.task_triple_subgraphs_1, d_test.task_triple_subgraphs_2)
    elif args.hop == 1:
        train_data = load_data_lg_hopwise(args, 'train', d_train.path_list['kg'], d_train.kg_edge_index, d_train.kg_edge_type, d_train.kg_triple_subgraphs_1)
        valid_data = load_data_lg_hopwise(args, 'valid', d_train.path_list['task'], d_train.task_edge_index, d_train.task_edge_type, d_train.task_triple_subgraphs_1)
        test_data = load_data_lg_hopwise(args, 'test', d_test.path_list['task'], d_test.task_edge_index, d_test.task_edge_type, d_test.task_triple_subgraphs_1)

    model = CBLiP(args, num_relation, device)
    model.to(device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    # weights = []
    # for param in model.parameters():
    #     weights.append(param.clone())
    # w_name = []
    # for name, p in model.named_parameters():
    #     w_name.append(name)
    if args.bce:
        raise ValueError('To do: Implement in the train/valid/test methods; Include label smoothing;')
        criterion = nn.BCELoss()
    else:
        criterion = nn.MarginRankingLoss(margin=args.margin)
    
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # decayRate = 1e-6
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'pytorch_total_params: {pytorch_total_params}')

    least_valid_loss = 1000
    best_valid_mrr = 0.
    best_valid_hits_1 = 0.

    best_model_hits_1 = None
    best_model_mrr = None
    best_model_loss = None

    best_epoch_loss = 0
    best_epoch_mrr = 0
    best_epoch_hits_1 = 0

    test_metrics_file = filepath + '_test_results_' + suffix + '.json'
    
    for epoch in tqdm(range(args.num_epochs), desc='epochs', ncols=150):
        
        train_loss = train(model, train_data, metrics, args, d_train.kg_num_triples)
        valid_loss, valid_mrr, valid_hits_1 = evaluate(model, valid_data, metrics, args, d_train.task_num_triples)
        

        if valid_loss < least_valid_loss:
            least_valid_loss = valid_loss
            best_model_loss=model
            best_model_loss = copy.deepcopy(model)
            best_epoch_loss = epoch
        
        if valid_mrr > best_valid_mrr:
            best_valid_mrr = valid_mrr
            best_model_mrr=model
            best_model_mrr = copy.deepcopy(model)
            best_epoch_mrr = epoch

        if valid_hits_1 > best_valid_hits_1:
            best_valid_hits_1 = valid_hits_1
            best_model_hits_1=model
            best_model_hits_1 = copy.deepcopy(model)
            best_epoch_hits_1 = epoch

        if (epoch > 0 and epoch % args.log == 0) or epoch == args.num_epochs-1:

            if os.path.isfile(test_metrics_file):
                test_metrics = json.load(open(test_metrics_file, 'r'))
            else:
                test_metrics = dict()
            
            # plot_one(metrics['train']['loss'], 'Training loss', 'loss', epoch+1, filepath + '_loss_'+ suffix+ '.jpg')
            test_metrics_cur_epoch = dict()
            test_metrics_cur_epoch['hits_1'] = evaluate_test(best_model_hits_1, test_data, args, d_test.task_num_triples, best_epoch_hits_1, d_train.e_idx)
            test_metrics_cur_epoch['loss'] = evaluate_test(best_model_loss, test_data, args, d_test.task_num_triples, best_epoch_loss, d_train.e_idx)
            test_metrics_cur_epoch['mrr'] = evaluate_test(best_model_mrr, test_data, args, d_test.task_num_triples, best_epoch_mrr, d_train.e_idx)
            execution_time = time.time()-st
            test_metrics_cur_epoch['time'] = str(timedelta(seconds=execution_time))

            test_metrics[epoch] = test_metrics_cur_epoch
            json_print(test_metrics_cur_epoch)
            
            with open(filepath + '_test_results_' + suffix + '.json', 'w', encoding='utf-8') as f:
                json.dump(test_metrics, f, ensure_ascii=False, indent=4)
                
            

    
    print('training done!')
    
    # weights_after = []
    # for param in model.parameters():
    #     weights_after.append(param.clone())
    # print('Param comp')
    # for i in zip(w_name, weights, weights_after):
    #     print(f'name: {i[0]}, status:{torch.equal(i[1],i[2])}')
        # we want all param to be changed (outcome-> False)
    print('Metrics\t\t\tHits_1\tloss\tMRR\t|| Metrics\t\tHits_1\tloss\tMRR')
    print(f'Best epoch:\t\t {best_epoch_hits_1}\t{best_epoch_loss}\t{best_epoch_mrr}')
    print(f"Training MR:\t\t {metrics['train']['mr'][best_epoch_hits_1]:.3f}\t{metrics['train']['mr'][best_epoch_loss]:.3f}\t{metrics['train']['mr'][best_epoch_mrr]:.3f}\t|| Validation MR:\t{metrics['eval']['mr'][best_epoch_hits_1]:.3f}\t{metrics['eval']['mr'][best_epoch_loss]:.3f}\t{metrics['eval']['mr'][best_epoch_mrr]:.3f}")
    print(f"Training MRR:\t\t {metrics['train']['mrr'][best_epoch_hits_1]:.3f}\t{metrics['train']['mrr'][best_epoch_loss]:.3f}\t{metrics['train']['mrr'][best_epoch_mrr]:.3f}\t|| Validation MRR:\t{metrics['eval']['mrr'][best_epoch_hits_1]:.3f}\t{metrics['eval']['mrr'][best_epoch_loss]:.3f}\t{metrics['eval']['mrr'][best_epoch_mrr]:.3f}")
    print(f"Training hits@1:\t {metrics['train']['hits_1'][best_epoch_hits_1]:.3f}\t{metrics['train']['hits_1'][best_epoch_loss]:.3f}\t{metrics['train']['hits_1'][best_epoch_mrr]:.3f}\t|| Validation hits@1:\t{metrics['eval']['hits_1'][best_epoch_hits_1]:.3f}\t{metrics['eval']['hits_1'][best_epoch_loss]:.3f}\t{metrics['eval']['hits_1'][best_epoch_mrr]:.3f}")
    print(f"Training hits@3:\t {metrics['train']['hits_3'][best_epoch_hits_1]:.3f}\t{metrics['train']['hits_3'][best_epoch_loss]:.3f}\t{metrics['train']['hits_3'][best_epoch_mrr]:.3f}\t|| Validation hits@3:\t{metrics['eval']['hits_3'][best_epoch_hits_1]:.3f}\t{metrics['eval']['hits_3'][best_epoch_loss]:.3f}\t{metrics['eval']['hits_3'][best_epoch_mrr]:.3f}")
    print(f"Training hits@10:\t {metrics['train']['hits_10'][best_epoch_hits_1]:.3f}\t{metrics['train']['hits_10'][best_epoch_loss]:.3f}\t{metrics['train']['hits_10'][best_epoch_mrr]:.3f}\t|| Validation hits@10:\t{metrics['eval']['hits_10'][best_epoch_hits_1]:.3f}\t{metrics['eval']['hits_10'][best_epoch_loss]:.3f}\t{metrics['eval']['hits_10'][best_epoch_mrr]:.3f}")
    print(f"Training AUCPR:\t\t {metrics['train']['auc'][best_epoch_hits_1]:.3f}\t{metrics['train']['auc'][best_epoch_loss]:.3f}\t{metrics['train']['auc'][best_epoch_mrr]:.3f}\t|| Validation AUCPR:\t{metrics['eval']['auc'][best_epoch_hits_1]:.3f}\t{metrics['eval']['auc'][best_epoch_loss]:.3f}\t{metrics['eval']['auc'][best_epoch_mrr]:.3f}")
    print(f"Training loss:\t\t {metrics['train']['loss'][best_epoch_hits_1]:.3f}\t{metrics['train']['loss'][best_epoch_loss]:.3f}\t{metrics['train']['loss'][best_epoch_mrr]:.3f}\t|| Validation loss:\t{metrics['eval']['loss'][best_epoch_hits_1]:.3f}\t{metrics['eval']['loss'][best_epoch_loss]:.3f}\t{metrics['eval']['loss'][best_epoch_mrr]:.3f}")
    
