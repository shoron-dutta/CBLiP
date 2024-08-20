import pprint, os, time, json
pp = pprint.PrettyPrinter(indent=4)
from collections import defaultdict
import random, pickle
import networkx as nx
from torch.utils.data import Dataset
import torch, os, pickle
from torch_geometric.utils import k_hop_subgraph as k_hop
from itertools import chain
from torch.nn.functional import pad
import matplotlib.pyplot as plt
from tqdm import tqdm

class InductiveData:


    def __init__(self, dataset, task_file, args, test_id2edge=None):
        
        self.dataset = dataset
        self.hop = args.hop
        self.max_path_len = args.max_path_len
        self.max_path_count = args.max_path_count
        self.up = args.up
        self.us = args.us
        self.node2id, self.edge2id = dict(), dict() # key: entity/relation string, value: int ID
        self.id2node, self.id2edge = [], [] # idx: ID, item: entity/relation string
        self.e_idx, self.r_idx = 0, 0 # count # of nodes, relations
        self.reuse = args.reuse
        if args.pe == 'basic':
            self.get_pe = self.get_basic_pe
        elif args.pe == 'eig':
            self.get_pe = self.get_eig_pe
        else:
            self.get_pe = self.get_svd_pe
        

        with open('./data/'+ dataset + '/train.txt', 'r') as f:
            kg_data = f.read().split() # list of all entities and relations in order
        with open('./data/'+ dataset + task_file, 'r') as f:
            task_data = f.read().split() # list of all entities and relations in order
        
        # read kg data
        # need not be stored in the class since we have the int <-> str mapping both ways
        kg_src_str = kg_data[::3] # every 3rd item starting at 0th entry of data, all head entities
        kg_dst_str = kg_data[2::3] # every 3rd item starting at 2nd entry of data, all tail entities
        kg_edge_type_str = kg_data[1::3] # # every 3rd item starting at 3rd entry of data, all relations

        # read task data
        task_src_str = task_data[::3] # every 3rd item starting at 0th entry of data, all head entities
        task_dst_str = task_data[2::3] # every 3rd item starting at 2nd entry of data, all tail entities
        task_edge_type_str = task_data[1::3] # # every 3rd item starting at 3rd entry of data, all relations
        
        # assign IDs to entities base on the kg graph
        if self.filesexist() and self.reuse:
            self.readfiles()
            self.e_idx = len(self.id2node)
            self.r_idx = len(self.id2edge)
        else:
            
            self.id2node = list(set(kg_src_str+kg_dst_str+task_src_str+task_dst_str))
            
            self.node2id = {self.id2node[i]: i for i in range(len(self.id2node))}
            self.e_idx = len(self.id2node)
            # assign IDs to edges if it is the test setting
            if self.dataset.endswith('_ind'):
                self.id2edge = list(set(kg_edge_type_str+task_edge_type_str))
                self.edge2id = {self.id2edge[i]: i for i in range(len(self.id2edge))}
                
                self.r_idx = len(self.id2edge)
            else:
                
                self.id2edge = test_id2edge
                self.edge2id = {self.id2edge[i]: i for i in range(len(self.id2edge))}
                self.r_idx = len(self.edge2id) # start from what we have seen
                for i in kg_edge_type_str+task_edge_type_str:
                    
                    if i not in self.edge2id:
                        self.edge2id[i] = self.r_idx
                        self.id2edge.append(i.strip())
                        self.r_idx += 1
               

            # store for valid and test files to retrieve
            if self.reuse:
                self.writefiles()
        
        kg_src = torch.tensor([self.node2id[i] for i in kg_src_str])
        kg_dst = torch.tensor([self.node2id[i] for i in kg_dst_str])
        self.kg_edge_type = torch.tensor([self.edge2id[i] for i in kg_edge_type_str])
        

        self.task_src = torch.tensor([self.node2id[i] for i in task_src_str])
        self.task_dst = torch.tensor([self.node2id[i] for i in task_dst_str])
        self.task_edge_type = torch.tensor([self.edge2id[i] for i in task_edge_type_str])
        
        
        self.kg_edge_index = torch.stack([kg_src, kg_dst], dim=0)
        self.task_edge_index = torch.stack([self.task_src, self.task_dst], dim=0)
        self.kg_num_triples = self.kg_edge_index.shape[1]
        self.task_num_triples = self.task_edge_index.shape[1]

        
        ## find common neighbors
        if args.common:

            self.create_induced_subgraphs() # all upto self.hop [not separating into 1 hop and two hop]
            self.count_t = []
            self.find_common_neighbors('task') # needs to be called after getting kg node subgraphs
            # print(f'MAX: {max(self.count_t)}, MIN: {min(self.count_t)}, AVG: {sum(self.count_t)/len(self.count_t)}')
            # c_0 = [1 for i in self.count_t if i==0]
            # c_1 = [1 for i in self.count_t if i==1]
            # print(f'# of Samples with 0: {sum(c_0)}, # of samples with 1: {sum(c_1)}')
            if not self.dataset.endswith('_ind'):
                self.count_k = []
                self.find_common_neighbors('kg')
                # print('FOR KG')
                # print(f'MAX: {max(self.count_k)}, MIN: {min(self.count_k)}, AVG: {sum(self.count_k)/len(self.count_k)}')
                # c_0 = [1 for i in self.count_k if i==0]
                # c_1 = [1 for i in self.count_k if i==1]
                # print(f'# of Samples with 0: {sum(c_0)}, # of samples with 1: {sum(c_1)}')
            self.gather_corrupted_subg = self.gather_common
        else:
            self.create_induced_subgraphs_hopwise()
            self.combine_neighborhoods_lg_hopwise('task') # needs to be called after getting kg node subgraphs
            if not self.dataset.endswith('_ind'):
                self.combine_neighborhoods_lg_hopwise('kg') # needs to be called after getting kg node subgraphs
            self.gather_corrupted_subg = self.gather_all
        self.path_list = {'kg': dict(), 'task': dict()}
        
        if self.up:
            
            self.create_paths()
        if not args.common:
            if self.hop == 1:
                self.node_subg = [self.kg_node_subgraphs_1[i] for i in range(len(self.kg_node_subgraphs_1))]
            if self.hop == 2:
                self.node_subg = [torch.cat((self.kg_node_subgraphs_1[i], self.kg_node_subgraphs_2[i]), dim=0) for i in range(len(self.kg_node_subgraphs_1))] 
            if self.hop == 3:
                self.node_subg = [torch.cat((self.kg_node_subgraphs_1[i], self.kg_node_subgraphs_2[i], self.kg_node_subgraphs_3[i]), dim=0) for i in range(len(self.kg_node_subgraphs_1))]
            if self.hop == 4:
                self.node_subg = [torch.cat((self.kg_node_subgraphs_1[i], self.kg_node_subgraphs_2[i], self.kg_node_subgraphs_3[i], self.kg_node_subgraphs_4[i]), dim=0) for i in range(len(self.kg_node_subgraphs_1))]
        if args.rp:
            self.get_neg_triples = self.get_neg_samples_rel
        elif args.hneg:
            self.get_neg_triples = self.get_neg_triples_hneg
            self.create_choice_entities()
        else:
            self.get_neg_triples = self.get_neg_triples_basic

    
    def create_choice_entities(self):
        # for a given hr pair or tr pair, it lists  what entities are valid corruption choices
        st_ = time.time()
        hr_pairs = torch.stack([self.task_edge_index[0,:], self.task_edge_type], dim=1) # [-, 2]
        tr_pairs = torch.stack([self.task_edge_index[1,:], self.task_edge_type], dim=1) # [-, 2]
        t_avoid=dict() # key: [head, rel], value: [list of true tails]
        h_avoid=dict() # key: [tair, rel], value: [list of true heads]
        
        for i in range(self.task_edge_type.shape[0]):
            if tuple(hr_pairs[i,:].tolist()) in t_avoid:
                t_avoid[tuple(hr_pairs[i,:].tolist())].append(self.task_edge_index[1,:][i].item())
            else:
                t_avoid[tuple(hr_pairs[i,:].tolist())] = [self.task_edge_index[1,:][i]]

            if tuple(tr_pairs[i,:].tolist()) in h_avoid:
                h_avoid[tuple(tr_pairs[i,:].tolist())].append(self.task_edge_index[0,:][i].item())
            else:
                h_avoid[tuple(tr_pairs[i,:].tolist())] = [self.task_edge_index[0,:][i].item()]
        self.t_choice_task = dict()
        self.h_choice_task = dict()
        for i in t_avoid.keys():
            self.t_choice_task[i] = [k for k in range(self.e_idx) if k not in t_avoid[i]]
        for i in h_avoid.keys():
            self.h_choice_task[i] = [k for k in range(self.e_idx) if k not in h_avoid[i]]
        
        if not self.dataset.endswith('_ind'):
            hr_pairs = torch.stack([self.kg_edge_index[0,:], self.kg_edge_type], dim=1) # [-, 2]
            tr_pairs = torch.stack([self.kg_edge_index[1,:], self.kg_edge_type], dim=1) # [-, 2]
            t_avoid=dict() # key: [head, rel], value: [list of true tails]
            h_avoid=dict() # key: [tair, rel], value: [list of true heads]
            
            for i in range(self.kg_edge_type.shape[0]):
                if tuple(hr_pairs[i,:].tolist()) in t_avoid:
                    t_avoid[tuple(hr_pairs[i,:].tolist())].append(self.kg_edge_index[1,:][i].item())
                else:
                    t_avoid[tuple(hr_pairs[i,:].tolist())] = [self.kg_edge_index[1,:][i]]

                if tuple(tr_pairs[i,:].tolist()) in h_avoid:
                    h_avoid[tuple(tr_pairs[i,:].tolist())].append(self.kg_edge_index[0,:][i].item())
                else:
                    h_avoid[tuple(tr_pairs[i,:].tolist())] = [self.kg_edge_index[0,:][i].item()]
            self.t_choice_kg = dict()
            self.h_choice_kg = dict()
            for i in t_avoid.keys():
                self.t_choice_kg[i] = [k for k in range(self.e_idx) if k not in t_avoid[i]]
            for i in h_avoid.keys():
                self.h_choice_kg[i] = [k for k in range(self.e_idx) if k not in h_avoid[i]]
        print(f'Time spent in Create Choice Entities method: {time.time()-st_}')
        #TODO save and load options since this is fixed for a dataset
        return
    def create_paths(self):
        st_ = time.time()
        print('Inside create_paths')
        filename_1 = './data/' + self.dataset + '/paths_mpl' + str(self.max_path_len) + '.pkl'
        filename_2 = './data/' + self.dataset + '/map.pkl'
        self.KG = nx.DiGraph()
        self.KG.add_edges_from(self.kg_edge_index.T.tolist() + self.kg_edge_index.T[:,[1,0]].tolist()) # add edges from both directions

        if self.reuse and os.path.isfile(filename_1) and os.path.isfile(filename_2):
            self.path_list = pickle.load(open(filename_1, 'rb'))
            self.ht_rel_map = pickle.load(open(filename_2, 'rb'))
        else:
            print(f'Did not find {filename_1} and {filename_2}. Creating now.')
            self.ht_rel_map = dict()
            for i in range(self.kg_edge_type.shape[0]):
                self.ht_rel_map[(self.kg_edge_index[0, i].item(), self.kg_edge_index[1, i].item())] = self.kg_edge_type[i].item()
                reverse_pair = torch.tensor([self.kg_edge_index[1, i].item(), self.kg_edge_index[0, i].item()])
                if not (reverse_pair.unsqueeze(1)==self.kg_edge_index).all(dim=0).any():
                # check if such a pair <t,-,h> is in original triple set
                    self.ht_rel_map[(self.kg_edge_index[1, i].item(), self.kg_edge_index[0, i].item())] = self.kg_edge_type[i].item() + self.r_idx # add reverse edge
            
            self.path_list['task'] = {tuple(k):[] for k in self.task_edge_index.T.tolist()}
            
            
            for pair in tqdm(self.task_edge_index.T.tolist()):
                # do not remove the target rel; this way we always have one len-1 path
                nx_paths = nx.all_simple_paths(self.KG, source=pair[0], target=pair[1], cutoff=self.max_path_len)
                for path in map(nx.utils.pairwise, nx_paths):
                    p = [self.ht_rel_map[i] for i in list(path)]
                    if p not in self.path_list['task'][tuple(pair)]:
                        self.path_list['task'][tuple(pair)].append(p)
                
            if not self.dataset.endswith('_ind'):
                self.path_list['kg'] = {tuple(k):[] for k in self.kg_edge_index.T.tolist()}
                for pair in tqdm(self.kg_edge_index.T.tolist()):
                    nx_paths = nx.all_simple_paths(self.KG, source=pair[0], target=pair[1], cutoff=self.max_path_len)
                    for path in map(nx.utils.pairwise, nx_paths):
                        p = [self.ht_rel_map[i] for i in list(path)]
                        if p not in self.path_list['kg'][tuple(pair)]:
                            self.path_list['kg'][tuple(pair)].append(p)
            if self.reuse:
                pickle.dump(self.path_list, open(filename_1, 'wb'))
                pickle.dump(self.ht_rel_map, open(filename_2, 'wb'))
        print(f'Exiting create_paths; time: {time.time()-st_}')
        return
            


    
    def create_induced_subgraphs_hopwise(self):
        
        if self.reuse:
            filename_1 = './data/' + self.dataset + '/kg_node_subgraphs_sep_1.pt'
            filename_2 = './data/' + self.dataset + '/kg_node_subgraphs_sep_2.pt'
            filename_3 = './data/' + self.dataset + '/kg_node_subgraphs_sep_3.pt'
            filename_4 = './data/' + self.dataset + '/kg_node_subgraphs_sep_4.pt'
            if self.hop == 1 and os.path.isfile(filename_1):
                self.kg_node_subgraphs_1 = torch.load(filename_1)
                return 
            if self.hop == 2 and os.path.isfile(filename_1) and os.path.isfile(filename_2):
                self.kg_node_subgraphs_1 = torch.load(filename_1)
                self.kg_node_subgraphs_2 = torch.load(filename_2)
                return
            if self.hop == 3 and os.path.isfile(filename_1) and os.path.isfile(filename_2) and os.path.isfile(filename_3):
                self.kg_node_subgraphs_1 = torch.load(filename_1)
                self.kg_node_subgraphs_2 = torch.load(filename_2)
                self.kg_node_subgraphs_3 = torch.load(filename_3)
                return 
            if self.hop == 4 and os.path.isfile(filename_1) and os.path.isfile(filename_2) and os.path.isfile(filename_3) and os.path.isfile(filename_4):
                self.kg_node_subgraphs_1 = torch.load(filename_1)
                self.kg_node_subgraphs_2 = torch.load(filename_2)
                self.kg_node_subgraphs_3 = torch.load(filename_3)
                self.kg_node_subgraphs_4 = torch.load(filename_4)
                return 
            
        
        edge_all = torch.cat((self.kg_edge_index, torch.stack([self.kg_edge_index[1], self.kg_edge_index[0]])), dim=1)
        self.kg_node_subgraphs_1 = []
        self.kg_node_subgraphs_2 = []
        self.kg_node_subgraphs_3 = []
        self.kg_node_subgraphs_4 = []
        print(f'Did not find kg_node_subgraphs_sep_{self.hop} files. Creaing now.')
        
        for node_id in tqdm(range(self.e_idx)):
            
            mask_1 = k_hop(node_id, 1, edge_all)[3][:self.kg_num_triples]
            if self.hop > 1:
                mask_2 = k_hop(node_id, 2, edge_all)[3][:self.kg_num_triples] # contains both one and two hops
                if self.hop > 2:
                    mask_3 = k_hop(node_id, 3, edge_all)[3][:self.kg_num_triples] # contains 1,2,3 all hop neighbors
                    if self.hop > 3:
                        mask_4 = k_hop(node_id, 4, edge_all)[3][:self.kg_num_triples]
                        mask_4 = torch.logical_xor(mask_3, mask_4) # only hop 4
                    mask_3 = torch.logical_xor(mask_2, mask_3) # mask_3 has to be computed using XOR before mask2, so we get rid of 1 and 2 hop at once
                mask_2 = torch.logical_xor(mask_1, mask_2) # k_hop returns all of 1+2 hop when k=2; remove hop1 and only hop2 remains.
            

            idx = mask_1.nonzero()# idx of edges that are in the subgraph
            rels = self.kg_edge_type[idx] # relations in the subgraph
            heads = self.kg_edge_index[0, idx]
            tails = self.kg_edge_index[1, idx]
            triples = torch.tensor(list(zip(heads, rels, tails))) # tensor [M,3] where M is number of triples in **1-hop** neighborhood
            self.kg_node_subgraphs_1.append(triples)

            if self.hop > 1:
                # repeat, using same variables to save space
                idx = mask_2.nonzero()# idx of edges that are in the subgraph
                rels = self.kg_edge_type[idx] # relations in the subgraph
                heads = self.kg_edge_index[0, idx]
                tails = self.kg_edge_index[1, idx]
                triples = torch.tensor(list(zip(heads, rels, tails))) # tensor [M,3] where M is number of triples in **1-hop** neighborhood
                self.kg_node_subgraphs_2.append(triples)

            if self.hop > 2:
                # repeat, using same variables to save space
                idx = mask_3.nonzero()# idx of edges that are in the subgraph
                rels = self.kg_edge_type[idx] # relations in the subgraph
                heads = self.kg_edge_index[0, idx]
                tails = self.kg_edge_index[1, idx]
                triples = torch.tensor(list(zip(heads, rels, tails))) # tensor [M,3] where M is number of triples in **1-hop** neighborhood
                self.kg_node_subgraphs_3.append(triples)
            if self.hop > 3:
                # repeat, using same variables to save space
                idx = mask_4.nonzero()# idx of edges that are in the subgraph
                rels = self.kg_edge_type[idx] # relations in the subgraph
                heads = self.kg_edge_index[0, idx]
                tails = self.kg_edge_index[1, idx]
                triples = torch.tensor(list(zip(heads, rels, tails))) # tensor [M,3] where M is number of triples in **1-hop** neighborhood
                self.kg_node_subgraphs_4.append(triples)
        
        if self.reuse:
            torch.save(self.kg_node_subgraphs_1, open(filename_1, 'wb'))
            if self.hop > 1:
                torch.save(self.kg_node_subgraphs_2, open(filename_2, 'wb'))
            if self.hop > 2:
                torch.save(self.kg_node_subgraphs_3, open(filename_3, 'wb'))
            if self.hop > 3:
                torch.save(self.kg_node_subgraphs_4, open(filename_4, 'wb'))
        if len(self.kg_node_subgraphs_1) != self.e_idx:
            raise ValueError('Inside induced subgraphs')
        return
    def create_induced_subgraphs(self):
        st_ = time.time()
        print('Inside create_induced_subgraphs')
        edge_all = torch.cat((self.kg_edge_index, torch.stack([self.kg_edge_index[1], self.kg_edge_index[0]])), dim=1)
        self.kg_node_subgraphs_upto_hop = []
        # print(f'Creating UPTO neighbors for each node using k_hop.')
        
        for node_id in tqdm(range(self.e_idx)):
            
            mask = k_hop(node_id, self.hop, edge_all)[3][:self.kg_num_triples]
            

            idx = mask.nonzero()# idx of edges that are in the subgraph
            rels = self.kg_edge_type[idx] # relations in the subgraph
            heads = self.kg_edge_index[0, idx]
            tails = self.kg_edge_index[1, idx]
            triples = torch.tensor(list(zip(heads, rels, tails))) # tensor [M,3] where M is number of triples in **1-hop** neighborhood
            self.kg_node_subgraphs_upto_hop.append(triples)
        print(f'Exiting create_induced_subgraphs; time: {time.time()-st_}')
        return
    def combine_neighborhoods_lg_hopwise(self, mode):
        # combines the neighborhood of a triple's head and tail to create line graphs later
        # for line graphs we do not remove target triple yet
        # in dataloader we choose some neighbor triples and make sure target triple is included
        # then we create the line graph and mask out its label
        st_ = time.time()
        print(f'Mode: {mode}; Inside combine_neighborhoods_lg_hopwise')
        if mode=='task':
            task_filename_1 = './data/' + self.dataset + '/task_triple_neighborhoods_sep_1.pt'
            task_filename_2 = './data/' + self.dataset + '/task_triple_neighborhoods_sep_2.pt'
            task_filename_3 = './data/' + self.dataset + '/task_triple_neighborhoods_sep_3.pt'
            task_filename_4 = './data/' + self.dataset + '/task_triple_neighborhoods_sep_4.pt'

            # type_filename_1 = './data/' + self.dataset + '/task_type_sep_1.pt'
            # type_filename_2 = './data/' + self.dataset + '/task_type_sep_2.pt'
            # type_filename_3 = './data/' + self.dataset + '/task_type_sep_3.pt'
            # type_filename_4 = './data/' + self.dataset + '/task_type_sep_4.pt'
            if self.reuse:
                if self.hop==4 and os.path.isfile(task_filename_1) and os.path.isfile(task_filename_2) and os.path.isfile(task_filename_3) and os.path.isfile(task_filename_4):
                    self.task_triple_subgraphs_1 = torch.load(task_filename_1)
                    self.task_triple_subgraphs_2 = torch.load(task_filename_2)
                    self.task_triple_subgraphs_3 = torch.load(task_filename_3)
                    self.task_triple_subgraphs_4 = torch.load(task_filename_4)

                    # self.task_triple_type_1 = torch.load(type_filename_1)
                    # self.task_triple_type_2 = torch.load(type_filename_2)
                    # self.task_triple_type_3 = torch.load(type_filename_3)
                    # self.task_triple_type_4 = torch.load(type_filename_4)
                    return
                
                if self.hop==3 and os.path.isfile(task_filename_1) and os.path.isfile(task_filename_2) and os.path.isfile(task_filename_3):
                    self.task_triple_subgraphs_1 = torch.load(task_filename_1)
                    self.task_triple_subgraphs_2 = torch.load(task_filename_2)
                    self.task_triple_subgraphs_3 = torch.load(task_filename_3)

                    # self.task_triple_type_1 = torch.load(type_filename_1)
                    # self.task_triple_type_2 = torch.load(type_filename_2)
                    # self.task_triple_type_3 = torch.load(type_filename_3)
                    return
                if self.hop==2 and os.path.isfile(task_filename_1) and os.path.isfile(task_filename_2):
                    self.task_triple_subgraphs_1 = torch.load(task_filename_1)
                    self.task_triple_subgraphs_2 = torch.load(task_filename_2)

                    # self.task_triple_type_1 = torch.load(type_filename_1)
                    # self.task_triple_type_2 = torch.load(type_filename_2)
                    return
                if self.hop==1 and os.path.isfile(task_filename_1):
                    self.task_triple_subgraphs_1 = torch.load(task_filename_1)

                    # self.task_triple_type_1 = torch.load(type_filename_1)
                    return
            
            self.task_triple_subgraphs_1 = []
            self.task_triple_subgraphs_2 = []
            self.task_triple_subgraphs_3 = []
            self.task_triple_subgraphs_4 = []
            
            self.task_triple_type_1 = []
            self.task_triple_type_2 = []
            self.task_triple_type_3 = []
            self.task_triple_type_4 = []
            
            for pair in tqdm(self.task_edge_index.T): # access columns
                
                h_triples = self.kg_node_subgraphs_1[pair[0]] # tuples of h,r,t from one hop neighbors of h
                t_triples = self.kg_node_subgraphs_1[pair[1]] # tuples of h,r,t from one hop neighbors of t
                combined_subg = torch.unique(torch.cat((h_triples, t_triples), dim=0), dim=0) # shape: M x 3, M = number of unique triples in combined subgraph
                
                # Keeping track of neighborhood type
                # h_n_idx = (h_triples.unsqueeze(1)==combined_subg).all(dim=2).nonzero()[:,1]
                # t_n_idx = (t_triples.unsqueeze(1)==combined_subg).all(dim=2).nonzero()[:,1]
                # type_idx = torch.ones(combined_subg.shape[0], dtype=torch.long) # 1: common, 2: h's neighb., 0: t's neighb.
                # type_idx[h_n_idx] += 1
                # type_idx[t_n_idx] -= 1
                
                # we are using the kg triples for subgraphs of task triples
                # NOTE we have verified that across the inductive datasets no test triple is present in the train files
                # we can omit the check for whether the target triple is included or not inside dataloader:getitem
                # we simply always put the target triple in the 0th position in each batch's in sample's neighborhood
                self.task_triple_subgraphs_1.append(combined_subg)
                # self.task_triple_type_1.append(type_idx)

                if self.hop > 1:
                    h_triples = self.kg_node_subgraphs_2[pair[0]] # tuples of h,r,t from two hop neighbors of h
                    t_triples = self.kg_node_subgraphs_2[pair[1]] # tuples of h,r,t from two hop neighbors of t
                    combined_subg = torch.unique(torch.cat((h_triples, t_triples), dim=0), dim=0) # shape: M x 3, M = number of unique triples in combined subgraph
                    self.task_triple_subgraphs_2.append(combined_subg)

                    # h_n_idx = (h_triples.unsqueeze(1)==combined_subg).all(dim=2).nonzero()[:,1]
                    # t_n_idx = (t_triples.unsqueeze(1)==combined_subg).all(dim=2).nonzero()[:,1]
                    # type_idx = torch.ones(combined_subg.shape[0], dtype=torch.long) # 1: common, 2: h's neighb., 0: t's neighb.
                    # type_idx[h_n_idx] += 1
                    # type_idx[t_n_idx] -= 1
                    # self.task_triple_type_2.append(type_idx)

                if self.hop > 2:
                    h_triples = self.kg_node_subgraphs_3[pair[0]] # tuples of h,r,t from two hop neighbors of h
                    t_triples = self.kg_node_subgraphs_3[pair[1]] # tuples of h,r,t from two hop neighbors of t
                    combined_subg = torch.unique(torch.cat((h_triples, t_triples), dim=0), dim=0) # shape: M x 3, M = number of unique triples in combined subgraph
                    self.task_triple_subgraphs_3.append(combined_subg)

                    # h_n_idx = (h_triples.unsqueeze(1)==combined_subg).all(dim=2).nonzero()[:,1]
                    # t_n_idx = (t_triples.unsqueeze(1)==combined_subg).all(dim=2).nonzero()[:,1]
                    # type_idx = torch.ones(combined_subg.shape[0], dtype=torch.long) # 1: common, 2: h's neighb., 0: t's neighb.
                    # type_idx[h_n_idx] += 1
                    # type_idx[t_n_idx] -= 1
                    # self.task_triple_type_3.append(type_idx)

                if self.hop > 3:
                    h_triples = self.kg_node_subgraphs_4[pair[0]] # tuples of h,r,t from two hop neighbors of h
                    t_triples = self.kg_node_subgraphs_4[pair[1]] # tuples of h,r,t from two hop neighbors of t
                    combined_subg = torch.unique(torch.cat((h_triples, t_triples), dim=0), dim=0) # shape: M x 3, M = number of unique triples in combined subgraph
                    self.task_triple_subgraphs_4.append(combined_subg)

                    # h_n_idx = (h_triples.unsqueeze(1)==combined_subg).all(dim=2).nonzero()[:,1]
                    # t_n_idx = (t_triples.unsqueeze(1)==combined_subg).all(dim=2).nonzero()[:,1]
                    # type_idx = torch.ones(combined_subg.shape[0], dtype=torch.long) # 1: common, 2: h's neighb., 0: t's neighb.
                    # type_idx[h_n_idx] += 1
                    # type_idx[t_n_idx] -= 1
                    # self.task_triple_type_4.append(type_idx)

            if self.reuse:
                torch.save(self.task_triple_subgraphs_1, open(task_filename_1, 'wb'))
                # torch.save(self.task_triple_type_1, open(type_filename_1, 'wb'))
                if self.hop > 1:
                    torch.save(self.task_triple_subgraphs_2, open(task_filename_2, 'wb'))
                    # torch.save(self.task_triple_type_2, open(type_filename_2, 'wb'))
                if self.hop > 2:    
                    torch.save(self.task_triple_subgraphs_3, open(task_filename_3, 'wb'))
                    # torch.save(self.task_triple_type_3, open(type_filename_3, 'wb'))
                if self.hop > 3:    
                    torch.save(self.task_triple_subgraphs_4, open(task_filename_4, 'wb'))
                    # torch.save(self.task_triple_type_4, open(type_filename_4, 'wb'))
            print(f'Exiting combine_neighborhoods_lg_hopwise; time: {time.time()-st_}')
            return
        elif mode=='kg':
            kg_filename_1 = './data/' + self.dataset + '/kg_triple_neighborhoods_sep_1.pt'
            kg_filename_2 = './data/' + self.dataset + '/kg_triple_neighborhoods_sep_2.pt'
            kg_filename_3 = './data/' + self.dataset + '/kg_triple_neighborhoods_sep_3.pt'
            kg_filename_4 = './data/' + self.dataset + '/kg_triple_neighborhoods_sep_4.pt'

            # type_filename_1 = './data/' + self.dataset + '/kg_type_sep_1.pt'
            # type_filename_2 = './data/' + self.dataset + '/kg_type_sep_2.pt'
            # type_filename_3 = './data/' + self.dataset + '/kg_type_sep_3.pt'
            # type_filename_4 = './data/' + self.dataset + '/kg_type_sep_4.pt'
            if self.reuse:
                if self.hop==4 and os.path.isfile(kg_filename_1) and os.path.isfile(kg_filename_2) and os.path.isfile(kg_filename_3) and os.path.isfile(kg_filename_4):
                    self.kg_triple_subgraphs_1 = torch.load(kg_filename_1) 
                    self.kg_triple_subgraphs_2 = torch.load(kg_filename_2)
                    self.kg_triple_subgraphs_3 = torch.load(kg_filename_3)
                    self.kg_triple_subgraphs_4 = torch.load(kg_filename_4)

                    # self.kg_triple_type_1 = torch.load(type_filename_1)
                    # self.kg_triple_type_2 = torch.load(type_filename_2)
                    # self.kg_triple_type_3 = torch.load(type_filename_3)
                    # self.kg_triple_type_4 = torch.load(type_filename_4)
                    return
                
                if self.hop==3 and os.path.isfile(kg_filename_1) and os.path.isfile(kg_filename_2) and os.path.isfile(kg_filename_3):
                    self.kg_triple_subgraphs_1 = torch.load(kg_filename_1) 
                    self.kg_triple_subgraphs_2 = torch.load(kg_filename_2)
                    self.kg_triple_subgraphs_3 = torch.load(kg_filename_3)

                    # self.kg_triple_type_1 = torch.load(type_filename_1)
                    # self.kg_triple_type_2 = torch.load(type_filename_2)
                    # self.kg_triple_type_3 = torch.load(type_filename_3)
                    return
                if self.hop==2 and os.path.isfile(kg_filename_1) and os.path.isfile(kg_filename_2):
                    self.kg_triple_subgraphs_1 = torch.load(kg_filename_1) 
                    self.kg_triple_subgraphs_2 = torch.load(kg_filename_2)

                    # self.kg_triple_type_1 = torch.load(type_filename_1)
                    # self.kg_triple_type_2 = torch.load(type_filename_2)
                    return
                if self.hop==1 and os.path.isfile(kg_filename_1):
                    self.kg_triple_subgraphs_1 = torch.load(kg_filename_1)

                    # self.kg_triple_type_1 = torch.load(type_filename_1)
                    return
            
            self.kg_triple_subgraphs_1 = []
            self.kg_triple_subgraphs_2 = []
            self.kg_triple_subgraphs_3 = []
            self.kg_triple_subgraphs_4 = []

            self.kg_triple_type_1 = []
            self.kg_triple_type_2 = []
            self.kg_triple_type_3 = []
            self.kg_triple_type_4 = []


            for pair in tqdm(self.kg_edge_index.T): # access columns
                
                h_triples = self.kg_node_subgraphs_1[pair[0]] # tuples of h,r,t of one hop neighbors of h
                t_triples = self.kg_node_subgraphs_1[pair[1]] # tuples of h,r,t of one hop neighbors of t
                combined_subg = torch.unique(torch.cat((h_triples, t_triples), dim=0), dim=0) # shape: M x 3, M = number of unique triples in combined subgraph
                
                
                # TODO: we cannot omit the check for whether the target triple is included or not inside dataloader:getitem
                # we simply always put the target triple in the 0th position in each batch's in sample's neighborhood
                self.kg_triple_subgraphs_1.append(combined_subg)

                # h_n_idx = (h_triples.unsqueeze(1)==combined_subg).all(dim=2).nonzero()[:,1]
                # t_n_idx = (t_triples.unsqueeze(1)==combined_subg).all(dim=2).nonzero()[:,1]
                # type_idx = torch.ones(combined_subg.shape[0], dtype=torch.long) # 1: common, 2: h's neighb., 0: t's neighb.
                # type_idx[h_n_idx] += 1
                # type_idx[t_n_idx] -= 1
                # self.kg_triple_type_1.append(type_idx)

                if self.hop > 1:
                    h_triples = self.kg_node_subgraphs_2[pair[0]] # tuples of h,r,t of two hop neighbors of h
                    t_triples = self.kg_node_subgraphs_2[pair[1]] # tuples of h,r,t of two hop neighbors of t
                    combined_subg = torch.unique(torch.cat((h_triples, t_triples), dim=0), dim=0) # shape: M x 3, M = number of unique triples in combined subgraph
                    self.kg_triple_subgraphs_2.append(combined_subg)

                    # h_n_idx = (h_triples.unsqueeze(1)==combined_subg).all(dim=2).nonzero()[:,1]
                    # t_n_idx = (t_triples.unsqueeze(1)==combined_subg).all(dim=2).nonzero()[:,1]
                    # type_idx = torch.ones(combined_subg.shape[0], dtype=torch.long) # 1: common, 2: h's neighb., 0: t's neighb.
                    # type_idx[h_n_idx] += 1
                    # type_idx[t_n_idx] -= 1
                    # self.kg_triple_type_2.append(type_idx)

                if self.hop > 2:
                    h_triples = self.kg_node_subgraphs_3[pair[0]] # tuples of h,r,t of two hop neighbors of h
                    t_triples = self.kg_node_subgraphs_3[pair[1]] # tuples of h,r,t of two hop neighbors of t
                    combined_subg = torch.unique(torch.cat((h_triples, t_triples), dim=0), dim=0) # shape: M x 3, M = number of unique triples in combined subgraph
                    self.kg_triple_subgraphs_3.append(combined_subg)

                    # h_n_idx = (h_triples.unsqueeze(1)==combined_subg).all(dim=2).nonzero()[:,1]
                    # t_n_idx = (t_triples.unsqueeze(1)==combined_subg).all(dim=2).nonzero()[:,1]
                    # type_idx = torch.ones(combined_subg.shape[0], dtype=torch.long) # 1: common, 2: h's neighb., 0: t's neighb.
                    # type_idx[h_n_idx] += 1
                    # type_idx[t_n_idx] -= 1
                    # self.kg_triple_type_3.append(type_idx)

                if self.hop > 3:
                    h_triples = self.kg_node_subgraphs_4[pair[0]] # tuples of h,r,t of two hop neighbors of h
                    t_triples = self.kg_node_subgraphs_4[pair[1]] # tuples of h,r,t of two hop neighbors of t
                    combined_subg = torch.unique(torch.cat((h_triples, t_triples), dim=0), dim=0) # shape: M x 3, M = number of unique triples in combined subgraph
                    self.kg_triple_subgraphs_4.append(combined_subg)

                    h_n_idx = (h_triples.unsqueeze(1)==combined_subg).all(dim=2).nonzero()[:,1]
                    t_n_idx = (t_triples.unsqueeze(1)==combined_subg).all(dim=2).nonzero()[:,1]
                    type_idx = torch.ones(combined_subg.shape[0], dtype=torch.long) # 1: common, 2: h's neighb., 0: t's neighb.
                    type_idx[h_n_idx] += 1
                    type_idx[t_n_idx] -= 1
                    self.kg_triple_type_4.append(type_idx)
            if self.reuse:
                torch.save(self.kg_triple_subgraphs_1, open(kg_filename_1, 'wb'))
                # torch.save(self.kg_triple_type_1, open(type_filename_1, 'wb'))
                if self.hop > 1:
                    torch.save(self.kg_triple_subgraphs_2, open(kg_filename_2, 'wb'))
                    # torch.save(self.kg_triple_type_2, open(type_filename_2, 'wb'))
                if self.hop > 2:
                    torch.save(self.kg_triple_subgraphs_3, open(kg_filename_3, 'wb'))
                    # torch.save(self.kg_triple_type_3, open(type_filename_3, 'wb'))
                if self.hop > 3:
                    torch.save(self.kg_triple_subgraphs_4, open(kg_filename_4, 'wb'))
                    # torch.save(self.kg_triple_type_4, open(type_filename_4, 'wb'))
            print(f'Exiting combine_neighborhoods_lg_hopwise; time: {time.time()-st_}')
            return
        
    def find_common_neighbors(self, mode):
        # combines the neighborhood of a triple's head and tail to create line graphs later
        # for line graphs we do not remove target triple yet
        # in dataloader we choose some neighbor triples and make sure target triple is included
        # then we create the line graph and mask out its label
        st_ = time.time()
        print(f'mode: {mode}; inside find_common_neighbors')
        if mode=='task':
            
            
            kg_filename = './data/' + self.dataset + '/task_common_'+str(self.hop)+'.pt'
            if self.reuse and os.path.isfile(kg_filename):
                self.task_common_neighbors = torch.load(kg_filename)
                return
            self.task_common_neighbors = []
            for  pair in tqdm(self.task_edge_index.T): # access columns
                h_triples = self.kg_node_subgraphs_upto_hop[pair[0]]
                t_triples = self.kg_node_subgraphs_upto_hop[pair[1]]
                # shape: M x 3, M = number of common triples in common subgraph
                common_subg = h_triples[(h_triples.unsqueeze(dim=1) == t_triples).all(dim=-1).any(dim=1)] 
                self.task_common_neighbors.append(common_subg)
                self.count_t.append(common_subg.shape[0])
            if self.reuse:
                torch.save(self.task_common_neighbors, open(kg_filename, 'wb'))
            print(f'Exiting find_common_neighbors; time: {time.time()-st_}')
            return
        elif mode=='kg':
            kg_filename = './data/' + self.dataset + '/kg_common_'+str(self.hop)+'.pt'
            if self.reuse and os.path.isfile(kg_filename):
                self.kg_common_neighbors = torch.load(kg_filename)
                return
            self.kg_common_neighbors = []
            for pair in tqdm(self.kg_edge_index.T): # access columns
                
                h_triples = self.kg_node_subgraphs_upto_hop[pair[0]]
                t_triples = self.kg_node_subgraphs_upto_hop[pair[1]]
                common_subg = h_triples[(h_triples.unsqueeze(dim=1) == t_triples).all(dim=-1).any(dim=1)]
                self.kg_common_neighbors.append(common_subg)
                self.count_k.append(common_subg.shape[0])
            if self.reuse:
                torch.save(self.kg_common_neighbors, open(kg_filename, 'wb'))
            print(f'Exiting find_common_neighbors; time: {time.time()-st_}')
            return
    
    def gather_common(self, neg_triples, num_samples, m):
        h_subg = [self.kg_node_subgraphs_upto_hop[i] for i in neg_triples[:,0]] # list of tensors, list len: batch_size * num_neg, tensor shape [*, 3]
        t_subg = [self.kg_node_subgraphs_upto_hop[i] for i in neg_triples[:,2]] # list of tensors, list len: batch_size * num_neg, tensor shape [*, 3]

        flag = [(h_subg[i].unsqueeze(dim=1)==t_subg[i]).all(dim=-1).any(dim=-1).any() for i in range(num_samples)]
        selected_with_true = []
        # Apparently loop with this flag check is consistently faster than list comprehension and also loop w/o flag
        # print('num_samples', num_samples)
        # for i in range(num_samples):
        #     if flag[i]:
        #         selected_with_true.append(torch.cat((neg_triples[i].unsqueeze(dim=0), \
        #                     h_subg[i][(h_subg[i].unsqueeze(dim=1)==t_subg[i]).all(dim=-1).any(dim=-1)]), dim=0)[:m])
        #     else:
        #         selected_with_true.append(neg_triples[i].unsqueeze(dim=0))
                
        # Add option to add more if space left

        selected_with_true = [torch.cat((neg_triples[i].unsqueeze(dim=0), h_subg[i][(h_subg[i].unsqueeze(dim=1)==\
                    t_subg[i]).all(dim=-1).any(dim=-1)]), dim=0)[:m] for i in range(num_samples)]
        
        return selected_with_true

    def gather_all(self, neg_triples, num_samples, m):
        

        h_subg = [self.node_subg[i] for i in neg_triples[:,0]] # list of tensors, list len: batch_size * num_neg, tensor shape [*, 3]
        t_subg = [self.node_subg[i] for i in neg_triples[:,2]] # list of tensors, list len: batch_size * num_neg, tensor shape [*, 3]
        selected_with_true = [  torch.cat((neg_triples[i].unsqueeze(dim=0), \
                                torch.unique(torch.cat((h_subg[i][:m], t_subg[i][:m]), dim=0), dim=0)))[:m] \
                                for i in range(num_samples)] 
        return selected_with_true
    def get_svd_pe(self, selected_with_true, k, m, type_info):

        og_unique_nodes = [torch.cat((a[:,0],a[:,2]),dim=0).unique() for a in selected_with_true]
        e1_idx = [(og_unique_nodes[i] == selected_with_true[i][:,0].unsqueeze(1)).nonzero()[:,1] for i in range(len(selected_with_true))]
        e2_idx = [(og_unique_nodes[i] == selected_with_true[i][:,2].unsqueeze(1)).nonzero()[:,1] for i in range(len(selected_with_true))]
        og_n = [i.numel() for i in og_unique_nodes]
        adj_list = [torch.eye(og_n[i], og_n[i]) for i in range(len(selected_with_true))] # Use eye for svd computation
        for i in range(len(adj_list)):
            adj_list[i][e1_idx[i], e2_idx[i]]=1
            adj_list[i][e2_idx[i], e1_idx[i]]=1
        # TODO pad, run batched svd
        singular_values = [torch.linalg.svd(a) for a in adj_list]


        singular_values = [torch.stack([torch.cat((singular_values[i][0][e1_idx[i], :k], singular_values[i][2].T[e1_idx[i], :k]), dim=1), \
                                        torch.cat((singular_values[i][0][e2_idx[i], :k], singular_values[i][2].T[e2_idx[i], :k]), dim=1)]) \
                                        for i in range(len(singular_values))]
        padding_len = [m-i.shape[1] for i in singular_values]
        padding_min_k = [2*k - i.shape[2] for i in singular_values]        
        singular_values = [pad(singular_values[i],(0, padding_min_k[i], 0, padding_len[i]),'constant',0) for i in range(len(singular_values))]
        return torch.stack(singular_values)
    def get_eig_pe(self, selected_with_true, k, m, type_info):
        og_unique_nodes = [torch.cat((a[:,0],a[:,2]),dim=0).unique() for a in selected_with_true]
        e1_idx = [(og_unique_nodes[i] == selected_with_true[i][:,0].unsqueeze(1)).nonzero()[:,1] for i in range(len(selected_with_true))]
        e2_idx = [(og_unique_nodes[i] == selected_with_true[i][:,2].unsqueeze(1)).nonzero()[:,1] for i in range(len(selected_with_true))]
        og_n = [i.numel() for i in og_unique_nodes]
        adj_list = [torch.eye(og_n[i], og_n[i]) for i in range(len(selected_with_true))] # Use eye for svd computation
            
        for i in range(len(adj_list)):
            adj_list[i][e1_idx[i], e2_idx[i]]=1
            adj_list[i][e2_idx[i], e1_idx[i]]=1
        
        degree_sqrt = [torch.sqrt(torch.diag(adj.sum(dim=0))) for adj in adj_list]
        # Compute laplacian eigenvectors: I - D^{-1/2}AD^{-1/2}
        laplacian = [torch.eye(og_n[i]) - torch.mm(torch.mm(torch.linalg.inv(degree_sqrt[i]), adj_list[i]), degree_sqrt[i]) for i in range(len(adj_list))]
        eig_vectors = [torch.linalg.eigh(l)[1] for l in laplacian] # eigenvalues are returned in ascending order # TODO this can be batchified
        
        # 0th slice in dim 0- laplacian eigenvectors for selected triples' head according to this subgraph
        # 1st slice in dim 0- laplacian eigenvectors for selected triples' tail according to this subgraph
        
        eig_vectors = [torch.stack([eig_vectors[i][e1_idx[i]],eig_vectors[i][e2_idx[i]]])[:,:,:k] for i in range(len(adj_list))]
        # pad so that each is same shape
        padding_len = [m-i.shape[1] for i in eig_vectors]
        padding_min_k = [k - min(k, i.shape[2]) for i in eig_vectors]
        eig_vectors = [pad(eig_vectors[i],(0, padding_min_k[i], 0, padding_len[i]),'constant',0) for i in range(len(eig_vectors))]

        return torch.stack(eig_vectors)
    def get_basic_pe(self, selected_with_true, k, m, type_info):
        return type_info
    def get_neg_triples_hneg(self, heads, relations, tails, num_neg, neg_h_choice, neg_t_choice):
        batch_size = heads.shape[0]
        mask_corrupt = torch.rand(batch_size).round().repeat(num_neg) # using uniform distribution to decide whether to corrupt the head or the tail
        n_h_cor = int(mask_corrupt.sum().item())

         # find an intersection of possible values for h and possible values for t
        # for this given batch
        # if num_neg == 1:
        #     # TODO explicitly pass mode variable and determine
        #     # as num_neg can change for demo experiments
        #     choice_h = self.h_choice_kg
        #     choice_t = self.t_choice_kg
        neg_head_idx=(mask_corrupt==1).nonzero(as_tuple=True) # which indices have their heads corrupted
        neg_tail_idx=(mask_corrupt==0).nonzero(as_tuple=True)
        neg_heads = heads.repeat(num_neg) 
        neg_tails = tails.repeat(num_neg)
        
        if num_neg == 1:
            neg_heads[neg_head_idx] = neg_h_choice[neg_head_idx]
            neg_tails[neg_tail_idx] = neg_t_choice[neg_tail_idx]

            neg_triples = torch.stack([neg_heads, relations.repeat(num_neg), neg_tails], dim=1)
            return neg_triples
        else:
            choice_h = self.h_choice_task
            choice_t = self.t_choice_task
        
            possible_values_t = set(choice_t[(heads[0].item(), relations[0].item())])
            possible_values_h = set(choice_h[(tails[0].item(), relations[0].item())])
            
            ## NOTE this is not updating the choices
            ## in this case, the first item is deciding choices for all items in this batch

            # for i in range(1, batch_size):
            #     if mask_corrupt[i]==0:
            #         pair = (heads[i].item(), relations[i].item())
            #         possible_values_t.intersection_update(set(choice_t[pair]))
            #     if mask_corrupt[i]==1:
            #         pair = (tails[i].item(), relations[i].item())
            #         possible_values_h.intersection_update(set(choice_h[pair]))

            possible_values_t = torch.tensor(list(possible_values_t))
            possible_values_h = torch.tensor(list(possible_values_h))

            mask_h = (~(neg_heads[neg_head_idx].unsqueeze(dim=1)==possible_values_h)).int() *  possible_values_h.repeat(n_h_cor,1) # False for avoid, True for neg samples
            mask_bool_h = ~(neg_heads[neg_head_idx].unsqueeze(dim=1)==possible_values_h)  
            neg_h = mask_h[mask_bool_h]
            neg_h = neg_h[torch.randperm(neg_h.shape[0])][:n_h_cor]

            mask_t = (~(neg_tails[neg_tail_idx].unsqueeze(dim=1)==possible_values_t)).int() *  possible_values_t.repeat(batch_size * num_neg - n_h_cor,1) # False for avoid, True for neg samples
            mask_bool_t = ~(neg_tails[neg_tail_idx].unsqueeze(dim=1)==possible_values_t)  
            neg_t = mask_t[mask_bool_t]
            neg_t = neg_t[torch.randperm(neg_t.shape[0])][:batch_size * num_neg - n_h_cor]

            neg_heads[neg_head_idx] = neg_h
            neg_tails[neg_tail_idx] = neg_t
            neg_triples = torch.stack([neg_heads, relations.repeat(num_neg), neg_tails], dim=1)
            return neg_triples
        
    
    def get_neg_triples_basic(self, heads, relations, tails, num_neg, neg_h_choice=None, neg_t_choice=None):
        batch_size = heads.shape[0]
        mask = torch.rand(batch_size).round().repeat(num_neg) # using uniform distribution to decide whether to corrupt the head or the tail

        neg_head_idx=(mask==1).nonzero(as_tuple=True) # which indices have their heads corrupted
        neg_tail_idx=(mask==0).nonzero(as_tuple=True) # which indices have their tails corrupted

        neg_heads = heads.repeat(num_neg) 
        neg_tails = tails.repeat(num_neg) 
        
        
        n_h_cor = int(mask.sum().item()) # number of heads corrupted
        
        neg_heads[mask == 1] = torch.randint(1, self.e_idx, (n_h_cor,))
        neg_heads[mask == 1][neg_heads[neg_head_idx] == heads.repeat(num_neg)[neg_head_idx]] += 1

        neg_tails[mask == 0] = torch.randint(1, self.e_idx, (batch_size * num_neg - n_h_cor,)) # batch_size * num_neg - n_h_cor is the number of tails corrupted
        neg_tails[mask == 0][neg_tails[neg_tail_idx] == tails.repeat(num_neg)[neg_tail_idx]] += 1
        
        neg_triples = torch.empty((batch_size * num_neg, 3), dtype=torch.long)
        neg_triples[neg_head_idx]=torch.stack([neg_heads[neg_head_idx],relations.repeat(num_neg)[neg_head_idx],\
                                tails.repeat(num_neg)[neg_head_idx]],dim=1) # triples with corrupted head
        neg_triples[neg_tail_idx]=torch.stack([heads.repeat(num_neg)[neg_tail_idx],relations.repeat(num_neg)[neg_tail_idx],\
                                neg_tails[neg_tail_idx]],dim=1) # triples with corrupted tail
        return neg_triples
    def get_neg_samples_rel(self, heads, relations, tails, num_neg, neg_h_choice=None, neg_t_choice=None):
        batch_size = heads.shape[0]
        possible_values = torch.arange(self.r_idx)
        mask_r = (~(relations.unsqueeze(dim=1)==possible_values)).int() *  possible_values.repeat(batch_size,1) # False for avoid, True for neg samples
        mask_bool = ~(relations.unsqueeze(dim=1)==possible_values)  
        neg_r = mask_r[mask_bool].reshape(batch_size,self.r_idx-1)
        
        if num_neg < self.r_idx-1:
            perm_idx = torch.randperm(neg_r.size(1))
            neg_r = neg_r[:, perm_idx][:, :num_neg]
        neg_triples = torch.stack([heads.repeat(num_neg), neg_r.flatten(), tails.repeat(num_neg)], dim=1) 
        return neg_triples
    def corrupt_batch(self, heads, relations, tails, num_neg, args, neg_h_choice=None, neg_t_choice=None):
        # st_ = time.time()
        # print('Inside corrupt_batch')
        batch_size = heads.shape[0]
        neg_triples = self.get_neg_triples(heads, relations, tails, num_neg, neg_h_choice, neg_t_choice)
            
        neg_path_info = dict()
        if self.up:
            print('Inside corrupt batch path count!')
            
            neg_paths = torch.full((batch_size * num_neg, self.max_path_count, self.max_path_len), -1)
            n_paths = torch.zeros(batch_size * num_neg)
            n_rels = torch.full((batch_size * num_neg, self.max_path_count), -1)
            for i in range(batch_size * num_neg):
                
                pair = neg_triples[i, [0,2]].tolist()
                nx_paths = nx.all_simple_paths(self.KG, source=pair[0], target=pair[1], cutoff=self.max_path_len)
                pc = 0
                for path in map(nx.utils.pairwise, nx_paths):
                    
                    p = [self.ht_rel_map[j] for j in list(path)]
                    
                    n_rels[i, pc] = len(p)
                    p = pad(torch.tensor(p), (0, self.max_path_len-len(p)), 'constant', -1)
                    neg_paths[i, pc] = p # omit check for duplicate path
                    pc += 1
                    
                    if pc == self.max_path_count:
                        break
                n_paths[i] = pc
            neg_path_info['paths'] = neg_paths
            neg_path_info['n_paths'] = n_paths
            neg_path_info['n_rels'] = n_rels
        
        selected_with_true = self.gather_corrupted_subg(neg_triples, batch_size * num_neg, args.m) 

        neg_n = torch.tensor([item.shape[0] for item in selected_with_true]) # n: list of INT, batch_size * num_neg; each element <= self.m, number of triples in original graph, number of nodes in line graph
        neg_type_info = torch.zeros((batch_size * num_neg, args.m,3), dtype=torch.int) # store start type, rel id, end type; rel id is actual rel ids so that we can access embeddings from model param

        neg_rels=[x[:,1] for x in selected_with_true]
        padded_neg_rels = [pad(x, (0, args.m-x.shape[0]), 'constant', 0) for x in neg_rels]
        
        neg_type_info[:,:,1] = torch.tensor(list(chain(*padded_neg_rels))).reshape(batch_size*num_neg, args.m)

        # NOTE: multiple OTHER type
        # unique_nodes = [torch.tensor(neg_ht[i,:].tolist() + list(set(selected_with_true[i][1:,ht_idx].flatten().tolist())- set(neg_ht[i,:].tolist()))) for i in range(batch_size*num_neg)]
        
        # st_type = [(unique_nodes[i] == selected_with_true[i][:,0].unsqueeze(dim=-1)).nonzero()[:,1] for i in range(batch_size*num_neg)]
        # en_type = [(unique_nodes[i] == selected_with_true[i][:,2].unsqueeze(dim=-1)).nonzero()[:,1] for i in range(batch_size*num_neg)]
        
        # NOTE: Single OTHER type
        st_type = [(selected_with_true[i][:,0]==neg_triples[i,0]).int()+ 2 * (selected_with_true[i][:,0]==neg_triples[i,2]).int() for i in range(batch_size*num_neg)]
        en_type = [(selected_with_true[i][:,2]==neg_triples[i,0]).int()+ 2 * (selected_with_true[i][:,2]==neg_triples[i,2]).int() for i in range(batch_size*num_neg)]

        padded_st = [pad(z, (0, args.m-z.shape[0]), 'constant', 0) for z in st_type]
        padded_en = [pad(z, (0, args.m-z.shape[0]), 'constant', 0) for z in en_type]

        neg_type_info[:,:,0] = torch.stack(padded_st)
        neg_type_info[:,:,2] = torch.stack(padded_en)

        # if there is a triple which has the same entity as head and tail, then by default: assign tail
        # necessary for single OTHER
        neg_type_info[:,:,0][neg_type_info[:,:,0]>2] = 2
        neg_type_info[:,:,2][neg_type_info[:,:,2]>2] = 2

        # compute edge bias adj info
        # r1-h == r2-t -> 1, r1-t == r2-h -> 2, r1-h == r2-h -> 4, r1-t == r2-t -> 5
        
        edge_bias = [(selected_with_true[i][:,0]==selected_with_true[i][:,2].unsqueeze(dim=1)).int() for i in range(batch_size*num_neg)]
        edge_bias = [adj_mat + (adj_mat.T)*2 for adj_mat in edge_bias]
        edge_bias = [edge_bias[i] + (selected_with_true[i][:,0]==selected_with_true[i][:,0].unsqueeze(dim=1)).int()*4 for i in range(batch_size*num_neg)]
        edge_bias = [edge_bias[i] + (selected_with_true[i][:,2]==selected_with_true[i][:,2].unsqueeze(dim=1)).int()*5 for i in range(batch_size*num_neg)]
        
        # pad the values with a large number like 16[a value we're not storing in adj] 
        # so that each tensor is [m,m] [accessing shape is faster than accessing neg_n]
        edge_bias = [pad(a,(0,args.m-a.shape[1],0,args.m-a.shape[0]),'constant',0) for a in edge_bias] # NOTE: to use neither h nor t option, replace with 16 here
        edge_bias = torch.stack(edge_bias) # so that we can find and replace non-diagonal zero values
        # NOTE: uncomment the following lines for neither h nor t type
        # edge_bias[edge_bias==0] = 13 # fill zero values with 13 which is shares neither head nor tail
        # edge_bias[edge_bias==16] = 0
        edge_bias = [a.fill_diagonal_(0) for a in edge_bias]
        
        pairs = [i[:,[0,2]] for i in selected_with_true]
        pairs = [pad(i, (0, 0, 0, args.m-i.shape[0]), 'constant', -1) for i in pairs]
        
        all_labels = torch.cat((torch.ones(batch_size), torch.zeros(batch_size * num_neg)), dim=0)
        corrupted_pe = self.get_pe(selected_with_true, args.k, args.m, neg_type_info)
        # print(f'exiting corrupt_batch; time: {time.time()-st_}')
        return corrupted_pe, neg_n, all_labels, torch.stack(edge_bias), neg_type_info, neg_path_info, torch.stack(pairs)



    def filesexist(self):
        path = './data/' + self.dataset + '/'
        return os.path.isfile(path + 'id2node.txt') and os.path.isfile(path + 'id2edge.txt')\
            and os.path.isfile(path + 'node2id.pkl') and os.path.isfile(path + 'edge2id.pkl')
    
    def writefiles(self):
        path = './data/' + self.dataset + '/'
        with open(path + 'id2node.txt', 'w') as file_:
            file_.write('\n'.join(self.id2node))
        with open(path + 'id2edge.txt', 'w') as file_:
            file_.write('\n'.join(self.id2edge))
        with open(path + '/node2id.pkl', 'wb') as file_:
            pickle.dump(self.node2id, file_)
        with open(path + '/edge2id.pkl', 'wb') as file_:
            pickle.dump(self.edge2id, file_)
        # with open(path + 'stat.txt', 'w') as file_:
        #     s = str(self.e_idx) + ' ' + str(self.r_idx)
        #     file_.write(s)
    
    def readfiles(self):
        path = './data/' + self.dataset + '/'
        with open(path + 'id2node.txt', 'r') as file_:
            self.id2node = file_.read().splitlines()
        with open(path + 'id2edge.txt', 'r') as file_:
            self.id2edge = file_.read().splitlines() 
        with open(path + 'node2id.pkl', 'rb') as file_:
            self.node2id = pickle.load(file_)
        with open(path + '/edge2id.pkl', 'rb') as file_:
            self.edge2id = pickle.load(file_)

        

