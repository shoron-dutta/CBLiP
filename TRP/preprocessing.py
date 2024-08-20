import pprint, os
pp = pprint.PrettyPrinter(indent=4)
import pickle
from torch.utils.data import Dataset
import torch, os, pickle
from torch_geometric.utils import k_hop_subgraph as k_hop
from torch.nn.functional import pad
from tqdm import tqdm

class TransductiveData(Dataset):
    
    def __init__(self, args, base):
        
        self.dataset = args.dataset
        self.hop = args.hop
        self.m = args.m
        
        self.mode = 'train' # initial assignment
        
        self.node2id, self.edge2id = dict(), dict() # key: entity/relation string, value: int ID
        self.id2node, self.id2edge = [], [] # idx: ID, item: entity/relation string
        self.e_idx, self.r_idx = 0, 0 # count # of nodes, relations
        

        with open('./data/'+ self.dataset + '/train.txt', 'r') as f:
            kg_data = f.read().split() # list of all entities and relations in order
        with open('./data/'+ self.dataset + '/valid.txt', 'r') as f:
            valid_triples = f.readlines() # list of triples
        with open('./data/'+ self.dataset + '/test.txt', 'r') as f:
            test_triples = f.readlines() # list of triples
        
        # read kg data
        # need not be stored in the class since we have the int <-> str mapping both ways
        kg_src_str = kg_data[::3] # every 3rd item starting at 0th entry of data, all head entities
        kg_dst_str = kg_data[2::3] # every 3rd item starting at 2nd entry of data, all tail entities
        kg_edge_type_str = kg_data[1::3] # # every 3rd item starting at 3rd entry of data, all relations

        # assign IDs to entities base on the kg graph
        if self.files_exist():
            self.readfiles()
        else:
            print("Inside init, generating IDs")
            self.id2node = list(set(kg_src_str+kg_dst_str))
            self.node2id = {self.id2node[i]: i for i in range(len(self.id2node))}
            self.e_idx = len(self.id2node)
            # assign IDs to edges
            self.id2edge = list(set(kg_edge_type_str))
            self.edge2id = {self.id2edge[i]: i for i in range(len(self.id2edge))}
            self.r_idx = len(self.id2edge)
            
            # store for valid and test files to retrieve
            self.writefiles()
        valid_src_str, valid_dst_str, valid_edge_type_str = [], [], []
        test_src_str, test_dst_str, test_edge_type_str = [], [], []

        for triple in valid_triples:
            h, r, t = triple.split()
            if h in self.node2id and t in self.node2id:
                valid_src_str.append(h)
                valid_dst_str.append(t)
                valid_edge_type_str.append(r)
        for triple in test_triples:
            h, r, t = triple.split()
            if h in self.node2id and t in self.node2id:
                test_src_str.append(h)
                test_dst_str.append(t)
                test_edge_type_str.append(r)
        
        kg_src = torch.tensor([self.node2id[i] for i in kg_src_str])
        kg_dst = torch.tensor([self.node2id[i] for i in kg_dst_str])
        self.kg_edge_type = torch.tensor([self.edge2id[i] for i in kg_edge_type_str])
        self.kg_unique_ids = list(range(self.e_idx))

        valid_src = torch.tensor([self.node2id[i] for i in valid_src_str])
        valid_dst = torch.tensor([self.node2id[i] for i in valid_dst_str])
        self.valid_edge_type = torch.tensor([self.edge2id[i] for i in valid_edge_type_str])

        test_src = torch.tensor([self.node2id[i] for i in test_src_str])
        test_dst = torch.tensor([self.node2id[i] for i in test_dst_str])
        self.test_edge_type = torch.tensor([self.edge2id[i] for i in test_edge_type_str])
        
        self.kg_edge_index = torch.stack([kg_src, kg_dst], dim=0)
        self.valid_edge_index = torch.stack([valid_src, valid_dst], dim=0)
        self.test_edge_index = torch.stack([test_src, test_dst], dim=0)
       
        self.kg_num_triples = self.kg_edge_index.shape[1]
        self.valid_num_triples = self.valid_edge_index.shape[1]
        self.test_num_triples = self.test_edge_index.shape[1]
        

        self.create_induced_subgraphs_hopwise() # list is created and populated
    def __len__(self):
        var_dict = {
                    'train': self.kg_num_triples,
                    'valid': self.valid_num_triples,
                    'test': self.test_num_triples
                    }
        return var_dict[self.mode]


    def create_induced_subgraphs_hopwise(self):
        filename_1 = './data/' + self.dataset + '/kg_node_subgraphs_sep_1.pt'
        filename_2 = './data/' + self.dataset + '/kg_node_subgraphs_sep_2.pt'
        filename_3 = './data/' + self.dataset + '/kg_node_subgraphs_sep_3.pt'
        filename_4 = './data/' + self.dataset + '/kg_node_subgraphs_sep_4.pt'
        # hop == 1 is minimum
        if self.hop == 1 and os.path.isfile(filename_1):
            self.kg_node_subgraphs_1 = torch.load(filename_1)
            return            
        if self.hop == 2 and os.path.isfile(filename_1) and os.path.isfile(filename_2):
            self.kg_node_subgraphs_1 = torch.load(filename_1)
            self.kg_node_subgraphs_2 = torch.load(filename_2)
            return
        if self.hop==3 and os.path.isfile(filename_1) and os.path.isfile(filename_2) and os.path.isfile(filename_3):
            self.kg_node_subgraphs_1 = torch.load(filename_1)
            self.kg_node_subgraphs_2 = torch.load(filename_2)
            self.kg_node_subgraphs_3 = torch.load(filename_3)
            return 
        if self.hop==4 and os.path.isfile(filename_1) and os.path.isfile(filename_2) and os.path.isfile(filename_3) and os.path.isfile(filename_4):
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
        print('inside create_induced_subgraphs_hopwise')
        for node_id in tqdm(self.kg_unique_ids):
            
            mask_1 = k_hop(node_id, 1, edge_all)[3][:self.kg_num_triples]
        
            idx = mask_1.nonzero()# idx of edges that are in the subgraph
            triples_all = torch.tensor(list(zip(self.kg_edge_index[0, idx], self.kg_edge_type[idx]\
                                    , self.kg_edge_index[1, idx]))) # tensor [M,3] where M is number of triples in k-hop neighborhood
            self.kg_node_subgraphs_1.append(triples_all)
            
            if self.hop ==2:
                mask_2 = k_hop(node_id, 2, edge_all)[3][:self.kg_num_triples] # includes both 1 and 2 hop
                mask_2 = torch.logical_xor(mask_1, mask_2) # only 2 hops
                # repeat, using same variables to save space
                idx = mask_2.nonzero()# idx of edges that are in the subgraph
                triples_all = torch.tensor(list(zip(self.kg_edge_index[0, idx], self.kg_edge_type[idx]\
                                    , self.kg_edge_index[1, idx]))) # tensor [M,3] where M is number of triples in k-hop neighborhood
                self.kg_node_subgraphs_2.append(triples_all)             
            
            elif self.hop == 3:
                mask_2 = k_hop(node_id, 2, edge_all)[3][:self.kg_num_triples] # includes both 1 and 2 hop
                mask_3 = k_hop(node_id, 3, edge_all)[3][:self.kg_num_triples] # includes 1, 2, 3 hop
                mask_3 = torch.logical_xor(mask_2, mask_3) # only 3 hops
                mask_2 = torch.logical_xor(mask_1, mask_2) # only 2 hops, this should come after xor of mask_2 and mask_3
                
                idx = mask_2.nonzero()# idx of edges that are in the subgraph
                triples_all = torch.tensor(list(zip(self.kg_edge_index[0, idx], self.kg_edge_type[idx]\
                                    , self.kg_edge_index[1, idx]))) # tensor [M,3] where M is number of triples in k-hop neighborhood
                self.kg_node_subgraphs_2.append(triples_all)             
            
                # repeat, using same variables to save space
                idx = mask_3.nonzero()# idx of edges that are in the subgraph
                triples_all = torch.tensor(list(zip(self.kg_edge_index[0, idx], self.kg_edge_type[idx]\
                                    , self.kg_edge_index[1, idx]))) # tensor [M,3] where M is number of triples in k-hop neighborhood
                self.kg_node_subgraphs_3.append(triples_all)
            elif self.hop == 4:
                mask_2 = k_hop(node_id, 2, edge_all)[3][:self.kg_num_triples] # includes both 1 and 2 hop
                mask_3 = k_hop(node_id, 3, edge_all)[3][:self.kg_num_triples] # includes 1, 2, 3 hop
                mask_4 = k_hop(node_id, 4, edge_all)[3][:self.kg_num_triples] # includes 1, 2, 3, 4 hop
                
                mask_4 = torch.logical_xor(mask_3, mask_4) # only 4 hops
                mask_3 = torch.logical_xor(mask_2, mask_3) # only 3 hops
                mask_2 = torch.logical_xor(mask_1, mask_2) # only 2 hops, this should come after xor of mask_2 and mask_3
                
                idx = mask_2.nonzero()# idx of edges that are in the subgraph
                triples_all = torch.tensor(list(zip(self.kg_edge_index[0, idx], self.kg_edge_type[idx]\
                                    , self.kg_edge_index[1, idx]))) # tensor [M,3] where M is number of triples in k-hop neighborhood
                self.kg_node_subgraphs_2.append(triples_all)             
            
                # repeat, using same variables to save space
                idx = mask_3.nonzero()# idx of edges that are in the subgraph
                triples_all = torch.tensor(list(zip(self.kg_edge_index[0, idx], self.kg_edge_type[idx]\
                                    , self.kg_edge_index[1, idx]))) # tensor [M,3] where M is number of triples in k-hop neighborhood
                self.kg_node_subgraphs_3.append(triples_all)

                # repeat, using same variables to save space
                idx = mask_4.nonzero()# idx of edges that are in the subgraph
                triples_all = torch.tensor(list(zip(self.kg_edge_index[0, idx], self.kg_edge_type[idx]\
                                    , self.kg_edge_index[1, idx]))) # tensor [M,3] where M is number of triples in k-hop neighborhood
                self.kg_node_subgraphs_4.append(triples_all)
            
        torch.save(self.kg_node_subgraphs_1, open(filename_1, 'wb'))
        if self.hop==2:
            torch.save(self.kg_node_subgraphs_2, open(filename_2, 'wb'))
        elif self.hop==3:
            torch.save(self.kg_node_subgraphs_2, open(filename_2, 'wb'))
            torch.save(self.kg_node_subgraphs_3, open(filename_3, 'wb'))
        elif self.hop==4:
            torch.save(self.kg_node_subgraphs_2, open(filename_2, 'wb'))
            torch.save(self.kg_node_subgraphs_3, open(filename_3, 'wb'))
            torch.save(self.kg_node_subgraphs_4, open(filename_4, 'wb'))
        if len(self.kg_node_subgraphs_1) != self.e_idx:
            raise ValueError('Inside induced subgraphs hopwise')
        return
    
    def shuffle_neighborhood(self):
        # shuffle the neighborhood once before each epoch
        # avoid shuffling per sample
        hop_1_size = [self.kg_node_subgraphs_1[i].shape[0] for i in range(self.e_idx)]
        for i in (range(self.e_idx)):
            self.kg_node_subgraphs_1[i] = self.kg_node_subgraphs_1[i][torch.randperm(hop_1_size[i])]
        if self.hop >= 2:
            hop_2_size = [self.kg_node_subgraphs_2[i].shape[0] for i in range(self.e_idx)]
            for i in (range(self.e_idx)):
                self.kg_node_subgraphs_2[i] = self.kg_node_subgraphs_2[i][torch.randperm(hop_2_size[i])]
        if self.hop == 3:
            hop_3_size = [self.kg_node_subgraphs_3[i].shape[0] for i in range(self.e_idx)]
            for i in (range(self.e_idx)):
                self.kg_node_subgraphs_3[i] = self.kg_node_subgraphs_3[i][torch.randperm(hop_3_size[i])]
        if self.hop == 4:
            hop_4_size = [self.kg_node_subgraphs_4[i].shape[0] for i in range(self.e_idx)]
            for i in (range(self.e_idx)):
                self.kg_node_subgraphs_4[i] = self.kg_node_subgraphs_4[i][torch.randperm(hop_4_size[i])]
        return

    def set_mode(self, mode):
        self.mode = mode
    
    
    def __getitem__(self, idx):
        
        variable_dict_edge_idx = {
                        'train':self.kg_edge_index, 
                        'valid':self.valid_edge_index, 
                        'test':self.test_edge_index
                        }
        variable_dict_edge_type = {
                        'train':self.kg_edge_type, 
                        'valid':self.valid_edge_type, 
                        'test':self.test_edge_type
                        }
        
        head = variable_dict_edge_idx[self.mode][0,idx].item()
        tail = variable_dict_edge_idx[self.mode][1,idx].item()
        target_rel = variable_dict_edge_type[self.mode][idx].item()
        target_triple = torch.tensor([head, target_rel, tail])
        
        ## Find neighborhood of head entity
        hop_1 = self.kg_node_subgraphs_1[head][:self.m+1, :] 
        
        if self.hop == 1:
            h_neighbors = hop_1
        elif self.hop == 2:
            hop_2 = self.kg_node_subgraphs_2[head][:self.m+1] # shuffled. pick m.
            h_neighbors = torch.cat((hop_1, hop_2), dim=0) 
            # h_dist_2 = (h_neighbors.shape[0] - len(h_dist_1)) * [self.d2]
        elif self.hop == 3:
            hop_2 = self.kg_node_subgraphs_2[head][:self.m+1] # shuffled. pick m.
            hop_3 = self.kg_node_subgraphs_3[head][:self.m+1]    
            h_neighbors = torch.cat((hop_1, hop_2, hop_3), dim=0) 
        elif self.hop == 4:
            hop_2 = self.kg_node_subgraphs_2[head][:self.m+1] # shuffled. pick m.
            hop_3 = self.kg_node_subgraphs_3[head][:self.m+1] 
            hop_4 = self.kg_node_subgraphs_4[head][:self.m+1]   
            h_neighbors = torch.cat((hop_1, hop_2, hop_3, hop_4), dim=0)          
        else:
            raise ValueError("Not recognized")
            
        h_neighbors = h_neighbors[~(h_neighbors==target_triple).all(dim=1)][:self.m]

        # repeat, for tail; 
        hop_1 = self.kg_node_subgraphs_1[tail][:self.m]
        if self.hop == 1:
            t_neighbors = hop_1
        elif self.hop == 2:
            hop_2 = self.kg_node_subgraphs_2[tail][:self.m+1]
            t_neighbors = torch.cat((hop_1, hop_2), dim=0)
        elif self.hop == 3:
            hop_2 = self.kg_node_subgraphs_2[tail][:self.m+1]
            hop_3 = self.kg_node_subgraphs_3[tail][:self.m+1]
            t_neighbors = torch.cat((hop_1, hop_2, hop_3), dim=0)
        elif self.hop == 4:
            hop_2 = self.kg_node_subgraphs_2[tail][:self.m+1]
            hop_3 = self.kg_node_subgraphs_3[tail][:self.m+1]
            hop_4 = self.kg_node_subgraphs_4[tail][:self.m+1]   
            t_neighbors = torch.cat((hop_1, hop_2, hop_3, hop_4), dim=0)
        else:
             raise ValueError("Not recognized")
            
        t_neighbors = t_neighbors[~(t_neighbors==target_triple).all(dim=1)][:self.m]
        k1, k2 = h_neighbors.shape[0], t_neighbors.shape[0]
        n = torch.tensor([k1, k2]) 
        h_neighbors = pad(h_neighbors, (0, 0, 0, self.m-n[0]), "constant", -1)
        t_neighbors = pad(t_neighbors, (0, 0, 0, self.m-n[1]), "constant", -1)
        res_dict = {
                    'target_triple': target_triple.long(),
                    'n':n,
                    'h_neighbors': h_neighbors.long(),
                    't_neighbors': t_neighbors.long()
                    }

        selected_triples = torch.cat((target_triple.unsqueeze(0), h_neighbors, t_neighbors), dim=0) # INCLUDE tgt triple as it is part of input token sequence
        n_st = selected_triples.shape[0]
        
        edge_bias_adj = torch.zeros((n_st, n_st), dtype=torch.long)
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
        
        res_dict['adj'] = edge_bias_adj
    
        return res_dict
    
    def files_exist(self):
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
        with open(path + 'stat.txt', 'w') as file_:
            s = str(self.e_idx) + ' ' + str(self.r_idx)
            file_.write(s)
    
    def readfiles(self):
        path = './data/' + self.dataset + '/'
        with open(path + 'id2node.txt', 'r') as file_:
            self.id2node = file_.readlines()
        with open(path + 'id2edge.txt', 'r') as file_:
            self.id2edge = file_.readlines()
        with open(path + 'node2id.pkl', 'rb') as file_:
            self.node2id = pickle.load(file_)
        with open(path + '/edge2id.pkl', 'rb') as file_:
            self.edge2id = pickle.load(file_)
        with open(path + '/stat.txt', 'r') as file_:
            a, b = file_.readline().split()
            self.e_idx, self.r_idx = int(a), int(b)
