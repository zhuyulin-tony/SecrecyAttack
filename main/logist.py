import torch
import torch.nn as nn

class Logistic(nn.Module):
    def __init__(self, n, mask_idx, device):
        super().__init__()
        self.n = n
        self.device = device
        self.mask_idx = mask_idx
        self.weight = nn.Parameter(torch.FloatTensor(9, 1).to(device))
        self.bias = nn.Parameter(torch.FloatTensor(1).to(device))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def tri_to_adj(self, triple):
        A = torch.sparse_coo_tensor(triple[:,:2].T, triple[:,2], size=[self.n,self.n], device = self.device).to_dense().to(self.device) # upper triangular matrix.
        A = A + A.T - torch.diag(torch.diag(A)) # symmetric.
        return A
    
    def weakly_balance_percentage(self, triple):
        # hide the sign of the edges in the test set.
        triple_mask = triple.clone()
        triple_mask[self.mask_idx,2] = 0
        A_mask = self.tri_to_adj(triple_mask)
        A_mask_unsign = torch.abs(A_mask)
        A_mask_plus = torch.relu(A_mask)
        # A_minus = A_plus - A
        A_mask_minus = A_mask_plus- A_mask
        violate_triads = torch.trace(A_mask_plus @ A_mask_plus @ A_mask_minus)/2
        total_triads = torch.trace(torch.matrix_power(A_mask_unsign,3))/6
        return  violate_triads/total_triads
    
    def sparse_matrix_power(self, A):
        n = len(A)
        A_sp = A.to_sparse()
        A_sp = torch.sparse_coo_tensor(A_sp.indices(), A_sp.values(), size=[n, n])
        return torch.sparse.mm(A_sp, A_sp).to_dense()
    
    def sparse_matrix_mul(self, A1, A2):
        n1 = len(A1)
        A1_sp = A1.to_sparse()
        A1_sp = torch.sparse_coo_tensor(A1_sp.indices(), A1_sp.values(), size=[n1,n1])
        n2= len(A2)
        A2_sp = A2.to_sparse()
        A2_sp = torch.sparse_coo_tensor(A2_sp.indices(), A2_sp.values(), size=[n2,n2])
        return torch.sparse.mm(A1_sp, A2_sp).to_dense()
    
    def expm(self, A):
        e, v = torch.linalg.eigh(A + 1e-10*torch.randn(A.shape).to(self.device))
        e_exp = torch.diag(torch.exp(e))
        return v @ e_exp @ v.T
    
    def polar(self, triple, t=10**0.8):
        triple_mask = triple.clone()
        triple_mask[self.mask_idx,2] = 0
        A_mask = self.tri_to_adj(triple_mask)
        A_abs = torch.abs(A_mask)
        D_05 = torch.diag(1/torch.sqrt(A_abs.sum(0)))
        L_sign = torch.eye(A_mask.size(0)).to(self.device) - self.sparse_matrix_mul(self.sparse_matrix_mul(D_05, A_mask), D_05)
        L_unsign = torch.eye(A_mask.size(0)).to(self.device) - self.sparse_matrix_mul(self.sparse_matrix_mul(D_05, A_abs), D_05)
        M = self.expm(- L_sign * t)
        M_abs = self.expm(- L_unsign * t)
        node_scores = torch.cat(([torch.corrcoef(torch.cat((signed.reshape(1,-1), unsigned.reshape(1,-1))))[1,0].reshape(-1,) for signed, unsigned in list(zip(M.T, M_abs.T))]))
        node_scores = torch.nan_to_num(node_scores, nan=0.0)  # Replace Nan with 0.0.
        return node_scores.mean()
    
    def feats(self, triple, idx):
        A = self.tri_to_adj(triple)
        # hide the sign of the edges in the test set.
        triple_mask = triple.clone()
        triple_mask[self.mask_idx,2] = 0
        A_mask = self.tri_to_adj(triple_mask)
        A_unsign = torch.abs(A)
        A_mask_plus = torch.relu(A_mask)
        # A_minus = A_plus - A
        A_mask_minus = A_mask_plus - A_mask
        
        # we remove the sign of test edges, only CN do not influenced by this kind of removement.
        pos_degree_all = A_mask_plus.sum(0)
        neg_degree_all = A_mask_minus.sum(0)
        cn_all = self.sparse_matrix_power(A_unsign)
        #cn_all = torch.matrix_power(A_unsign, 2)
        
        # all the edges {u,v} in the train/test dataset sampled from the original signed graph.
        us = triple[idx][:,0].int().tolist()
        vs = triple[idx][:,1].int().tolist()
        
        # degree features.
        pos_degree_us = pos_degree_all[us].reshape(-1,1)
        neg_degree_us = neg_degree_all[us].reshape(-1,1)
        pos_degree_vs = pos_degree_all[vs].reshape(-1,1)
        neg_degree_vs = neg_degree_all[vs].reshape(-1,1)
        cn_uv = cn_all[us,vs].reshape(-1,1)
        
        # triad features.
        type1 = self.sparse_matrix_mul(A_mask_plus,A_mask_plus)[us,vs].reshape(-1,1)
        type2 = self.sparse_matrix_mul(A_mask_plus,A_mask_minus)[us,vs].reshape(-1,1)
        type3 = self.sparse_matrix_mul(A_mask_minus,A_mask_plus)[us,vs].reshape(-1,1)
        type4 = self.sparse_matrix_mul(A_mask_minus,A_mask_minus)[us,vs].reshape(-1,1)
        #type1 = torch.mm(A_mask_plus,A_mask_plus)[us,vs].reshape(-1,1)
        #type2 = torch.mm(A_mask_plus,A_mask_minus)[us,vs].reshape(-1,1)  
        #type3 = torch.mm(A_mask_minus,A_mask_plus)[us,vs].reshape(-1,1)  
        #type4 = torch.mm(A_mask_minus,A_mask_minus)[us,vs].reshape(-1,1)  
        
        features = torch.cat((pos_degree_us, neg_degree_us, pos_degree_vs, neg_degree_vs, cn_uv,\
                              type1, type2, type3, type4),1).float()
        return features
    
    def OLS(self, triple, idx, y):
        logit_features = self.feats(triple, idx)
        logit_features1 = torch.cat((torch.ones((len(logit_features),1)).to(self.device), logit_features), 1)
        theta = torch.inverse(logit_features1.T @ logit_features1) @ logit_features1.T @ y
        self.bias = nn.Parameter(theta[:1].reshape(1))
        self.weight = nn.Parameter(theta[1:].reshape(9,1))
        return torch.sigmoid(torch.mm(logit_features, self.weight) + self.bias)
        
    def forward(self, triple, idx):
        logit_features = self.feats(triple, idx)
        return torch.sigmoid(torch.mm(logit_features, self.weight) + self.bias)