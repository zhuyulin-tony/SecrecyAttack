import torch
import torch.nn as nn

class POLE(nn.Module):
    def __init__(self, n, mask_idx, device):
        super().__init__()
        self.n = n
        self.device = device
        self.mask_idx = mask_idx
        self.epochs = 500
        self.lr = 0.1
        self.t = 10**0.8
        self.embs = nn.Parameter(torch.rand(self.n, 64).float()) 
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embs)
        
    def tri_to_adj(self, triple):
        A = torch.sparse_coo_tensor(triple[:,:2].T, triple[:,2], size=[self.n,self.n], device = self.device).to_dense().to(self.device) # upper triangular matrix.
        A = A + A.T - torch.diag(torch.diag(A)) # symmetric.
        return A
    
    def sparse_matrix_power(self, A):
        n = len(A)
        A_sp = A.to_sparse()
        A_sp = torch.sparse_coo_tensor(A_sp.indices(), A_sp.values(), size=[n, n])
        return torch.sparse.mm(A_sp, A_sp).to_dense()
    
    def sparse_matrix_mul(self, A1, A2):
        n1 = A1.size(0)
        k1 = A1.size(1)
        A1_sp = A1.to_sparse()
        A1_sp = torch.sparse_coo_tensor(A1_sp.indices(), A1_sp.values(), size=[n1,k1])
        n2 = A2.size(0)
        k2 = A2.size(1)
        A2_sp = A2.to_sparse()
        A2_sp = torch.sparse_coo_tensor(A2_sp.indices(), A2_sp.values(), size=[n2,k2])
        return torch.sparse.mm(A1_sp, A2_sp).to_dense()
    
    def expm(self, A):
        e, v = torch.linalg.eigh(A + 1e-10*torch.randn(A.shape).to(self.device))
        e_exp = torch.diag(torch.exp(e))
        return v @ e_exp @ v.T
    
    def sym_signed_autocovariance_matrix(self, triple):
        triple_mask = triple.clone()
        triple_mask[self.mask_idx,2] = 0
        A_mask = self.tri_to_adj(triple_mask)
        A_abs = torch.abs(A_mask)
        d_abs = A_abs.sum(0)
        D_05 = torch.diag(1/torch.sqrt(A_abs.sum(0)))
        pi = d_abs/torch.sum(d_abs)
        Pi = torch.diag(d_abs/torch.sum(d_abs))
        W = Pi - torch.outer(pi, pi)
        L_rw_sym = torch.eye(A_mask.size(0)).to(self.device) - self.sparse_matrix_mul(self.sparse_matrix_mul(D_05, A_mask), D_05)
        M_t = self.expm(- L_rw_sym * self.t)
        return M_t.T @ W @ M_t
    
    def signed_autocovariance_matrix(self, triple):
        triple_mask = triple.clone()
        triple_mask[self.mask_idx,2] = 0
        A_mask = self.tri_to_adj(triple_mask)
        A_abs = torch.abs(A_mask)
        d_abs = A_abs.sum(0)
        D_1 = torch.diag(1/A_abs.sum(0))
        pi = d_abs/torch.sum(d_abs)
        Pi = torch.diag(d_abs/torch.sum(d_abs))
        W = Pi - torch.outer(pi, pi)
        L_rw = torch.eye(A_mask.size(0)).to(self.device) - self.sparse_matrix_mul(D_1, A_mask)
        M_t = torch.matrix_exp(- L_rw * self.t)
        return M_t.T @ W @ M_t
    
    def unsigned_autocovariance_matrix(self, triple):
        triple_mask = triple.clone()
        triple_mask[self.mask_idx,2] = 0
        A_mask = self.tri_to_adj(triple_mask)
        A_abs = torch.abs(A_mask)
        d_abs = A_abs.sum(0)
        D_1 = torch.diag(1/A_abs.sum(0))
        pi = d_abs/torch.sum(d_abs)
        Pi = torch.diag(d_abs/torch.sum(d_abs))
        W = Pi - torch.outer(pi, pi)
        L_rw = torch.eye(A_mask.size(0)).to(self.device) - self.sparse_matrix_mul(D_1, A_abs)
        M_t = torch.matrix_exp(- L_rw * self.t)
        return M_t.T @ W @ M_t
    
    def polar(self, triple):
        triple_mask = triple.clone()
        triple_mask[self.mask_idx,2] = 0
        A_mask = self.tri_to_adj(triple_mask)
        A_abs = torch.abs(A_mask)
        D_05 = torch.diag(1/torch.sqrt(A_abs.sum(0)))
        L_sign = torch.eye(A_mask.size(0)).to(self.device) - self.sparse_matrix_mul(self.sparse_matrix_mul(D_05, A_mask), D_05)
        L_unsign = torch.eye(A_mask.size(0)).to(self.device) - self.sparse_matrix_mul(self.sparse_matrix_mul(D_05, A_abs), D_05)
        M = self.expm(- L_sign * self.t)
        M_abs = self.expm(- L_unsign * self.t)
        node_scores = torch.cat(([torch.corrcoef(torch.cat((signed.reshape(1,-1), unsigned.reshape(1,-1))))[1,0].reshape(-1,) for signed, unsigned in list(zip(M.T, M_abs.T))]))
        node_scores = torch.nan_to_num(node_scores, nan=0.0)  # Replace Nan with 0.0.
        return node_scores.mean()
    
    def forward(self, triple):
        R_t = self.sym_signed_autocovariance_matrix(triple)
        R_t = R_t.float()
        for e in range(self.epochs):
            loss = torch.dist(self.embs @ self.embs.T, R_t)
            loss.backward()
            self.embs.data -= self.lr * self.embs.grad.data
            self.embs.grad.zero_()