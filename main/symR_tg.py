import argparse
import os
import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from pole import POLE

parser = argparse.ArgumentParser(description='')
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=int, default=0.1, help='Learning rate for training.')
parser.add_argument('--dataset', type=str, default='bitcoin_alpha', choices=['word', 'bitcoin_alpha', 'bitcoin_otc'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.1,  help='pertubation rate')
parser.add_argument('--clf', type=str, default='pole', help='model')
parser.add_argument('--device', type=str, default='cuda:2', choices=['cuda:0','cpu'], help='device')
args = parser.parse_args()

root_dir = os.getcwd().replace('\\', '/')
lam_lst = [0.01]
###########################################################################
######################## load dataset #####################################
###########################################################################
for lam in lam_lst:
    for t in [1,2,3,4,5]:
        np.random.seed(t)
        torch.manual_seed(t)
        triple = np.loadtxt(root_dir +'/'+args.dataset+'/'+args.dataset+'_tri.txt')
        n_nodes = int(triple[:,:2].max() + 1)
        n_edges = len(triple)
        B = int(args.ptb_rate * n_edges)
        
        train_idx = np.loadtxt(root_dir+'/'+args.dataset+'/train_idx_'+str(t)+'.txt',dtype='int32')
        test_idx = np.loadtxt(root_dir+'/'+args.dataset+'/test_idx_'+str(t)+'.txt',dtype='int32')
        train_label = np.loadtxt(root_dir+'/'+args.dataset+'/train_label_'+str(t)+'.txt')
        test_pred = np.loadtxt(root_dir+'/'+args.dataset+'/test_pred_pole_ce_'+str(t)+'.txt')
        save_dir = root_dir+'/'+args.dataset+'/symR_tg_'+str(lam)+'/'+str(t)
        try:
            os.makedirs(save_dir)
        except:
            pass
        #######################################################################################
        ############################# load target model #######################################
        ####################################################################################### 
        if args.clf == 'pole':
            clf = POLE(n_nodes, test_idx, args.device).to(args.device)
        #######################################################################################
        ############################# set up attack model #####################################
        #######################################################################################
        class gradmax(nn.Module):
            def __init__(self, train_model, train_iters, test_label, optimizer, loss_fn, B, device):
                super().__init__()
                self.model = train_model
                self.iters = train_iters
                self.optimizer = optimizer
                self.loss_fn = loss_fn
                self.test_label = test_label
                self.B = B
                self.device = device
            
            def sparse_matrix_power(self, A):
                n = A.size(0)
                A_sp = A.to_sparse()
                A_sp = torch.sparse_coo_tensor(A_sp.indices(), A_sp.values(), size=[n,n])
                A2 = torch.sparse.mm(A_sp, A_sp).to_dense()  
                A3 = torch.mm(A2, A)
                return A3
                #return torch.sparse.mm(torch.sparse.mm(A_sp, A_sp), A_sp).to_dense() 
            
            def inner_train(self, triple):
                self.model.reset_parameters()
                R_t = self.model.sym_signed_autocovariance_matrix(triple)
                for epoch in range(self.iters):
                    loss = torch.dist(self.model.embs @ self.model.embs.T, R_t)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            def get_meta_grad(self, triple_copy):
                edges = Variable(triple_copy[:,2:], requires_grad = True)
                triple_torch = torch.cat((triple_copy[:,:2], edges),1)
                triple_mask = triple_torch.clone().to(self.device)
                triple_mask[test_idx,2] = 0
                adj_mask = self.model.tri_to_adj(triple_mask)
                adj = self.model.tri_to_adj(triple_torch)
                adj_unsign = torch.abs(adj)
                R_t = self.model.sym_signed_autocovariance_matrix(triple_torch)
                triu_idx = triple_torch[:,:2].cpu().detach().numpy().copy().astype('int')
                sims = (R_t / (self.model.embs.norm(dim=1, keepdim=True) @ self.model.embs.norm(dim=1, keepdim=True).T))[triu_idx[test_idx,0],triu_idx[test_idx,1]]
                atk_loss = -self.loss_fn(torch.clamp(0.5*(sims+1), 1e-3, 1.-1e-3),torch.from_numpy(self.test_label).to(self.device)) \
                           -(lam/2)*(1+(torch.trace(self.sparse_matrix_power(adj_mask))/torch.trace(self.sparse_matrix_power(adj_unsign))))
                atk_loss.backward()
                meta_grad = edges.grad.data.cpu().numpy()
                meta_grad[test_idx] = 0
                #print(meta_grad)
                return np.concatenate((triple_copy[:,:2], meta_grad), 1)
                
            def forward(self, triple):
                triple_copy = torch.from_numpy(triple.copy())
                perturb = []
                for i in tqdm(range(self.B), desc = 'Perturbing Graph'):
                    if i != 0:
                        triple_copy = torch.from_numpy(triple_copy)
                    self.inner_train(triple_copy)
                    meta_grad = self.get_meta_grad(triple_copy) 
                    v_grad = np.zeros((len(meta_grad),3))
                    for j in range(len(meta_grad)):
                        v_grad[j,0] = meta_grad[j,0]
                        v_grad[j,1] = meta_grad[j,1]
                        if triple_copy[j,2] == -1 and meta_grad[j,2] < 0:
                            v_grad[j,2] = meta_grad[j,2]
                        elif triple_copy[j,2] == 1 and meta_grad[j,2] > 0:
                            v_grad[j,2] = meta_grad[j,2]
                        else:
                            continue
                     
                    v_grad = v_grad[np.abs(v_grad[:,2]).argsort()]
                    # attack w.r.t gradient information.
                    K = -1
                    while v_grad[K][:2].astype('int').tolist() in perturb:
                        K -= 1
                    target_grad = v_grad[int(K)]
                    target_index = np.where(np.all((triple[:,:2] == target_grad[:2]), axis = 1))[0][0]
                    triple_copy = triple_copy.data.numpy()
                    triple_copy[target_index,2] -= 2 * np.sign(target_grad[2])
                    perturb.append([int(target_grad[0]),int(target_grad[1])])
                    
                    if i in [int(0.01*n_edges)-1,int(0.03*n_edges)-1,int(0.05*n_edges)-1,int(0.07*n_edges)-1,int(0.1*n_edges)-1]:
                        np.savetxt(save_dir+'/'+args.dataset+'_mtri_'+str(int(i+1))+'.txt',triple_copy,fmt='%d')
            
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.SGD(clf.parameters(),lr=args.lr)
        greedysearch = gradmax(clf, args.epochs, test_pred, optimizer, loss_fn, B, args.device).to(args.device)
        grads = greedysearch(triple)