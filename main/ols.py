import argparse
import os
import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataloader import dataset
from logist import Logistic

parser = argparse.ArgumentParser(description='')
parser = argparse.ArgumentParser()
#parser.add_argument('--seed', type=int, default=666, help='Random seed.')
parser.add_argument('--dataset', type=str, default='bitcoin_alpha', choices=['word', 'bitcoin_alpha', 'bitcoin_otc'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.25,  help='pertubation rate')
parser.add_argument('--clf', type=str, default='logist', help='model')
parser.add_argument('--device', type=str, default='cuda:0', choices=['cuda:1','cpu'], help='device')
args = parser.parse_args()

#np.random.seed(args.seed)
#torch.manual_seed(args.seed)
root_dir = os.getcwd().replace('\\', '/')
###########################################################################
######################## load dataset #####################################
###########################################################################
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
    save_dir = root_dir+'/'+args.dataset+'/ols/'+str(t)
    try:
        os.makedirs(save_dir)
    except:
        pass
    #######################################################################################
    ############################# load target model #######################################
    ####################################################################################### 
    if args.clf == 'logist':
        clf = Logistic(n_nodes, test_idx, args.device).to(args.device)
    
    #######################################################################################
    ############################# set up attack model #####################################
    #######################################################################################
    
    class gradmaxOLS(nn.Module):
        def __init__(self, train_model, train_dataloader, test_dataloader, loss_fn, B, device):
            super().__init__()
            self.model = train_model
            self.train_loader = train_dataloader
            self.test_loader = test_dataloader
            self.loss_fn = loss_fn
            self.B = B
            self.device = device
            
        def inner_train(self, triple):
            self.model.reset_parameters()
            for i, (x_train, y_train) in enumerate(self.train_loader):
                x_train = x_train.type(torch.LongTensor)
                y_train = y_train.to(self.device)
                outputs = self.model.OLS(triple, x_train, y_train)
                #loss = self.loss_fn(outputs, y_train.reshape(-1,1))
                #print('train_loss:',loss.item())
    
        def get_meta_grad(self, triple_copy):
            edges = Variable(triple_copy[:,2:], requires_grad = True)
            triple_torch = torch.cat((triple_copy[:,:2], edges),1)
            for i, (x_test, y_test) in enumerate(self.test_loader): 
                y_test = y_test.to(self.device)
                x_test = x_test.type(torch.LongTensor)
                outputs = self.model(triple_torch, x_test)
                # attack loss = -training loss.
                atk_loss = -self.loss_fn(outputs, y_test.reshape(-1,1))
                atk_loss.backward()
                meta_grad = edges.grad.data.cpu().numpy()
                meta_grad[test_idx] = 0
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
                #print(K, target_grad)
                target_index = np.where(np.all((triple[:,:2] == target_grad[:2]), axis = 1))[0][0]
                triple_copy = triple_copy.data.numpy()
                triple_copy[target_index,2] -= 2 * np.sign(target_grad[2])
                perturb.append([int(target_grad[0]),int(target_grad[1])])
                
                if i in [int(0.01*n_edges)-1,int(0.05*n_edges)-1,int(0.1*n_edges)-1,int(0.15*n_edges)-1,int(0.2*n_edges)-1]:
                    np.savetxt(save_dir+'/'+args.dataset+'_mtri_'+str(int(i+1))+'.txt',triple_copy,fmt='%d')
    
    trainset = dataset(train_idx, train_label)
    testset = dataset(test_idx, test_pred)
    trainloader = DataLoader(trainset, batch_size = len(train_idx), shuffle = True)
    testloader = DataLoader(testset, batch_size = len(test_idx),shuffle = False)
    loss_fn = nn.BCELoss()
    greedysearch = gradmaxOLS(clf, trainloader, testloader, loss_fn, B, args.device).to(args.device)
    grads = greedysearch(triple)