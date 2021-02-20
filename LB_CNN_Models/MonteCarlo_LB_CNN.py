# -*- coding: utf-8 -*-
# @Author: Tim Reese
# @Date:   2021-02-20

import numpy as np
import torch
import torch.nn as nn
import os

torch.set_default_tensor_type(torch.DoubleTensor)

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes) 
    return y[labels]


class LB_CNN_MonteCarlo(nn.Module):
    def __init__(self,num_features,num_classes,latent_size,num_samples,debug_mode = False):
        super(LB_CNN_MonteCarlo,self).__init__()
        self.num_classes = num_classes
        self.M = latent_size
        self.num_features = num_features
        self.num_samples = num_samples
        
        self.prior = (0.15-0.85)*torch.rand(self.M)+0.85
        self.prior = self.prior[None,:]
        
        
        init_constant = 2.0/np.sqrt(self.num_features)
        self.alpha_0_tensor = torch.nn.Parameter(torch.normal(torch.zeros(self.M),torch.ones(self.M)).type(torch.DoubleTensor))
        
        self.alpha_tensor = torch.normal(torch.zeros([self.M,self.num_features]),init_constant*torch.ones([self.M,self.num_features])).type(torch.DoubleTensor)
        self.alpha_tensor = torch.nn.Parameter(self.alpha_tensor / torch.norm(self.alpha_tensor,dim=0,keepdim = True))
        
        init_constant2 = 2.0/np.sqrt(self.M)
        self.beta_0_tensor = torch.nn.Parameter(torch.normal(torch.zeros(self.num_classes),torch.ones(self.num_classes)).type(torch.DoubleTensor))
        self.beta_tensor = torch.nn.Parameter(torch.normal(torch.zeros([self.num_classes,self.M]),init_constant2*torch.ones([self.num_classes,self.M])).type(torch.DoubleTensor))

        self.debug_mode = debug_mode
        
        
    def compute_linear_eta(self,x_in):
        self.linear_eta = self.alpha_0_tensor[None,:] + torch.matmul(x_in,self.alpha_tensor.T)
    
    def compute_eta(self):
        with torch.no_grad():
            self.eta = torch.exp(self.linear_eta)/(1.0 + torch.exp(self.linear_eta))    
                      
    def sample_Z(self):
        with torch.no_grad():
            bern = torch.distributions.bernoulli.Bernoulli(self.eta)
            ind_ZgivenX = bern.sample(sample_shape=torch.Size([self.num_samples]))
            self.Z = 2.0*ind_ZgivenX-1.0
            
    def sample_Z_via_prior(self):
        with torch.no_grad():
            device = next(self.parameters()).device
            self.prior = self.prior.to(device)
            self.prior = self.prior*torch.ones_like(self.eta)
            bern = torch.distributions.bernoulli.Bernoulli(self.prior)
            ind_ZgivenX = bern.sample(sample_shape=torch.Size([self.num_samples]))
            self.Z = 2.0*ind_ZgivenX-1.0
        
    
    def compute_pi_j(self,y_in):
        with torch.no_grad():
            device = next(self.parameters()).device
            y_onehot = one_hot_embedding(y_in, self.num_classes).type(torch.DoubleTensor).to(device)
            
            pi_pos_den = []
            pi_neg_den = []            
            pi_pos_n = []
            pi_neg_n = []
            for j in range(self.M):
                Z_sample_pos = self.Z.clone()
                Z_sample_pos[:,:,j] = 1.0

                #RxNxK
                linear_pi_pos_j = self.beta_0_tensor[None,:] + torch.matmul(Z_sample_pos,self.beta_tensor.T) 
                
                max_pos, _ = torch.max(linear_pi_pos_j, dim = 2, keepdim = True)
                linear_pi_pos_j = linear_pi_pos_j +  max_pos
                
                linear_pi_pos_j_n = torch.sum(torch.mul(y_onehot,linear_pi_pos_j),dim = 2)
                pi_pos_j_den = torch.sum(torch.exp(linear_pi_pos_j),dim = 2)
                pi_pos_j = torch.mul(torch.exp(linear_pi_pos_j_n),1.0/pi_pos_j_den)
                
                pi_pos_den.append(pi_pos_j_den)
                pi_pos_n.append(pi_pos_j)
                
                
                
                Z_sample_neg = self.Z.clone()
                Z_sample_neg[:,:,j] = -1.0
                linear_pi_neg_j = self.beta_0_tensor[None,:] + torch.matmul(Z_sample_neg,self.beta_tensor.T) 
                
                max_neg, _ = torch.max(linear_pi_neg_j, dim = 2, keepdim = True)
                linear_pi_neg_j =linear_pi_neg_j +  max_neg
                
                linear_pi_neg_j_n = torch.sum(torch.mul(y_onehot,linear_pi_neg_j),dim =2)
                pi_neg_j_den = torch.sum(torch.exp(linear_pi_neg_j),dim = 2) 
                pi_neg_j = torch.mul(torch.exp(linear_pi_neg_j_n),1.0/pi_neg_j_den)
                
                pi_neg_den.append(pi_neg_j_den)
                pi_neg_n.append(pi_neg_j)

            self.pi_pos_n = pi_pos_n
            self.pi_neg_n = pi_neg_n
            
            self.pi_pos = torch.stack(pi_pos_n,dim = 2)
            self.pi_neg = torch.stack(pi_neg_n,dim = 2)
            
            self.pi_pos_den = torch.stack(pi_pos_den,dim = 2)
            self.pi_neg_den = torch.stack(pi_neg_den,dim = 2)
            
    def compute_pi(self,y_in):
        with torch.no_grad():
            device = next(self.parameters()).device
            y_onehot = one_hot_embedding(y_in, self.num_classes).type(torch.DoubleTensor).to(device)
            self.linear_pi = self.beta_0_tensor[None,None,:] + torch.matmul(self.Z,self.beta_tensor.T)
            self.denom_pi = torch.sum(torch.exp(self.linear_pi),dim=2)
            self.pi = torch.sum(torch.mul(torch.exp(self.linear_pi),y_onehot),dim=2) / self.denom_pi        
       
    def compute_EY_X_j(self):
        with torch.no_grad():
            pos_part = torch.mul(self.pi_pos,self.eta)
            neg_part = torch.mul(self.pi_neg,1.0-self.eta)
            self.EY_X_j = torch.sum(pos_part + neg_part,dim = 0)
            
    def compute_EY_X(self):
        with torch.no_grad():
            self.EY_X = torch.sum(self.pi,dim = 0)
            
    def compute_EI(self):
        with torch.no_grad():
            pos_part = torch.mul(self.pi_pos,self.eta)
            self.EI = torch.sum(pos_part,dim = 0)/self.EY_X_j
          
    def compute_EII(self):
        with torch.no_grad():
            self.EII = 2.0*self.EI - 1.0   
            
    def compute_EIII(self,y_in):
        with torch.no_grad():
            device = next(self.parameters()).device
            y_onehot = one_hot_embedding(y_in, self.num_classes).to(device)
            numerator = torch.sum(torch.mul(y_onehot,torch.exp(self.linear_pi).squeeze()), dim =2)
            pi = numerator/self.denom_pi.squeeze()
        linear_pi = self.beta_0_tensor[None,None,:] + torch.matmul(self.Z,self.beta_tensor.T)
        denom = torch.sum(torch.exp(linear_pi),dim=2)
        log_terms = torch.log(denom)
        self.EIII = torch.sum(torch.mul(log_terms,pi),dim=0)/self.EY_X

   

    def compute_TermI(self,y_in):
        device = next(self.parameters()).device
        y_onehot = one_hot_embedding(y_in, self.num_classes).type(torch.DoubleTensor).to(device)
        
        linear_term = self.beta_0_tensor[None,:] + torch.matmul(self.EII,self.beta_tensor.T)
        self.TermI = torch.sum(torch.sum(torch.mul(y_onehot,linear_term),dim = 1), dim = 0)
        
    def compute_TermII(self):
        self.TermII = torch.sum(self.EIII,dim = 0)
        
        
    def compute_TermIII(self):
        linear_term = torch.mul(self.EI,self.linear_eta)
        self.TermIII = torch.sum(torch.sum(linear_term, dim = 1), dim = 0)
        
    def compute_TermIV(self):
        log_term = torch.log(1.0 + torch.exp(self.linear_eta))
        self.TermIV = torch.sum(torch.sum(log_term, dim = 1), dim = 0)
                

    def compute_E_logLikelihood(self,y_in):
        self.compute_TermI(y_in)
        self.compute_TermII()
        self.compute_TermIII()
        self.compute_TermIV()
        
        self.E_log_likelihood = self.TermI - self.TermII + self.TermIII - self.TermIV
        if self.debug_mode:
            self.printNetworkInfo()

        
    def predict_Z(self,x_in):
        with torch.no_grad():
            self.compute_linear_eta(x_in)
            self.compute_eta()
            Z = 2*torch.round(self.eta)-1
            EZ = 2*self.eta-1
        return Z,EZ
    
    def predict_Y(self,x_in):
        with torch.no_grad():
            Z, EZ = self.predict_Z(x_in)
            numerator_Z =  torch.exp(self.beta_0_tensor[None,:] + torch.matmul(Z,self.beta_tensor.T))
            denominator_Z = torch.sum(numerator_Z,dim=1)
            
            numerator_EZ =  torch.exp(self.beta_0_tensor[None,:] + torch.matmul(EZ,self.beta_tensor.T))
            denominator_EZ = torch.sum(numerator_EZ,dim=1)
            
            P_YgivenZi_Z = numerator_Z/denominator_Z[:,None]
            P_YgivenZi_EZ = numerator_EZ/denominator_EZ[:,None]

            
            _, Y_pred_Z = torch.max(P_YgivenZi_Z,1)
            _, Y_pred_EZ = torch.max(P_YgivenZi_EZ,1)
            return Y_pred_Z,Y_pred_EZ    
      
    
    def predict_Y_Top5(self,x_in):
        with torch.no_grad():
            Z, EZ = self.predict_Z(x_in)
            
            numerator_Z =  torch.exp(self.beta_0_tensor[None,:] + torch.matmul(Z,self.beta_tensor.T))
            denominator_Z = torch.sum(numerator_Z,dim=1)
            
            numerator_EZ =  torch.exp(self.beta_0_tensor[None,:] + torch.matmul(EZ,self.beta_tensor.T))
            denominator_EZ = torch.sum(numerator_EZ,dim=1)
            
            P_YgivenZi_Z = numerator_Z/denominator_Z[:,None]
            P_YgivenZi_EZ = numerator_EZ/denominator_EZ[:,None]
           
            top_5_ps, top_5_classes = P_YgivenZi_Z.topk(5, dim=1)
            top_5_psEZ, top_5_classesEZ = P_YgivenZi_EZ.topk(5, dim=1)
            
        return top_5_classes, top_5_classesEZ
    
    def printNetworkInfo(self):
            print('Alpha_0: Min: {:.2f}\tMax: {:.2f}'.format(torch.min(self.alpha_0_tensor),torch.max(self.alpha_0_tensor)))
            print('Alpha: Min: {:.2f}\tMax: {:.2f}'.format(torch.min(self.alpha_tensor),torch.max(self.alpha_tensor)))
            print('beta_0: Min: {:.2f}\tMax: {:.2f}'.format(torch.min(self.beta_0_tensor),torch.max(self.beta_0_tensor)))
            print('beta: Min: {:.2f}\tMax: {:.2f}'.format(torch.min(self.beta_tensor),torch.max(self.beta_tensor)))
            print('Eta: Min: {:.4f}\tMax: {:.4f}'.format(torch.min(self.eta),torch.max(self.eta)))
            
            print('Z: {}'.format(torch.sum(torch.mean(self.Z,dim = 0),dim = 1)))
            
            print('Pi+: Min: {:.4f}\tMax: {:.4f}'.format(torch.min(self.pi_pos),torch.max(self.pi_pos))) 
            print('Pi-: Min: {:.4f}\tMax: {:.4f}'.format(torch.min(self.pi_neg),torch.max(self.pi_neg))) 
            print('Pi: Min: {:.4f}\tMax: {:.4f}'.format(torch.min(self.pi),torch.max(self.pi))) 
            
            print('EY_X: Min: {:.4f}\tMax: {:.4f}'.format(torch.min(self.EY_X/self.num_samples),torch.max(self.EY_X/self.num_samples))) 
            print('EY_X_j: Min: {:.4f}\tMax: {:.4f}'.format(torch.min(self.EY_X_j/self.num_samples),torch.max(self.EY_X_j/self.num_samples))) 
            
            print('EI: Min: {:.4f}\tMax: {:.4f}'.format(torch.min(self.EI),torch.max(self.EI))) 
            print('EII: Min: {:.4f}\tMax: {:.4f}'.format(torch.min(self.EII),torch.max(self.EII))) 
            print('EIII: Min: {:.4f}\tMax: {:.4f}'.format(torch.min(self.EIII),torch.max(self.EIII))) 
            
            print('TermI: {:.2f}'.format(self.TermI)) 
            print('TermII: {:.2f}'.format(self.TermII))
            print('TermIII: {:.2f}'.format(self.TermIII))
            print('TermIV: {:.2f}'.format(self.TermIV))
            print('Expected Log Likelihood: {:.2f}\n\n'.format(self.E_log_likelihood))


    def save_models(self,epoch,optimizer_dict,save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if optimizer_dict is not None:
            state = {
                'epoch': epoch,
                'state_dict': self.state_dict(),
                'optimizer': optimizer_dict,
                }
        else:
            state = {
                'epoch': epoch,
                'state_dict': self.state_dict(),
                }
        torch.save(state, save_dir + "/LBCNN" + "_{}.model".format(epoch))
        print("Check Point Saved") 
         
    def forward(self,x_in,y_in,epoch,num_epochs):  
        self.BATCH_SIZE = y_in.shape[0]
        #print('Input: {}'.format(x_in.shape))
        self.compute_linear_eta(x_in)
        self.compute_eta()
        
        
        if epoch < 0.25*num_epochs:
            p = torch.tensor([0.25])
            coin_flip = torch.bernoulli(torch.tensor(p))
            if coin_flip == 1:
                self.sample_Z()
            if coin_flip == 0:
                self.sample_Z_via_prior()
        else:
            self.sample_Z()
        
        self.compute_pi_j(y_in)
        self.compute_pi(y_in)
        
        self.compute_EY_X()
        self.compute_EY_X_j()
        
        self.compute_EI()
        self.compute_EII()
        self.compute_EIII(y_in)
        
        
        self.compute_E_logLikelihood(y_in)


        
        return self.E_log_likelihood