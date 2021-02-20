# -*- coding: utf-8 -*-
# @Author: Tim Reese
# @Date:   2021-02-20

import numpy as np
import torch
import torch.nn as nn
import os
import itertools

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

def binaryValues(k):
    bins = list(itertools.product([0, 1], repeat=k))
    bins = np.array(bins).reshape(-1,k).astype(dtype = np.float32)
    return bins

        
class LB_CNN(nn.Module):
    def __init__(self,num_features,num_classes,latent_size,debug_mode = False):
        super(LB_CNN,self).__init__()
        
        self.num_classes = num_classes
        self.M = latent_size
        
        self.num_features = num_features
        
        init_constant = 2.0/np.sqrt(self.num_features)
        self.alpha_0_tensor = torch.nn.Parameter(torch.normal(torch.zeros(self.M),torch.ones(self.M)))
        self.alpha_tensor = torch.nn.Parameter(torch.normal(torch.zeros([self.M,self.num_features]),init_constant*torch.ones([self.M,self.num_features])))
        
        init_constant2 = 2.0/np.sqrt(self.M)
        self.beta_0_tensor = torch.nn.Parameter(torch.normal(torch.zeros(self.num_classes),torch.ones(self.num_classes)))
        self.beta_tensor = torch.nn.Parameter(torch.normal(torch.zeros([self.num_classes,self.M]),init_constant2*torch.ones([self.num_classes,self.M])))


        self.code_book = binaryValues(self.M)
        self.debug_mode = debug_mode

    def compute_linear_eta(self,x_input):
        #self.linear_eta = torch.clamp(self.alpha_0_tensor + torch.matmul(x_input,self.alpha_tensor.t()) , min=-30.0, max=30.0)
        self.linear_eta = self.alpha_0_tensor + torch.matmul(x_input,self.alpha_tensor.t())

    
    def compute_eta(self):
        #self.P_ZjgivenXi <-- n x M
        with torch.no_grad():
            self.eta = torch.exp(self.linear_eta)/(1.0+torch.exp(self.linear_eta))
            
    
    def compute_pi_codes(self):
        mod_codes = 2*self.code_book_tensor-1
        self.linear_pi = self.beta_0_tensor[:,None] +torch.matmul(self.beta_tensor,mod_codes.t())


    def compute_pi(self):
        with torch.no_grad():
            exp_pi = torch.exp(self.linear_pi)
            sum_codes = torch.sum(exp_pi,dim=0)
            self.pi = exp_pi/sum_codes   

    #Computed using Independence
    def compute_P_ZgivenXi(self):
        #self.code_book_tensor
        with torch.no_grad():
            prod_eta_codes = torch.matmul(self.code_book_tensor,self.linear_eta.t())
            ones_matrix = torch.ones_like(self.code_book_tensor)
            
            log_one_eta = torch.log(1.0 + torch.exp(self.linear_eta))
            prod_log_term = torch.matmul(ones_matrix,log_one_eta.t())
            
            self.P_ZgivenX = torch.exp(prod_eta_codes - prod_log_term).t()
    
    def compute_E_IndZ(self,y_in):
        with torch.no_grad():
            device = next(self.parameters()).device
            self.E_IndZ = torch.zeros([self.BATCH_SIZE,self.M]).to(device)
            y_onehot = one_hot_embedding(y_in, self.num_classes).to(device)
            
            numerator = torch.mul(torch.matmul(y_onehot,self.pi),self.P_ZgivenX)
            denominator = torch.sum(numerator,dim=1) #sum over codes
            #Consider numerator for j=1,2,...,M all other codes summed over
            for j in range(self.M):
                index_ones = (self.code_book_tensor[:,j]==1).nonzero()
                self.E_IndZ[:,j] = torch.sum(numerator[:,index_ones].squeeze(),dim=1)/denominator


    def compute_EI(self,y_in):
        with torch.no_grad():
            device = next(self.parameters()).device
            y_onehot = one_hot_embedding(y_in, self.num_classes).to(device)
        EZ = 2*self.E_IndZ-1
        linear = self.beta_0_tensor[None,:] + torch.matmul(EZ,self.beta_tensor.t())
        self.EI = torch.sum(torch.sum(torch.mul(y_onehot,linear),dim = 1),dim = 0)
        
    def compute_EII(self,y_in):
        with torch.no_grad():
            device = next(self.parameters()).device
            y_onehot = one_hot_embedding(y_in, self.num_classes).to(device)
            
            numerator = torch.mul(torch.matmul(y_onehot,self.pi),self.P_ZgivenX)
            denominator = torch.sum(numerator,dim=1,keepdim = True) 
            prob_log = numerator / denominator #n x 2^M
        
        log_terms = torch.log(torch.sum(torch.exp(self.linear_pi),dim=0)) #2^M
        self.EII = torch.sum(torch.sum(torch.mul(log_terms[None,:],prob_log),dim=1),dim=0)
        
    
    def compute_EIII(self):
        linear = torch.mul(self.E_IndZ,self.linear_eta)
        self.EIII = torch.sum(torch.sum(linear,dim = 1),dim = 0)
        
    def compute_EIV(self):
        log_term = torch.log(1.0 + torch.exp(self.linear_eta))
        self.EIV = torch.sum(torch.sum(log_term,dim=1),dim=0)
                
                
                

    def compute_E_logLikelihood(self,y_in):
        self.compute_E_IndZ(y_in)
        self.compute_EI(y_in)
        self.compute_EII(y_in)
        self.compute_EIII()
        self.compute_EIV()
        
        self.E_log_likelihood = self.EI - self.EII + self.EIII - self.EIV
        if self.debug_mode:
            print('Alpha: Min: {:.2f}\tMax: {:.2f}'.format(torch.min(self.alpha_0_tensor),torch.max(self.alpha_tensor)))
            print('beta: Min: {:.2f}\tMax: {:.2f}'.format(torch.min(self.beta_0_tensor),torch.max(self.beta_tensor)))
            print('Eta: Min: {:.4f}\tMax: {:.4f}'.format(torch.min(self.eta),torch.max(self.eta)))
            print('Pi: Min: {:.4f}\tMax: {:.4f}'.format(torch.min(self.pi),torch.max(self.pi))) 
            print('P(Z|X): Min: {:.4f}\tMax: {:.4f}'.format(torch.min(self.P_ZgivenX),torch.max(self.P_ZgivenX)))   
            print('P(Z|X,Y): Min: {:.4f}\tMax: {:.4f}'.format(torch.min(self.E_IndZ),torch.max(self.E_IndZ))) 
            print('E_I: {:.2f}'.format(self.EI)) 
            print('E_II: {:.2f}'.format(self.EII))
            print('E_III: {:.2f}'.format(self.EIII))
            print('E_IV: {:.2f}'.format(self.EIV))
            print('Expected Log Likelihood: {:.2f}\n\n'.format(self.E_log_likelihood))

                 
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
            numerator_Z =  torch.exp(self.beta_0_tensor[:,None] + torch.matmul(self.beta_tensor,Z.t()))
            denominator_Z = torch.sum(numerator_Z,dim=0)
            
            numerator_EZ =  torch.exp(self.beta_0_tensor[:,None] + torch.matmul(self.beta_tensor,EZ.t()))
            denominator_EZ = torch.sum(numerator_EZ,dim=0)
            
            P_YgivenZi_Z = numerator_Z/denominator_Z
            P_YgivenZi_EZ = numerator_EZ/denominator_EZ
            
            _, Y_pred_Z = torch.max(P_YgivenZi_Z,0)
            _, Y_pred_EZ = torch.max(P_YgivenZi_EZ,0)
            return Y_pred_Z,Y_pred_EZ
    def printNetworkInfo(self):
        print('Alpha: Min: {:.2f}\tMax: {:.2f}'.format(torch.min(self.alpha_0_tensor),torch.max(self.alpha_tensor)))
        print('beta: Min: {:.2f}\tMax: {:.2f}'.format(torch.min(self.beta_0_tensor),torch.max(self.beta_tensor)))
        print('Eta: Min: {:.4f}\tMax: {:.4f}'.format(torch.min(self.eta),torch.max(self.eta)))
        print('Pi: Min: {:.4f}\tMax: {:.4f}'.format(torch.min(self.pi),torch.max(self.pi))) 
        print('P(Z|X): Min: {:.4f}\tMax: {:.4f}'.format(torch.min(self.P_ZgivenX),torch.max(self.P_ZgivenX)))   
        print('P(Z|X,Y): Min: {:.4f}\tMax: {:.4f}'.format(torch.min(self.E_IndZ),torch.max(self.E_IndZ))) 
        print('E_I: {:.2f}'.format(self.EI)) 
        print('E_II: {:.2f}'.format(self.EII))
        print('E_III: {:.2f}'.format(self.EIII))
        print('E_IV: {:.2f}'.format(self.EIV))
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
        torch.save(state, save_dir + "/LBCNN " + "_{}.model".format(epoch))
        print("Check Point Saved") 
        
        
    def forward(self,x_in,y_in):
        self.BATCH_SIZE = y_in.shape[0]
        device = next(self.parameters()).device
        self.code_book_tensor = torch.from_numpy(self.code_book).type(torch.FloatTensor).to(device)        
        
        
        self.compute_linear_eta(x_in)
        self.compute_eta()
        
        self.compute_pi_codes()
        self.compute_pi()
        self.compute_P_ZgivenXi()
        
        self.compute_E_logLikelihood(y_in)

        
        return self.E_log_likelihood

    
    