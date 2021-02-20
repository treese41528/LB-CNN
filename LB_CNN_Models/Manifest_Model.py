# -*- coding: utf-8 -*-
# @Author: Tim Reese
# @Date:   2021-02-20

import numpy as np
import torch
import torch.nn as nn
import os

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


class Manifest_Model(nn.Module):
    def __init__(self,num_features,num_classes,code_size,debug_mode = False):
        super(Manifest_Model,self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.M = code_size
        
        init_constant = 0.1/np.sqrt(self.num_features)
        init_constant2 = 0.1/np.sqrt(self.M)
        self.alpha_0_tensor = torch.nn.Parameter(torch.normal(torch.zeros([self.M]),torch.ones([self.M])).type(torch.DoubleTensor))
        self.alpha_tensor = torch.nn.Parameter(torch.normal(torch.zeros([self.M,self.num_features]),init_constant*torch.ones([self.M,self.num_features])).type(torch.DoubleTensor))
        
        self.beta_0_tensor = torch.nn.Parameter(torch.normal(torch.zeros([self.num_classes]),torch.ones([self.num_classes])).type(torch.DoubleTensor))
        self.beta_tensor = torch.nn.Parameter(torch.normal(torch.zeros([self.num_classes,self.M]),init_constant2*torch.ones([self.num_classes,self.M])).type(torch.DoubleTensor))
        
        
        
        self.debug_mode = debug_mode

        
    def compute_linear_alpha(self,x_in):
        self.linear_alpha = self.alpha_0_tensor[None,:] + torch.matmul(x_in,self.alpha_tensor.t())

    def compute_linear_beta(self,z_in):
        self.linear_beta = self.beta_0_tensor[:,None] + torch.matmul(self.beta_tensor,z_in.t())
        self.linear_beta = self.linear_beta.t()

    def compute_eta(self):
        with torch.no_grad():
            self.eta = torch.exp(self.linear_alpha)/(1.0+torch.exp(self.linear_alpha))

    def compute_pi(self):
        with torch.no_grad():
            self.pi = torch.exp(self.linear_beta)/torch.sum(torch.exp(self.linear_beta),dim=1,keepdim = True)

    def evaluate_linearq(self,z_in):
        with torch.no_grad():
            ones = torch.ones_like(z_in)
            ind_z = (z_in+1.0)/2.0
            self.linearQ = torch.mul(ind_z,self.linear_alpha) -torch.mul(ones,torch.log(1.0+torch.exp(self.linear_alpha))) #n 
            
    def evaluate_Q(self,z_in):
        self.evaluate_linearq(z_in)
        return torch.exp(torch.sum(self.linearQ,dim=1))
        
    def compute_TermI(self,y_in):
        with torch.no_grad():
            device = next(self.parameters()).device
            y_onehot = one_hot_embedding(y_in, self.num_classes).to(device)
        
        self.termI = torch.sum(torch.sum(torch.mul(y_onehot,self.linear_beta),axis=1),axis=0)


    def compute_TermII(self):
        self.termII = torch.sum(torch.log(torch.sum(torch.exp(self.linear_beta),dim=1)),dim=0) 

    def compute_TermIII(self,z_in):
        with torch.no_grad():
            Z_onehot = (z_in+1.0)/2.0
        self.termIII = torch.sum(torch.sum(torch.mul(Z_onehot,self.linear_alpha),dim = 1),dim = 0)

    def compute_TermIV(self):
        self.termIV = torch.sum(torch.sum(torch.log(1.0+torch.exp(self.linear_alpha)),dim=1),dim=0)

    
    def compute_logLikelihood(self,z_in,y_in):
        self.compute_TermI(y_in)
        self.compute_TermII()
        self.compute_TermIII(z_in)
        self.compute_TermIV()
        
        self.log_likelihood = self.termI-self.termII+self.termIII-self.termIV
        if self.debug_mode:
            print('Alpha: Min: {:.2f}\tMax: {:.2f}'.format(torch.min(self.alpha_0_tensor),torch.max(self.alpha_tensor)))
            print('beta: Min: {:.2f}\tMax: {:.2f}'.format(torch.min(self.beta_0_tensor),torch.max(self.beta_tensor)))
            print('Eta: Min: {:.4f}\tMax: {:.4f}'.format(torch.min(self.eta),torch.max(self.eta)))
            print('Pi: Min: {:.4f}\tMax: {:.4f}'.format(torch.min(self.pi),torch.max(self.pi))) 

            print('I: {:.2f}'.format(self.termI)) 
            print('II: {:.2f}'.format(self.termII))
            print('III: {:.2f}'.format(self.termIII))
            print('IV: {:.2f}'.format(self.termIV))
            print('Log Likelihood: {:.2f}\n\n'.format(self.log_likelihood))
                 
    def predict_Z(self,x_in):
        with torch.no_grad():
            self.compute_linear_alpha(x_in)
            self.compute_eta()
            
            Z = 2*torch.round(self.eta)-1
            EZ = 2*self.eta-1
        return Z,EZ
    
    
    def predict_Y(self,x_in):
        with torch.no_grad():
            Z, EZ = self.predict_Z(x_in)
        
            self.compute_linear_beta(Z)
            self.compute_pi()
            P_Ygiven_Z = self.pi
            _, Y_pred_Z = torch.max(P_Ygiven_Z,1)
        
            self.compute_linear_beta(EZ)
            self.compute_pi()
            P_Ygiven_EZ = self.pi
            _, Y_pred_EZ = torch.max(P_Ygiven_EZ,1)
            
            return Y_pred_Z,Y_pred_EZ
   
    def predict_Y_Top5(self,x_in):
        with torch.no_grad():
            Z, EZ = self.predict_Z(x_in)
            Q = self.evaluate_Q(Z)
            
            self.compute_linear_beta(Z)
            self.compute_pi()
            
            P_Ygiven_Z = torch.mul(self.pi,Q[:,None])
            
            top_5_ps, top_5_classes = P_Ygiven_Z.topk(5, dim=1)
        
            self.compute_linear_beta(EZ)
            self.compute_pi()
            P_Ygiven_EZ = self.pi
            top_5_psEZ, top_5_classesEZ = P_Ygiven_EZ.topk(5, dim=1)
        return top_5_classes, top_5_classesEZ


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
            
        torch.save(state, save_dir + "ManifestBCNN_{}.model".format(epoch))
        print("Check Point Saved")

    def forward(self,x_in,z_in,y_in):
        self.compute_linear_alpha(x_in)
        self.compute_eta()
        
        self.compute_linear_beta(z_in)
        self.compute_pi()
        
        self.compute_logLikelihood(z_in,y_in)
        return self.log_likelihood
    
    
    