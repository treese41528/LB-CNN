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


class LB_CNN_Gibbs(nn.Module):
    def __init__(self,num_features,num_classes,latent_size,num_samples,sample_Type = 1,debug_mode = False):
        super(LB_CNN_Gibbs,self).__init__()
        self.num_classes = num_classes
        self.M = latent_size
        self.num_features = num_features
        self.num_samples = num_samples
        self.sample_Type = sample_Type
        
        init_constant = 2.0/np.sqrt(self.num_features)
        self.alpha_0_tensor = torch.nn.Parameter(torch.normal(torch.zeros(self.M),torch.ones(self.M)).type(torch.DoubleTensor))
        self.alpha_tensor = torch.nn.Parameter(torch.normal(torch.zeros([self.M,self.num_features]),init_constant*torch.ones([self.M,self.num_features])).type(torch.DoubleTensor))
        
        init_constant2 = 2.0/np.sqrt(self.M)
        self.beta_0_tensor = torch.nn.Parameter(torch.normal(torch.zeros(self.num_classes),torch.ones(self.num_classes)).type(torch.DoubleTensor))
        self.beta_tensor = torch.nn.Parameter(torch.normal(torch.zeros([self.num_classes,self.M]),init_constant2*torch.ones([self.num_classes,self.M])).type(torch.DoubleTensor))

        self.debug_mode = debug_mode
        
        
    def compute_linear_pi(self,x_in):
        self.linear_pi = self.alpha_0_tensor[None,:] + torch.matmul(x_in,self.alpha_tensor.T)
    
    def compute_pi_pos(self):
        with torch.no_grad():
  
            self.pi_pos = torch.exp(self.linear_pi)/(1.0 + torch.exp(self.linear_pi))    
            
    def init_sample1(self):
        with torch.no_grad():
            device = next(self.parameters()).device
            p = 0.5*torch.ones_like(self.pi_pos).to(device)
            bern = torch.distributions.bernoulli.Bernoulli(p)
            self.ind_ZgivenX = bern.sample(sample_shape=torch.Size([self.num_samples])) #Indicator
            self.Z = 2.0*self.ind_ZgivenX-1.0 #+1/-1 
            
    def init_sample2(self):
        with torch.no_grad():
            bern = torch.distributions.bernoulli.Bernoulli(self.pi_pos)
            self.ind_ZgivenX = bern.sample(sample_shape=torch.Size([self.num_samples])) #Indicator
            self.Z = 2.0*self.ind_ZgivenX-1.0 #+1/-1 
            
    def init_sample3(self,y_in):
        with torch.no_grad():
            device = next(self.parameters()).device
            y_onehot = one_hot_embedding(y_in, self.num_classes).type(torch.DoubleTensor).to(device)
            linear_gamma_n_pos =  torch.matmul(y_onehot,self.beta_0_tensor[:,None] +self.beta_tensor) #N x M
            linear_gamma_n_neg =  torch.matmul(y_onehot,self.beta_0_tensor[:,None] -self.beta_tensor) #N x M
            
            linear_gamma_pos = self.beta_0_tensor[None,:] + self.beta_tensor.T #MxK
            linear_gamma_neg = self.beta_0_tensor[None,:] - self.beta_tensor.T #MxK
            
            p_gamma_pos = torch.exp(linear_gamma_n_pos) / torch.sum(torch.exp(linear_gamma_pos),dim =1)[None,:]
            p_gamma_neg = torch.exp(linear_gamma_n_neg) / torch.sum(torch.exp(linear_gamma_neg),dim =1)[None,:]
            
            numerator = torch.mul(p_gamma_pos,self.pi_pos)
            denominator = numerator + torch.mul(p_gamma_neg,1.0 - self.pi_pos)
            p = numerator / denominator
            bern = torch.distributions.bernoulli.Bernoulli(p)
            self.ind_ZgivenX = bern.sample(sample_shape=torch.Size([self.num_samples])) #Indicator
            self.Z = 2.0*self.ind_ZgivenX-1.0 #+1/-1

    def gibbs_update(self,y_in):
        with torch.no_grad():
            device = next(self.parameters()).device
            y_onehot = one_hot_embedding(y_in, self.num_classes).type(torch.DoubleTensor).to(device)
            
            gibbsOrder = np.arange(self.M)
            np.random.shuffle(gibbsOrder)
            for j in gibbsOrder:
                Z_sample_pos = self.Z.clone()
                Z_sample_pos[:,:,j] = 1.0
                
                Z_sample_neg = self.Z.clone()
                Z_sample_neg[:,:,j] = -1.0
                
                
                linear_gamma_pos_j = self.beta_0_tensor[None,:] + torch.matmul(Z_sample_pos,self.beta_tensor.T) 
                
                linear_gamma_pos_j_n = torch.sum(torch.mul(y_onehot,linear_gamma_pos_j),dim = 2)
                gamma_pos_j_den = torch.sum(torch.exp(linear_gamma_pos_j),dim = 2)
                gamma_pos_j = torch.mul(torch.exp(linear_gamma_pos_j_n),1.0/gamma_pos_j_den)
                
                
                linear_gamma_neg_j = self.beta_0_tensor[None,:] + torch.matmul(Z_sample_neg,self.beta_tensor.T) 
                
                linear_gamma_neg_j_n = torch.sum(torch.mul(y_onehot,linear_gamma_neg_j),dim =2)
                gamma_neg_j_den = torch.sum(torch.exp(linear_gamma_neg_j),dim = 2) 
                gamma_neg_j = torch.mul(torch.exp(linear_gamma_neg_j_n),1.0/gamma_neg_j_den)
                
                gamma_pi_pos_numerator = torch.mul(gamma_pos_j,self.pi_pos[:,j])
                gamma_pi_denom = gamma_pi_pos_numerator + torch.mul(gamma_neg_j,1.0-self.pi_pos[:,j])
                gamma_pi_pos = gamma_pi_pos_numerator/gamma_pi_denom
                
                bern = torch.distributions.bernoulli.Bernoulli(gamma_pi_pos)
                ind_ZgivenXYj = bern.sample() #Indicator
                self.Z[:,:,j] = 2.0*ind_ZgivenXYj-1.0
                
                
    def compute_EI(self):
        with torch.no_grad():
            self.EI = torch.mean((self.Z+1.0)/2.0,dim = 0) #Expected value of indicator
          
    def compute_EII(self):
        with torch.no_grad():
            self.EII = torch.mean(self.Z,dim = 0)   #Expected Value of Z
            
    def compute_EIII(self,y_in):
        linear_gamma = self.beta_0_tensor[None,None,:] + torch.matmul(self.Z,self.beta_tensor.T)
        log_term = torch.log(torch.sum(torch.exp(linear_gamma),dim=2))
        self.EIII = torch.mean(log_term,dim = 0)

    def compute_TermI(self,y_in):
        device = next(self.parameters()).device
        y_onehot = one_hot_embedding(y_in, self.num_classes).type(torch.DoubleTensor).to(device)
        
        linear_term = self.beta_0_tensor[None,:] + torch.matmul(self.EII,self.beta_tensor.T)
        self.TermI = torch.sum(torch.sum(torch.mul(y_onehot,linear_term),dim = 1), dim = 0)
        
    def compute_TermII(self):
        self.TermII = torch.sum(self.EIII,dim = 0)
        
        
    def compute_TermIII(self):
        linear_term = torch.mul(self.EI,self.linear_pi)
        self.TermIII = torch.sum(torch.sum(linear_term, dim = 1), dim = 0)
        
    def compute_TermIV(self):
        log_term = torch.log(1.0 + torch.exp(self.linear_pi))
        self.TermIV = torch.sum(torch.sum(log_term, dim = 1), dim = 0)
                

    def compute_E_logLikelihood(self,y_in):
        self.compute_TermI(y_in)
        self.compute_TermII()
        self.compute_TermIII()
        self.compute_TermIV()
        
        self.E_log_likelihood = self.TermI - self.TermII + self.TermIII - self.TermIV
        if self.debug_mode:
            self.printNetworkInfo() 
        
    def printNetworkInfo(self):
            print('Alpha_0: Min: {:.2f}\tMax: {:.2f}'.format(torch.min(self.alpha_0_tensor),torch.max(self.alpha_0_tensor)))
            print('Alpha: Min: {:.2f}\tMax: {:.2f}'.format(torch.min(self.alpha_tensor),torch.max(self.alpha_tensor)))
            print('beta_0: Min: {:.2f}\tMax: {:.2f}'.format(torch.min(self.beta_0_tensor),torch.max(self.beta_0_tensor)))
            print('beta: Min: {:.2f}\tMax: {:.2f}'.format(torch.min(self.beta_tensor),torch.max(self.beta_tensor)))
            print('Pi: Min: {:.4f}\tMax: {:.4f}'.format(torch.min(self.pi_pos),torch.max(self.pi_pos)))

        
            
            print('EI: Min: {:.4f}\tMax: {:.4f}'.format(torch.min(self.EI),torch.max(self.EI))) 
            print('EII: Min: {:.4f}\tMax: {:.4f}'.format(torch.min(self.EII),torch.max(self.EII))) 
            print('EIII: Min: {:.4f}\tMax: {:.4f}'.format(torch.min(self.EIII),torch.max(self.EIII))) 
            
            print('TermI: {:.2f}'.format(self.TermI)) 
            print('TermII: {:.2f}'.format(self.TermII))
            print('TermIII: {:.2f}'.format(self.TermIII))
            print('TermIV: {:.2f}'.format(self.TermIV))
            print('Expected Log Likelihood: {:.2f}\n\n'.format(self.E_log_likelihood))
        
    def predict_Z(self,x_in):
        with torch.no_grad():
            self.compute_linear_pi(x_in)
            self.compute_pi_pos()
            Z = 2*torch.round(self.pi_pos)-1
            EZ = 2*self.pi_pos-1
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
            return Z, EZ, Y_pred_Z, Y_pred_EZ  
      
    
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
        torch.save(state, save_dir + "/GibbsSampler_LBCNN " + "_{}.model".format(epoch))
        print("Check Point Saved") 
         
    def forward(self,x_in,y_in):  
        
        self.compute_linear_pi(x_in)
        self.compute_pi_pos()
        if self.sample_Type == 1:
            self.init_sample1()
        if self.sample_Type == 2:
            self.init_sample2()
        if self.sample_Type == 3:
            self.init_sample3(y_in)
            
        self.gibbs_update(y_in)
        
        self.compute_EI()
        self.compute_EII()
        self.compute_EIII(y_in)
        
        
        self.compute_E_logLikelihood(y_in)

        
        return self.E_log_likelihood

                
            
    
        
        
        
        
        