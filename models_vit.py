import timm
#timm.list_models('vit*', pretrained=True)

#import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final

from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_, resample_patch_embed, \
    resample_abs_pos_embed, RmsNorm, PatchDropout, use_fused_attn, SwiGLUPacked

from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import named_apply, checkpoint_seq, adapt_input_conv
from timm.models._registry import generate_default_cfgs, register_model, register_model_deprecations    
    

import random
import numpy as np
from attrdict import AttrDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import parameters_to_vector
from torch.autograd import Variable
        
from functorch import vmap,grad,jacfwd,jacrev
from functorch import make_functional, make_functional_with_buffers

from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal as LowMVN
from torch.distributions.kl import kl_divergence
from attrdict import AttrDict



## https://pytorch.org/blog/Accelerating-Hugging-Face-and-TIMM-models/
## https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py / baseline
# import timm
# model = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=10)

from timm.models.registry import register_model
from timm.models.vision_transformer import _create_vision_transformer
from timm.models.vision_transformer import checkpoint_filter_fn
from timm.models.vision_transformer import VisionTransformer


## --------------------------------------------------------------------------------------------
## get_vit_model
## --------------------------------------------------------------------------------------------

def _get_vit_model(configs):
    modelname = configs.model.modelname
    pretrained = configs.model.pretrained
    infer = configs.model.infer
    outdim = configs.model.outdim
    
    priorstdout = configs.model.priorstdout if 'priorstdout' in configs.model.keys() else 0.0  
    temppeak = configs.model.temppeak if 'temppeak' in configs.model.keys() else 0.5  

    # if modelname == 'vittp16r224':
    #     model = vit_bayes_tiny_patch16_224(pretrained=pretrained,infer=infer,num_classes=outdim)
    
    if modelname == 'vitbp16r224':
        model = vit_bayes_base_patch16_224(pretrained=pretrained,infer=infer,num_classes=outdim, priorstdout = priorstdout, temppeak=temppeak)
    
    
    global_pool = model.global_pool 
    model.reset_classifier(num_classes=outdim, global_pool=global_pool)
    
    return model



def _create_vision_transformer_Bayes(variant, pretrained=False, infer='map', **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    if 'flexi' in variant:
        # FIXME Google FlexiViT pretrained models have a strong preference for bilinear patch / embed
        # interpolation, other pretrained models resize better w/ anti-aliased bicubic interpolation.
        _filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear', antialias=False)
    else:
        _filter_fn = checkpoint_filter_fn

    if infer == 'map':
        vit_cls = VisionTransformer_Map

    if infer == 'vbw':
        vit_cls = VisionTransformer_Bayes_W

    if infer == 'vbf':
        vit_cls = VisionTransformer_Bayes_F

    if infer == 'vbfwojkde':
        vit_cls = VisionTransformer_Bayes_F_WO_Jacobian_KDE

    if infer == 'vbfwojkde2':
        vit_cls = VisionTransformer_Bayes_F_WO_Jacobian_KDE_ADVH
        

    return build_model_with_cfg(
        vit_cls,
        variant,
        pretrained,
        pretrained_filter_fn=_filter_fn,
        **kwargs,
    )


@register_model
def vit_bayes_tiny_patch16_224(pretrained=False,infer='map', **kwargs) -> VisionTransformer:
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_vision_transformer_Bayes('vit_tiny_patch16_224', pretrained=pretrained, infer=infer,**dict(model_args, **kwargs))
    return model

@register_model
def vit_bayes_base_patch16_224(pretrained=False,infer='map', **kwargs) -> VisionTransformer:
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer_Bayes('vit_base_patch16_224', pretrained=pretrained, infer=infer, **dict(model_args, **kwargs))
    return model

@register_model
def vit_bayes_base_patch16_384(pretrained=False,infer='map', **kwargs) -> VisionTransformer:
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer_Bayes('vit_base_patch16_384', pretrained=pretrained, infer=infer, **dict(model_args, **kwargs))
    return model




## --------------------------------------------------------------------------------------------
## vit map 
## reference model: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
## --------------------------------------------------------------------------------------------

from timm.models.vision_transformer import (VisionTransformer,
                                            Attention, 
                                            Block,
                                            Mlp, 
                                            PatchEmbed,
                                            get_init_weights_vit,
                                            _load_weights
)


#class VisionTransformer_LastFBayes(nn.Module):
class VisionTransformer_Map(VisionTransformer):

    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            no_embed_class: bool = False,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: str = '',
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[Callable] = None,
            act_layer: Optional[Callable] = None,
            block_fn: Callable = Block,
            mlp_layer: Callable = Mlp,
            priorstdout: float = 10.,
            temppeak: float = 1.,  
    ):
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super(VisionTransformer_Map,self).__init__(   img_size,
                                                    patch_size,
                                                    in_chans,
                                                    num_classes,
                                                    global_pool,
                                                    embed_dim,
                                                    depth,
                                                    num_heads,
                                                    mlp_ratio,
                                                    qkv_bias,
                                                    qk_norm,
                                                    init_values,
                                                    class_token,
                                                    no_embed_class,
                                                    pre_norm,
                                                    fc_norm,
                                                    drop_rate,
                                                    pos_drop_rate,
                                                    patch_drop_rate,
                                                    proj_drop_rate,
                                                    attn_drop_rate,
                                                    drop_path_rate,
                                                    weight_init,
                                                    embed_layer,
                                                    norm_layer,
                                                    act_layer,
                                                    block_fn,
                                                    mlp_layer,            
                                )

        ## Classifier Head
        self.global_pool = global_pool        
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head_bayes = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        #self.head_logstd = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()



        if weight_init != 'skip':
            self.init_weights(weight_init)


    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_logstd = nn.Parameter(torch.zeros(self.embed_dim*num_classes))
        return 




    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        # return x if pre_logits else self.head(x)
        return x if pre_logits else self.head_bayes(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x



    ##------------------------------------------------------------------------------------------------------------------------------
    ## add implementation for map
    ##------------------------------------------------------------------------------------------------------------------------------
    def predict_train(self, x , xc, nfsample=1 ,p_params=None, configs=None):        
        if configs is not None:
            reglambda = configs.train.reglambda
            nfsample = configs.train.nfsample
        else:            
            reglambda = 1e-3
            nfsample = 1

        
        self.train()    
            
        # compute logits
        logits = self.forward(x) #(nb,nout)                
        
        # compute reg
        #sup_kl = torch.tensor(0.0).to(x.device)                    
        param_vec = parameters_to_vector(self.parameters())
        reg = 0.5 * torch.norm(param_vec, p = 2)
        outs = AttrDict()
        outs.logits = logits
        outs.reg = reg                
        return outs        
        
        
    def predict_eval(self, x ,topk=10, nfsample=1):        
        self.eval()        
        y = self.forward(x)
        pred = F.softmax(y,dim=-1)           
        
        outs = AttrDict()
        outs.pred = pred  #prob               
        outs.logits = y.unsqueeze(dim=0)       
        
        return outs         









## --------------------------------------------------------------------------------------------
## vit last layer vb on weight space 
## --------------------------------------------------------------------------------------------

scalelogstd = -4
eps_log     = 1e-8
eps_zitter  = 1e-8
eps_clamp   = 1e-8



def transform_logstd(logstd):
    return 1e-6 + F.softplus(logstd, beta=1, threshold=20)

def build_MVN_Low(params:list , asprior=True):
    qf_mean,qf_Lcov = params
    zitter_mat = eps_zitter*torch.ones_like(qf_mean).to(qf_mean.device)                    
    if asprior:
        qf_dist = LowMVN(qf_mean.detach().clone(), qf_Lcov.detach().clone(), zitter_mat.detach().clone() )                        
    else:
        qf_dist = LowMVN(qf_mean ,qf_Lcov, zitter_mat)                                
    return qf_dist

def build_MVN_Full(params:list , asprior=True):
    qf_mean,qf_cov = params
    zitter_mat = eps_zitter*torch.eye(qf_mean.shape[-1]).to(qf_mean.device)[None,...]                    
    if asprior:    
        qf_dist = MVN(qf_mean.detach().clone() , (qf_cov +  zitter_mat).detach().clone()   )                        
    else:
        qf_dist = MVN(qf_mean , qf_cov +  zitter_mat)                               
    return qf_dist

def build_MVN_Diag(params:list , asprior=True, eps_zitter=1e-4):
    qf_mean,qf_std = params
    zitter_mat = eps_zitter*torch.ones_like(qf_mean).to(qf_mean.device)                    
    if asprior:    
        qf_dist = Normal(qf_mean.detach().clone() , (qf_std +  zitter_mat).detach().clone()   )                        
    else:
        qf_dist = Normal(qf_mean , qf_std +  zitter_mat)                               
    return qf_dist

    
def build_MVN(params:list, asprior=True, Lowrank=True, Diag=False):
    if Diag:
        return  build_MVN_Diag(params,asprior)   
    else:
        return  build_MVN_Low(params,asprior) if Lowrank  else build_MVN_Full(params,asprior)        
    
def compute_kl(params_qf, params_pf , Lowrank=True, Diag=False):
    qf_dist = build_MVN( params_qf, asprior=False, Lowrank=Lowrank, Diag=Diag)
    pf_dist = build_MVN( params_pf, asprior=True,  Lowrank=Lowrank, Diag=Diag)
    #breakpoint()
    return kl_divergence(qf_dist,pf_dist).mean(dim=0,keepdim=True)


def compute_kl2(params_qf, params_pf , Lowrank=True, Diag=False):
    qf_dist = build_MVN( params_qf, asprior=False, Lowrank=Lowrank, Diag=Diag)
    pf_dist = build_MVN( params_pf, asprior=True,  Lowrank=Lowrank, Diag=Diag)
    #breakpoint()
    return kl_divergence(qf_dist,pf_dist).sum(dim=-1,keepdim=True)


def _num_parameters(model):
    return sum([p.numel() for p in model.parameters()])


## ------------------------------
## vit_bayes_w
## ------------------------------
class VisionTransformer_Bayes_W(VisionTransformer):

    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            no_embed_class: bool = False,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: str = '',
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[Callable] = None,
            act_layer: Optional[Callable] = None,
            block_fn: Callable = Block,
            mlp_layer: Callable = Mlp,
            priorstdout: float = 10.,
            temppeak: float = 1.,              
    ):
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super(VisionTransformer_Bayes_W,self).__init__(img_size,
                                                    patch_size,
                                                    in_chans,
                                                    num_classes,
                                                    global_pool,
                                                    embed_dim,
                                                    depth,
                                                    num_heads,
                                                    mlp_ratio,
                                                    qkv_bias,
                                                    qk_norm,
                                                    init_values,
                                                    class_token,
                                                    no_embed_class,
                                                    pre_norm,
                                                    fc_norm,
                                                    drop_rate,
                                                    pos_drop_rate,
                                                    patch_drop_rate,
                                                    proj_drop_rate,
                                                    attn_drop_rate,
                                                    drop_path_rate,
                                                    weight_init,
                                                    embed_layer,
                                                    norm_layer,
                                                    act_layer,
                                                    block_fn,
                                                    mlp_layer,            
                                )

        ## Classifier Head
        self.scalelogstd = scalelogstd
        self.global_pool = global_pool        
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        #self.head_logstd = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)


    # def forward_head(self, x, pre_logits: bool = False):
    #     if self.global_pool:
    #         x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    #     x = self.fc_norm(x)
    #     x = self.head_drop(x)
    #     # return x if pre_logits else self.head(x)
    #     return x if pre_logits else self.head_bayes(x)

    # def forward(self, x):
    #     x = self.forward_features(x)
    #     x = self.forward_head(x)
    #     return x

    # def forward(self, x):
    #     x = self.forward_features(x)
    #     x = self.forward_head_bayes(x)
    #     return x

    ##------------------------------------------------------------------------------------------------------------------------------
    ## add implementation for map
    ##------------------------------------------------------------------------------------------------------------------------------
    
    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
            
        self.head_q_mu     = nn.Linear(self.embed_dim, num_classes, bias=False) if num_classes > 0 else nn.Identity()
        self.head_q_logstd = nn.ParameterList( [nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd,self.scalelogstd+1)  ,requires_grad=True )  for ith_param in self.head_q_mu.parameters()] )
        
        self.head_p_mu     = nn.ParameterList( [nn.Parameter(torch.zeros_like(ith_param) , requires_grad=False )  for ith_param in self.head_q_mu.parameters()]  )
        self.head_p_logstd = nn.ParameterList( [nn.Parameter(torch.ones_like(ith_param)  , requires_grad=True )   for ith_param in self.head_p_mu.parameters()]  )          
        
        func_model, _ , buffers = make_functional_with_buffers(self.head_q_mu)                
        self.func_model =   lambda params, x_in : func_model(params, buffers, x_in) 
        return 

    
    def forward_bayes(self, x, nfsample: int= 1):
        x = self.forward_features(x)
        x = self.forward_head_bayes(x,nfsample=nfsample)
        return x


    def forward_head_bayes(self, x, pre_logits: bool = False, nfsample: int= 1):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)

        if pre_logits:
            return x
        else:
            #return self.head(x)
            param_q_mean, param_q_logstd =  list(self.head_q_mu.parameters()) ,  list(self.head_q_logstd)     
            logits_sample                =  self.compute_func_samples(x,  param_q_mean, param_q_logstd , nfsample=nfsample) #(nfsample,nb,nout)            
            return logits_sample
        
                                        


                                                                                                                                                
    def compute_func_samples(self, x,  param_q_mean_s, param_q_logstd_s, nfsample=5):                
        def _sample_weight(param_q_mean, param_q_logstd):        
            weight_sample = []
            for ith_q_mean, ith_q_logstd in zip(param_q_mean, param_q_logstd): 
                ith_weight_sample = ith_q_mean + transform_logstd(ith_q_logstd) * torch.randn_like(ith_q_mean).to(ith_q_mean.device)  
                weight_sample.append(ith_weight_sample)
            return weight_sample         
                                
        func_samples = []
        for _ in range(nfsample):
            ith_weight_sample_s = _sample_weight(param_q_mean_s, param_q_logstd_s)                        
            ith_func_samples = self.func_model(ith_weight_sample_s , x)                            
            func_samples.append(  ith_func_samples[None,...]  )  

        func_samples = torch.cat(func_samples,dim=0) #(nfsample,nb,ydim)                   
        return func_samples
    
    

        
    def compute_weight_kl(self, reglambda=1e-4, eps=1e-8):
        if reglambda > 0.0:                                            
            param_q_mean, param_q_logstd    =  self.head_q_mu.parameters() ,  self.head_q_logstd  
            param_p_mean, param_p_logstd    =  self.head_p_mu              ,  self.head_p_logstd     
            
            # compute kl on weight space        
            kl_list = []
            num_params = 0
            for i_q_m,i_q_logstd,i_p_m,i_p_logstd in zip(param_q_mean,param_q_logstd,param_p_mean,param_p_logstd):
                i_kl = compute_kl([i_q_m,  transform_logstd(i_q_logstd) ], 
                                  [i_p_m,  transform_logstd(i_p_logstd) ] ,
                                  Lowrank=False, Diag=True).sum().unsqueeze(0)
                kl_list.append(i_kl)
                num_params += i_q_m.numel()
                
            #breakpoint()
            weight_kl = torch.cat(kl_list,dim=0).sum(dim=0) #(nparams)
            weight_kl = weight_kl / num_params

        else :
            weight_kl = torch.tensor(0.0).to(x.device)            

        return weight_kl


                
            
    #def predict_train(self, x , xc , nfsample=10 ,p_params=None, nctset= 1, reglambda=0.1):        
    def predict_train(self, x , xc ,p_params=None,configs=None):                
        if configs is not None:
            #nctset = configs.train.nctset 
            reglambda = configs.train.reglambda
            nfsample = 1
            
        else:            
            #nctset = 2 
            reglambda = 1e-3
            nfsample = 1
        
        
        self.train()                    
        # compute function samples        
        logits_sample = self.forward_bayes(x)
        
        # compute functional dist on context set        
        reglambda = float(reglambda)
        weight_kl = self.compute_weight_kl(reglambda=reglambda)
        
        ## outputs        
        outs = AttrDict()
        outs.logits = logits_sample
        outs.reg = weight_kl               
        
        
        #breakpoint()
        
        return outs        
    
        
        
    def predict_eval(self, x , topk=10, nfsample=1):        
        self.eval()        
        # compute function samples        
        logits_sample = self.forward_bayes(x , nfsample=nfsample)
        pred = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 
                                
        outs = AttrDict()
        outs.pred = pred.mean(dim=0)  #(nb,nout)        
        outs.logits = logits_sample       
        return outs       
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
## --------------------------------------------------------------------------------------------
## baseline: vit last layer vb on function space 
## --------------------------------------------------------------------------------------------
class VisionTransformer_Bayes_F(VisionTransformer):

    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            no_embed_class: bool = False,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: str = '',
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[Callable] = None,
            act_layer: Optional[Callable] = None,
            block_fn: Callable = Block,
            mlp_layer: Callable = Mlp,
            priorstdout: float = 10.,
            temppeak : float = 0.9,  
    ):
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super(VisionTransformer_Bayes_F,self).__init__(img_size,
                                                    patch_size,
                                                    in_chans,
                                                    num_classes,
                                                    global_pool,
                                                    embed_dim,
                                                    depth,
                                                    num_heads,
                                                    mlp_ratio,
                                                    qkv_bias,
                                                    qk_norm,
                                                    init_values,
                                                    class_token,
                                                    no_embed_class,
                                                    pre_norm,
                                                    fc_norm,
                                                    drop_rate,
                                                    pos_drop_rate,
                                                    patch_drop_rate,
                                                    proj_drop_rate,
                                                    attn_drop_rate,
                                                    drop_path_rate,
                                                    weight_init,
                                                    embed_layer,
                                                    norm_layer,
                                                    act_layer,
                                                    block_fn,
                                                    mlp_layer,          
                                )

        ## Classifier Head
        self.scalelogstd = scalelogstd
        self.global_pool = global_pool        
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        #self.head_logstd = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()




        if weight_init != 'skip':
            self.init_weights(weight_init)



    ##------------------------------------------------------------------------------------------------------------------------------
    ## add implementation for map
    ##------------------------------------------------------------------------------------------------------------------------------
    
    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
            
        self.head_q_mu     = nn.Linear(self.embed_dim, num_classes, bias=False) if num_classes > 0 else nn.Identity()
        self.head_q_logstd = nn.ParameterList( [nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd,self.scalelogstd+1)  ,requires_grad=True )  for ith_param in self.head_q_mu.parameters()] )
        
        self.head_p_mu     = nn.ParameterList( [nn.Parameter(torch.zeros_like(ith_param) , requires_grad=False )  for ith_param in self.head_q_mu.parameters()]  )
        self.head_p_logstd = nn.ParameterList( [nn.Parameter(torch.ones_like(ith_param)  , requires_grad=True )   for ith_param in self.head_p_mu.parameters()]  )          
        
        func_model, _ , buffers = make_functional_with_buffers(self.head_q_mu)                
        self.func_model =   lambda params, x_in : func_model(params, buffers, x_in) 
        return 



    def forward_features_bayes(self,x):
        x = self.forward_features(x)
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x 


    def forward_head_bayes(self, h_x, use_bayes: bool = False,  nfsample: int= 1):
        # chanage as bayes 
        param_q_mean, param_q_logstd  =  list(self.head_q_mu.parameters()) ,  list(self.head_q_logstd)    
        f_mean, f_diagcov             = self.compute_lastfuncdist_wo_jacobian_from_h( h_x , param_q_mean, param_q_logstd , no_grad_jacobian=True)
        f_diagstd = torch.sqrt(f_diagcov)

        # build sample functions            
        if use_bayes:
            logits_sample = f_mean[None,...] + f_diagstd[None,...]*torch.randn(nfsample,*f_mean.shape).to(h_x.device)  #(nfsample,nb,nydim)
        else:
            logits_sample = f_mean[None,...]
            
        return logits_sample

    
    
    def forward_bayes(self, x, y=None,  use_bayes=False, nfsample: int= 1):
        h_x = self.forward_features_bayes(x)        
        yhat = self.forward_head_bayes(h_x, use_bayes=use_bayes, nfsample=nfsample)
        return yhat
        
    
    def compute_logit_samples(self,x , y=None, use_bayes=True, nfsample=1, configs=None):
        if use_bayes:
            nfsample = configs.train.nfsample
        else:
            nfsample = 1
        return self.forward_bayes(x , y=y, use_bayes=use_bayes,nfsample=nfsample)        
        

                
    # def compute_jacobian_f_vec(self,q_params, x):        
    #     with torch.no_grad():
    #         jacobian_f_list = jacrev(self.func_model,argnums=0)(q_params, x)                
            
    #     jacobian_f_vec = torch.cat([jacobian.flatten(start_dim=2).detach() for jacobian in jacobian_f_list],dim=2)
    #     return jacobian_f_vec #(nb,ydim,nparams)
        
        
    # def compute_lastfuncdist(self, x , param_q_mean, param_q_logstd,eps=1e-8 ):
    #     #self.setup_functiondist() if not self.flag_setup else 1
        
    #     # compute f_mean
    #     #h = self.deterministic_nn(x)        
    #     f_mean = self.func_model(param_q_mean, x)  
        
    #     # compute f_Lcov
    #     jacobian_f_vec = self.compute_jacobian_f_vec(param_q_mean, x) #(nb,ydim,nparams))                
    #     with torch.no_grad():        
    #         U,S,VT = torch.linalg.svd(jacobian_f_vec, full_matrices=False, driver='gesvda') if x.is_cuda else torch.linalg.svd(jacobian_f_vec, full_matrices=False)
    #     param_q_logstd_vec = parameters_to_vector( list(param_q_logstd) )
    #     param_q_std_vec = transform_logstd(param_q_logstd_vec)[None,None,:] #(1,1,nparams)                        
                 
    #     sum_VT2_qstd2 = ( torch.pow(VT * param_q_std_vec , 2)).sum(dim=-1)  #(nb,nydim)
    #     eigen2_sum_VT2_qstd2 = torch.pow(S,2)*sum_VT2_qstd2 #(nb,nydim)
    #     std_on_fs = torch.sqrt(  torch.clamp(eigen2_sum_VT2_qstd2, min=eps_clamp) ) #(nb,nydim)        
        
    #     #f_Lcov = U*(std_on_fs[:,None,:])   #(nb,ydim,ydim)*(nb,1,ydim) = (nb,ydim,ydim)        
    #     return f_mean, U , std_on_fs
    
    
    
    
        # computation checked     
    def compute_lastfuncdist_wo_jacobian_from_h(self, h , param_q_mean, param_q_logstd ,no_grad_jacobian=True ):
        #self.setup_functiondist() if not self.flag_setup else 1        
        assert len(param_q_logstd) <=2
        
        # compute f_mean
        #h = self.deterministic_nn(x) 
        f_mean = self.func_model(param_q_mean, h ) 


        if no_grad_jacobian:
            h = h.detach().clone()
            
        # comopute f_std
        param_q_std        = [transform_logstd(ith_q_logstd) for ith_q_logstd in param_q_logstd ]                
        f_var_linear_inner = param_q_std[0][None,...]*h[:,None,:] #(1,ydim,nparams)*(nb,1,nparams) = (nb,ydim,nparams)
        f_var_linear       = torch.pow(f_var_linear_inner,2).sum(dim=-1) #(nb,ydim)

        if len(param_q_logstd) == 2:               
            f_var_bias = torch.pow( param_q_std[1][None,...], 2)  
        else :  
            f_var_bias = 0.0
        f_diagcov = f_var_linear + f_var_bias #(nb,ydim)
    
        return f_mean, f_diagcov
    


    # #def predict_train(self, x , xc , nfsample=10 ,p_params=None, nctset= 1, reglambda=0.1):        
    # def compute_fdist(self, xc ,p_params=None,configs=None):                

    #     # get hypparams from configs
    #     priorstdout = configs.model.priorstdout
        
    #     self.train()             
           
    #     # get last layer features
    #     h_xc = self.forward_features_bayes(xc)        
        
    #     # compute "q" functional dist on context set        
    #     param_q_mean, param_q_logstd  = list(self.head_q_mu.parameters()) ,  list(self.head_q_logstd)             
    #     qf_mean_xc , qf_diagcov_xc    = self.compute_lastfuncdist_wo_jacobian_from_h( h_xc, param_q_mean, param_q_logstd, no_grad_jacobian=True )   #without jacobian computation         
    #     qf_std_xc                     = torch.sqrt(qf_diagcov_xc)
        
    #     # compute "p" functional dist on context set
    #     if p_params is None:            
    #         pf_mean_xc                =  torch.zeros_like(qf_mean_xc).to(xc.device)
    #         pf_std_xc                 =  priorstdout*torch.ones_like(qf_mean_xc).to(xc.device)            


    #     # if configs.train.topk < pf_mean_xc.shape[-1]:
    #     #     # mariginalize posterior on nbd
    #     #     nbd_idx    = torch.topk(qf_mean_xc, k = configs.train.topk,dim=-1)[1]
    #     #     qf_mean_xc, qf_std_xc = torch.gather(qf_mean_xc,1,nbd_idx), torch.gather(qf_std_xc ,1,nbd_idx)    
    #     #     pf_mean_xc, pf_std_xc = torch.gather(pf_mean_xc,1,nbd_idx), torch.gather(pf_std_xc ,1,nbd_idx)    
            
            
    #     #param_qf = (qf_mean_xc, qf_std_xc)    
    #     param_qf = (qf_mean_xc, qf_std_xc)    
    #     param_pf = (pf_mean_xc, pf_std_xc)

    #     #return param_qf, param_pf 
    #     return param_qf, param_pf, None 




    #def predict_train(self, x , xc , nfsample=10 ,p_params=None, nctset= 1, reglambda=0.1):        
    def compute_fdist(self, xc ,p_params=None,configs=None):                

        # get hypparams from configs
        priorstdout = configs.model.priorstdout
        
        self.train()             
           
        # get last layer features
        h_xc = self.forward_features_bayes(xc)        
        
        # compute "q" functional dist on context set        
        param_q_mean, param_q_logstd  = list(self.head_q_mu.parameters()) ,  list(self.head_q_logstd)             
        qf_mean_xc , qf_diagcov_xc    = self.compute_lastfuncdist_wo_jacobian_from_h( h_xc, param_q_mean, param_q_logstd, no_grad_jacobian=True )   #without jacobian computation         
        qf_std_xc                     = torch.sqrt(qf_diagcov_xc)
        
        # compute "p" functional dist on context set
        if p_params is None:            
            pf_mean_xc                =  torch.zeros_like(qf_mean_xc).to(xc.device)
            pf_std_xc                 =  priorstdout*torch.ones_like(qf_mean_xc).to(xc.device)            

            
            
        #param_qf = (qf_mean_xc, qf_std_xc)    
        param_qf = (qf_mean_xc, qf_std_xc)    
        param_pf = (pf_mean_xc, pf_std_xc)

        #return param_qf, param_pf 
        return param_qf, param_pf, None 


        
    def compute_sup_kl(self, params_qf, params_pf, weight=None, Lowrank=True, Diag=True, nctset=2, eps=1e-8):        
        def _compute_kl(params_qf, params_pf , Lowrank=True, Diag=False):
            qf_dist = build_MVN( params_qf, asprior=False, Lowrank=Lowrank, Diag=Diag)
            pf_dist = build_MVN( params_pf, asprior=True,  Lowrank=Lowrank, Diag=Diag)
            kl_div_qf_pf = kl_divergence(qf_dist,pf_dist)
            kl_div = kl_div_qf_pf.sum(dim=-1)
            return kl_div #(nb,)
        
        
        # set params
        qf_mean,qf_cov = params_qf
        pf_mean,pf_cov = params_pf

        # split ctxset
        num_batch = qf_mean.shape[0] // nctset
        idx_set = [torch.randperm(qf_mean.shape[0])[i*num_batch:(i+1)*num_batch] for i in range(nctset) ]

        # compute kl
        kl_divergence_list = []
        for i_set in idx_set:
            i_params_q = qf_mean[i_set] , qf_cov[i_set]
            i_params_p = pf_mean[i_set] , pf_cov[i_set]       
            i_kl_batch = _compute_kl(i_params_q, i_params_p, Lowrank=Lowrank, Diag=Diag )       
            if weight is None:            
                i_weight_kl_batch = i_kl_batch                
            else:
                i_weight          = weight[i_set]                       
                i_weight_kl_batch = i_weight*i_kl_batch
            kl_divergence_list.append( (i_weight_kl_batch).sum(dim=-1,keepdim=True) )

        kl_divergence_list = torch.cat(kl_divergence_list,dim=0)
        sup_kl = (1 / num_batch)*kl_divergence_list.max()
        sup_kl = torch.clamp(sup_kl,min=eps)  
            
        return sup_kl
    
    
                    
            
            
            
    #def predict_train(self, x , xc , nfsample=10 ,p_params=None, nctset= 1, reglambda=0.1):        
    def predict_train(self, x , xc ,p_params=None,configs=None):                
        if configs is not None:
            #nctset = configs.train.nctset 
            reglambda = configs.train.reglambda
            nfsample = configs.train.nfsample
            #nfsample = 1
            
        else:            
            #nctset = 2 
            reglambda = 1e-3
            nfsample = 1
        
        
        self.train()                    
        # compute function samples        
        logits_sample = self.forward_bayes(x , nfsample=nfsample)
        
        # compute functional dist on context set        
        reglambda = float(reglambda)
        func_kl = torch.tensor(0.0).to(x.device)
        
        ## outputs        
        outs = AttrDict()
        outs.logits = logits_sample
        outs.reg = func_kl               
        return outs        
    
        
        
    # def predict_eval(self, x , nfsample=1):        
    #     self.eval()        
    #     # compute function samples        
    #     logits_sample = self.forward_bayes(x , nfsample=nfsample)
    #     pred = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 
                                
    #     outs = AttrDict()
    #     outs.pred = pred.mean(dim=0)  #(nb,nout)        
    #     outs.logits = logits_sample       
    #     return outs       
    
        
    def predict_eval(self, x ,use_bayes=True, topk=1,  nfsample=1):        
        self.eval()        
        # compute function samples        
        #logits_sample = self.forward_bayes(x , use_bayes =True, nfsample=nfsample,topk=topk)
        logits_sample = self.forward_bayes(x , use_bayes =use_bayes, nfsample=nfsample)
        pred = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 
                                
        outs = AttrDict()
        outs.pred = pred.mean(dim=0)  #(nb,nout)        
        outs.logits = logits_sample       
        return outs               
    
    
    
    







# ## --------------------------------------------------------------------------------------------
# ## vit last layer vb on function space; w
# ## without jacobian computation 
# ## --------------------------------------------------------------------------------------------
# class VisionTransformer_Bayes_F_WO_Jacobian(VisionTransformer):

#     """ Vision Transformer
#     A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
#         - https://arxiv.org/abs/2010.11929
#     """

#     def __init__(
#             self,
#             img_size: Union[int, Tuple[int, int]] = 224,
#             patch_size: Union[int, Tuple[int, int]] = 16,
#             in_chans: int = 3,
#             num_classes: int = 1000,
#             global_pool: str = 'token',
#             embed_dim: int = 768,
#             depth: int = 12,
#             num_heads: int = 12,
#             mlp_ratio: float = 4.,
#             qkv_bias: bool = True,
#             qk_norm: bool = False,
#             init_values: Optional[float] = None,
#             class_token: bool = True,
#             no_embed_class: bool = False,
#             pre_norm: bool = False,
#             fc_norm: Optional[bool] = None,
#             drop_rate: float = 0.,
#             pos_drop_rate: float = 0.,
#             patch_drop_rate: float = 0.,
#             proj_drop_rate: float = 0.,
#             attn_drop_rate: float = 0.,
#             drop_path_rate: float = 0.,
#             weight_init: str = '',
#             embed_layer: Callable = PatchEmbed,
#             norm_layer: Optional[Callable] = None,
#             act_layer: Optional[Callable] = None,
#             block_fn: Callable = Block,
#             mlp_layer: Callable = Mlp,
#             priorstdout: float = 10.,
#     ):
#         """
#         Args:
#             img_size: Input image size.
#             patch_size: Patch size.
#             in_chans: Number of image input channels.
#             num_classes: Mumber of classes for classification head.
#             global_pool: Type of global pooling for final sequence (default: 'token').
#             embed_dim: Transformer embedding dimension.
#             depth: Depth of transformer.
#             num_heads: Number of attention heads.
#             mlp_ratio: Ratio of mlp hidden dim to embedding dim.
#             qkv_bias: Enable bias for qkv projections if True.
#             init_values: Layer-scale init values (layer-scale enabled if not None).
#             class_token: Use class token.
#             fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
#             drop_rate: Head dropout rate.
#             pos_drop_rate: Position embedding dropout rate.
#             attn_drop_rate: Attention dropout rate.
#             drop_path_rate: Stochastic depth rate.
#             weight_init: Weight initialization scheme.
#             embed_layer: Patch embedding layer.
#             norm_layer: Normalization layer.
#             act_layer: MLP activation layer.
#             block_fn: Transformer block layer.
#         """
#         super(VisionTransformer_Bayes_F_WO_Jacobian,self).__init__(img_size,
#                                                                     patch_size,
#                                                                     in_chans,
#                                                                     num_classes,
#                                                                     global_pool,
#                                                                     embed_dim,
#                                                                     depth,
#                                                                     num_heads,
#                                                                     mlp_ratio,
#                                                                     qkv_bias,
#                                                                     qk_norm,
#                                                                     init_values,
#                                                                     class_token,
#                                                                     no_embed_class,
#                                                                     pre_norm,
#                                                                     fc_norm,
#                                                                     drop_rate,
#                                                                     pos_drop_rate,
#                                                                     patch_drop_rate,
#                                                                     proj_drop_rate,
#                                                                     attn_drop_rate,
#                                                                     drop_path_rate,
#                                                                     weight_init,
#                                                                     embed_layer,
#                                                                     norm_layer,
#                                                                     act_layer,
#                                                                     block_fn,
#                                                                     mlp_layer,            
#                                                 )

#         ## Classifier Head
#         self.scalelogstd = scalelogstd
#         self.global_pool = global_pool        
#         use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
#         self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
#         self.head_drop = nn.Dropout(drop_rate)
#         self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
#         #self.head_logstd = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

#         if weight_init != 'skip':
#             self.init_weights(weight_init)


#     # def forward_head(self, x, pre_logits: bool = False):
#     #     if self.global_pool:
#     #         x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
#     #     x = self.fc_norm(x)
#     #     x = self.head_drop(x)
#     #     # return x if pre_logits else self.head(x)
#     #     return x if pre_logits else self.head_bayes(x)

#     # def forward(self, x):
#     #     x = self.forward_features(x)
#     #     x = self.forward_head(x)
#     #     return x


#     # def forward(self, x):
#     #     x = self.forward_features(x)
#     #     x = self.forward_head_bayes(x)
#     #     return x

#     ##------------------------------------------------------------------------------------------------------------------------------
#     ## revise implementation 
#     ##------------------------------------------------------------------------------------------------------------------------------
    
#     def reset_classifier(self, num_classes: int, global_pool=None):
#         self.num_classes = num_classes
#         if global_pool is not None:
#             assert global_pool in ('', 'avg', 'token')
#             self.global_pool = global_pool
            
#         self.head_q_mu     = nn.Linear(self.embed_dim, num_classes, bias=False) if num_classes > 0 else nn.Identity()
#         self.head_q_logstd = nn.ParameterList( [nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd,self.scalelogstd+1)  ,requires_grad=True )  for ith_param in self.head_q_mu.parameters()] )
        
#         self.head_p_mu     = nn.ParameterList( [nn.Parameter(torch.zeros_like(ith_param) , requires_grad=False )  for ith_param in self.head_q_mu.parameters()]  )
#         self.head_p_logstd = nn.ParameterList( [nn.Parameter(torch.ones_like(ith_param)  , requires_grad=True )   for ith_param in self.head_p_mu.parameters()]  )          
        
#         func_model, _ , buffers = make_functional_with_buffers(self.head_q_mu)                
#         self.func_model =   lambda params, x_in : func_model(params, buffers, x_in) 
#         return 



#     def forward_features_bayes(self,x):
#         x = self.forward_features(x)
#         if self.global_pool:
#             x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
#         x = self.fc_norm(x)
#         x = self.head_drop(x)
#         return x 

    
#     def forward_bayes(self, x, use_bayes=False, nfsample: int= 1):
#         h = self.forward_features_bayes(x)
#         y = self.forward_head_bayes(h, use_bayes=use_bayes, nfsample=nfsample)
#         return y
        
    
#     def compute_logit_samples(self,x,use_bayes=True, nfsample=1,configs=None):
#         if use_bayes:
#             nfsample = configs.train.nfsample
#         else:
#             nfsample = 1
#         return self.forward_bayes(x,use_bayes=use_bayes,nfsample=nfsample)        
        
        
        
        
#     # computation checked     
#     def compute_lastfuncdist_wo_jacobian(self, h , param_q_mean, param_q_logstd ):
#         #self.setup_functiondist() if not self.flag_setup else 1        
#         assert len(param_q_logstd) <=2
        
#         # compute f_mean
#         #h = self.deterministic_nn(x) 
#         f_mean = self.func_model(param_q_mean, h ) 

#         # comopute f_std
#         param_q_std  = [transform_logstd(ith_q_logstd) for ith_q_logstd in param_q_logstd ]                
#         f_var_linear_inner = param_q_std[0][None,...]*h[:,None,:] #(1,ydim,nparams)*(nb,1,nparams) = (nb,ydim,nparams)
#         f_var_linear       = torch.pow(f_var_linear_inner,2).sum(dim=-1) #(nb,ydim)

#         if len(param_q_logstd) == 2:               
#             f_var_bias = torch.pow( param_q_std[1][None,...], 2)  
#         else :  
#             f_var_bias = 0.0
#         f_diagcov = f_var_linear + f_var_bias #(nb,ydim)
    
#         return f_mean, f_diagcov
    



#     def forward_head_bayes(self, x, use_bayes: bool = False,  nfsample: int= 1):
#         # chanage as bayes 
#         param_q_mean, param_q_logstd  =  list(self.head_q_mu.parameters()) ,  list(self.head_q_logstd)    
#         f_mean, f_diagcov             = self.compute_lastfuncdist_wo_jacobian( x , param_q_mean, param_q_logstd )
#         f_diagstd = torch.sqrt(f_diagcov)

#         # build sample functions            
#         if use_bayes:
#             logits_sample = f_mean[None,...] + f_diagstd[None,...]*torch.randn(nfsample,*f_mean.shape).to(x.device)  #(nfsample,nb,nydim)
#         else:
#             logits_sample = f_mean[None,...]
            
#         return logits_sample

    
#     #def predict_train(self, x , xc , nfsample=10 ,p_params=None, nctset= 1, reglambda=0.1):        
#     def compute_fdist(self, xc ,p_params=None,configs=None):                

#         # get hypparams from configs
#         priorstdout = configs.model.priorstdout
        
#         self.train()             
           
#         # get last layer features
#         hc = self.forward_features_bayes(xc)        
        
#         # compute "q" functional dist on context set        
#         param_q_mean, param_q_logstd        = list(self.head_q_mu.parameters()) ,  list(self.head_q_logstd)             
#         qf_mean_xc , qf_diagcov_xc          = self.compute_lastfuncdist_wo_jacobian( hc, param_q_mean, param_q_logstd )   #without jacobian computation         
#         qf_std_xc                           = torch.sqrt(qf_diagcov_xc)
        
#         # compute "p" functional dist on context set
#         if p_params is None:            
#             pf_mean_xc =                1*torch.zeros_like(qf_mean_xc).to(xc.device)
#             pf_std_xc  = priorstdout*torch.ones_like(qf_mean_xc).to(xc.device)            
#         else:        
#             param_p_mean, param_p_logstd          = p_params

#             ## swag diag            
#             ## failed diag swag for training            
#             # pf_mean_xc , pf_diagcov_xc, _  = self.compute_funcdist_wo_jacobian( xc, param_p_mean, param_p_logstd )   #without jacobian computation         
#             # # pf_std_xc                      = torch.sqrt(pf_diagcov_xc)            
#             # pf_mean_xc, pf_std_xc          = pf_mean_xc.detach().clone(), pf_std_xc.detach().clone() #.detach().clone()

#             ## swa diag
#             pf_mean_xc , _   = self.compute_lastfuncdist_wo_jacobian( hc, param_p_mean, param_p_logstd )   #without jacobian computation         
#             pf_mean_xc = pf_mean_xc.detach().clone()
#             pf_std_xc  = priorstdout*torch.ones_like(qf_mean_xc).to(xc.device)            


#             # pf_std_xc                      = torch.sqrt(pf_diagcov_xc)            
 
#             if configs.train.topk < pf_mean_xc.shape[-1]:
#                 # mariginalize posterior on nbd
#                 nbd_idx    = torch.topk(pf_mean_xc,k = configs.train.topk,dim=-1)[1]
#                 qf_mean_xc, qf_std_xc = torch.gather(qf_mean_xc,1,nbd_idx), torch.gather(qf_std_xc ,1,nbd_idx)    
#                 pf_mean_xc, pf_std_xc = torch.gather(pf_mean_xc,1,nbd_idx), torch.gather(pf_std_xc ,1,nbd_idx)    
            
#         #param_qf = (qf_mean_xc, qf_std_xc)    
#         param_qf = (qf_mean_xc, qf_std_xc)    
#         param_pf = (pf_mean_xc, pf_std_xc)

#         #return param_qf, param_pf 
#         return param_qf, param_pf, None 
    
        
        
        
#     def compute_sup_kl(self, params_qf, params_pf, weight=None, Lowrank=True, Diag=True, nctset=2, eps=1e-8):        
#         def _compute_kl(params_qf, params_pf , Lowrank=True, Diag=False):
#             qf_dist = build_MVN( params_qf, asprior=False, Lowrank=Lowrank, Diag=Diag)
#             pf_dist = build_MVN( params_pf, asprior=True,  Lowrank=Lowrank, Diag=Diag)
#             kl_div_qf_pf = kl_divergence(qf_dist,pf_dist)
#             kl_div = kl_div_qf_pf.sum(dim=-1)
#             return kl_div #(nb,)
        
        
#         # set params
#         qf_mean,qf_cov = params_qf
#         pf_mean,pf_cov = params_pf

#         # split ctxset
#         num_batch = qf_mean.shape[0] // nctset
#         idx_set = [torch.randperm(qf_mean.shape[0])[i*num_batch:(i+1)*num_batch] for i in range(nctset) ]

#         # compute kl
#         kl_divergence_list = []
#         for i_set in idx_set:
#             i_params_q = qf_mean[i_set] , qf_cov[i_set]
#             i_params_p = pf_mean[i_set] , pf_cov[i_set]       
#             i_kl_batch = _compute_kl(i_params_q, i_params_p, Lowrank=Lowrank, Diag=Diag )       
#             if weight is None:            
#                 i_weight_kl_batch = i_kl_batch                
#             else:
#                 i_weight          = weight[i_set]                       
#                 i_weight_kl_batch = i_weight*i_kl_batch
#             kl_divergence_list.append( (i_weight_kl_batch).sum(dim=-1,keepdim=True) )

#         kl_divergence_list = torch.cat(kl_divergence_list,dim=0)
#         sup_kl = (1 / num_batch)*kl_divergence_list.max()
#         sup_kl = torch.clamp(sup_kl,min=eps)  
            
#         return sup_kl
    
    
            

            
            
#     #def predict_train(self, x , xc , nfsample=10 ,p_params=None, nctset= 1, reglambda=0.1):        
#     def predict_train(self, x , xc ,p_params=None,configs=None):                
#         if configs is not None:
#             #nctset = configs.train.nctset 
#             reglambda = configs.train.reglambda
#             nfsample = configs.train.nfsample
#             #nfsample = 1
            
#         else:            
#             #nctset = 2 
#             reglambda = 1e-3
#             nfsample = 1
        
        
#         self.train()                    
#         # compute function samples        
#         logits_sample = self.forward_bayes(x , nfsample=nfsample)
        
#         # compute functional dist on context set        
#         reglambda = float(reglambda)
#         func_kl = torch.tensor(0.0).to(x.device)
        
#         ## outputs        
#         outs = AttrDict()
#         outs.logits = logits_sample
#         outs.reg = func_kl               
#         return outs        
    
        
        
#     def predict_eval(self, x ,use_bayes=True , nfsample=1):        
#         self.eval()        
#         # compute function samples        
#         logits_sample = self.forward_bayes(x , use_bayes =use_bayes, nfsample=nfsample)
#         pred = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 
                                
#         outs = AttrDict()
#         outs.pred = pred.mean(dim=0)  #(nb,nout)        
#         outs.logits = logits_sample       
#         return outs           
    
    
    
    
    
    
    
    
    








## --------------------------------------------------------------------------------------------
## proposed models
## --------------------------------------------------------------------------------------------

from torch.distributions.categorical import Categorical

class Hiddenfeature_Mixture_Module(nn.Module):     
    def __init__(self, nfeature: int, nlabel: int , usediagcov=True, gamma=0.99,eps_zitter=1e-4):
        super(Hiddenfeature_Mixture_Module,self).__init__()
        
        self.nlabel = nlabel
        self.nfeature = nfeature
        self.eps_zitter = eps_zitter
        self.gamma = gamma
        self.usediagcov = usediagcov
        
        # per epoch        
        self.static_cnt_sum    =  nn.Parameter( torch.zeros(self.nlabel), requires_grad=False )        
        self.static_mean_sum   =  nn.Parameter( torch.zeros(self.nlabel, self.nfeature), requires_grad=False )
        self.static_cov_sum    =  nn.Parameter( torch.zeros(self.nlabel, self.nfeature, self.nfeature) , requires_grad=False )        
        return 

    
    #def __repr__(self):
    def extra_repr(self):        
        msg = 'nlabel: {}, \n'.format(self.nlabel)
        msg += 'nfeature: {}, \n'.format(self.nfeature)
        return msg[:-1]
    
           
       
    def update_layers_static_batch(self, h_feature, y, eps_zitter=1e-4):
        # set idx_per_label        
        idx_per_label =  [(y == k)  for k in range(self.nlabel) ]  
        
        # compute emirical mean and cov
        static_mean_sum, static_cov_sum = [],[]        
        for idx_current in idx_per_label:
            if idx_current.sum() > 0 :        
                # compute mean
                k_label_mean_sum = h_feature[idx_current].sum(dim=0,keepdim=True) #(1,nfilters)
                k_label_mean     = k_label_mean_sum / idx_current.sum()
                
                # compute cov
                k_delta_cov      = h_feature[idx_current] - k_label_mean   #(nb,nfilters)
                k_label_cov_sum  = (k_delta_cov[:,:,None]*k_delta_cov[:,None,:]).sum(dim=0,keepdim=True)  #(1, nfilters,nfilters)
            else:
                k_label_mean_sum = torch.zeros(1,self.nfeature).to(y.device) 
                k_label_cov_sum  = torch.zeros(1,self.nfeature,self.nfeature).to(y.device)                               
                
            static_mean_sum.append(k_label_mean_sum)     
            static_cov_sum.append(k_label_cov_sum) 
        
        static_mean_sum = torch.cat(static_mean_sum, dim=0)   #(nlables, nfilters)
        static_cov_sum  = torch.cat(static_cov_sum, dim=0)    #(nlables, nfilters, nfilters)
        static_cnt      = torch.tensor([idx_current.sum() for idx_current in idx_per_label]).to(y.device) #(nlables)        
        

        # compute layer feature
        nb = len(y)        
        
        ## update static params per batch
        self.static_cnt_sum.data     +=  static_cnt.data
        self.static_mean_sum.data    +=  static_mean_sum.data
        self.static_cov_sum.data     +=  static_cov_sum.data
        

        return 
    
    def get_static_params(self):
        mean               = self.static_mean_sum.data                          / self.static_cnt_sum.data.view(-1,1)
        mean_shared        = self.static_mean_sum.data.sum(dim=0,keepdims=True) / self.static_cnt_sum.data.sum()
        cov_shared         = self.static_cov_sum.data.sum(dim=0)                / self.static_cnt_sum.data.sum()  
        
        if self.usediagcov:
            cov_shared = torch.diag(cov_shared) + self.eps_zitter
            precision_shared   = 1 / cov_shared
        else:
            precision_shared   = torch.linalg.inv( cov_shared + self.eps_zitter*torch.eye(self.nfeature).to(mean.device) )
        # return mean,mean_shared,cov_shared,precision_shared
        return mean,mean_shared,precision_shared
        
      
    def compute_mh_distance_batch(self, h_feature , usenormalize = True ,overclass=False):
        # layers_mean = self.static_mean.to(h_feature.device)
        # layers_precision = self.static_precision.to(h_feature.device)        
        layers_mean, leayers_mean_shared , layers_precision = self.get_static_params()

        if self.usediagcov:
            delta_h = h_feature[:,None,:] - layers_mean[None,:,:]                      #(nb,nlabel,nhidden)
            square_term = ((delta_h**2)*layers_precision[None,None,...])   #(nb,nlabel)
            mh_distance_over_class  = -0.5*square_term.sum(dim=-1)

        else:                           
            delta_h = h_feature[:,None,:] - layers_mean[None,:,:]
            squareterm_left_product = torch.einsum('bij,bjk->bik',delta_h, layers_precision[None,...].repeat(delta_h.size(0),1,1))
            mh_distance_over_class  = -0.5*(squareterm_left_product*delta_h).sum(dim=-1)

        if  usenormalize:
            mh_distance_over_class = mh_distance_over_class / h_feature.size(-1)
        
        mh_distance, assigned_label = mh_distance_over_class.max(dim=-1)            
        if overclass:
            return mh_distance_over_class
        else:
            return mh_distance


    def compute_mh_distance_shared_batch(self, h_feature , datatype = 'x',overclass=False):
        ## compute mh distance per class
        _, layers_mean, layers_precision = self.get_static_params()

        #breakpoint()
        delta_h = h_feature - layers_mean  #(nb,nfilters)
        squareterm_left_product = torch.einsum('bi,ik->bk',delta_h, layers_precision)  #(nb,nfilters)
        mh_distance_shared_over_class = -0.5*(squareterm_left_product*delta_h).sum(dim=-1)
        mh_distance_shared_over_class = h_feature.size(-1) 
        mh_distance, assigned_label = mh_distance_shared_over_class.max(dim=-1) 
        if overclass:
            return mh_distance_shared_over_class
        else:
            return mh_distance



    # def get_nbd_idx(self, h_feature, topk=3):        
    #     mh_dist_over_cls = self.compute_mh_distance_batch(h_feature , datatype = 'x' ,overclass=True)
    #     mh_dist,mh_idx = torch.topk(mh_dist_over_cls,k=topk)
    #     return mh_dist,mh_idx




class VisionTransformer_Bayes_F_WO_Jacobian_KDE(VisionTransformer):

    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            no_embed_class: bool = False,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: str = '',
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[Callable] = None,
            act_layer: Optional[Callable] = None,
            block_fn: Callable = Block,
            mlp_layer: Callable = Mlp,
            priorstdout: float = 10.,
            temppeak : float = 0.9,
    ):
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super(VisionTransformer_Bayes_F_WO_Jacobian_KDE,self).__init__(img_size,
                                                                                patch_size,
                                                                                in_chans,
                                                                                num_classes,
                                                                                global_pool,
                                                                                embed_dim,
                                                                                depth,
                                                                                num_heads,
                                                                                mlp_ratio,
                                                                                qkv_bias,
                                                                                qk_norm,
                                                                                init_values,
                                                                                class_token,
                                                                                no_embed_class,
                                                                                pre_norm,
                                                                                fc_norm,
                                                                                drop_rate,
                                                                                pos_drop_rate,
                                                                                patch_drop_rate,
                                                                                proj_drop_rate,
                                                                                attn_drop_rate,
                                                                                drop_path_rate,
                                                                                weight_init,
                                                                                embed_layer,
                                                                                norm_layer,
                                                                                act_layer,
                                                                                block_fn,
                                                                                mlp_layer,            
                                                            )

        ## Classifier Head
        self.scalelogstd = scalelogstd
        self.global_pool = global_pool        
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if weight_init != 'skip':
            self.init_weights(weight_init)



        ## hidden feature mixture model
        self.mixture_module = Hiddenfeature_Mixture_Module(nfeature=self.embed_dim, nlabel=num_classes, usediagcov=True)

        ## temperature hyperparams
        #self.temperature_peak = 0.5
        self.temperature_peak = temppeak        
        self.temperature_soft = 1. 
        print('temppeak: {}, tempsoft: {}'.format(self.temperature_peak,self.temperature_soft))
        
        
        self.priorstdout      = priorstdout       
        

        self.temperature_distance = 1


    ##------------------------------------------------------------------------------------------------------------------------------
    ## revise implementation 
    ##------------------------------------------------------------------------------------------------------------------------------
    
    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
            
        self.head_q_mean     = nn.Linear(self.embed_dim, num_classes, bias=False) if num_classes > 0 else nn.Identity()
        self.head_q_logstd   = nn.ParameterList( [nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd,self.scalelogstd+1)  ,requires_grad=True )  for ith_param in self.head_q_mean.parameters()] )
        
        self.head_p_mean     = nn.ParameterList( [nn.Parameter(torch.zeros_like(ith_param) , requires_grad=False )   for ith_param in self.head_q_mean.parameters()]  )
        self.head_p_logstd   = nn.ParameterList( [nn.Parameter(torch.ones_like(ith_param)  , requires_grad=False )   for ith_param in self.head_q_mean.parameters()]  )          
        
        func_model, _ , buffers = make_functional_with_buffers(self.head_q_mean)                
        self.func_model =   lambda params, x_in : func_model(params, buffers, x_in) 
        return 



    def forward_features_bayes(self,x):
        x = self.forward_features(x)
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x 
    
    
    
    # computation checked     
    def compute_lastfuncdist_wo_jacobian(self, h , param_q_mean, param_q_logstd ):
        #self.setup_functiondist() if not self.flag_setup else 1        
        assert len(param_q_logstd) <=2
        
        # compute f_mean
        #h = self.deterministic_nn(x) 
        f_mean = self.func_model(param_q_mean, h ) 

        # comopute f_std
        param_q_std        = [transform_logstd(ith_q_logstd) for ith_q_logstd in param_q_logstd ]                
        f_var_linear_inner = param_q_std[0][None,...]*h[:,None,:] #(1,ydim,nparams)*(nb,1,nparams) = (nb,ydim,nparams)
        f_var_linear       = torch.pow(f_var_linear_inner,2).sum(dim=-1) #(nb,ydim)

        if len(param_q_logstd) == 2:               
            f_var_bias = torch.pow( param_q_std[1][None,...], 2)  
        else :  
            f_var_bias = 0.0
        f_diagcov = f_var_linear + f_var_bias #(nb,ydim)
    
        return f_mean, f_diagcov




    def compute_kde_pfuncdist_from_h(self, ht, param_q_mean, param_q_logstd , requires_grad=False ):
        #import torch.nn.functional as F
        h_mean, _, _ = self.mixture_module.get_static_params()
        h_mean = h_mean.detach().clone()

        kde_ht = self.mixture_module.compute_mh_distance_batch(ht ,overclass=True)
        kde_weight = F.softmax( kde_ht , dim=-1)

        proj_ht = (kde_weight[...,None]*h_mean[None,...]).sum(dim=1) #(nb,ncls,1)*(nb,ncls,ndim) = (nb,ndim)

        f_mean = self.func_model(param_q_mean, proj_ht)
        param_q_std        = [transform_logstd(ith_q_logstd) for ith_q_logstd in param_q_logstd ]                
        f_var_linear_inner = param_q_std[0][None,...]*proj_ht[:,None,:] #(1,ydim,nparams)*(nb,1,nparams) = (nb,ydim,nparams)
        f_var_linear       = torch.pow(f_var_linear_inner,2).sum(dim=-1) #(nb,ydim)
        f_var_bias         = torch.pow( param_q_std[1][None,...], 2) if len(param_q_logstd) == 2 else 0.0                
        f_diagcov          = f_var_linear + f_var_bias #(nb,ydim)
        
        
        if requires_grad:        
            # stop gradient
            return f_mean, f_diagcov, proj_ht
        else:
            # stop gradient
            return f_mean.detach().clone(), f_diagcov.detach().clone(), proj_ht.detach().clone()
    
    
    


    def forward_head_bayes(self, h_x, use_bayes: bool = False,  nfsample: int= 1, topk: int=3):
        # chanage as bayes 
        param_q_mean, param_q_logstd  =  list(self.head_q_mean.parameters()) ,  list(self.head_q_logstd)    
        qf_mean, qf_diagcov             = self.compute_lastfuncdist_wo_jacobian( h_x , param_q_mean, param_q_logstd )
        qf_diagstd = torch.sqrt(qf_diagcov)


        # build sample functions            
        if not use_bayes:
            logits_sample = qf_mean[None,...]
        else:
            logits_sample = qf_mean[None,...] + qf_diagstd[None,...]*torch.randn(nfsample,*qf_mean.shape).to(h_x.device)  #(nfsample,nb,nydim)


            ## compute phat based on MPA approximation
            phat_nominator   =  qf_mean      
            nmh_distance     = -self.mixture_module.compute_mh_distance_batch(h_x,overclass=True, usenormalize = False)      

            phat_denominator =  torch.sqrt( 1 + torch.pi/8 * nmh_distance )         
            post_logits_phat =  (phat_nominator / phat_denominator).detach().clone() 


            ## build z \sim q(z): categorical distribution
            # (2): topk<10            
            post_logits_phat_topk,logit_idx = post_logits_phat.topk(topk, dim=1)
            cat_h_sparse_samples = Categorical(logits=post_logits_phat_topk).sample((nfsample,))
            cat_h_samples = torch.cat([ logit_idx[range(len(logit_idx)),i_samples].view(1,-1) for i_samples in cat_h_sparse_samples],dim=0)                               
                                
            ## get logits samples
            peak_idx            =  torch.cat( [F.one_hot(i_sample, num_classes=logits_sample.shape[-1])[None] for i_sample in cat_h_samples] , dim = 0 ) 
            soft_idx            =  1- peak_idx
            temperature_samples =  peak_idx/self.temperature_peak + soft_idx/self.temperature_soft
            logits_sample       =  logits_sample * temperature_samples

            
            #breakpoint()#
                        
        return logits_sample
    
    

    
    def forward_bayes(self, x, y=None,  use_bayes=False, nfsample: int= 1, topk: int=3):
        h_x = self.forward_features_bayes(x)        
        #if y is not None:
        #    self.update_hidden_module_from_hx(h_x, y)
        yhat = self.forward_head_bayes(h_x, use_bayes=use_bayes, nfsample=nfsample, topk=topk)
        return yhat
        
    
    def compute_logit_samples(self,x , y=None, use_bayes=True, nfsample=1, topk = 10, configs=None):
        if use_bayes:
            nfsample = configs.train.nfsample
        else:
            nfsample = 1
        return self.forward_bayes(x , y=y, use_bayes=use_bayes,nfsample=nfsample, topk= topk)        
        
            



    ## mixture model 
    def update_hidden_module(self, x, y):
        #h = self.deterministic_nn(x)                
        h = self.forward_features_bayes(x)
        self.mixture_module.update_layers_static_batch(h, y)
        return 
    
    def update_hidden_module_from_hx(self, h_x, y):
        # update mixture feature
        self.mixture_module.update_layers_static_batch(h_x, y)
        return     







    ## seems to be revised : self.compute_lastfuncdist_wo_jacobian seems to wrong
    def compute_fdist(self, xc ,p_params=None,configs=None):                

        # get hypparams from configs
        priorstdout = configs.model.priorstdout
        
        self.train()             
           
        # get last layer features
        hc = self.forward_features_bayes(xc)        
        
        # compute "q" functional dist on context set        
        param_q_mean, param_q_logstd        = list(self.head_q_mean.parameters()) ,  list(self.head_q_logstd)             
        qf_mean_xc , qf_diagcov_xc          = self.compute_lastfuncdist_wo_jacobian( hc, param_q_mean, param_q_logstd )   #without jacobian computation , ## seems to be revised   
        qf_std_xc                           = torch.sqrt(qf_diagcov_xc)
        
        # compute "p" functional dist on context set
        if p_params is None:            
            pf_mean_xc =                1*torch.zeros_like(qf_mean_xc).to(xc.device)
            pf_std_xc  = priorstdout*torch.ones_like(qf_mean_xc).to(xc.device)            
        else:        
            # param_p_mean, param_p_logstd          = p_params
            param_p_mean, param_p_logstd    = list(self.head_p_mean) ,  list(self.head_p_logstd)      

            ## swag diag            
            pf_mean_xc , pf_diagcov_xc     = self.compute_lastfuncdist_wo_jacobian( hc, param_p_mean, param_p_logstd )   #without jacobian computation         
            pf_std_xc                      = priorstdout*torch.sqrt(pf_diagcov_xc)            
            pf_mean_xc, pf_std_xc          = pf_mean_xc.detach().clone(), pf_std_xc.detach().clone() #.detach().clone()
 
            if configs.train.topk < pf_mean_xc.shape[-1]:
                # mariginalize posterior on nbd
                nbd_idx    = torch.topk(qf_mean_xc, k = configs.train.topk,dim=-1)[1]
                qf_mean_xc, qf_std_xc = torch.gather(qf_mean_xc,1,nbd_idx), torch.gather(qf_std_xc ,1,nbd_idx)    
                pf_mean_xc, pf_std_xc = torch.gather(pf_mean_xc,1,nbd_idx), torch.gather(pf_std_xc ,1,nbd_idx)    
            
        #param_qf = (qf_mean_xc, qf_std_xc)    
        param_qf = (qf_mean_xc, qf_std_xc)    
        param_pf = (pf_mean_xc, pf_std_xc)

        #return param_qf, param_pf 
        return param_qf, param_pf, None 
    
        
        
        
    def compute_sup_kl(self, params_qf, params_pf, weight=None, Lowrank=True, Diag=True, nctset=2, eps=1e-8):        
        def _compute_kl(params_qf, params_pf , Lowrank=True, Diag=False):
            qf_dist = build_MVN( params_qf, asprior=False, Lowrank=Lowrank, Diag=Diag)
            pf_dist = build_MVN( params_pf, asprior=True,  Lowrank=Lowrank, Diag=Diag)
            kl_div_qf_pf = kl_divergence(qf_dist,pf_dist)
            kl_div = kl_div_qf_pf.sum(dim=-1)
            return kl_div #(nb,)
        
        
        # set params
        qf_mean,qf_cov = params_qf
        pf_mean,pf_cov = params_pf

        # split ctxset
        num_batch = qf_mean.shape[0] // nctset
        idx_set = [torch.randperm(qf_mean.shape[0])[i*num_batch:(i+1)*num_batch] for i in range(nctset) ]

        # compute kl
        kl_divergence_list = []
        for i_set in idx_set:
            i_params_q = qf_mean[i_set] , qf_cov[i_set]
            i_params_p = pf_mean[i_set] , pf_cov[i_set]       
            i_kl_batch = _compute_kl(i_params_q, i_params_p, Lowrank=Lowrank, Diag=Diag )       
            if weight is None:            
                i_weight_kl_batch = i_kl_batch                
            else:
                i_weight          = weight[i_set]                       
                i_weight_kl_batch = i_weight*i_kl_batch
            kl_divergence_list.append( (i_weight_kl_batch).sum(dim=-1,keepdim=True) )

        kl_divergence_list = torch.cat(kl_divergence_list,dim=0)
        sup_kl = (1 / num_batch)*kl_divergence_list.max()
        sup_kl = torch.clamp(sup_kl,min=eps)  
            
        return sup_kl
    
    
    

    def get_current_p_params(self):
        #return [i_param.data.detach().clone() for i_param in self.stochastic_nn.parameters()]  
        return [i_param.data.detach().clone() for i_param in self.head_q_mean.parameters()]  


    def update_params_p(self,param_p_mean,param_p_logstd,base_nan=-12.):
        for i_p_mu,i_p_logstd,i_u_p_mu,i_u_p_logstd in     zip(self.head_p_mean,
                                                               self.head_p_logstd,
                                                               param_p_mean,
                                                               param_p_logstd):
            i_p_mu.data     = i_u_p_mu.data
            i_p_logstd.data = torch.nan_to_num(i_u_p_logstd.data, nan= base_nan) 
        return             
            

            
        
    def predict_eval(self, x ,use_bayes=True ,topk=3, nfsample=1):        
        self.eval()        
        # compute function samples        
        #logits_sample = self.forward_bayes(x , use_bayes =True, nfsample=nfsample,topk=topk)
        logits_sample = self.forward_bayes(x , use_bayes =use_bayes, nfsample=nfsample,topk=topk)

        pred = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 
                                
        outs = AttrDict()
        outs.pred = pred.mean(dim=0)  #(nb,nout)        
        outs.logits = logits_sample       
        return outs               
    
    

    
    
    
    
    
    
    
    
## -----------------------
# revise  VisionTransformer_Bayes_F_WO_Jacobian_KDE
## -----------------------    
class VisionTransformer_Bayes_F_WO_Jacobian_KDE_ADVH(VisionTransformer):

    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            no_embed_class: bool = False,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: str = '',
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[Callable] = None,
            act_layer: Optional[Callable] = None,
            block_fn: Callable = Block,
            mlp_layer: Callable = Mlp,
            priorstdout: float = 10.,
            temppeak : float = 0.9,
    ):
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super(VisionTransformer_Bayes_F_WO_Jacobian_KDE_ADVH,self).__init__(img_size,
                                                                            patch_size,
                                                                            in_chans,
                                                                            num_classes,
                                                                            global_pool,
                                                                            embed_dim,
                                                                            depth,
                                                                            num_heads,
                                                                            mlp_ratio,
                                                                            qkv_bias,
                                                                            qk_norm,
                                                                            init_values,
                                                                            class_token,
                                                                            no_embed_class,
                                                                            pre_norm,
                                                                            fc_norm,
                                                                            drop_rate,
                                                                            pos_drop_rate,
                                                                            patch_drop_rate,
                                                                            proj_drop_rate,
                                                                            attn_drop_rate,
                                                                            drop_path_rate,
                                                                            weight_init,
                                                                            embed_layer,
                                                                            norm_layer,
                                                                            act_layer,
                                                                            block_fn,
                                                                            mlp_layer,            
                                                                        )

        ## Classifier Head
        self.scalelogstd = scalelogstd
        self.global_pool = global_pool        
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if weight_init != 'skip':
            self.init_weights(weight_init)



        ## hidden feature mixture model
        self.mixture_module = Hiddenfeature_Mixture_Module(nfeature=self.embed_dim, nlabel=num_classes, usediagcov=True)

        ## temperature hyperparams
        #self.temperature_peak = 0.5
        self.temperature_peak = temppeak        
        self.temperature_soft = 1. 
        print('temppeak: {}, tempsoft: {}'.format(self.temperature_peak,self.temperature_soft))
        
        
        self.priorstdout      = priorstdout       
        

        self.temperature_distance = 1e-2 




    ##------------------------------------------------------------------------------------------------------------------------------
    ## revise implementation 
    ##------------------------------------------------------------------------------------------------------------------------------
    
    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
            
        self.head_q_mean     = nn.Linear(self.embed_dim, num_classes, bias=False) if num_classes > 0 else nn.Identity()
        self.head_q_logstd   = nn.ParameterList( [nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd,self.scalelogstd+1)  ,requires_grad=True )  for ith_param in self.head_q_mean.parameters()] )
        
        self.head_p_mean     = nn.ParameterList( [nn.Parameter(torch.zeros_like(ith_param) , requires_grad=False )   for ith_param in self.head_q_mean.parameters()]  )
        self.head_p_logstd   = nn.ParameterList( [nn.Parameter(torch.ones_like(ith_param)  , requires_grad=False )   for ith_param in self.head_q_mean.parameters()]  )          
        
        func_model, _ , buffers = make_functional_with_buffers(self.head_q_mean)                
        self.func_model =   lambda params, x_in : func_model(params, buffers, x_in) 
        return 



    def forward_features_bayes(self,x):
        x = self.forward_features(x)
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x 
    
    
    
    # computation checked     
    def compute_lastfuncdist_wo_jacobian(self, h , param_q_mean, param_q_logstd ):
        assert len(param_q_logstd) <=2
        
        # compute f_mean
        f_mean = self.func_model(param_q_mean, h ) 

        # comopute f_std
        param_q_std        = [transform_logstd(ith_q_logstd) for ith_q_logstd in param_q_logstd ]                
        f_var_linear_inner = param_q_std[0][None,...]*h[:,None,:] #(1,ydim,nparams)*(nb,1,nparams) = (nb,ydim,nparams)
        f_var_linear       = torch.pow(f_var_linear_inner,2).sum(dim=-1) #(nb,ydim)

        if len(param_q_logstd) == 2:               
            f_var_bias = torch.pow( param_q_std[1][None,...], 2)  
        else :  
            f_var_bias = 0.0
        f_diagcov = f_var_linear + f_var_bias #(nb,ydim)
    
        return f_mean, f_diagcov




    ## updated at 091823
    def compute_kde_pfuncdist_from_h(self, ht, param_q_mean, param_q_logstd , requires_grad=False ):
        h_mean, _, _ = self.mixture_module.get_static_params()
        h_mean = h_mean.detach().clone()

        kde_ht = self.mixture_module.compute_mh_distance_batch(ht ,overclass=True)
        kde_weight = F.softmax( kde_ht /self.temperature_distance, dim=-1)
        proj_ht = (kde_weight[...,None]*h_mean[None,...]).sum(dim=1) #(nb,ncls,1)*(nb,ncls,ndim) = (nb,ndim)

        f_mean = self.func_model(param_q_mean, proj_ht)
        param_q_std        = [transform_logstd(ith_q_logstd) for ith_q_logstd in param_q_logstd ]                
        f_var_linear_inner = param_q_std[0][None,...]*proj_ht[:,None,:] #(1,ydim,nparams)*(nb,1,nparams) = (nb,ydim,nparams)
        f_var_linear       = torch.pow(f_var_linear_inner,2).sum(dim=-1) #(nb,ydim)
        f_var_bias         = torch.pow( param_q_std[1][None,...], 2) if len(param_q_logstd) == 2 else 0.0                
        f_diagcov          = f_var_linear + f_var_bias #(nb,ydim)
        
        
        if requires_grad:        
            # stop gradient
            return f_mean, f_diagcov, proj_ht
        else:
            # stop gradient
            return f_mean.detach().clone(), f_diagcov.detach().clone(), proj_ht.detach().clone()
    
    
    
    







    def forward_head_bayes(self, h_x, use_bayes: bool = False,  nfsample: int= 1, topk: int=3):
        # chanage as bayes 
        param_q_mean, param_q_logstd  =  list(self.head_q_mean.parameters()) ,  list(self.head_q_logstd)    
        qf_mean, qf_diagcov             = self.compute_lastfuncdist_wo_jacobian( h_x , param_q_mean, param_q_logstd )
        qf_diagstd = torch.sqrt(qf_diagcov)


        # build sample functions            
        if not use_bayes:
            logits_sample = qf_mean[None,...]
        else:
            logits_sample = qf_mean[None,...] + qf_diagstd[None,...]*torch.randn(nfsample,*qf_mean.shape).to(h_x.device)  #(nfsample,nb,nydim)


            ## compute phat by MPA             
            phat_nominator   =  qf_mean      
            nmh_distance     = -self.mixture_module.compute_mh_distance_batch(h_x,overclass=True, usenormalize = False)      

            phat_denominator =  torch.sqrt( 1 + torch.pi/8 * nmh_distance )         
            post_logits_phat =  (phat_nominator / phat_denominator).detach().clone() 


            ## build z \sim q(z): categorical distribution
            # (2): topk<10            
            post_logits_phat_topk,logit_idx = post_logits_phat.topk(topk, dim=1)
            cat_h_sparse_samples = Categorical(logits=post_logits_phat_topk).sample((nfsample,))
            cat_h_samples = torch.cat([ logit_idx[range(len(logit_idx)),i_samples].view(1,-1) for i_samples in cat_h_sparse_samples],dim=0)                               
                                
            ## get logits samples
            peak_idx            =  torch.cat( [F.one_hot(i_sample, num_classes=logits_sample.shape[-1])[None] for i_sample in cat_h_samples] , dim = 0 ) 
            soft_idx            =  1- peak_idx
            temperature_samples =  peak_idx/self.temperature_peak + soft_idx/self.temperature_soft
            logits_sample       =  logits_sample * temperature_samples

            
            #breakpoint()#
                        
        return logits_sample
    
    
    ## mixture model 
    def update_hidden_module(self, x, y):
        #h = self.deterministic_nn(x)                
        h = self.forward_features_bayes(x)
        self.mixture_module.update_layers_static_batch(h, y)
        return 
    
    def update_hidden_module_from_hx(self, h_x, y):
        # update mixture feature
        self.mixture_module.update_layers_static_batch(h_x, y)
        return     
    

    
    def forward_bayes(self, x, y=None,  use_bayes=False, nfsample: int= 1, topk: int=3):
        h_x = self.forward_features_bayes(x)        
        yhat = self.forward_head_bayes(h_x, use_bayes=use_bayes, nfsample=nfsample, topk=topk)
        return yhat
        
    
    def compute_logit_samples(self,x , y=None, use_bayes=True, nfsample=1, topk = 10, configs=None):
        if use_bayes:
            nfsample = configs.train.nfsample
        else:
            nfsample = 1
        return self.forward_bayes(x , y=y, use_bayes=use_bayes,nfsample=nfsample, topk= topk)        
    
    
    
    def compute_fdist(self, xc ,p_params=None,configs=None):                

        # get hypparams from configs
        priorstdout = configs.model.priorstdout
        
        self.train()             
           
        # get last layer features
        hc = self.forward_features_bayes(xc)        
        
        # compute "q" function space dist on context set        
        param_q_mean, param_q_logstd        = list(self.head_q_mean.parameters()) ,  list(self.head_q_logstd)             
        qf_mean_xc , qf_diagcov_xc          = self.compute_lastfuncdist_wo_jacobian( hc, param_q_mean, param_q_logstd )   #without jacobian computation , ## seems to be revised   
        qf_std_xc                           = torch.sqrt(qf_diagcov_xc)
        
        # compute "p" function space dist on context set
        if p_params is None:            
            pf_mean_xc =                1*torch.zeros_like(qf_mean_xc).to(xc.device)
            pf_std_xc  = priorstdout*torch.ones_like(qf_mean_xc).to(xc.device)            
        else:        
            param_p_mean, param_p_logstd    = list(self.head_p_mean) ,  list(self.head_p_logstd)      
            pf_mean_xc , pf_diagcov_xc, _  = self.compute_kde_pfuncdist_from_h( hc, param_p_mean, param_p_logstd , requires_grad=False )                    
            pf_std_xc                      = priorstdout*torch.sqrt(pf_diagcov_xc)            
 
            if configs.train.topk < pf_mean_xc.shape[-1]:
                # mariginalize posterior on nbd
                nbd_idx    = torch.topk(qf_mean_xc, k = configs.train.topk,dim=-1)[1]
                qf_mean_xc, qf_std_xc = torch.gather(qf_mean_xc,1,nbd_idx), torch.gather(qf_std_xc ,1,nbd_idx)    
                pf_mean_xc, pf_std_xc = torch.gather(pf_mean_xc,1,nbd_idx), torch.gather(pf_std_xc ,1,nbd_idx)    
            
        #param_qf = (qf_mean_xc, qf_std_xc)    
        param_qf = (qf_mean_xc, qf_std_xc)    
        param_pf = (pf_mean_xc, pf_std_xc)

        #return param_qf, param_pf 
        return param_qf, param_pf, None 
    
        
        
        
    def compute_sup_kl(self, params_qf, params_pf, weight=None, Lowrank=True, Diag=True, nctset=2, eps=1e-8):        
        def _compute_kl(params_qf, params_pf , Lowrank=True, Diag=False):
            qf_dist = build_MVN( params_qf, asprior=False, Lowrank=Lowrank, Diag=Diag)
            pf_dist = build_MVN( params_pf, asprior=True,  Lowrank=Lowrank, Diag=Diag)
            kl_div_qf_pf = kl_divergence(qf_dist,pf_dist)
            kl_div = kl_div_qf_pf.sum(dim=-1)
            return kl_div #(nb,)
        
        
        # set params
        qf_mean,qf_cov = params_qf
        pf_mean,pf_cov = params_pf

        # split ctxset
        num_batch = qf_mean.shape[0] // nctset
        idx_set = [torch.randperm(qf_mean.shape[0])[i*num_batch:(i+1)*num_batch] for i in range(nctset) ]

        # compute kl
        kl_divergence_list = []
        for i_set in idx_set:
            i_params_q = qf_mean[i_set] , qf_cov[i_set]
            i_params_p = pf_mean[i_set] , pf_cov[i_set]       
            i_kl_batch = _compute_kl(i_params_q, i_params_p, Lowrank=Lowrank, Diag=Diag )       
            if weight is None:            
                i_weight_kl_batch = i_kl_batch                
            else:
                i_weight          = weight[i_set]                       
                i_weight_kl_batch = i_weight*i_kl_batch
            kl_divergence_list.append( (i_weight_kl_batch).sum(dim=-1,keepdim=True) )

        kl_divergence_list = torch.cat(kl_divergence_list,dim=0)
        sup_kl = (1 / num_batch)*kl_divergence_list.max()
        sup_kl = torch.clamp(sup_kl,min=eps)  
            
        return sup_kl
    
    
    

    def get_current_p_params(self):
        return [i_param.data.detach().clone() for i_param in self.head_q_mean.parameters()]  


    def update_params_p(self,param_p_mean,param_p_logstd,base_nan=-12.):
        for i_p_mu,i_p_logstd,i_u_p_mu,i_u_p_logstd in     zip(self.head_p_mean,
                                                               self.head_p_logstd,
                                                               param_p_mean,
                                                               param_p_logstd):
            i_p_mu.data     = i_u_p_mu.data
            i_p_logstd.data = torch.nan_to_num(i_u_p_logstd.data, nan= base_nan) 
        return             
            

            
        
    def predict_eval(self, x ,use_bayes=True ,topk=3, nfsample=1):        
        self.eval()        
        # compute function samples        
        logits_sample = self.forward_bayes(x , use_bayes =use_bayes, nfsample=nfsample,topk=topk)

        pred = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 
                                
        outs = AttrDict()
        outs.pred = pred.mean(dim=0)  #(nb,nout)        
        outs.logits = logits_sample       
        return outs               
    
    

    
    ##-------------------------------------------
    ## build Adversarial hidden feature
    ##-------------------------------------------

    def forward_bayes2(self, x, y=None,  use_bayes=False, nfsample: int= 1, topk: int=3):
        h_x = self.forward_features_bayes(x)        
        #if y is not None:
        #    self.update_hidden_module_from_hx(h_x, y)
        yhat = self.forward_head_bayes(h_x, use_bayes=use_bayes, nfsample=nfsample, topk=topk)
        return yhat, h_x
        
    
    def compute_logit_samples2(self,x , y=None, use_bayes=True, nfsample=1, topk = 10, configs=None):
        if use_bayes:
            nfsample = configs.train.nfsample
        else:
            nfsample = 1
        return self.forward_bayes2(x , y=y, use_bayes=use_bayes,nfsample=nfsample, topk= topk)       


        
    def compute_h_x_grad(self,h_x):
        h_x_copy = h_x.detach().clone().requires_grad_(True)                                                                                          
        mh_distance = -self.mixture_module.compute_mh_distance_batch(h_x_copy, overclass=False, usenormalize=False)  
        mh_distance.mean().backward()   
        h_x_grad = h_x_copy.grad.data 
        return h_x_copy, h_x_grad
    
    

    
    def compute_fdist_from_h(self, h_xc , p_params=None, configs=None):                                
        self.train()                
        
        
        # get last layer features
        priorstdout = configs.model.priorstdout

        
        # compute "q" functional dist on context set        
        param_q_mean, param_q_logstd        = list(self.head_q_mean.parameters()) ,  list(self.head_q_logstd)             
        qf_mean_xc , qf_diagcov_xc          = self.compute_lastfuncdist_wo_jacobian( h_xc, param_q_mean, param_q_logstd )   #without jacobian computation         
        qf_std_xc                           = torch.sqrt(qf_diagcov_xc)
        
        # compute "p" functional dist on context set
        if p_params is None:            
            pf_mean_xc                      = torch.zeros_like(qf_mean_xc).to(h_xc.device)
            pf_std_xc                       = priorstdout*torch.ones_like(qf_mean_xc).to(h_xc.device)            
        else:        
            param_p_mean, param_p_logstd    = list(self.head_p_mean) ,  list(self.head_p_logstd)      

            
            ## swag diag            
            pf_mean_xc , pf_diagcov_xc, _  = self.compute_kde_pfuncdist_from_h( h_xc, param_p_mean, param_p_logstd , requires_grad=False )  #without jacobian computation         
            pf_std_xc                      = priorstdout*torch.sqrt(pf_diagcov_xc)            

 
            if configs.train.topk < pf_mean_xc.shape[-1]:
                # mariginalize posterior on nbd
                nbd_idx    = torch.topk(qf_mean_xc, k = configs.train.topk,dim=-1)[1]
                qf_mean_xc, qf_std_xc = torch.gather(qf_mean_xc,1,nbd_idx), torch.gather(qf_std_xc ,1,nbd_idx)    
                pf_mean_xc, pf_std_xc = torch.gather(pf_mean_xc,1,nbd_idx), torch.gather(pf_std_xc ,1,nbd_idx)    

            
        param_qf = (qf_mean_xc, qf_std_xc)    
        param_pf = (pf_mean_xc, pf_std_xc)
                
        return param_qf, param_pf, None        
            

    
            

