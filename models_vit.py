
import numpy as np
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

from torch.nn.utils import parameters_to_vector        
from functorch import vmap,grad,jacfwd,jacrev
from functorch import make_functional, make_functional_with_buffers

from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal as LowMVN
from torch.distributions.kl import kl_divergence


## --------------------------------------------
## timm==0.9.2
## --------------------------------------------

import timm
from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, RmsNorm, PatchDropout, use_fused_attn, SwiGLUPacked

from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import named_apply, checkpoint_seq, adapt_input_conv
from timm.models._registry import generate_default_cfgs, register_model, register_model_deprecations    
    


## https://pytorch.org/blog/Accelerating-Hugging-Face-and-TIMM-models/
## https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py / baseline
# import timm
# model = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=10)

from timm.models.registry import register_model
from timm.models.vision_transformer import _create_vision_transformer
from timm.models.vision_transformer import checkpoint_filter_fn
from timm.models.vision_transformer import VisionTransformer


from addict import Dict as AttrDict



## --------------------------------------------------------------------------------------------
## get_vit_model
## --------------------------------------------------------------------------------------------

def _get_vit_model(configs):
    
    # ---------------------------------------------- #
    # configure to model 
    # ---------------------------------------------- #
    modelname = configs.model.modelname
    pretrained = configs.model.pretrained
    infer = configs.model.infer
    outdim = configs.model.outdim
    
    priorstdout = configs.model.priorstdout if 'priorstdout' in configs.model.keys() else 10.0  
    mixalpha    = configs.train.mixalpha if 'mixalpha' in configs.train.keys() else 0.05 
    scalelogstd = configs.model.scalelogstd


    model = vit_bayes_base_patch16_224(modelname,pretrained=pretrained,infer=infer, num_classes=outdim,
                                        priorstdout = priorstdout,
                                        mixalpha    = mixalpha, 
                                        scalelogstd = scalelogstd)

    
    global_pool = model.global_pool 
    model.reset_classifier(num_classes=outdim, global_pool=global_pool)
    
    return model



@register_model
def vit_bayes_base_patch16_224(modelname='vitbp16r224',pretrained=False,infer='map',use_mixture_diag=True, **kwargs) -> VisionTransformer:
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    
    ## has been conducted; guess pretrained from imagenet 21k finetuned imagenet 1k
   
    ## pretrained imagenet 21k and finetuned imagenet 1k
    if modelname == 'vitbp16r224':
        model = _create_vision_transformer_Bayes('vit_base_patch16_224',
                                                pretrained=True, 
                                                infer=infer,
                                                **dict(model_args, **kwargs)
                                                )

    ## pretrained imagenet 21k
    if modelname == 'vitbp16r224-21k':    
        model = _create_vision_transformer_Bayes('vit_base_patch16_224.augreg_in21k',
                                                pretrained=True, 
                                                infer=infer,
                                                **dict(model_args, **kwargs)
                                                )

    ## pretrained imagenet 1k
    if modelname == 'vitbp16r224-1k':    
        model = _create_vision_transformer_Bayes('vit_base_patch16_224.augreg_in1k',
                                                pretrained=True, 
                                                infer=infer,
                                                **dict(model_args, **kwargs)
                                                )

    return model





def _create_vision_transformer_Bayes(variant, pretrained=False, infer='map', **kwargs):    
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented fo Vision Transformer models.')

    if 'flexi' in variant:
        ## FIXME Google FlexiViT pretrained models have a strong preference for bilinear patch / embed
        # interpolation, other pretrained models resize better w/ anti-aliased bicubic interpolation.
        _filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear', antialias=False)
    else:
        _filter_fn = checkpoint_filter_fn

    if infer == 'map':
        vit_cls = VisionTransformer_Map
    elif infer == 'vbp':
        vit_cls = VisionTransformer_Bayes_Proposed
    else:
        raise ValueError(f"Unsupported inference method {infer!r}; choose 'map' or 'vbp'")

    return build_model_with_cfg(vit_cls,variant,pretrained,pretrained_filter_fn=_filter_fn,**kwargs,)







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
            mixalpha : float = 0.05,            
            scalelogstd: float = -4,
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
        super(VisionTransformer_Map,self).__init__(  img_size,
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

#scalelogstd = -4 #for existing dataset

# scalelogstd = -4 #for  imagenet v20
# scalelogstd = -8  #for imagenet v22,v23
#scalelogstd = -6  #for imagenet v24,v25


#scalelogstd = -6  #v41,42
#scalelogstd = -3  #v43


# scalelogstd = -4  #v44


#scalelogstd = -4  #v1 for jounal submission


eps_log     = 1e-8
eps_zitter  = 1e-8
eps_clamp   = 1e-8



def transform_logstd(logstd):
    return F.softplus(logstd, beta=1, threshold=20)

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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
## --------------------------------------------------------------------------------------------
## baseline: vit last layer vb on function space 
## --------------------------------------------------------------------------------------------
    





## --------------------------------------------------------------------------------------------
## proposed models
## + vit last layer vb on function space,
## + without jacobian computation 
## + feature space mh, 
## --------------------------------------------------------------------------------------------

from torch.distributions.categorical import Categorical





## use diag cov
class Hiddenfeature_Mixture_Diag_Module(nn.Module):     
    def __init__(self, nfeature: int, nlabel: int , usediagcov=True, gamma=0.99,eps_zitter=1e-4):
        super(Hiddenfeature_Mixture_Diag_Module,self).__init__()
        
        self.nfeature = nfeature
        self.nlabel = nlabel
        self.eps_zitter = eps_zitter
        #self.gamma = gamma
        self.usediagcov = usediagcov
        
        # per epoch        
        self.static_cnt_sum    =  nn.Parameter( torch.zeros(self.nlabel), requires_grad=False )        
        self.static_mean_sum   =  nn.Parameter( torch.zeros(self.nlabel, self.nfeature), requires_grad=False )
        self.static_cov_sum    =  nn.Parameter( torch.zeros(self.nlabel, self.nfeature) , requires_grad=False )    
        
        print('-'*100)
        print('Hiddenfeature_Mixture_Diag_Module is initialized')    
        print('-'*100)
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
                k_delta_cov      = h_feature[idx_current] - k_label_mean    #(nb,nfilters)
                k_label_cov_sum  = (k_delta_cov**2).sum(dim=0,keepdim=True)  #(1,nfilters)
            else:
                k_label_mean_sum = torch.zeros(1,self.nfeature).to(y.device) 
                k_label_cov_sum  = torch.zeros(1,self.nfeature).to(y.device)                               
                
            static_mean_sum.append(k_label_mean_sum)     
            static_cov_sum.append(k_label_cov_sum) 
        
        static_mean_sum = torch.cat(static_mean_sum, dim=0)   #(nlables, nfilters)
        static_cov_sum  = torch.cat(static_cov_sum, dim=0)    #(nlables, nfilters)
        static_cnt      = torch.tensor([idx_current.sum() for idx_current in idx_per_label]).to(y.device) #(nlables)        
        

        # compute layer feature
        nb = len(y)        
        
        ## update static params per batch
        self.static_cnt_sum.data     +=  static_cnt.data
        self.static_mean_sum.data    +=  static_mean_sum.data
        self.static_cov_sum.data     +=  static_cov_sum.data
        

        return 
    
    def get_static_params(self, eps=1e-4):

        mean               = self.static_mean_sum.data                          / (self.static_cnt_sum.data.view(-1,1) + eps)
        mean_shared        = self.static_mean_sum.data.sum(dim=0,keepdims=True) / (self.static_cnt_sum.data.sum() + eps)
        cov_shared         = self.static_cov_sum.data.sum(dim=0)                / (self.static_cnt_sum.data.sum() + eps) 
        
        
        # if self.usediagcov:
        cov_shared         = cov_shared + self.eps_zitter
        precision_shared   = 1 / cov_shared 

        return mean,mean_shared,precision_shared
        

    def compute_mh_distance_batch(self, h_feature , datatype = 'x' ,overclass=False, usenormalize=True):    
        layers_mean, _ , layers_precision = self.get_static_params()
                           

        ## compute distance                           
        # if self.usediagcov :
        delta_h = h_feature[:,None,:] - layers_mean[None,:,:]
        squareterm_inner =  torch.pow(delta_h,2) * layers_precision[None,None,...]  #(nb,ncls,nhdim)*(1,1,nhdim) = (nb,ncls,nhdim)
        mh_distance_over_class  = -1*squareterm_inner.sum(dim=-1)       

        if usenormalize:
            mh_distance_over_class  = mh_distance_over_class / h_feature.size(-1)
        mh_distance, assigned_label = mh_distance_over_class.max(dim=-1)            

        if overclass:
            return mh_distance_over_class
        else:
            return mh_distance
        





    


class VisionTransformer_Bayes_Proposed(VisionTransformer):

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
            mixalpha : float = 0.05,
            scalelogstd: float = -4,
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
        super(VisionTransformer_Bayes_Proposed,self).__init__(img_size,
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


        self.priorstdout = priorstdout
        self.mixalpha    = mixalpha
        self.scalelogstd = scalelogstd
        self.global_pool = global_pool        
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        

        if weight_init != 'skip':
            self.init_weights(weight_init)


        ## hidden feature mixture model
        self.mixture_module = Hiddenfeature_Mixture_Diag_Module(nfeature=embed_dim, nlabel=num_classes, usediagcov=True)

        #breakpoint()

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
            
        self.head_q_mu     = nn.Linear(self.embed_dim, num_classes, bias=False) if num_classes > 0 else nn.Identity()
        self.head_q_logstd = nn.ParameterList( [nn.Parameter(torch.empty_like(ith_param).uniform_(self.scalelogstd,self.scalelogstd+1)  ,requires_grad=True )  for ith_param in self.head_q_mu.parameters()] )
        
        self.head_p_mu     = nn.ParameterList( [nn.Parameter(torch.zeros_like(ith_param) , requires_grad=False )  for ith_param in self.head_q_mu.parameters()]  )
        self.head_p_logstd = nn.ParameterList( [nn.Parameter(torch.ones_like(ith_param)  , requires_grad=False )   for ith_param in self.head_p_mu.parameters()]  )          
        
        func_model, _ , buffers = make_functional_with_buffers(self.head_q_mu)                
        self.func_model         = lambda params, x_in : func_model(params, buffers, x_in) 

        return 

    


    def forward_features_bayes(self,x):
        x = self.forward_features(x)
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x 


    
        # computation checked     
    def compute_lastfuncdist_wo_jacobian_from_h(self, h , param_q_mean, param_q_logstd ,no_grad_jacobian=True ):
        assert len(param_q_logstd) <=2
        
        # compute f_mean
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

    
    
    
    def forward_bayes(self, x, y=None,  use_bayes=False, nfsample: int= 1):
        h_x = self.forward_features_bayes(x)        
        yhat = self.forward_head_bayes(h_x, use_bayes=use_bayes, nfsample=nfsample)
        return yhat
        
    
    def compute_logit_samples(self, x , y=None, update_h=False, use_bayes=True, nfsample=1,  configs=None):
        # if use_bayes:
        #     nfsample = configs.train.nfsample
        # else:
        #     nfsample = 1

        if use_bayes:
            nfsample = configs.train.nfsample
        else:
            nfsample = 1

        h_x       = self.forward_features_bayes(x)
        yhat      = self.forward_head_bayes(h_x ,  use_bayes=use_bayes,nfsample=nfsample)            

        if (y is not None) and (update_h == True):
            self.update_hidden_module_from_hx(h_x,y)

        return yhat, h_x   
        


    
    

            



    #def predict_train(self, x , xc , nfsample=10 ,p_params=None, nctset= 1, reglambda=0.1):        
    def compute_fdist(self, xc ,configs=None):                

        # get hypparams from configs
        priorstdout = configs.model.priorstdout
        
        self.train()             
           
        # get last layer features
        h_xc = self.forward_features_bayes(xc)        
        
        # compute "q" functional dist on context set        
        param_q_mean, param_q_logstd  = list(self.head_q_mu.parameters()) ,  list(self.head_q_logstd)             
        qf_mean_xc , qf_diagcov_xc    = self.compute_lastfuncdist_wo_jacobian_from_h( h_xc, param_q_mean, param_q_logstd, no_grad_jacobian=True )   #without jacobian computation         
        qf_std_xc                     = torch.sqrt(qf_diagcov_xc)
        
     
        # param_p_mean, param_p_logstd          = p_params
        param_p_mean, param_p_logstd    = list(self.head_p_mu) ,  list(self.head_p_logstd)      
        
        #put the priorstdout into the model            
        pf_mean_xc , pf_diagcov_xc, _  = self.compute_kde_pfuncdist_from_h( h_xc, param_p_mean, param_p_logstd , requires_grad=False, priorstdout=priorstdout )  #put the priorstdout into the model
        pf_std_xc                      = torch.sqrt(pf_diagcov_xc)            
                       
                        

            
        #param_qf = (qf_mean_xc, qf_std_xc)    
        param_qf = (qf_mean_xc, qf_std_xc)    
        param_pf = (pf_mean_xc, pf_std_xc)

        #return param_qf, param_pf 
        return param_qf, param_pf, None 


    

    def compute_fdist_from_h(self, h_xc , p_params=None, configs=None):        
        self.train()                
    
        
        # get last layer features
        priorstdout = configs.model.priorstdout

        
        # compute "q" functional dist on context set        
        param_q_mean, param_q_logstd        = list(self.head_q_mu.parameters()) ,  list(self.head_q_logstd)             
        qf_mean_xc , qf_diagcov_xc          = self.compute_lastfuncdist_wo_jacobian_from_h( h_xc, param_q_mean, param_q_logstd )   #without jacobian computation         
        qf_std_xc                           = torch.sqrt(qf_diagcov_xc)


        # param_p_mean, param_p_logstd          = p_params
        param_p_mean, param_p_logstd    = list(self.head_p_mu) ,  list(self.head_p_logstd)      
        
        #put the priorstdout into the model            
        pf_mean_xc , pf_diagcov_xc, _  = self.compute_kde_pfuncdist_from_h( h_xc, param_p_mean, param_p_logstd , requires_grad=False, priorstdout=priorstdout )  #put the priorstdout into the model
        pf_std_xc                      = torch.sqrt(pf_diagcov_xc)            
                    

            
        param_qf = (qf_mean_xc, qf_std_xc)    
        param_pf = (pf_mean_xc, pf_std_xc)
                
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

        # Partition the context batch without overlap or omitted samples.
        batch_size = qf_mean.shape[0]
        if not 1 <= nctset <= batch_size:
            raise ValueError("nctset must be between 1 and the actual context batch size")
        indices = torch.randperm(batch_size, device=qf_mean.device)
        idx_set = torch.tensor_split(indices, nctset)

        # compute kl
        kl_divergence_list = []
        for i_set in idx_set:
            i_params_q = qf_mean[i_set] , qf_cov[i_set]
            i_params_p = pf_mean[i_set] , pf_cov[i_set]       
            i_kl_batch = _compute_kl(i_params_q, i_params_p, Lowrank=Lowrank, Diag=Diag )       

            i_weight_kl_batch = i_kl_batch                
            # Normalize each group separately, including uneven final batches.
            kl_divergence_list.append(i_weight_kl_batch.mean().reshape(1))

        kl_divergence_list = torch.cat(kl_divergence_list,dim=0)
        sup_kl = kl_divergence_list.max()
        sup_kl = torch.clamp(sup_kl,min=eps)  
            
        return sup_kl
    
    
                    
        
                            
    # ## Journal version-1 ##
    # ## this seems better than version-2
    # def compute_kde_pfuncdist_from_h(self, ht, param_p_mean, param_p_logstd ,requires_grad=False ,priorstdout = 1.):


    #     h_mean, _, _ = self.mixture_module.get_static_params()
    #     h_mean = h_mean.detach().clone()

    #     kde_ht = self.mixture_module.compute_mh_distance_batch(ht ,overclass=True)
    #     kde_weight = F.softmax( kde_ht , dim=-1)
    #     proj_ht = (kde_weight[...,None]*h_mean[None,...]).sum(dim=1) #(nb,ncls,1)*(nb,ncls,ndim) = (nb,ndim)

    
    #     m_qx               = h_mean[kde_weight.argmax(dim=-1)]
    #     f_mean             = self.func_model(param_p_mean, proj_ht)


    #     param_p_std        = [transform_logstd(ith_q_logstd) for ith_q_logstd in param_p_logstd ]                
    #     f_var_linear_inner = (param_p_std[0][None,...])*((m_qx - proj_ht)[:,None,:]) #(1,ydim,nparams)*(nb,1,nparams) = (nb,ydim,nparams)
    #     f_var_linear       = (self.priorstdout**2)*torch.pow(f_var_linear_inner,2).sum(dim=-1) #(nb,ydim)
    #     f_var_linear      += torch.pow((param_p_std[0][None,...])*(m_qx[:,None,:]),2).sum(dim=-1)
        
            
        
    #     if len(param_p_logstd) == 2 : #bias term exists
    #         f_var_bias         = torch.pow( param_p_std[1][None,...], 2)                 
    #     else:
    #         f_var_bias         = torch.tensor([0.0]).to(ht.device)

    #     f_diagcov              = f_var_linear + f_var_bias #(nb,ydim)



    #     if requires_grad:        
    #         # stop gradient
    #         return f_mean, f_diagcov, proj_ht
    #     else:
    #         # stop gradient
    #         return f_mean.detach().clone(), f_diagcov.detach().clone(), proj_ht.detach().clone()   
    # 



        
                            
    # ## Journal version-2 ##
    # def compute_kde_pfuncdist_from_h(self, ht, param_p_mean, param_p_logstd ,requires_grad=False ,priorstdout = 1.):


    #     h_mean, _, _ = self.mixture_module.get_static_params()
    #     h_mean = h_mean.detach().clone()

    #     kde_ht = self.mixture_module.compute_mh_distance_batch(ht ,overclass=True)
    #     kde_weight = F.softmax( kde_ht , dim=-1)
    #     proj_ht = (kde_weight[...,None]*h_mean[None,...]).sum(dim=1) #(nb,ncls,1)*(nb,ncls,ndim) = (nb,ndim)

    
    #     m_qx               = h_mean[kde_weight.argmax(dim=-1)]
    #     f_mean             = self.func_model(param_p_mean, proj_ht)


    #     param_p_std        = [transform_logstd(ith_q_logstd) for ith_q_logstd in param_p_logstd ]                
    #     f_var_linear_inner = (param_p_std[0][None,...])*((m_qx - proj_ht)[:,None,:]) #(1,ydim,nparams)*(nb,1,nparams) = (nb,ydim,nparams)
    #     f_var_linear       = torch.pow(f_var_linear_inner,2).sum(dim=-1) #(nb,ydim)
    #     f_var_linear      += torch.pow((param_p_std[0][None,...])*(m_qx[:,None,:]),2).sum(dim=-1)
        
            
        
    #     if len(param_p_logstd) == 2 : #bias term exists
    #         f_var_bias         = torch.pow( param_p_std[1][None,...], 2)                 
    #     else:
    #         f_var_bias         = torch.tensor([0.0]).to(ht.device)

    #     f_diagcov              = (self.priorstdout**2)*(f_var_linear + f_var_bias) #(nb,ydim)

    #     if requires_grad:        
    #         # stop gradient
    #         return f_mean, f_diagcov, proj_ht
    #     else:
    #         # stop gradient
    #         return f_mean.detach().clone(), f_diagcov.detach().clone(), proj_ht.detach().clone()   
            
    

                         
    ## Journal version-3 ##
    def compute_kde_pfuncdist_from_h(self, ht, param_p_mean, param_p_logstd ,requires_grad=False ,priorstdout = 1.):


        h_mean, _, _ = self.mixture_module.get_static_params()
        h_mean = h_mean.detach().clone()

        kde_ht = self.mixture_module.compute_mh_distance_batch(ht ,overclass=True)
        kde_weight = F.softmax( kde_ht , dim=-1)
        proj_ht = (kde_weight[...,None]*h_mean[None,...]).sum(dim=1) #(nb,ncls,1)*(nb,ncls,ndim) = (nb,ndim)

    
        m_qx               = h_mean[kde_weight.argmax(dim=-1)]
        f_mean             = self.func_model(param_p_mean, proj_ht)


        param_p_std        = [transform_logstd(ith_q_logstd) for ith_q_logstd in param_p_logstd ]                
        # param_p_std = [10 * transform_logstd(ith_q_logstd) for ith_q_logstd in param_p_logstd]  # Outer variance factor = 100 (including bias)
        f_var_linear_inner = self.priorstdout*(param_p_std[0][None,...])*((m_qx - proj_ht)[:,None,:]) #(1,ydim,nparams)*(nb,1,nparams) = (nb,ydim,nparams)
        f_var_linear       = torch.pow(f_var_linear_inner,2).sum(dim=-1) #(nb,ydim)
        f_var_linear      += torch.pow((param_p_std[0][None,...])*(m_qx[:,None,:]),2).sum(dim=-1)
        
            
        
        if len(param_p_logstd) == 2 : #bias term exists
            f_var_bias         = torch.pow( param_p_std[1][None,...], 2)                 
        else:
            f_var_bias         = torch.tensor([0.0]).to(ht.device)

        f_diagcov = f_var_linear + f_var_bias #(nb,ydim)

        if requires_grad:        
            # stop gradient
            return f_mean, f_diagcov, proj_ht
        else:
            # stop gradient
            return f_mean.detach().clone(), f_diagcov.detach().clone(), proj_ht.detach().clone()   
    

    
    # ## updated at 011324                        
    def compute_h_x_grad(self,h_x):
        h_x_copy          = h_x.detach().clone().requires_grad_(True)                                                        
        mh_distance       = self.mixture_module.compute_mh_distance_batch(h_x_copy, overclass=True, usenormalize=True) 
        mh_distance_topk1 = torch.topk(mh_distance,k=1)[0]                  
        loss              = mh_distance_topk1 - torch.logsumexp(mh_distance, dim=1, keepdim=True)
        loss.mean().backward()   


            
        h_x_grad = -h_x_copy.grad.data 
        return h_x_copy, h_x_grad    
    



    def update_hidden_module_from_hx(self, h_x, y):
        # update mixture feature
        self.mixture_module.update_layers_static_batch(h_x, y)
        return     


    
    def get_current_p_params(self):
        return [i_param.data.detach().clone() for i_param in self.head_q_mu.parameters()]  


    def update_params_p(self,param_p_mu,param_p_logstd,base_nan=-12.):
        for i_p_mu,i_p_logstd,i_u_p_mu,i_u_p_logstd in     zip(self.head_p_mu,
                                                               self.head_p_logstd,
                                                               param_p_mu,
                                                               param_p_logstd):
            i_p_mu.data     = i_u_p_mu.data
            i_p_logstd.data = torch.nan_to_num(i_u_p_logstd.data, nan= base_nan) 
        return             
            


        
    def predict_eval(self, x ,use_bayes=True , nfsample=1, topk=None):        
        self.eval()        
        # compute function samples        
        h_x           = self.forward_features_bayes(x)        
        logits_sample = self.forward_head_bayes(h_x, use_bayes=True, nfsample=nfsample)

        pred          = F.softmax(logits_sample,dim=-1) #(nfsample,nb,nout) 
                                
        outs = AttrDict()
        outs.pred = pred.mean(dim=0)  #(nb,nout)        
        outs.logits = logits_sample       
        return outs         
