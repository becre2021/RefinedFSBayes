#!/bin/bash


## last stochastic-layer 
#paramtrace=1
# CUDA_VISIBLE_DEVICES=0 python3 train_v4_bmafbnn_adv_hidden.py  --configs ./exp_cifar10_cifar100 \
#                                                                 --modelname bmafbresnet18 \
                                                                # --reglambda 1e-2  \
                                                                # --nepochratio .7 \
                                                                # --priorstdout 1e1 \
                                                                # --mixalpha   -4  \
                                                                # --paramtrace $paramtrace \
                                                                # --seeds 1111 \
                                                                # --v 1  ;   

## fully stochastic layer 
paramtrace=1
CUDA_VISIBLE_DEVICES=0 python3 train_v4_bmafbnn_adv_hidden.py  --configs ./exp_cifar10_cifar100 \
                                                                --modelname bmafbresnet18st \
                                                                --reglambda 1e-2  \
                                                                --nepochratio .7 \
                                                                --priorstdout 1e1 \
                                                                --mixalpha   -4  \
                                                                --paramtrace $paramtrace \
                                                                --seeds 1111 \
                                                                --v 1  ;            





