#!/bin/bash


## last stochastic-layer 
#paramtrace=1
# CUDA_VISIBLE_DEVICES=0 python3 train_v4_bmafbnn_adv_hidden.py  --configs ./exp_cifar100_tinyimagenet \
#                                                                 --modelname bmafbresnet50 \
                                                                # --reglambda 1e-4  \
                                                                # --nepochratio .75 \
                                                                # --mixalpha   -4  \
                                                                # --paramtrace $paramtrace \
                                                                # --seeds 1111 \
                                                                # --v 1  ;


## fully stochastic layer 
paramtrace=1
CUDA_VISIBLE_DEVICES=0 python3 train_v4_bmafbnn_adv_hidden.py  --configs ./exp_cifar100_tinyimagenet \
                                                                --modelname bmafbresnet50st \
                                                                --reglambda 1e-4  \
                                                                --nepochratio .75 \
                                                                --mixalpha   -4  \
                                                                --paramtrace $paramtrace \
                                                                --seeds 1111 \
                                                                --v 1  ;