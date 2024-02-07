#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python3 train_v3_bmafbnn_adv_hidden.py  --configs ./exp_cifar10_cifar100 \
                                                                --modelname bmafbresnet18v10 \
                                                                --nexp  1 \
                                                                --scalelogstd -6 \
                                                                --nfsample 10 \
                                                                --nctset 4 \
                                                                --reglambda 1e-3  \
                                                                --nepochratio .8 \
                                                                --losstype elbo \
                                                                --priorstdout 1e1 \
                                                                --topk  3   \
                                                                --temppeak   1.  \
                                                                --mixalpha   .1  \
                                                                --paramtrace 9 \
                                                                --seeds 1111 \
                                                                --v 1  ;