#!/bin/bash






## last stochastic-layer 
#paramtrace=4
# CUDA_VISIBLE_DEVICES=0 python3 train_v4_bmafbnn_adv_hidden.py  --configs ./exp_cifar100_tinyimagenet \
#                                                                 --modelname bmafbresnet50 \
#                                                                 --nexp  1 \
#                                                                 --scalelogstd -6 \
#                                                                 --nfsample 10 \
#                                                                 --nctset 4 \
#                                                                 --reglambda 1e-4  \
#                                                                 --nepochratio .7 \
#                                                                 --priorstdout 1e1 \
#                                                                 --mixalpha   .1  \
#                                                                 --paramtrace 1 \
#                                                                 --seeds 1111 \
#                                                                 --v 1  ;

## fully stochastic layer 

paramtrace=1
CUDA_VISIBLE_DEVICES=1 python3 train_v4_bmafbnn_adv_hidden.py  --configs ./exp_cifar100_tinyimagenet \
                                                                --modelname bmafbresnet50st \
                                                                --nexp  1 \
                                                                --scalelogstd -6 \
                                                                --nfsample 10 \
                                                                --nctset 4 \
                                                                --reglambda 1e-4  \
                                                                --nepochratio .7 \
                                                                --priorstdout 1e1 \
                                                                --mixalpha   -4  \
                                                                --paramtrace $paramtrace \
                                                                --seeds 1111 \
                                                                --v 1  &            

CUDA_VISIBLE_DEVICES=1 python3 train_v4_bmafbnn_adv_hidden.py  --configs ./exp_cifar100_tinyimagenet \
                                                                --modelname bmafbresnet50st \
                                                                --nexp  1 \
                                                                --scalelogstd -6 \
                                                                --nfsample 10 \
                                                                --nctset 4 \
                                                                --reglambda 1e-4  \
                                                                --nepochratio .75 \
                                                                --priorstdout 1e1 \
                                                                --mixalpha   -4  \
                                                                --paramtrace $paramtrace \
                                                                --seeds 1111 \
                                                                --v 1  &            

CUDA_VISIBLE_DEVICES=1 python3 train_v4_bmafbnn_adv_hidden.py  --configs ./exp_cifar100_tinyimagenet \
                                                                --modelname bmafbresnet50st \
                                                                --nexp  1 \
                                                                --scalelogstd -6 \
                                                                --nfsample 10 \
                                                                --nctset 4 \
                                                                --reglambda 1e-4  \
                                                                --nepochratio .8 \
                                                                --priorstdout 1e1 \
                                                                --mixalpha   -4  \
                                                                --paramtrace $paramtrace \
                                                                --seeds 1111 \
                                                                --v 1  ;     