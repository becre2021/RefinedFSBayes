# #!/bin/bash


# ## last stochastic-layer 
# # CUDA_VISIBLE_DEVICES=0 python3 train_bmafbnn.py  --configs ./configs/exp_cifar100_tinyimagenet \
# #                                                                 --modelname bmafbresnet50 \
#                                                                 # --reglambda 1e-4  \
#                                                                 # --nepochratio .75 \
#                                                                 # --mixalpha   .1  \
#                                                                 # --seeds 1111 \
#                                                                 # --v 1  ;


# ## fully stochastic layer 
# CUDA_VISIBLE_DEVICES=0 python3 train_bmafbnn_st.py  --configs ./configs/exp_cifar100_tinyimagenet \
#                                                                 --modelname bmafbresnet50st \
#                                                                 --reglambda 1e-4  \
#                                                                 --nepochratio .75 \
#                                                                 --mixalpha   .1  \
#                                                                 --seeds 1111 \
#                                                                 --v 1  ;
