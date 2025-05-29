## Effective Function-Space Regularization for Bayesian Neural Networks via Learning Trajectories

This is an implemenation for "Effective Function-Space Regularization for Bayesian Neural Networks via Learning Trajectories"



## Description for implementation


* models_deepnet.py     : classification algorithms  
* models_vit.py         : transfer learning algorithms for VIT 
* models_regression.py  : regression algorithms 
* train_v4_bmafbnn_adv_hidden.py  : train R-FVI with informative priors for classification

* example_twomoon_main.ipynb      : describe the construction of the proposed distribution on Two-moon sets 
* example_ablation_proposed.ipynb : investigate the sensitiviy across varying trajectory on Cifar-10 with ResNet 18 



## Example to run code 
    # BNNs
    python3 train_v4_bmafbnn_adv_hidden.py  --configs ./exp_cifar10_cifar100 \
                                            --modelname bmafbresnet18st \
                                            --nfsample 10 \
                                            --reglambda 1e-2  \
                                            --nepochratio .7 \
                                            --temppeak   1.  \
                                            --mixalpha   0.1  \
                                            --paramtrace 1 \
                                            --seeds 1111 \
                                            --v 1  ;


    # last-layer BNNs (last-layer is only stochastic)
    python3 train_v4_bmafbnn_adv_hidden.py  --configs ./exp_cifar10_cifar100 \
                                            --modelname bmafbresnet18 \
                                            --reglambda 1e-2 \
                                            --nepochratio .7 \
                                            --priorstdout 1e1 \
                                            --mixalpha   .1  \
                                            --paramtrace 1 \
                                            --seeds 1111 \
                                            --v 1  ;
