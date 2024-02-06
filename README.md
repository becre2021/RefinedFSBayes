## Improving Function-Space Bayesian Neural networks with Informative Priors

This is implemenation for Improving Function-Space Bayesian Neural networks with Informative Priors.



## Description for implementation


* models_deepnet_v3.py     : classification algorithms  
* models_vit.py            : transfer learning algorithms for VIT 
* models_regression_v3.py  : regression algorithms 

* train_v3_bmafbnn_adv_hidden.py  : train R-FVI with informative priors for classifation
* train_v3_bmafbnn_adv_hidden.py  : train R-FVI with informative priors for classifation



## Example to run code 

    python3 train_v3_bmafbnn_adv_hidden.py  --configs ./exp_cifar10_cifar100 \
                                            --modelname bmafbresnet18 \
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
                                            --mixalpha   .05  \
                                            --paramtrace 9 \
                                            --seeds 1111 \
                                            --v 1  ;

