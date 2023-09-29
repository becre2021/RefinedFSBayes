

## basic setting
nexp=1
nchosenexp=3
nctset=4
nctset2=8


##-------------------------------------------------
## mvfi
##-------------------------------------------------

DEVICES=0

#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_mfvibnn.py   --modelname bresnet20   --configs ./exp_cifar10  --scalelogstd -4  --nexp $nexp  --reglambda  1e-1 --nfsample 10 ;
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_mfvibnn.py   --modelname bresnet20   --configs ./exp_cifar10  --scalelogstd -6  --nexp $nexp  --reglambda  1e-1 --nfsample 10 ;
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_mfvibnn.py   --modelname bresnet20   --configs ./exp_cifar10  --scalelogstd -6  --nexp $nexp  --reglambda  1e-4 --nfsample 10 ;
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_mfvibnn.py   --modelname bresnet20   --configs ./exp_cifar10  --scalelogstd -6  --nexp $nexp  --reglambda  1e-4 --nfsample 10 ;

#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_mfvibnn.py   --modelname bresnet20   --configs ./exp_cifar10_cifar100   --scalelogstd -8  --nexp $nexp  --reglambda  1e-4 --nfsample 10 ;
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_mfvibnn.py   --modelname bresnet20   --configs ./exp_cifar10_cifar100   --scalelogstd -8  --nexp $nexp  --reglambda  1e-6 --nfsample 10 ;



##-------------------------------------------------
## MFVI FAILED
##-------------------------------------------------
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_mfvibnn.py   --modelname bresnet20   --configs ./exp_cifar10_cifar100   --scalelogstd -4  --nexp $nexp  --reglambda  1e-6 --nfsample 1 ;





##-------------------------------------------------
## proposed
##-------------------------------------------------


#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareplusclsnn.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100   --nexp 1 --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 20 --targetacc 0.5 --seeds 1111;
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100   --nexp 1 --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 20 --targetacc 0.5 --seeds 1111;



# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat2.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100   --nexp 1 --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --seeds 1111;
#modelnamefbresnet20/scalelogstd-6/nepoch250/nbatch128/lr0.1/Tmax250/nctset4/reglambda0.001/nfsample1/nexp3/v5        (0.91, 0.0)     (0.314, 0.0)    (0.012, 0.0)    (0.902, 0.0)    (1.493, 0.0)    (2.128, 0.0)


##-------------------------------------------------
## proposed validation for temporary with high prior from .5,5 -->1.,10. because classfication works as expected, but aucroc for calibrated temperature is not good
## based on, experiments fmnist on kmnist, high variance, i.e around entrophty 2 on ood seems works in regardless of the calibrated temperature.
## Therefore, I impose high variance for after training inductive bias,
##-------------------------------------------------
nexp=1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat2.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100   --nexp 1 --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin 1. --priorstdout 10. &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat2.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_monochrome    --nexp 1 --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin 1. --priorstdout 10.;






##-------------------------------------------------
## if above experiments works as expected, runs all experiments to get final results., not run
##-------------------------------------------------
nchosenexp=1
seeds=1111 ##if results are good, seeds-->1112,1113 run

## failed, not obatained good results
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat2.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --seeds $seeds&
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat2.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 20 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --seeds $seeds;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat2.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_monochrome  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --seeds $seeds&
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat2.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_monochrome  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 20 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --seeds $seeds;


##----------------------
## failed
##----------------------
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn.py     --modelname fbresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda 0.001 --scalelogstd -6 --nctset 4 --nfsample 10 --seeds $seeds &   ## prior on xc only
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn.py     --modelname fb2resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda 0.001 --scalelogstd -6 --nctset 4 --nfsample 10 --seeds $seeds;   ## prior on xc only
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn.py     --modelname fbresnet20   --configs ./exp_cifar10_monochrome  --nexp $nchosenexp --reglambda 0.001 --scalelogstd -6 --nctset 4 --nfsample 10 --seeds $seeds &   ## prior on x and xc together
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn.py     --modelname fb2resnet20  --configs ./exp_cifar10_monochrome  --nexp $nchosenexp --reglambda 0.001 --scalelogstd -6 --nctset 4 --nfsample 10 --seeds $seeds;   ## prior on xc only


#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat2.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --seeds $seeds&
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat2.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --seeds $seeds --v 19 ;
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn.py     --modelname fbresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda 0.001 --scalelogstd -6 --nctset 4 --nfsample 1 --seeds $seeds --v 17;   ## prior on xc only

#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn.py     --modelname fbresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda 1e-4 --scalelogstd -6 --nctset 4 --nfsample 1 --seeds $seeds &   ## prior on xc only
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn.py     --modelname fbresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda 1e-4 --scalelogstd -8 --nctset 4 --nfsample 1 --seeds $seeds ;   ## prior on xc only
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn.py     --modelname fbresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda 1e-4 --scalelogstd -6 --nctset 4 --nfsample 1 --seeds $seeds ;   ## prior on xc only
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn.py     --modelname fbresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda 0.0 --scalelogstd -6 --nctset 4 --nfsample 1 --seeds $seeds ;   ## prior on xc only



#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat2.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 20 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --seeds $seeds --v 20 ;




#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_mfvibnn.py   --modelname bresnet20   --configs ./exp_cifar10_cifar100   --scalelogstd -4  --nexp $nexp  --reglambda  1e-4 --nfsample 10 --seeds 1111 --v 17; seems failed
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_mfvibnn.py   --modelname bresnet20   --configs ./exp_cifar10_cifar100   --scalelogstd -6  --nexp $nexp  --reglambda  1e-3 --nfsample 10 --seeds 1111 --v 17;
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_mfvibnn.py   --modelname bresnet20   --configs ./exp_cifar10_cifar100   --scalelogstd -6  --nexp $nexp  --reglambda  0 --nfsample 10 --seeds 1111 --v 17;
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_mfvibnn.py   --modelname bresnet20   --configs ./exp_cifar10_cifar100   --scalelogstd -2  --nexp $nexp  --reglambda  0 --nfsample 10 --seeds 1111 --v 17;


#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_mfvibnn.py   --modelname bresnet20   --configs ./exp_cifar10_cifar100   --scalelogstd -6  --nexp $nexp  --reglambda  1e-4 --nfsample 10 --seeds 1111 --v 17;




nchosenexp=1
seeds=1111

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat2.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --seeds $seeds --v 20  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat2.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_monochrome  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --seeds $seeds --v 20 &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn.py     --modelname fb2resnet20  --configs ./exp_cifar10_monochrome  --nexp $nchosenexp --reglambda 0.001 --scalelogstd -6 --nctset 4 --nfsample 10  --seeds $seeds --v 20 ;   ## prior on xc only



# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat2.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --seeds 1112 --v 20  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat2.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --seeds 1113 --v 20  ;

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat2.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 20 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --seeds 1111 --v 20  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat2.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 20 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --seeds 1112 --v 20  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat2.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 20 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --seeds 1113 --v 20  ;


#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat2.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.91 --priorstdin .1 --priorstdout 10. --seeds $seeds --v 21  &
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat2.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_monochrome  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.91 --priorstdin .1 --priorstdout 10. --seeds $seeds --v 20 &



#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat3.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.91 --priorstdin .1 --priorstdout 10. --seeds $seeds --v 22  ;
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat3.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.40 --priorstdin .1 --priorstdout 10. --seeds $seeds --v 22  ;




# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat3.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-1 --scalelogstd -4 --nctset 4 --nfsample 5 --targetacc 0.9 --priorstdin 1 --priorstdout 10. --seeds $seeds --v 22  ;


# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat3.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 5 --targetacc 0.85 --priorstdin .1 --priorstdout 10. --seeds $seeds --v 23  ;


# seeds=1111
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat3.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100_v2  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin 1. --priorstdout 5. --seeds $seeds --v 24  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat3.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_monochrome_v2  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin 1. --priorstdout 5. --seeds $seeds --v 24 ;


## 10
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat3.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 10. --seeds 1111 --v 23  ;
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat3.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 10. --seeds 1112 --v 23  ;
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat3.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 10. --seeds 1113 --v 23  ;

## 5
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat3.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 5 --targetacc 0.85 --priorstdin .1 --priorstdout 10. --seeds $seeds --v 23  ;
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat3.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 5 --targetacc 0.85 --priorstdin .1 --priorstdout 10. --seeds 1112 --v 23  ;


#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat3.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 5 --targetacc 0.85 --priorstdin .1 --priorstdout 20. --seeds 1113 --v 23  ;





##--------------------------------------------------------------------
## 6
## after submission, to make complete paper
##--------------------------------------------------------------------

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat4.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.8 --priorstdin .1 --priorstdout 10. --seeds 1111 --v 25  ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat4.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 10. --seeds 1111 --v 25  ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat4.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 10. --seeds 1111 --v 25  ;


# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat4.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.8 --priorstdin .1 --priorstdout 10. --seeds 1111 --v 25  ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat4.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 10. --seeds 1111 --v 25  ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat4.py   --modelname fbrpawareclscatresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 10. --seeds 1111 --v 25  ;




# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat4.py   --modelname fbrpawareclscat2resnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 5.  --seeds 1111 --v 26  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat4.py   --modelname fbrpawareclscat2resnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 5.  --seeds 1111 --v 26  ;

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat4.py   --modelname fbrpawareclscat2resnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 10. --seeds 1111 --v 26  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat4.py   --modelname fbrpawareclscat2resnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 10. --seeds 1111 --v 26  ;

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat4.py   --modelname fbrpawareclscat2resnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 20. --seeds 1111 --v 26  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat4.py   --modelname fbrpawareclscat2resnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 20. --seeds 1111 --v 26  ;


#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat4.py   --modelname fbrpawareclscat2resnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 1. --seeds 1111 --v 27  &
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat4.py   --modelname fbrpawareclscat2resnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 1. --seeds 1111 --v 28  ;
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat4.py   --modelname fbrpawareclscat2resnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.40 --priorstdin .1 --priorstdout 1. --seeds 1111 --v 29  ;



# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat4.py   --modelname fbrpawareclscat2resnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 1. --seeds 1111 --v 30  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat4.py   --modelname fbrpawareclscat3resnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 1. --seeds 1111 --v 30  ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat4.py   --modelname fbrpawareclscat2resnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 1. --seeds 1111 --v 30  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat4.py   --modelname fbrpawareclscat3resnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 1. --seeds 1111 --v 30  ;



# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat4.py   --modelname fbrpawareclscat2resnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 5. --seeds 1111 --v 31  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat4.py   --modelname fbrpawareclscat2resnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --seeds 1111 --v 31  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat4.py   --modelname fbrpawareclscat3resnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --seeds 1111 --v 31  ;



# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat4.py   --modelname fbrpawareclscat1resnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 1. --seeds 1111 --v 32  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpawareclscat4.py   --modelname fbrpawareclscat1resnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --seeds 1111 --v 32  ;





##-------------
## 05/23/22 at midnight, latent posteior
##-------------

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --updateperiod 1 --seeds 1111 --v 33  ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --updateperiod 1 --seeds 1111 --v 33  ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --updateperiod 3 --seeds 1111 --v 33  ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 5. --updateperiod 3 --seeds 1111 --v 33  ;

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 10. --updateperiod 1 --seeds 1111 --v 34  ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 10. --updateperiod 3 --seeds 1111 --v 34  ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 5. --updateperiod 1 --seeds 1111 --v 34  ;


##-------------
## 05/23/22 at lunch, stochastic label fails to train
##-------------
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 5. --updateperiod 1 --seeds 1111 --v 35  ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 5. --updateperiod 3 --seeds 1111 --v 35  ;


#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.8 --priorstdin .1 --priorstdout 5. --updateperiod 3 --seeds 1111 --v 36  ;


## at 5:00 pm, before dinner.
## when eval, use high temperature, for extreme sampling
## when train, does not consider $z$ for sampling function.
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 5. --updateperiod 1 --seeds 1111 --v 37  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 5. --updateperiod 3 --seeds 1111 --v 37  ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --updateperiod 3 --seeds 1111 --v 37  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 20 --targetacc 0.9 --priorstdin .1 --priorstdout 5. --updateperiod 3 --seeds 1111 --v 37   ;


## at 1:00 am, midnight
## reverse kl still is not trained, and numerical issues occur.
# #CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 5. --updateperiod 3 --seeds 1111 --v 38  ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 10. --updateperiod 3 --seeds 1111 --v 40  ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --updateperiod 3 --seeds 1111 --v 40  ;



#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.40 --priorstdin .1 --priorstdout 5. --updateperiod 3 --seeds 1111 --v 41  ;



# #CUDA_VISIBLE_DEVICES=$DEVICES python3 train_mfvibnn.py   --modelname b2resnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nfsample 10  --seeds 1111 --v 42  ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_mfvibnn.py   --modelname b3resnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --scalelogstd -4 --nfsample 10  --seeds 1111 --v 42  ;


# ## at 3:30 am 
# ## edit regression loss and label, and check assumption
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 5. --updateperiod 1 --seeds 1111 --v 42  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 5. --updateperiod 3 --seeds 1111 --v 42  ;

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 5. --updateperiod 1 --seeds 1111 --v 42  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 5. --updateperiod 3 --seeds 1111 --v 42  ;

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e0 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 5. --updateperiod 1 --seeds 1111 --v 42  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e0 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 5. --updateperiod 3 --seeds 1111 --v 42  ;

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e0 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 5. --updateperiod 1 --seeds 1111 --v 42  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e0 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 5. --updateperiod 3 --seeds 1111 --v 42  ;


## at 3:30 pm  : edit regression loss and label, and check assumption .5
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 5. --updateperiod 1 --seeds 1111 --v 43  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 10. --updateperiod 1 --seeds 1111 --v 43  ;

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 20 --targetacc 0.90 --priorstdin .1 --priorstdout 5. --updateperiod 1 --seeds 1111 --v 43  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 20 --targetacc 0.90 --priorstdin .1 --priorstdout 10. --updateperiod 1 --seeds 1111 --v 43  ;


## at 11:00 pm  : denormalize mh, and set qunatile as .9, update static parms every 10 epochs instead of 5 epochs
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 5. --updateperiod 1 --seeds 1111 --v 44  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 10. --updateperiod 1 --seeds 1111 --v 44  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 20. --updateperiod 1 --seeds 1111 --v 44  ;


# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 5. --updateperiod 1 --seeds 1112 --v 44  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 10. --updateperiod 1 --seeds 1112 --v 44  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 20. --updateperiod 1 --seeds 1112 --v 44  ;

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 5. --updateperiod 1 --seeds 1113 --v 44  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 10. --updateperiod 1 --seeds 1113 --v 44  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 20. --updateperiod 1 --seeds 1113 --v 44  ;


# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 20 --targetacc 0.90 --priorstdin .1 --priorstdout 5. --updateperiod 1 --seeds 1111 --v 44  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 20 --targetacc 0.90 --priorstdin .1 --priorstdout 10. --updateperiod 1 --seeds 1111 --v 44  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 20 --targetacc 0.90 --priorstdin .1 --priorstdout 20. --updateperiod 1 --seeds 1111 --v 44  ;


# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 20 --targetacc 0.90 --priorstdin .1 --priorstdout 5. --updateperiod 1 --seeds 1112 --v 44  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 20 --targetacc 0.90 --priorstdin .1 --priorstdout 10. --updateperiod 1 --seeds 1112 --v 44  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 20 --targetacc 0.90 --priorstdin .1 --priorstdout 20. --updateperiod 1 --seeds 1112 --v 44  ;


# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 20 --targetacc 0.90 --priorstdin .1 --priorstdout 5. --updateperiod 1 --seeds 1113 --v 44  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 20 --targetacc 0.90 --priorstdin .1 --priorstdout 10. --updateperiod 1 --seeds 1113 --v 44  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrplatawarenn.py   --modelname fbrplatawareresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 20 --targetacc 0.90 --priorstdin .1 --priorstdout 20. --updateperiod 1 --seeds 1113 --v 44  ;



# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-2 --scalelogstd -4 --nctset 4 --nfsample 20 --targetacc 0.4 --priorstdin .1 --priorstdout 10. --updateperiod 1 --seeds 1111 --v 45  ;



# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --updateperiod 1 --seeds 1111 --v 45  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 5. --updateperiod 1 --seeds 1111 --v 45  ;

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --updateperiod 1 --seeds 1111 --v 45  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 5. --updateperiod 1 --seeds 1111 --v 45  ;


#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 10. --updateperiod 1 --seeds 1111 --v 46 ;


# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --updateperiod 1 --seeds 1111 --v 48 ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --updateperiod 1 --seeds 1111 --v 48 ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --updateperiod 1 --seeds 1111    --v 48 ;

# # check for classication loss
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --updateperiod 1 --seeds 1111 --v 49 & # 1e2, mseloss for classfication
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 5. --updateperiod 1 --seeds 1111 --v 49  ; # 1e2, mseloss for classfication



#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilat2resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --updateperiod 1 --seeds 1111 --v 51 ;


#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --updateperiod 1 --seeds 1111 --v 52 ;

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 5. --updateperiod 1 --seeds 1111 --v 53 ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --updateperiod 1 --seeds 1111 --v 53 ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 20. --updateperiod 1 --seeds 1111 --v 53 ;


# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilat2resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --updateperiod 1 --seeds 1111 --v 53 ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilat2resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 20. --updateperiod 1 --seeds 1111 --v 53 ;


#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --nmhlayer 2 --updateperiod 1 --seeds 1111 --v 54 ;
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --nmhlayer 1 --updateperiod 1 --seeds 1111 --v 54 ;


#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.8 --priorstdin .1 --priorstdout 10. --nmhlayer 1 --updateperiod 1 --seeds 1111 --v 54 ;

#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --nmhlayer 1 --updateperiod 1 --seeds 1111 --v 55 ; 1e1,1e1
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10. --nmhlayer 1 --updateperiod 1 --seeds 1111 --v 56 ;  1e1,1e2

#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.4 --priorstdin .1 --priorstdout 10. --betahyp .5 --updateperiod 1 --seeds 1111 --v 58 ; #1e1,1e1

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10.  --updateperiod 1 --betahyp .1 --seeds 1111 --v 58 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10.  --updateperiod 1 --betahyp 1. --seeds 1111 --v 58 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10.  --updateperiod 1 --betahyp 5. --seeds 1111 --v 58 ; #1e1,1e1



##-----------
## v59
## regularization on uniform prior using context sets 
##-----------

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10.  --updateperiod 1 --betahyp .1 --seeds 1111 --v 59 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10.  --updateperiod 1 --betahyp 1. --seeds 1111 --v 59 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10.  --updateperiod 1 --betahyp 5. --seeds 1111 --v 59 ; #1e1,1e1

#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.95 --priorstdin .1 --priorstdout 10.  --updateperiod 1 --betahyp 1. --seeds 1111 --v 59 ; #1e1,1e1




# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10.  --updateperiod 1 --betahyp .1 --seeds 1111 --v 60 & #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-3 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10.  --updateperiod 1 --betahyp .1 --seeds 1111 --v 60 ; #1e1,1e1

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.95 --priorstdin .1 --priorstdout 10.  --updateperiod 1 --betahyp .1 --seeds 1111 --v 60 & #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-3 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.95 --priorstdin .1 --priorstdout 10.  --updateperiod 1 --betahyp .1 --seeds 1111 --v 60 ; #1e1,1e1

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-2 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 10.  --updateperiod 1 --betahyp .1 --seeds 1111 --v 60 & #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-3 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 10.  --updateperiod 1 --betahyp .1 --seeds 1111 --v 60 ; #1e1,1e1





##-----------
## v61
## no regularization on uniform prior 
##-----------

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e0 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10.  --updateperiod 1 --betahyp .1 --seeds 1111 --v 61 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10.  --updateperiod 1 --betahyp .1 --seeds 1111 --v 61 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e0 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10.  --updateperiod 1 --betahyp 1. --seeds 1111 --v 61 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10.  --updateperiod 1 --betahyp 1. --seeds 1111 --v 61 ; #1e1,1e1

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e0 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 10.  --updateperiod 1 --betahyp .1 --seeds 1111 --v 61 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 10.  --updateperiod 1 --betahyp .1 --seeds 1111 --v 61 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e0 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 10.  --updateperiod 1 --betahyp 1. --seeds 1111 --v 61 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 10.  --updateperiod 1 --betahyp 1. --seeds 1111 --v 61 ; #1e1,1e1


# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e0 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.95 --priorstdin .1 --priorstdout 10.  --updateperiod 1 --betahyp .1 --seeds 1111 --v 61 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.95 --priorstdin .1 --priorstdout 10.  --updateperiod 1 --betahyp .1 --seeds 1111 --v 61 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e0 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.95 --priorstdin .1 --priorstdout 10.  --updateperiod 1 --betahyp 1. --seeds 1111 --v 61 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-1 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.95 --priorstdin .1 --priorstdout 10.  --updateperiod 1 --betahyp 1. --seeds 1111 --v 61 ; #1e1,1e1



# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-1 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 62 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e0 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 62 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 62 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e0 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 62 ; #1e1,1e1

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-1 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 62 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e0 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 62 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 62 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e0 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 62 ; #1e1,1e1

#savepath: ./results/cifar10_cifar100/trainsetcifar10_contextsetcifar100_modelnamefbrpcontilatresnet20_scalelogstd-4_priorstdin0.1_priorstdout10.0_betahyp1.0_nepoch250_nbatch128_lr0.1_Tmax250_nctset4_reglambda0.001_regcls0.1_userelregFalse_nfsample10_targetacc0.85_updateperiod1_nexp1_v62/seeds1111_gentime06030113.pth



## after fitting check linear inductive bais, apply the bayesian inference 
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-1 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.4 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 63 ; #1e1,1e1

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-1 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 63 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 5e-1 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 63 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e0 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 63 ; #1e1,1e1

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-1 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 63 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 5e-1 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 63 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e0 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 63 ; #1e1,1e1


#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-1 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.4 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 64 ; #1e1,1e1

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-1 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 64 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 5e-1 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 64 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e0  --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 64 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-1 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 20 --targetacc 0.9 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 64 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-1 --betahyp .5 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.9 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 64 ; #1e1,1e1

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-1 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 64 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 5e-1 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 64 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e0  --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 64 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-1 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 20 --targetacc 0.85 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 64 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-1 --betahyp .5 --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 64 ; #1e1,1e1



# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-1 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 65 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-1 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 65 ; #1e1,1e1

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e0 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 65 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e0 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.85 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 65 ; #1e1,1e1




# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-2 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 66 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-2 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 66 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-2 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 66 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 66 ; #1e1,1e1

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-2 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 20 --targetacc 0.90 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 66 ; #1e1,1e1



# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-2 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 67 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-1 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 67 ; #1e1,1e1


# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-2 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 68 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-2 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 68 ; #1e1,1e1

#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-2 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 69 ; #1e1,1e1


# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 70 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-2 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 70 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-3 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 70 ; #1e1,1e1

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-1 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 70 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-2 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 70 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --regcls 1e-3 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 70 ; #1e1,1e1


# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 71 ; #1e1,1e1



#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontisimp.py   --modelname fbrpcontsimpresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.35 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 72 ; #1e1,1e1

## after revising functioanl variance term 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 72 ; #1e1,1e1
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-2 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin .1 --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 72 ; #1e1,1e1



# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin 1. --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 72 ; #1e1,1e1


#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcontilatnn.py   --modelname fbrpcontilatresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --regcls 1e-1 --betahyp 1. --scalelogstd -4 --nctset 4 --nfsample 10 --targetacc 0.90 --priorstdin 1. --priorstdout 10.  --updateperiod 1  --seeds 1111 --v 73 ; #use jacobian directly 









# ## revisit the regularizer of the functional bnnn 

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn2.py   --modelname fb3resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 1 --priorstdout 5.  --seeds 1111 --v 74 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn2.py   --modelname fb3resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-4  --scalelogstd -4 --nctset 4 --nfsample 1 --priorstdout 5.  --seeds 1111 --v 74 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn3.py   --modelname fb3resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 1 --priorstdout 5.  --seeds 1111 --v 74 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn3.py   --modelname fb3resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-4  --scalelogstd -4 --nctset 4 --nfsample 1 --priorstdout 5.  --seeds 1111 --v 74 ; #use jacobian directly 

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn2.py   --modelname fb3resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 1 --priorstdout 10.  --seeds 1111 --v 74 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn2.py   --modelname fb3resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-4  --scalelogstd -4 --nctset 4 --nfsample 1 --priorstdout 10.  --seeds 1111 --v 74 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn3.py   --modelname fb3resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 1 --priorstdout 10.  --seeds 1111 --v 74 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn3.py   --modelname fb3resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-4  --scalelogstd -4 --nctset 4 --nfsample 1 --priorstdout 10.  --seeds 1111 --v 74 ; #use jacobian directly 


# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn2.py   --modelname fb3resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 1 --priorstdout 5.  --seeds 1111 --v 75 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn2.py   --modelname fb3resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-4  --scalelogstd -4 --nctset 4 --nfsample 1 --priorstdout 5.  --seeds 1111 --v 75 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn2.py   --modelname fb3resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 1 --priorstdout 10.  --seeds 1111 --v 75 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn2.py   --modelname fb3resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-4  --scalelogstd -4 --nctset 4 --nfsample 1 --priorstdout 10.  --seeds 1111 --v 75 ; #use jacobian directly 


# ## negative prior zeros logits not goods
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn3.py   --modelname fb3resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 1 --priorstdout 2.  --seeds 1111 --v 76 ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn2.py   --modelname fb3resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 1 --priorstdout 2.  --seeds 1111 --v 77 ;

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn2.py   --modelname fb3resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0.0  --scalelogstd -4 --nctset 4 --nfsample 10 --priorstdout 2.  --seeds 1111 --v 80 ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn2.py   --modelname fb3resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0.0  --scalelogstd -4 --nctset 4 --nfsample 20 --priorstdout 2.  --seeds 1111 --v 80 ;
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn2.py   --modelname fb3resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0.0  --scalelogstd -4 --nctset 4 --nfsample 50 --priorstdout 2.  --seeds 1111 --v 80 ;
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn2.py   --modelname fb3resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0.0  --scalelogstd -4 --nctset 4 --nfsample 200 --priorstdout 2.  --seeds 1111 --v 80 ;



## training without regularizer
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpnn2.py   --modelname fbrp2resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 10  --seeds 1111 --v 79 ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpnn2.py   --modelname fbrpresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4  --nfsample 10  --seeds 1111 --v 79 ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpnn2.py   --modelname fbrp2resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 20  --seeds 1111 --v 79 ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpnn2.py   --modelname fbrpresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4  --nfsample 20  --seeds 1111 --v 79 ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpnn2.py   --modelname fbrp2resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 50  --seeds 1111 --v 79 ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpnn2.py   --modelname fbrpresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4  --nfsample 50  --seeds 1111 --v 79 ;


#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpnn2.py   --modelname fbrp2resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 100  --seeds 1111 --v 79 ;
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpnn2.py   --modelname fbrpresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 100  --seeds 1111 --v 79 ;

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpnn2.py   --modelname fbrp2resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4 --nctset 4 --nfsample 200  --seeds 1111 --v 79 ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpnn2.py   --modelname fbrpresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4  --nctset 4 --nfsample 200  --seeds 1111 --v 79 ;


#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcdresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-1  --scalelogstd -4  --nfsample 20 --seeds 1111 --v 82 ; #use jacobian directly 
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcdresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2  --scalelogstd -4  --nfsample 20 --seeds 1111 --v 82 ; #use jacobian directly 
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcdresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2  --scalelogstd -4  --nfsample 20 --seeds 1111 --v 83 ; #use jacobian directly 

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcdresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-1  --scalelogstd -4  --nfsample 50 --seeds 1111 --v 82 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcdresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2  --scalelogstd -4  --nfsample 50 --seeds 1111 --v 82 ; #use jacobian directly 

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcdresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-1  --scalelogstd -4  --nfsample 100 --seeds 1111 --v 82 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcdresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2  --scalelogstd -4  --nfsample 100 --seeds 1111 --v 82 ; #use jacobian directly 




## use pretrained network as prior model, and train with predictive sample = 10
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn4.py   --modelname fb4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2  --scalelogstd -4  --nctset 4 --nfsample 10 --seeds 1111 --v 83 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn4.py   --modelname fb4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4  --nctset 4 --nfsample 10 --seeds 1111 --v 83 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn4.py   --modelname fb4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-4  --scalelogstd -4  --nctset 4 --nfsample 10 --seeds 1111 --v 83 ; #use jacobian directly 

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn4.py   --modelname fb4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2  --scalelogstd -4  --nctset 8 --nfsample 10 --seeds 1111 --v 83 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn4.py   --modelname fb4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4  --nctset 8 --nfsample 10 --seeds 1111 --v 83 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn4.py   --modelname fb4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-4  --scalelogstd -4  --nctset 8 --nfsample 10 --seeds 1111 --v 83 ; #use jacobian directly 

## use pretrained network as prior model, and train joint sup kl
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn4.py   --modelname fb4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-4  --scalelogstd -4  --nctset 4 --nfsample 10 --seeds 1111 --v 84 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn4.py   --modelname fb4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-5  --scalelogstd -4  --nctset 4 --nfsample 10 --seeds 1111 --v 84 ; #use jacobian directly 


## use pretrained network as prior model, with mixup contrained
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn4.py   --modelname fb4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-1  --scalelogstd -4  --nctset 4 --nfsample 10 --seeds 1111 --v 86 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn4.py   --modelname fb4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2  --scalelogstd -4  --nctset 4 --nfsample 10 --seeds 1111 --v 86 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnn4.py   --modelname fb4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3  --scalelogstd -4  --nctset 4 --nfsample 10 --seeds 1111 --v 86 ; #use jacobian directly 






## use functional prior controled by alpha
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcdresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-1  --scalelogstd -4  --nfsample 20 --seeds 1111 --v 85 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcdresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-1  --scalelogstd -4  --nfsample 50 --seeds 1111 --v 85 ; #use jacobian directly 

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcdresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2  --scalelogstd -4  --nfsample 20 --seeds 1111 --v 85 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcdresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2  --scalelogstd -4  --nfsample 50 --seeds 1111 --v 85 ; #use jacobian directly 



# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd2resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-1  --scalelogstd -4  --nfsample 20 --seeds 1111 --v 85 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd2resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-1  --scalelogstd -4  --nfsample 50 --seeds 1111 --v 85 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd2resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-1  --scalelogstd -4  --nfsample 100 --seeds 1111 --v 85 ; #use jacobian directly 

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd2resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2  --scalelogstd -4  --nfsample 20 --seeds 1111 --v 85 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd2resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2  --scalelogstd -4  --nfsample 50 --seeds 1111 --v 85 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd2resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2  --scalelogstd -4  --nfsample 100 --seeds 1111 --v 85 ; #use jacobian directly 


# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcdresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  5e-1  --scalelogstd -4  --nfsample 20 --seeds 1111 --v 85 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcdresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  5e-1  --scalelogstd -4  --nfsample 50 --seeds 1111 --v 85 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcdresnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  5e-1  --scalelogstd -4  --nfsample 100 --seeds 1111 --v 85 ; #use jacobian directly 

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd2resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  5e-1  --scalelogstd -4  --nfsample 20 --seeds 1111 --v 85 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd2resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  5e-1  --scalelogstd -4  --nfsample 50 --seeds 1111 --v 85 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd2resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  5e-1  --scalelogstd -4  --nfsample 100 --seeds 1111 --v 85 ; #use jacobian directly 


# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcdresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-1  --scalelogstd -4  --nfsample 100 --seeds 1111 --v 85 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcdresnet20   --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-1  --scalelogstd -4  --nfsample 500 --seeds 1111 --v 85 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd2resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-1  --scalelogstd -4  --nfsample 500 --seeds 1111 --v 85 ; #use jacobian directly 

#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd2resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-1  --scalelogstd -4  --nfsample 1000 --seeds 1111 --v 85 ; #use jacobian directly 



## check the effectiveness of the spectral normalization, and specifically how it affects feature collapase, and the following mh and density 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd3resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-1  --scalelogstd -4  --nfsample 100 --seeds 1111 --v 87 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd3resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-1  --scalelogstd -4  --nfsample 20 --seeds 1111 --v 87 ; #use jacobian directly 

#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd3resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nfsample 100 --seeds 1111 --v 88 ; #use jacobian directly 

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .1 --nfsample 20 --seeds 1111 --v 88 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .1 --nfsample 50 --seeds 1111 --v 88 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .1 --nfsample 100 --seeds 1111 --v 88 ; #use jacobian directly 

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .5 --nfsample 20 --seeds 1111 --v 88 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .5 --nfsample 50 --seeds 1111 --v 88 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .5 --nfsample 100 --seeds 1111 --v 88 ; #use jacobian directly 

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .9 --nfsample 20 --seeds 1111 --v 88 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .9 --nfsample 50 --seeds 1111 --v 88 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .9 --nfsample 100 --seeds 1111 --v 88 ; #use jacobian directly 


# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .9 --nfsample 100 --seeds 1112 --v 88 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .9 --nfsample 100 --seeds 1113 --v 88 ; #use jacobian directly 

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .9 --nfsample 50 --seeds 1112 --v 88 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .9 --nfsample 50 --seeds 1113 --v 88 ; #use jacobian directly 


## add distance prior on variational distribution as well  +  revising log marginal likelihood estimation
## fbrpcd4resnet20: 
## fbrpcd5resnet20: distance as well 

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd5resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .9 --nfsample 20 --seeds 1111 --v 89 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd5resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .9 --nfsample 50 --seeds 1111 --v 89 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd5resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .9 --nfsample 100 --seeds 1111 --v 89 ; #use jacobian directly 

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .9 --nfsample 20 --seeds 1111 --v 90 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .9 --nfsample 50 --seeds 1111 --v 90 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .9 --nfsample 100 --seeds 1111 --v 90 ; #use jacobian directly 

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd5resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .9 --nfsample 20 --seeds 1111 --v 90 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd5resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .9 --nfsample 50 --seeds 1111 --v 90 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd5resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .9 --nfsample 100 --seeds 1111 --v 90 ; #use jacobian directly 

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .9 --nfsample 200 --seeds 1111 --v 90 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd5resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .9 --nfsample 200 --seeds 1111 --v 90 ; #use jacobian directly 

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .9 --nfsample 5 --seeds 1111 --v 90 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd5resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .9 --nfsample 5 --seeds 1111 --v 90 ; #use jacobian directly 



## resnet20 via spectral normalization
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_base.py   --modelname res2net20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --seeds 1111 --v 91 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_base.py   --modelname res2net20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-3 --seeds 1111 --v 91 ; #use jacobian directly 

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_base.py   --modelname res2net20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --seeds 1112 --v 91 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_base.py   --modelname res2net20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  1e-2 --seeds 1113 --v 91 ; #use jacobian directly 


# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .9 --nfsample 1 --seeds 1111 --v 90 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd5resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .9 --nfsample 1 --seeds 1111 --v 90 ; #use jacobian directly 


## resnet20 via spectral normalization + mhprior
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_basemh.py   --modelname res3net20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda 1e-3  --seeds 1111 --nepochratio 0.1 --v 93 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_basemh.py   --modelname res3net20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda 1e-3  --seeds 1111 --nepochratio 0.5 --v 93 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_basemh.py   --modelname res3net20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda 1e-3  --seeds 1111 --nepochratio 0.9 --v 93 ; #use jacobian directly 

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_basemh.py   --modelname res3net20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda 1e-3  --seeds 1112 --nepochratio 0.5 --v 93 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_basemh.py   --modelname res3net20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda 1e-3  --seeds 1113 --nepochratio 0.5 --v 93 ; #use jacobian directly 

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_basemh.py   --modelname res3net20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda 1e-3  --seeds 1112 --nepochratio 0.9 --v 93 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_basemh.py   --modelname res3net20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda 1e-3  --seeds 1113 --nepochratio 0.9 --v 93 ; #use jacobian directly 





## revise the error of function variaince 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .9 --nfsample 10 --seeds 1111 --v 92 ; #use jacobian directly 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbrpcdnn.py   --modelname fbrpcd4resnet20  --configs ./exp_cifar10_cifar100  --nexp $nchosenexp --reglambda  0  --scalelogstd -4  --nepochratio .9 --nfsample 50 --seeds 1111 --v 92 ; #use jacobian directly 






## revisit the function prior with spectral regularization 07/06
DEVICES=0
nexp=1
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp2.py   --modelname fb2spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda 1. --priorstdout 5 --seeds 1111 --v 93 ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp2.py   --modelname fb2spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda 1. --priorstdout 10 --seeds 1111 --v 94 &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp2.py   --modelname fb2spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10 --seeds 1111 --v 94 ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp2.py   --modelname fb2spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .01 --priorstdout 10 --seeds 1111 --v 94 &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp2.py   --modelname fb2spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 10 --seeds 1111 --v 94 &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp2.py   --modelname fb2spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .0001 --priorstdout 10 --seeds 1111 --v 94 ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp2.py   --modelname fb2spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .00001 --priorstdout 10 --seeds 1111 --v 94 ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp2.py   --modelname fb2spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .000001 --priorstdout 10 --seeds 1111 --v 94 ;


# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp2.py   --modelname fb2spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10 --seeds 1111 --v 94 &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp2.py   --modelname fb2spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda .01 --priorstdout 10 --seeds 1111 --v 94 &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp2.py   --modelname fb2spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 10 --seeds 1111 --v 94 &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp2.py   --modelname fb2spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda .0001 --priorstdout 10 --seeds 1111 --v 94 ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp2.py   --modelname fb2spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda .00001 --priorstdout 10 --seeds 1111 --v 94 ;

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp2.py   --modelname fb2spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 10 --seeds 1112 --v 94 &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp2.py   --modelname fb2spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 10 --seeds 1113 --v 94 &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp2.py   --modelname fb2spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 10 --seeds 1112 --v 94 &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp2.py   --modelname fb2spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 10 --seeds 1113 --v 94 ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp2.py   --modelname fb2spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .000001 --priorstdout 10 --seeds 1112 --v 94 &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp2.py   --modelname fb2spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .000001 --priorstdout 10 --seeds 1113 --v 94 ;



# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp2.py   --modelname fb3spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .0001 --priorstdout 10 --seeds 1111 --v 94 &

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp2.py   --modelname fb3spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .0001 --priorstdout 5 --seeds 1111 --v 94 &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp2.py   --modelname fb3spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .001  --priorstdout 5 --seeds 1111 --v 94 &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp2.py   --modelname fb3spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .01   --priorstdout 5 --seeds 1111 --v 94 ;


# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp2.py   --modelname fb3spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .01   --priorstdout 1 --seeds 1111 --v 94 &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp2.py   --modelname fb3spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .001  --priorstdout 1 --seeds 1111 --v 94 ;
#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp2.py   --modelname fb3spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .0001 --priorstdout 1 --seeds 1111 --v 94 &


## use prior after pretrained model 80% == 200 epochs 
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp3.py   --modelname fb3spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 1 --seeds 1111 --v 95 &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp3.py   --modelname fb3spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .0001 --priorstdout 1 --seeds 1111 --v 95 ;

#CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp3.py   --modelname fb3spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 5 --seeds 1111 --v 95 ;


# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp3.py   --modelname fb3spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 5 --usepretrain --seeds 1111  --v 96 &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp3.py   --modelname fb3spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 5 --seeds 1111                --v 96 &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp3.py   --modelname fb3spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 10 --usepretrain --seeds 1111 --v 96 &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp3.py   --modelname fb3spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 10 --seeds 1111               --v 96 ;

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp3.py   --modelname fb3spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 20 --usepretrain --seeds 1111 --v 96 &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp3.py   --modelname fb3spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 20 --seeds 1111               --v 96 ;


## use prior after pretrained model 80% == 200 epochs and get params after 80% trainings
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp3.py   --modelname fb3spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 5 --usepretrain --seeds 1111  --v 97 &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp3.py   --modelname fb3spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 5 --seeds 1111                --v 97 ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp3.py   --modelname fb3spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 5 --usepretrain --seeds 1111  --v 98 &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp3.py   --modelname fb3spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 5 --seeds 1111                --v 98 ;

# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp3.py   --modelname fb3spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 5 --usepretrain --seeds 1111  --v 99 &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_fbnnsp3.py   --modelname fb3spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 5 --seeds 1111                --v 99 ;


# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp4.py   --modelname fb4spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 10 --nepochratio .5 --seeds 1111  --v 100 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp4.py   --modelname fb4spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 10 --nepochratio .8 --seeds 1111  --v 100 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp4.py   --modelname fb4spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 20 --nepochratio .5 --seeds 1111  --v 100 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp4.py   --modelname fb4spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 20 --nepochratio .8 --seeds 1111  --v 100 ;


# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp4.py   --modelname fb4spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 10 --nepochratio .8 --seeds 1111  --v 101 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp4.py   --modelname fb4spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 10 --nepochratio .8 --seeds 1111  --v 101 ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp4.py   --modelname fb4spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 10 --nepochratio .8 --seeds 1111  --v 102 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp4.py   --modelname fb4spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 20 --nepochratio .8 --seeds 1111  --v 102 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp4.py   --modelname fb4spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .01 --priorstdout 10 --nepochratio .8 --seeds 1111  --v 102 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp4.py   --modelname fb4spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .01 --priorstdout 20 --nepochratio .8 --seeds 1111  --v 102 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp4.py   --modelname fb4spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10 --nepochratio .8 --seeds 1111  --v 102 ;


# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp3.py   --modelname fb3spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10 --nepochratio .9 --seeds 1111  --v 103 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp3.py   --modelname fb3spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .01 --priorstdout 10 --nepochratio .9 --seeds 1111  --v 103 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp4.py   --modelname fb4spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10 --nepochratio .9 --seeds 1111  --v 103 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp4.py   --modelname fb4spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .01 --priorstdout 10 --nepochratio .9 --seeds 1111  --v 103 ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp3.py   --modelname fb3spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10 --nepochratio .95 --seeds 1111  --v 103 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp3.py   --modelname fb3spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .01 --priorstdout 10 --nepochratio .95 --seeds 1111  --v 103 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp4.py   --modelname fb4spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10 --nepochratio .95 --seeds 1111  --v 103 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp4.py   --modelname fb4spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .01 --priorstdout 10 --nepochratio .95 --seeds 1111  --v 103 ;



#CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp3.py   --modelname fb3spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10 --nepochratio .9 --seeds 1111  --v 104 ;
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp4.py   --modelname fb4spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10 --nepochratio .9 --seeds 1111  --v 104 ;
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp3.py   --modelname fb3spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .01 --priorstdout 10 --nepochratio .9 --seeds 1111  --v 104 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp4.py   --modelname fb4spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .01 --priorstdout 10 --nepochratio .9 --seeds 1111  --v 104 ;


# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .01 --priorstdout 10 --nepochratio .9 --seeds 1111  --v 105 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10  --nepochratio .9 --seeds 1111  --v 105 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .01 --priorstdout 10 --nepochratio .95 --seeds 1111  --v 105 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .0 --priorstdout 10 --nepochratio .95 --seeds 1111  --v 105 ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10  --nepochratio .95 --seeds 1111  --v 105 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .01 --priorstdout 10  --nepochratio .85 --seeds 1111  --v 105 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10  --nepochratio .85 --seeds 1111  --v 105 ;


# temperature by predicive entopy, obtained by mhd
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10  --nepochratio .85 --seeds 1111  --v 106 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .01 --priorstdout 10  --nepochratio .85 --seeds 1111  --v 106 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 10  --nepochratio .85 --seeds 1111  --v 106 ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10  --nepochratio .9 --seeds 1111  --v 106 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .01 --priorstdout 10  --nepochratio .9 --seeds 1111  --v 106 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 10  --nepochratio .9 --seeds 1111  --v 106 ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10  --nepochratio .95 --seeds 1111  --v 106 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .01 --priorstdout 10  --nepochratio .95 --seeds 1111  --v 106 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .001 --priorstdout 10  --nepochratio .95 --seeds 1111  --v 106 ;


# revise the temperature by predicive entopy, obtained by mhd
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10  --nepochratio .85 --seeds 1111  --v 107 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10  --nepochratio .9 --seeds 1111  --v 107 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10  --nepochratio .95 --seeds 1111  --v 107 ;

# revise the prior distribution, fix mean and modify std only
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10  --nepochratio .8 --seeds 1111  --v 108 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10  --nepochratio .85 --seeds 1111  --v 108 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10  --nepochratio .9 --seeds 1111  --v 108 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10  --nepochratio .95 --seeds 1111  --v 108 ;


# revise the prior distribution, zeros mean and modify std only
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10  --nepochratio .8 --seeds 1111  --v 109 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10  --nepochratio .85 --seeds 1111  --v 109 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10  --nepochratio .9 --seeds 1111  --v 109 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10  --nepochratio .95 --seeds 1111  --v 109 ;


# revise the prior distribution, zeros mean and modify std only,
# and topk 3 -> 2, tmperature 1e1 -> 1e2
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10  --nepochratio .8 --seeds 1111  --v 110 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10  --nepochratio .85 --seeds 1111  --v 110 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10  --nepochratio .9 --seeds 1111  --v 110 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 10  --nepochratio .95 --seeds 1111  --v 110 ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 20  --nepochratio .8 --seeds 1111  --v 110 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 20  --nepochratio .85 --seeds 1111  --v 110 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 20  --nepochratio .9 --seeds 1111  --v 110 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 20  --nepochratio .95 --seeds 1111  --v 110 ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 40  --nepochratio .8 --seeds 1111  --v 110 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 40  --nepochratio .85 --seeds 1111  --v 110 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 40  --nepochratio .9 --seeds 1111  --v 110 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 40  --nepochratio .95 --seeds 1111  --v 110 ;

# revise the prior distribution, zeros mean and modify std only,
# and topk 2 -> 2, tmperature 1e2 -> 1e3
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 40  --nepochratio .8 --useregweight --seeds 1111  --v 111 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1 --priorstdout 40  --nepochratio .95 --useregweight --seeds 1111  --v 111 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp5.py   --modelname fb5spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda 1e-8 --priorstdout 40  --nepochratio .01 --seeds 1111  --v 111 ;


# revise the prior distribution as dir distribution, use dir priors
# after fixing the numerical error of dirichlet regularization, fix 
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1    --nepochratio .8 --useregweight   --topk 3 --seeds 1111  --v 112 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1    --nepochratio .85 --useregweight  --topk 3  --seeds 1111  --v 112 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1    --nepochratio .90 --useregweight  --topk 3  --seeds 1111  --v 112 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1    --nepochratio .95 --useregweight  --topk 3  --seeds 1111  --v 112 ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1    --nepochratio .8 --useregweight   --topk 10  --seeds 1111  --v 112 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1    --nepochratio .85 --useregweight  --topk 10  --seeds 1111  --v 112 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1    --nepochratio .90 --useregweight  --topk 10  --seeds 1111  --v 112 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1    --nepochratio .95 --useregweight  --topk 10  --seeds 1111  --v 112 ;

# # check cold posterior effect
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda 1e-8  --nepochratio .01 --topk 10 --seeds 1111  --v 112 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda 1e-6  --nepochratio .01 --topk 10 --seeds 1111  --v 112 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda 1e-4  --nepochratio .01 --topk 10 --seeds 1111  --v 112 ;


# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1    --nepochratio .8 --useregweight   --topk 3 --seeds 1111  --v 113 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1    --nepochratio .8 --useregweight   --topk 10  --seeds 1111  --v 113 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .01    --nepochratio .8 --useregweight   --topk 3 --seeds 1111  --v 113 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .01    --nepochratio .8 --useregweight   --topk 10  --seeds 1111  --v 113 ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1    --nepochratio .85 --useregweight  --topk 3  --seeds 1111  --v 113 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1    --nepochratio .85 --useregweight  --topk 10  --seeds 1111  --v 113 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .01    --nepochratio .85 --useregweight  --topk 3  --seeds 1111  --v 113 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .01    --nepochratio .85 --useregweight  --topk 10  --seeds 1111  --v 113 ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1    --nepochratio .90 --useregweight  --topk 3  --seeds 1111  --v 113 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1    --nepochratio .90 --useregweight  --topk 10  --seeds 1111  --v 113 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .01    --nepochratio .90 --useregweight  --topk 3  --seeds 1111  --v 113 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .01    --nepochratio .90 --useregweight  --topk 10  --seeds 1111  --v 113 ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1    --nepochratio .95 --useregweight  --topk 3  --seeds 1111   --v 113 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1    --nepochratio .95 --useregweight  --topk 10  --seeds 1111  --v 113 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .01    --nepochratio .95 --useregweight  --topk 3  --seeds 1111   --v 113 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .01    --nepochratio .95 --useregweight  --topk 10  --seeds 1111  --v 113 ;


# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1    --nepochratio .8    --topk 3 --seeds 1111  --v 114 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1    --nepochratio .8    --topk 10  --seeds 1111  --v 114 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .01    --nepochratio .8   --topk 3 --seeds 1111  --v 114 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .01    --nepochratio .8   --topk 10  --seeds 1111  --v 114 ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1    --nepochratio .85   --topk 3  --seeds 1111  --v 114 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .1    --nepochratio .85   --topk 10  --seeds 1111  --v 114 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .01    --nepochratio .85  --topk 3  --seeds 1111  --v 114 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda .01    --nepochratio .85  --topk 10  --seeds 1111  --v 114 ;


# ## check functional diversity durining training.
# #CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 20 --nctset 4 --reglambda .0    --nepochratio .08    --topk 3 --seeds 1111  --v 115 ;
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6_parampath.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 20 --nctset 4 --reglambda .0  --nepochratio .75  --seeds 1111  --losstype elbo  --v 115 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp6_parampath.py   --modelname fb6spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 20 --nctset 4 --reglambda .0  --nepochratio .75  --seeds 1111  --losstype iwae  --v 115 ;




## after check functional diversity durining training, functional diversity prior
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb7spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1  --nepochratio .8  --seeds 1111  --losstype elbo  --v 116  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb7spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1  --nepochratio .85  --seeds 1111  --losstype elbo  --v 116 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb7spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1  --nepochratio .9  --seeds 1111  --losstype elbo  --v 116  ;
#CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb7spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1  --nepochratio .85  --seeds 1111  --losstype elbo  --v 116 ;

## re
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb7spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1  --nepochratio .8   --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 117  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb8spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1  --nepochratio .8  --losstype elbo --priorstdout 5 --topk 2 --seeds 1111 --v 117    &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb7spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1  --nepochratio .8   --losstype elbo --priorstdout 20 --topk 10 --seeds 1111 --v 117  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb8spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1  --nepochratio .8  --losstype elbo --priorstdout 20 --topk 2 --seeds 1111 --v 117    ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb7spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .01  --nepochratio .8   --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 117  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb8spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .01  --nepochratio .8  --losstype elbo --priorstdout 5 --topk 2 --seeds 1111 --v 117    &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb7spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .01  --nepochratio .8   --losstype elbo --priorstdout 20 --topk 10 --seeds 1111 --v 117  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb8spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .01  --nepochratio .8  --losstype elbo --priorstdout 20 --topk 2 --seeds 1111 --v 117    ;


# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb7spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1  --nepochratio .9   --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 117  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb8spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1  --nepochratio .9  --losstype elbo --priorstdout 5 --topk 2 --seeds 1111 --v 117    &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb7spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1  --nepochratio .9   --losstype elbo --priorstdout 20 --topk 10 --seeds 1111 --v 117  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb8spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1  --nepochratio .9  --losstype elbo --priorstdout 20 --topk 2 --seeds 1111 --v 117    ;

# # failed to run 
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb7spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .01  --nepochratio .9   --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 117  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb8spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .01  --nepochratio .9  --losstype elbo --priorstdout 5 --topk 2 --seeds 1111 --v 117    &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb7spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .01  --nepochratio .9   --losstype elbo --priorstdout 20 --topk 10 --seeds 1111 --v 117  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb8spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .01  --nepochratio .9  --losstype elbo --priorstdout 20 --topk 2 --seeds 1111 --v 117    ;


#CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb9spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1  --nepochratio .8  --losstype elbo --priorstdout 5 --topk 2 --seeds 1111 --v 117    ;


# # CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb9spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 2 --seeds 1111 --v 117    ;
# # CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb9spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .01   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 2 --seeds 1111 --v 117    &
# # CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb9spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1    --nepochratio .8  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 117    &
# # CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb9spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .01   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 117    ;

# # CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb9spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1   --nepochratio  .9  --losstype elbo --priorstdout 5 --topk 2 --seeds 1111 --v 117    &
# # CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb9spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .01   --nepochratio .9  --losstype elbo --priorstdout 5 --topk 2 --seeds 1111 --v 117    &
# # CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb9spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1   --nepochratio  .9  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 117    &
# # CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb9spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .01   --nepochratio .9  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 117    ;


# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb10spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 2 --seeds 1111 --v 117    &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb10spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .01   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 2 --seeds 1111 --v 117    &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb10spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1    --nepochratio .8  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 117    &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb10spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .01   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 117    ;



# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb9spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 2 --seeds 1111 --v 117    ;
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb9spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .01   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 2 --seeds 1111 --v 117    &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb9spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1    --nepochratio .8  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 117    &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb9spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .01   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 117    ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb9spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1   --nepochratio  .9  --losstype elbo --priorstdout 5 --topk 2 --seeds 1111 --v 117    &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb9spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .01   --nepochratio .9  --losstype elbo --priorstdout 5 --topk 2 --seeds 1111 --v 117    &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb9spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1   --nepochratio  .9  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 117    &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb9spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .01   --nepochratio .9  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 117    ;


# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb7spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1   --nepochratio .8  --losstype elbo --priorstdout 10 --topk 2 --seeds 1111 --v 118    &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb7spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1   --nepochratio .8  --losstype elbo --priorstdout 10 --topk 10 --seeds 1111 --v 118  & 


## set priorstdout 10 instead of 5; expect to improve nll
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb8spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1   --nepochratio .8  --losstype elbo --priorstdout 10 --topk 2 --seeds 1111 --v 118   &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb8spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1   --nepochratio .8  --losstype elbo --priorstdout 10 --topk 5 --seeds 1111 --v 118   &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb8spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1   --nepochratio .8  --losstype elbo --priorstdout 10 --topk 10 --seeds 1111 --v 118  ; 

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb9spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1   --nepochratio .8  --losstype elbo  --priorstdout 10 --topk 2 --seeds 1111 --v 118    &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb9spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1   --nepochratio .8  --losstype elbo  --priorstdout 10 --topk 5 --seeds 1111 --v 118    &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb9spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1    --nepochratio .8  --losstype elbo --priorstdout 10 --topk 10 --seeds 1111 --v 118  ; 

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb10spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1   --nepochratio .8  --losstype elbo --priorstdout 10 --topk 2 --seeds 1111 --v 118    &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb10spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1   --nepochratio .8  --losstype elbo --priorstdout 10 --topk 5 --seeds 1111 --v 118    &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb10spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1    --nepochratio .8  --losstype elbo --priorstdout 10 --topk 10 --seeds 1111 --v 118   ;


# ## set priorstdout 1 instead of 5; expect to improve nll
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb8spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1   --nepochratio .8  --losstype elbo --priorstdout 1 --topk 2 --seeds 1111 --v 119   &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb8spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1   --nepochratio .8  --losstype elbo --priorstdout 1 --topk 10 --seeds 1111 --v 119  ; 

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb9spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1   --nepochratio .8  --losstype elbo  --priorstdout 1 --topk 2 --seeds 1111 --v 119   &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb9spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1   --nepochratio .8  --losstype elbo --priorstdout 1 --topk 10 --seeds 1111 --v 119  ; 

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb10spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1   --nepochratio .8  --losstype elbo --priorstdout 1 --topk 2 --seeds 1111 --v 119    &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb10spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1   --nepochratio .8  --losstype elbo --priorstdout 1 --topk 10 --seeds 1111 --v 119   ;
#CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7.py   --modelname fb10spresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 3 --nctset 4 --reglambda .1   --nepochratio .8  --losstype elbo --priorstdout 1 --topk 10 --seeds 1111 --v 119   ;











##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------##
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------##
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------##
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------##
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------##

## set 200 epochs ##
## set no jacobian computation for last layer ; 
## 1:L-1 layers are regularized by not mean function of f_nn, but also covariance function


## resnet20 wo and wi
# CUDA_VISIBLE_DEVICES=0  python3 train_base.py   --modelname resnet20     --configs ./exp_cifar10_cifar100  --nexp 1 --reglambda  1e-3 --seeds 1111 --v 120 &  
# CUDA_VISIBLE_DEVICES=0  python3 train_base.py   --modelname spresnet20  --configs ./exp_cifar10_cifar100  --nexp 1 --reglambda  1e-3 --seeds 1111 --v 120 ; 


## fbnn cold posterior 
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_coldposterior.py   --modelname fb4spnjresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda 1e-8   --nepochratio .1  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 120  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_coldposterior.py   --modelname fb4njresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4   --reglambda 1e-8   --nepochratio .1  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 120  ;

## fbnn cold posterior + 0.75*empirical prior --> failed
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_swag.py   --modelname fb4spnjresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda 1e-1   --nepochratio .75  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 120  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_swag.py   --modelname fb4njresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4   --reglambda 1e-1   --nepochratio .75  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 120  ;

## fbnn cold posterior + 0.8*empirical prior, swag param failed
#CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_swag.py   --modelname fb4spnjresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4   --reglambda 1e-1   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 120  ;
#CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_swag.py   --modelname fb4njresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4   --reglambda 1e-1   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 120  ;



# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_swag.py   --modelname fb4spnjresnet20  --configs ./exp_cifar10_cifar100   --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4   --reglambda 1e-1   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 120  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_swag.py   --modelname fb4njresnet20    --configs ./exp_cifar10_cifar100   --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4   --reglambda 1e-1   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 120  ;
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_swag.py   --modelname fb4spnjresnet20  --configs ./exp_cifar10_cifar100   --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4   --reglambda 1e-1   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 120  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_swag.py   --modelname fb4njresnet20    --configs ./exp_cifar10_cifar100   --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4   --reglambda 1e-1   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 120  ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag.py   --modelname fb4njkderesnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda 1e-1   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 120  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag.py   --modelname fb4spnjkderesnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda 1e-1   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 120  ;
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag.py   --modelname fb4njkderesnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda 1e-1   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 120  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag.py   --modelname fb4spnjkderesnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda 1e-1   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 120  ;


# ##---------------------------------------------------------

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_swag.py   --modelname fb4spnjresnet20  --configs ./exp_cifar10_cifar100   --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4   --reglambda 1e-2   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 120  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_swag.py   --modelname fb4njresnet20    --configs ./exp_cifar10_cifar100   --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4   --reglambda 1e-2   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 120  ;
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_swag.py   --modelname fb4spnjresnet20  --configs ./exp_cifar10_cifar100   --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4   --reglambda 1e-2   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 120  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_swag.py   --modelname fb4njresnet20    --configs ./exp_cifar10_cifar100   --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4   --reglambda 1e-2   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 120  ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag.py   --modelname fb4njkderesnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda 1e-2   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 120  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag.py   --modelname fb4spnjkderesnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda 1e-2   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 120  ;
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag.py   --modelname fb4njkderesnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda 1e-2   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 120  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag.py   --modelname fb4spnjkderesnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda 1e-2   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 120  ;

# ##---------------------------------------------------------


# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_swag.py   --modelname fb4spnjresnet20  --configs ./exp_cifar10_cifar100   --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4   --reglambda 1e-2   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 120  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_swag.py   --modelname fb4njresnet20    --configs ./exp_cifar10_cifar100   --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4   --reglambda 1e-2   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 120  ;
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_swag.py   --modelname fb4spnjresnet20  --configs ./exp_cifar10_cifar100   --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4   --reglambda 1e-4   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 120  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_swag.py   --modelname fb4njresnet20    --configs ./exp_cifar10_cifar100   --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4   --reglambda 1e-4   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 120  &

# # CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag.py   --modelname fb4njkderesnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda 1e-2   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 120  &
# # CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag.py   --modelname fb4spnjkderesnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda 1e-2   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 120  ;
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag.py   --modelname fb4njkderesnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda 1e-4   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 120  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag.py   --modelname fb4spnjkderesnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda 1e-4   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 120  ;


#CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag.py   --modelname fb4spnjkderesnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4 --reglambda 1e-4   --nepochratio .9  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 120  ;


# ##---------------------------------------------------------
## after meeting
#In pretraining use map estimator with small prior
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4spnjkderesnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4 --reglambda 1e-2   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 121  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4spnjkderesnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4 --reglambda 1e-2   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 121  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4spnjkderesnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4 --reglambda 1e-2   --nepochratio .9  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 121  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4spnjkderesnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4 --reglambda 1e-2   --nepochratio .9  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 121  ;



# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4njkderesnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4 --reglambda 1e-2   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 121  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4njkderesnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4 --reglambda 1e-2   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 121  ;
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4njkderesnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4 --reglambda 1e-2   --nepochratio .9  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 121  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4njkderesnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4 --reglambda 1e-2   --nepochratio .9  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 121  ;



# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4spnjkderesnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 121  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4spnjkderesnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 121  ;
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4spnjkderesnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4 --reglambda 1e-3   --nepochratio .9  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 121  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4spnjkderesnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4 --reglambda 1e-3   --nepochratio .9  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 121  ;


# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4njkderesnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 121  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4njkderesnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 121  ;
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4njkderesnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4 --reglambda 1e-3   --nepochratio .9  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 121  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4njkderesnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4 --reglambda 1e-3   --nepochratio .9  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 121  ;


## temperature mixutre prior
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4spnjkdetmpresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 122  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4spnjkdetmpresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 122  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 122  ;


# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4spnjkdetmpresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-3   --nepochratio .9  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 122  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4spnjkdetmpresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-3   --nepochratio .9  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 122  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-3   --nepochratio .9  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111 --v 122  ;


## marginalization is appled to sampling as well
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4spnjkdetmpresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-3   --nepochratio .9  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111  --v 123  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4spnjkdetmpresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-3   --nepochratio .9  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 123  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-3   --nepochratio .9  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111  --v 123  ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4spnjkdetmpresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111  --v 123  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4spnjkdetmpresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 123  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111  --v 123  ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4spnjkdetmpresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-3   --nepochratio .9  --losstype iwae --priorstdout 5 --topk 3 --seeds 1111  --v 123  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4spnjkdetmpresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-3   --nepochratio .9  --losstype iwae --priorstdout 5 --topk 10 --seeds 1111 --v 123  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-3   --nepochratio .9  --losstype iwae --priorstdout 5 --topk 3 --seeds 1111  --v 123  ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4spnjkdetmpresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype iwae --priorstdout 5 --topk 3 --seeds 1111  --v 123  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4spnjkdetmpresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype iwae --priorstdout 5 --topk 10 --seeds 1111 --v 123  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswag_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype iwae --priorstdout 5 --topk 3 --seeds 1111  --v 123  ;


# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-3   --nepochratio .8 --losstype elbo --priorstdout 5 --topk 3 --seeds 1111  --v 124  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-3   --nepochratio .9  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111  --v 124  ;

#CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswagtemp_dpretrain.py   --modelname fb4spnjkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-3   --nepochratio .9  --losstype elbo --priorstdout 5 --topk 3 --seeds 1111  --v 123  ;




# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-2   --nepochratio .8 --losstype elbo --priorstdout 5 --topk 3 --seeds 1111  --v 125  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-3   --nepochratio .8 --losstype elbo --priorstdout 5 --topk 3 --seeds 1111  --v 125  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-3   --nepochratio .8 --losstype elbo --priorstdout 1 --topk 3 --seeds 1111  --v 125  ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-4   --nepochratio .8 --losstype elbo --priorstdout 5 --topk 3 --seeds 1111  --v 125  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-4   --nepochratio .8 --losstype elbo --priorstdout 10 --topk 3 --seeds 1111  --v 125  ;


# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-2   --nepochratio .9 --losstype elbo --priorstdout 5 --topk 3 --seeds 1111  --v 125  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-3   --nepochratio .9 --losstype elbo --priorstdout 5 --topk 3 --seeds 1111  --v 125  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-3   --nepochratio .9 --losstype elbo --priorstdout 1 --topk 3 --seeds 1111  --v 125  ;


#CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-2   --nepochratio .07 --losstype elbo --priorstdout 5 --topk 3 --seeds 1111  --v 126  ;
#CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-2   --nepochratio .07 --losstype elbo --priorstdout 5 --topk 3 --seeds 1111  --v 126  ;
#CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-2   --nepochratio .07 --losstype elbo --priorstdout 5 --topk 3 --seeds 1111  --v 126  ;
#CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-2   --nepochratio .07 --losstype elbo --priorstdout 5 --topk 3 --seeds 1111  --v 126  --usepretrain;

## save pretrain params
#CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-2   --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 127  ;
#CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-2   --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 127 --usepretrain  ;
#CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-2   --nepochratio .8 --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 127 ;

#CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-2   --nepochratio .8 --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 127 --usepretrain;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 128 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8 --losstype iwae  --priorstdout 1e1 --topk 3 --seeds 1111  --v 128 ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4 --reglambda 1e-3   --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 128 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4 --reglambda 1e-3   --nepochratio .8 --losstype iwae  --priorstdout 1e1 --topk 3 --seeds 1111  --v 128 ;




# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb5njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-2   --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 130 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb5njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-2   --nepochratio .8 --losstype elbo  --priorstdout 1e0 --topk 3 --seeds 1111  --v 130 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb5njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 130 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb5njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8 --losstype elbo  --priorstdout 1e0 --topk 3 --seeds 1111  --v 130 ;
#CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb5njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6   --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 130 --usepretrain;

#CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb5njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6   --nepochratio .06  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 131 ;
#CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb5njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6   --nepochratio .06  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 131 --usepretrain;




# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-2   --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 130 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-2   --nepochratio .8 --losstype elbo  --priorstdout 1e0 --topk 3 --seeds 1111  --v 130 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 130 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8 --losstype elbo  --priorstdout 1e0 --topk 3 --seeds 1111  --v 130 ;


# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-2   --nepochratio .8 --losstype iwae  --priorstdout 1e1 --topk 3 --seeds 1111  --v 131 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-2   --nepochratio .8 --losstype iwae  --priorstdout 1e0 --topk 3 --seeds 1111  --v 131 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-3   --nepochratio .8 --losstype iwae  --priorstdout 1e1 --topk 3 --seeds 1111  --v 131 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb4njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-3   --nepochratio .8 --losstype iwae  --priorstdout 1e0 --topk 3 --seeds 1111  --v 131 ;




#CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb5njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 132 ;

#CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb5njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 132 --usepretrain;
#CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb5njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 133 --usepretrain;
#CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb5njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 134 --usepretrain;
#CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb5njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-2   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 135 --usepretrain;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb5njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 136 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb5njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 136 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb5njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-5   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 136 ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb5njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .9  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 136 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb5njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4   --nepochratio .9  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 136 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb5njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-5   --nepochratio .9  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 136 ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb5njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 10 --seeds 1111  --v 136 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb5njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 10 --seeds 1111  --v 136 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb5njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-5   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 10 --seeds 1111  --v 136 ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb5njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .9  --losstype elbo  --priorstdout 1e2 --topk 10 --seeds 1111  --v 136 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb5njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4   --nepochratio .9  --losstype elbo  --priorstdout 1e2 --topk 10 --seeds 1111  --v 136 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb5njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-5   --nepochratio .9  --losstype elbo  --priorstdout 1e2 --topk 10 --seeds 1111  --v 136 ;




# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb6njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 137 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb6njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .9  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 137 ;

#CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb6njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .9  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 138 --usepretrain;


#CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb6njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 139 ;
#CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb6njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .5  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 139 ;


#CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb6njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .9  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 137 ;





## report prof lee; this solves kde functional prior, but have some jump issue.
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb6njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 139 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb6njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .9  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 139 ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb5njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 139 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb5njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .9  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 139 ;


# ## chekc iwae
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb6njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype iwae  --priorstdout 1e2 --topk 3 --seeds 1111  --v 139 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb6njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype iwae  --priorstdout 1e2 --topk 3 --seeds 1111  --v 139 ;

# ## chekc jump
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb6njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 0   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 140 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb6njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 140 --mixcontext ;

# ## 
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb7njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 140 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb7njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 140 --mixcontext;

# # ## chekc jump
# # CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb6njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 140 --mixcontext ;
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb6njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 10 --seeds 1111  --v 141 --mixcontext &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb6njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 141 --mixcontext ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb6njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 141 --mixcontext &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb6njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-5   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 141 --mixcontext &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb6njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-5   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 141 --mixcontext ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb6njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 2 --nctset 4 --reglambda 1e-4   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 141 --mixcontext &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb6njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 2 --nctset 4 --reglambda 1e-5   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 141 --mixcontext &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb6njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 2 --nctset 4 --reglambda 1e-6   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 141 --mixcontext ;

# # scale logstd topk 10
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb6njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 1 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 10 --seeds 1111  --v 141 --mixcontext &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb6njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 1 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 10 --seeds 1111  --v 141 --mixcontext ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb6njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 2 --nctset 4 --reglambda 0   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 141 ; 


## check q(w^{L}) and mh distance for mask
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb8njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 0.   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 141 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb8njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 141 &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb8njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 141 --mixcontext ;

#CUDA_VISIBLE_DEVICES=0  python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb8njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 142 --mixcontext &
# CUDA_VISIBLE_DEVICES=0  python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb8njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e2 --topk 3 --seeds 1111  --v 142 --mixcontext &
# CUDA_VISIBLE_DEVICES=0  python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb8njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype iwae  --priorstdout 1e1 --topk 3 --seeds 1111  --v 142 --mixcontext &
# CUDA_VISIBLE_DEVICES=0  python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb8njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype iwae  --priorstdout 1e1 --topk 3 --seeds 1111  --v 142 --mixcontext ;




# ## baseline revision for cifar10 and context
# CUDA_VISIBLE_DEVICES=0  python3 train_base.py   --modelname resnet20     --configs ./exp_cifar10_cifar100  --nexp 1 --reglambda  1e-3 --seeds 1112 --v 120 &  
# CUDA_VISIBLE_DEVICES=0  python3 train_base.py   --modelname resnet20     --configs ./exp_cifar10_cifar100  --nexp 1 --reglambda  1e-3 --seeds 1113 --v 120 ;
# CUDA_VISIBLE_DEVICES=0  python3 train_base.py   --modelname resnet20     --configs ./exp_cifar10_cifar100  --nexp 1 --reglambda  1e-3 --seeds 1113 --v 120 ;
# CUDA_VISIBLE_DEVICES=0  python3 train_base.py   --modelname resnet20     --configs ./exp_cifar10_cifar100  --nexp 1 --reglambda  1e-3 --seeds 1113 --v 120 ;


# #CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_coldposterior.py   --modelname fb4njresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4   --reglambda 1e-8   --nepochratio .1  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 120  ;
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_coldposterior.py   --modelname fb4njresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4   --reglambda 1e-8   --nepochratio .1  --losstype elbo --priorstdout 5 --topk 10 --seeds 1112 --v 120  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_coldposterior.py   --modelname fb4njresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -4  --nfsample 10 --nctset 4   --reglambda 1e-8   --nepochratio .1  --losstype elbo --priorstdout 5 --topk 10 --seeds 1113 --v 120  ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_coldposterior.py   --modelname fb4njresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4   --reglambda 1e-8   --nepochratio .1  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 120  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_coldposterior.py   --modelname fb4njresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4   --reglambda 1e-8   --nepochratio .1  --losstype elbo --priorstdout 5 --topk 10 --seeds 1112 --v 120  &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_coldposterior.py   --modelname fb4njresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4   --reglambda 1e-8   --nepochratio .1  --losstype elbo --priorstdout 5 --topk 10 --seeds 1113 --v 120  ;

# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_coldposterior.py   --modelname fb4nj2resnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4   --reglambda 1e-8   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 120 --mixcontext &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_coldposterior.py   --modelname fb4nj2resnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4   --reglambda 1e-8   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 10 --seeds 1112 --v 120 --mixcontext &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_coldposterior.py   --modelname fb4nj2resnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4   --reglambda 1e-8   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 10 --seeds 1113 --v 120 --mixcontext ;
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_coldposterior.py   --modelname fb4nj2resnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4   --reglambda 1e-4   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 120 --mixcontext &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_coldposterior.py   --modelname fb4nj2resnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4   --reglambda 1e-4   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 10 --seeds 1112 --v 120 --mixcontext &
# CUDA_VISIBLE_DEVICES=0 python3 train_fbnnsp7_coldposterior.py   --modelname fb4nj2resnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4   --reglambda 1e-4   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 10 --seeds 1113 --v 120 --mixcontext ;








# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_mfvibnn.py   --modelname b2resnet20   --configs ./exp_cifar10_cifar100  --nexp 1 --scalelogstd -6 --reglambda  1e-8  --nfsample 10  --seeds 1111 --v 120  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_mfvibnn.py   --modelname b2resnet20   --configs ./exp_cifar10_cifar100  --nexp 1 --scalelogstd -6 --reglambda  1e-8  --nfsample 10  --seeds 1112 --v 120  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_mfvibnn.py   --modelname b2resnet20   --configs ./exp_cifar10_cifar100  --nexp 1 --scalelogstd -6 --reglambda  1e-8  --nfsample 10  --seeds 1113 --v 120  ;
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_mfvibnn.py   --modelname b2resnet20   --configs ./exp_cifar10_cifar100  --nexp 1 --scalelogstd -6 --reglambda  1e-4  --nfsample 10  --seeds 1111 --v 120  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_mfvibnn.py   --modelname b2resnet20   --configs ./exp_cifar10_cifar100  --nexp 1 --scalelogstd -6 --reglambda  1e-4  --nfsample 10  --seeds 1112 --v 120  &
# CUDA_VISIBLE_DEVICES=$DEVICES python3 train_mfvibnn.py   --modelname b2resnet20   --configs ./exp_cifar10_cifar100  --nexp 1 --scalelogstd -6 --reglambda  1e-4  --nfsample 10  --seeds 1113 --v 120  ;





# ##p(h) use for kde
# CUDA_VISIBLE_DEVICES=0  python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb8njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-2   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 143  &
# CUDA_VISIBLE_DEVICES=0  python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb8njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-2   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1112  --v 143  &
# CUDA_VISIBLE_DEVICES=0  python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb8njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-2   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1113  --v 143  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb8njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-2   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 143 --mixcontext &
# CUDA_VISIBLE_DEVICES=0  python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb8njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-2   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1112  --v 143 --mixcontext &
# CUDA_VISIBLE_DEVICES=0  python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb8njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-2   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1113  --v 143 --mixcontext ;

# CUDA_VISIBLE_DEVICES=0  python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb8njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 143 &
# CUDA_VISIBLE_DEVICES=0  python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb8njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1112  --v 143 &
# CUDA_VISIBLE_DEVICES=0  python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb8njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1113  --v 143 ;

# CUDA_VISIBLE_DEVICES=0  python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb8njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 143 --mixcontext &
# CUDA_VISIBLE_DEVICES=0  python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb8njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1112  --v 143 --mixcontext &
# CUDA_VISIBLE_DEVICES=0  python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb8njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1113  --v 143 --mixcontext ;

# CUDA_VISIBLE_DEVICES=0  python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb8njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 143 &
# CUDA_VISIBLE_DEVICES=0  python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb8njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1112  --v 143 &
# CUDA_VISIBLE_DEVICES=0  python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb8njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1113  --v 143 ;

# CUDA_VISIBLE_DEVICES=0  python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb8njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 143 --mixcontext &
# CUDA_VISIBLE_DEVICES=0  python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb8njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1112  --v 143 --mixcontext &
# CUDA_VISIBLE_DEVICES=0  python3 train_fbnnsp8_kdeswagtemp_dpretrain.py   --modelname fb8njkdetmpresnet20    --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1113  --v 143 --mixcontext ;




## revision all implementations and add resnet 18: from models_deepnet_v3.py import *    , not  models_deepnet_v2.py
#CUDA_VISIBLE_DEVICES=0  python3 train_base_v3.py   --modelname resnet20     --configs ./exp_cifar10_cifar100  --nexp 1 --reglambda  1e-3 --seeds 1111 --v 145 ;  
#CUDA_VISIBLE_DEVICES=0  python3 train_base_v3.py   --modelname resnet18     --configs ./exp_cifar10_cifar100  --nexp 1 --reglambda  1e-3 --seeds 1111 --v 145 ;  

#CUDA_VISIBLE_DEVICES=0 python3 train_v3_mfvibnn_coldp.py   --modelname b2resnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --reglambda  1e-8 --nfsample 10  --reglambda 1e-8  --seeds 1111 --v 120 ;
#CUDA_VISIBLE_DEVICES=0 python3 train_v3_mfvibnn_coldp.py   --modelname b2resnet18  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --reglambda  1e-8 --nfsample 10  --reglambda 1e-8  --seeds 1111 --v 120 ;


#CUDA_VISIBLE_DEVICES=0 python3 train_v3_fbnn_coldp.py   --modelname fbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4   --reglambda 1e-8   --nepochratio .05  --losstype elbo --priorstdout 5 --topk 10 --seeds 1111 --v 120 ;
#CUDA_VISIBLE_DEVICES=0 python3 train_v3_fbnn_coldp.py   --modelname fbresnet18  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 10 --nctset 4   --reglambda 1e-8   --nepochratio .8  --losstype elbo --priorstdout 5 --topk 10 --seeds 1112 --v 120 --mixcontext &


# CUDA_VISIBLE_DEVICES=0  python3 train_bmafbnn_v3.py   --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 145 --mixcontext ;
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py   --modelname bmafbresnet18  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 120 --mixcontext &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py   --modelname bmafbresnet18  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype iwae  --priorstdout 1e1 --topk 3 --seeds 1111  --v 120 --mixcontext ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py   --modelname bmafbresnet18  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 120 --mixcontext &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py   --modelname bmafbresnet18  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6   --nepochratio .8  --losstype iwae  --priorstdout 1e1 --topk 3 --seeds 1111  --v 120 --mixcontext ;


# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py        --modelname bmafbresnet18  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-5   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 120 --mixcontext &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py        --modelname bmafbresnet18  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-5   --nepochratio .8  --losstype iwae  --priorstdout 1e1 --topk 3 --seeds 1111  --v 120 --mixcontext &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_mfvibnn_coldp.py  --modelname b2resnet18     --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --reglambda  1e-8 --nfsample 10  --reglambda 1e-8  --seeds 1111 --v 120 ;




# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py        --modelname bmafbresnet18  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype iwae  --priorstdout 1e0 --topk 3 --seeds 1111  --v 120 --mixcontext &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py        --modelname bmafbresnet18  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e0 --topk 3 --seeds 1111  --v 120 --mixcontext ;
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py        --modelname bmafbresnet18  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype iwae  --priorstdout 1e0 --topk 3 --seeds 1111  --v 120 --mixcontext &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py        --modelname bmafbresnet18  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e0 --topk 3 --seeds 1111  --v 120 --mixcontext ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py        --modelname bmafbresnet18  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4   --nepochratio .8  --losstype iwae  --priorstdout 1e1 --topk 3 --seeds 1111  --v 120 --mixcontext &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py        --modelname bmafbresnet18  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 120 --mixcontext ;
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py        --modelname bmafbresnet18  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-4   --nepochratio .8  --losstype iwae  --priorstdout 1e1 --topk 3 --seeds 1111  --v 120 --mixcontext &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py        --modelname bmafbresnet18  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 1e-4   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 120 --mixcontext ;


# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bma2fbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 120 --mixcontext &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bma2fbresnet18  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 120 --mixcontext ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bma2fbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-5   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 120 --mixcontext &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bma2fbresnet18  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-5   --nepochratio .8  --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 120 --mixcontext ;

#CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .75 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 120 --mixcontext ;


# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .75 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 120 --mixcontext &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .80 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 120 --mixcontext &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3   --nepochratio .80 --losstype elbo  --priorstdout 5e0 --topk 3 --seeds 1111  --v 120 --mixcontext ;


# no grad for jacobian gradient 
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6   --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 120 --mixcontext  --temppeak 0.9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6   --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 120 --mixcontext  --temppeak 0.5  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 0.     --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 120 --mixcontext  --temppeak 0.9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 0.     --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 120 --mixcontext  --temppeak 0.5  ;


# # CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-5   --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 120 --mixcontext  --temppeak 0.9  &
# # CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-5   --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 120 --mixcontext  --temppeak 0.5  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 0.     --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 120 --mixcontext  --temppeak 0.9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 0.     --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 120 --mixcontext  --temppeak 0.5  ;




# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 0.     --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 121 --mixcontext  --temppeak 0.9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 0.     --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 121 --mixcontext  --temppeak 0.75 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 0.     --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 121 --mixcontext  --temppeak 0.5  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 0.     --nepochratio .8 --losstype iwae  --priorstdout 1e1 --topk 3 --seeds 1111  --v 121 --mixcontext  --temppeak 0.9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 0.     --nepochratio .8 --losstype iwae  --priorstdout 1e1 --topk 3 --seeds 1111  --v 121 --mixcontext  --temppeak 0.75 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 20 --nctset 4 --reglambda 0.     --nepochratio .8 --losstype iwae  --priorstdout 1e1 --topk 3 --seeds 1111  --v 121 --mixcontext  --temppeak 0.5  ;



## use not normalize 
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 0.     --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 121 --mixcontext  --temppeak 0.9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4  --reglambda 1e-4     --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 121 --mixcontext  --temppeak 0.9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bma2fbnn.py     --modelname bma2fbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 0.     --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 121 --mixcontext  --temppeak 0.9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bma2fbnn.py     --modelname bma2fbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4     --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 121 --mixcontext  --temppeak 0.9  ;


## cifar 10: evaluate not normalized model on   v1 and v2

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 0.     --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 151 --mixcontext  --temppeak 0.9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 0.     --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1112  --v 151 --mixcontext  --temppeak 0.9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 0.     --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1113  --v 151 --mixcontext  --temppeak 0.9  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4  --reglambda 1e-4     --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 151 --mixcontext  --temppeak 0.9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4  --reglambda 1e-4     --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1112  --v 151 --mixcontext  --temppeak 0.9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4  --reglambda 1e-4     --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1113  --v 151 --mixcontext  --temppeak 0.9  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4  --reglambda 1e-3     --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 151 --mixcontext  --temppeak 0.9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4  --reglambda 1e-3     --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1112  --v 151 --mixcontext  --temppeak 0.9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4  --reglambda 1e-3     --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1113  --v 151 --mixcontext  --temppeak 0.9  ;


# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bma2fbnn.py      --modelname bma2fbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 0.     --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 151 --mixcontext  --temppeak 0.9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bma2fbnn.py      --modelname bma2fbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 0.     --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1112  --v 151 --mixcontext  --temppeak 0.9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bma2fbnn.py      --modelname bma2fbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 0.     --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1113  --v 151 --mixcontext  --temppeak 0.9  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bma2fbnn.py      --modelname bma2fbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4  --reglambda 1e-4     --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 151 --mixcontext  --temppeak 0.9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bma2fbnn.py      --modelname bma2fbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4  --reglambda 1e-4     --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1112  --v 151 --mixcontext  --temppeak 0.9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bma2fbnn.py      --modelname bma2fbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4  --reglambda 1e-4     --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1113  --v 151 --mixcontext  --temppeak 0.9  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bma2fbnn.py      --modelname bma2fbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4  --reglambda 1e-3     --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 151 --mixcontext  --temppeak 0.9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bma2fbnn.py      --modelname bma2fbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4  --reglambda 1e-3     --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1112  --v 151 --mixcontext  --temppeak 0.9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bma2fbnn.py      --modelname bma2fbresnet20  --configs ./exp_cifar10_cifar100  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4  --reglambda 1e-3     --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1113  --v 151 --mixcontext  --temppeak 0.9  ;



## cifar 100

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 0.   --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 152 --mixcontext  --temppeak 0.9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bma2fbnn.py     --modelname bma2fbresnet18  --configs ./exp_cifar100_cifar10  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 0.   --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1111  --v 152 --mixcontext   --temppeak 0.9  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 0.   --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1112  --v 152 --mixcontext  --temppeak 0.9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bma2fbnn.py      --modelname bma2fbresnet18  --configs ./exp_cifar100_cifar10  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 0.   --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1112  --v 152 --mixcontext   --temppeak 0.9  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 0.   --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1113  --v 152 --mixcontext  --temppeak 0.9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bma2fbnn.py      --modelname bma2fbresnet18  --configs ./exp_cifar100_cifar10  --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 0.   --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3 --seeds 1113  --v 152 --mixcontext   --temppeak 0.9  ;


## mixup xtrain
#CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5 --seeds 1111  --v 151 --mixcontext  --temppeak 0.9  &
#CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5 --seeds 1112  --v 151 --mixcontext  --temppeak 0.9  ;
#CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5 --seeds 1113  --v 151 --mixcontext  --temppeak 0.9  &

#CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5 --seeds 1111  --v 151 --mixcontext  --temppeak 0.75 &
#CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5 --seeds 1112  --v 151 --mixcontext  --temppeak 0.75 ;
#CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-3  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5 --seeds 1113  --v 151 --mixcontext  --temppeak 0.75 ;

#CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5 --seeds 1111  --v 151 --mixcontext  --temppeak 0.9  &
#CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5 --seeds 1112  --v 151 --mixcontext  --temppeak 0.9  ;
#CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5 --seeds 1113  --v 151 --mixcontext  --temppeak 0.9  &

#CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5 --seeds 1111  --v 151 --mixcontext  --temppeak 0.75 &
#CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5 --seeds 1112  --v 151 --mixcontext  --temppeak 0.75 ;
#CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5 --seeds 1113  --v 151 --mixcontext  --temppeak 0.75 ;



## mixup xtrain, prior scale 1e1 -> 1e2 to improve ood
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e2 --topk 5 --seeds 1111  --v 151 --mixcontext  --temppeak 0.9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e2 --topk 5 --seeds 1112  --v 151 --mixcontext  --temppeak 0.9  ;
# #CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e2 --topk 5 --seeds 1113  --v 151 --mixcontext  --temppeak 0.9  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e2 --topk 5 --seeds 1111  --v 151 --mixcontext  --temppeak 0.75 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e2 --topk 5 --seeds 1112  --v 151 --mixcontext  --temppeak 0.75 ;
# #CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e2 --topk 5 --seeds 1113  --v 151 --mixcontext  --temppeak 0.75 ;


# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 10 --seeds 1111  --v 151 --mixcontext  --temppeak 0.9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 10 --seeds 1112  --v 151 --mixcontext  --temppeak 0.9  ;


# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 10 --seeds 1111  --v 151 --mixcontext  --temppeak 0.75 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 10 --seeds 1112  --v 151 --mixcontext  --temppeak 0.75 ;


# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e2 --topk 10 --seeds 1111  --v 151 --mixcontext  --temppeak 0.9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e2 --topk 10 --seeds 1112  --v 151 --mixcontext  --temppeak 0.9  ;
# #CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e2 --topk 10 --seeds 1113  --v 151 --mixcontext  --temppeak 0.9  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e2 --topk 10 --seeds 1111  --v 151 --mixcontext  --temppeak 0.75 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e2 --topk 10 --seeds 1112  --v 151 --mixcontext  --temppeak 0.75 ;
# #CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e2 --topk 10 --seeds 1113  --v 151 --mixcontext  --temppeak 0.75 ;


## need to run
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 10 --seeds 1113  --v 151 --mixcontext  --temppeak 0.9  ;
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_cifar10    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 10 --seeds 1113  --v 151 --mixcontext  --temppeak 0.75 ;



## mixup cifar100 + tinyimagenet, 
## alpha = 0.5*torch.rand((x.shape[0],)).view(-1,1,1,1)
## xc    =  (1-alpha)* + alpha*xc  

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5 --seeds 1111  --v 6 --mixcontext  --temppeak 0.9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5 --seeds 1112  --v 6 --mixcontext  --temppeak 0.9  ;
#CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5 --seeds 1113  --v 6 --mixcontext  --temppeak 0.9  ;

#CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5 --seeds 1111  --v 6 --mixcontext  --temppeak 0.9  &


# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5 --seeds 1111  --v 6 --mixcontext  --temppeak 0.75  &
#CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5 --seeds 1112  --v 6 --mixcontext  --temppeak 0.75  ;
#CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5 --seeds 1113  --v 6 --mixcontext  --temppeak 0.75  ;

#CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5 --seeds 1111  --v 6 --mixcontext  --temppeak 0.75  ;


# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5 --seeds 1111  --v 6 --mixcontext  --temppeak 0.5  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5 --seeds 1112  --v 6 --mixcontext  --temppeak 0.5  ;
#CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5 --seeds 1113  --v 6 --mixcontext  --temppeak 0.5  ;
#savepath: ./results/cifar100_tinyimagenet/trainsetcifar100_contextsettinyimagenet_modelnamebmafbresnet18_scalelogstd-6_priorstdout10.0_temppeak0.5_lr0.1_nctset4_reglambda0.0001_nfsample2_nepochratio0.8_usepretrainFalse_mixcontextTrue_topk5_losstypeelbo_nexp1_v6/seeds1111_gentime08272235.pth






# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py       --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .05 --losstype elbo  --priorstdout 1e1  --seeds 1111  --v 6  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py       --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .05 --losstype elbo  --priorstdout 1e1  --seeds 1112  --v 6  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py       --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1111  --v 6  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py       --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1112  --v 6  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py       --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .05 --losstype elbo  --priorstdout 1e1  --seeds 1113  --v 6  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py       --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1113   --v 6  ;

# ## mix context fb baseline
# # alpha = .5*torch.rand(xc.shape[0]).view(-1,1,1,1)
# # xc = alpha*xc + (1-alpha)*x        
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py       --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .05 --losstype elbo  --priorstdout 1e1  --seeds 1111  --v 6  --mixcontext &  ## one of these .py does not run.
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py       --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .05 --losstype elbo  --priorstdout 1e1  --seeds 1112  --v 6  --mixcontext ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py       --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1111  --v 6  --mixcontext &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py       --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1112  --v 6  --mixcontext ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py       --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .05 --losstype elbo  --priorstdout 1e1  --seeds 1113  --v 6  --mixcontext &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py       --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1113   --v 6  --mixcontext  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --mixcontext --mixself --mixalpha .5  --seeds 1111  --v 7 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --mixcontext --mixself --mixalpha .5  --seeds 1112  --v 7 ;
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --mixcontext --mixself --mixalpha .5  --seeds 1113  --v 7 ;


# ## mix context proposedline
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5  --mixcontext  --temppeak 0.9  --seeds 1111  --v 7 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5  --mixcontext  --temppeak 0.9  --seeds 1112  --v 7 ;
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5  --mixcontext  --temppeak 0.9  --seeds 1113  --v 7 ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5  --mixcontext  --temppeak 0.75  --seeds 1111  --v 7 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5  --mixcontext  --temppeak 0.75  --seeds 1112  --v 7 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5  --mixcontext  --temppeak 0.75  --seeds 1113  --v 7 ;


# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5  --mixcontext  --temppeak 0.9  --seeds 1111 --mixalpha .1  --v 7 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5  --mixcontext  --temppeak 0.9  --seeds 1112 --mixalpha .1  --v 7 ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5  --mixcontext  --temppeak 0.75  --seeds 1111 --mixalpha .1 --v 7 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5  --mixcontext  --temppeak 0.75  --seeds 1112 --mixalpha .1 --v 7 ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5  --mixcontext  --temppeak 0.9   --seeds 1113 --mixalpha .1  --v 7 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5  --mixcontext  --temppeak 0.75  --seeds 1113 --mixalpha .1  --v 7 ;



# # 09/01: check proposed methods for mixalpha0.9 similar to context set
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5  --mixcontext  --temppeak 0.9  --seeds 1111 --mixalpha .9  --v 7 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5  --mixcontext  --temppeak 0.9  --seeds 1112 --mixalpha .9  --v 7 ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5  --mixcontext  --temppeak 0.75  --seeds 1111 --mixalpha .9 --v 7 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5  --mixcontext  --temppeak 0.75  --seeds 1112 --mixalpha .9 --v 7 ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5  --mixcontext  --temppeak 0.9   --seeds 1113 --mixalpha .9  --v 7 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5  --mixcontext  --temppeak 0.75  --seeds 1113 --mixalpha .9  --v 7 ;

# #        check baseline methods that  0.1 similar to train set , 0.9 similar to context set
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --mixcontext --mixself --mixalpha .1  --seeds 1111  --v 7 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --mixcontext --mixself --mixalpha .1  --seeds 1112  --v 7 ;
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --mixcontext --mixself --mixalpha .1  --seeds 1113  --v 7 ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --mixcontext --mixself --mixalpha .9  --seeds 1111  --v 7 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --mixcontext --mixself --mixalpha .9  --seeds 1112  --v 7 ;
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --mixcontext --mixself --mixalpha .9  --seeds 1113  --v 7 ;



# ## 09/02: check proposed methods adversarial contextsets
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp_adv.py    --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --mixcontext  --mixalpha .01  --seeds 1111  --v 8  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp_adv.py    --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --mixcontext  --mixalpha .01  --seeds 1112  --v 8  ;
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp_adv.py    --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --mixcontext  --mixalpha .01  --seeds 1113  --v 8  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp_adv.py    --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --mixcontext  --mixalpha .1  --seeds 1111  --v 8  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp_adv.py    --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --mixcontext  --mixalpha .1  --seeds 1112  --v 8  ;
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp_adv.py    --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --mixcontext  --mixalpha .1  --seeds 1113  --v 8  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp_adv.py    --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --mixcontext  --mixalpha .5  --seeds 1111  --v 8  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp_adv.py    --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --mixcontext  --mixalpha .5  --seeds 1112  --v 8  ;
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp_adv.py    --modelname fbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --mixcontext  --mixalpha .5  --seeds 1113  --v 8  ;



# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn_adv.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5  --mixcontext  --temppeak 0.9  --seeds 1111 --mixalpha .01  --v 8 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn_adv.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5  --mixcontext  --temppeak 0.9  --seeds 1112 --mixalpha .01  --v 8 ;
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn_adv.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5  --mixcontext  --temppeak 0.9  --seeds 1113 --mixalpha .01  --v 8 ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn_adv.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5  --mixcontext  --temppeak 0.9  --seeds 1111 --mixalpha .1  --v 8 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn_adv.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5  --mixcontext  --temppeak 0.9  --seeds 1112 --mixalpha .1  --v 8 ;
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn_adv.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5  --mixcontext  --temppeak 0.9  --seeds 1113 --mixalpha .1  --v 8 ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn_adv.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5  --mixcontext  --temppeak 0.9  --seeds 1111 --mixalpha .5  --v 8 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn_adv.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5  --mixcontext  --temppeak 0.9  --seeds 1112 --mixalpha .5  --v 8 ;
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn_adv.py       --modelname bmafbresnet18  --configs ./exp_cifar100_tinyimagenet    --nexp  1  --scalelogstd -6  --nfsample 2 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 5  --mixcontext  --temppeak 0.9  --seeds 1113 --mixalpha .5  --v 8 ;


# ## 09/04 trade-off experiments of fbresnet20
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-8  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1111  --v 9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-8  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1112  --v 9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-8  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1113  --v 9  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-7  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1111  --v 9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-7  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1112  --v 9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-7  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1113  --v 9  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-6  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1111  --v 9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-6  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1112  --v 9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-6  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1113  --v 9  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-5  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1111  --v 9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-5  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1112  --v 9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-5  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1113  --v 9  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-4  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1111  --v 9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-4  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1112  --v 9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-4  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1113  --v 9  ;

## 09/05 re-trade-off experiments of fbresnet20
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-8  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1111  --v 9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-8  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1112  --v 9  &
#CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-8  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1113  --v 9  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-7  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1111  --v 9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-7  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1112  --v 9  &
#CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-7  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1113  --v 9  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-6  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1111  --v 9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-6  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1112  --v 9  &
#CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-6  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1113  --v 9  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-5  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1111  --v 9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-5  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1112  --v 9  &
#CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-5  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1113  --v 9  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-4  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1111  --v 9  ;
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-4  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1112  --v 9  ;
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-4  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1113  --v 9  ;


# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-3  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1111  --v 9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-3  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1112  --v 9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-3  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1113  --v 9  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-2  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1111  --v 9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-2  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1112  --v 9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-2  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1113  --v 9  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-1  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1111  --v 9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-1  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1112  --v 9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-1  --nepochratio .1 --losstype elbo  --priorstdout 1e1  --seeds 1113  --v 9  ;


# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 5e-1  --nepochratio .1 --losstype elbo  --priorstdout 1e1   --seeds 1111  --v 9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 5e-1  --nepochratio .1 --losstype elbo  --priorstdout 1e1   --seeds 1112  --v 9  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 5e-1  --nepochratio .1 --losstype elbo  --priorstdout 1e1   --seeds 1113  --v 9  ;






## 09/05 re-trade-off experiments of fbresnet20 with nepochratio .8

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1111  --v 10  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1112  --v 10  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1113  --v 10  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-5  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1111  --v 10  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-5  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1112  --v 10  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-5  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1113  --v 10  ;



# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1111  --v 10  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1112  --v 10  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1113  --v 10  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-3  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1111  --v 10  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-3  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1112  --v 10  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-3  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1113  --v 10  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-2  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1111  --v 10  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-2  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1112  --v 10  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-2  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1113  --v 10  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-1  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1111  --v 10  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-1  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1112  --v 10  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e-1  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1113  --v 10  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 5e-1  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1111  --v 10  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 5e-1  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1112  --v 10  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 5e-1  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1113  --v 10  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 5e-2  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1111  --v 10  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 5e-2  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1112  --v 10  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 5e-2  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1113  --v 10  ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 5e-3  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1111  --v 10  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 5e-3  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1112  --v 10  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 5e-3  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1113  --v 10  ;


# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e0  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1111  --v 10  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e0  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1112  --v 10  &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_fbnn_coldp.py    --modelname fbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10 --nctset 4 --reglambda 1e0  --nepochratio .8 --losstype elbo  --priorstdout 1e1  --seeds 1113  --v 10  ;





# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 5e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1111   --v 11 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 5e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1112    --v 11 ;
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 5e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1113    --v 11 &


# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1111   --v 11 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1112    --v 11 &
#CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-4  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1113    --v 11 ;


# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 5e-3  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1111   --v 11 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 5e-3  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1112    --v 11 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 5e-3  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1113    --v 11 ;


# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-3  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1111   --v 11 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-3  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1112    --v 11 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-3  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1113    --v 11 ;


# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 5e-3  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1111   --v 11 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 5e-3  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1112    --v 11 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 5e-3  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1113    --v 11 ;


# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 5e-2  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1111   --v 11 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 5e-2  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1112    --v 11 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 5e-2  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1113    --v 11 ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-2  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1111   --v 11 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-2  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1112    --v 11 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-2  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1113    --v 11 ;


# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 5e-1  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1111   --v 11 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 5e-1  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1112    --v 11 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 5e-1  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1113    --v 11 ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-1  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1111   --v 11 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-1  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1112    --v 11 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-1  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1113    --v 11 ;




# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-5  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1111   --v 11 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-5  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1112    --v 11 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-5  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1113    --v 11 ;


# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-5  --nepochratio .8 --losstype elbo  --priorstdout 1e0 --topk 3  --temppeak 0.9   --seeds 1111   --v 11 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-5  --nepochratio .8 --losstype elbo  --priorstdout 1e0 --topk 3  --temppeak 0.9   --seeds 1112    --v 11 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-5  --nepochratio .8 --losstype elbo  --priorstdout 1e0 --topk 3  --temppeak 0.9   --seeds 1113    --v 11 ;


# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1111   --v 11 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1112    --v 11 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1113    --v 11 ;


# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e0 --topk 3  --temppeak 0.9   --seeds 1111   --v 11 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e0 --topk 3  --temppeak 0.9   --seeds 1112    --v 11 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-6  --nepochratio .8 --losstype elbo  --priorstdout 1e0 --topk 3  --temppeak 0.9   --seeds 1113    --v 11 ;



# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-7  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1111   --v 11 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-7  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1112    --v 11 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-7  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --seeds 1113    --v 11 ;


# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-7  --nepochratio .8 --losstype elbo  --priorstdout 1e0 --topk 3  --temppeak 0.9   --seeds 1111   --v 11 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-7  --nepochratio .8 --losstype elbo  --priorstdout 1e0 --topk 3  --temppeak 0.9   --seeds 1112    --v 11 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-7  --nepochratio .8 --losstype elbo  --priorstdout 1e0 --topk 3  --temppeak 0.9   --seeds 1113    --v 11 ;



# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-7  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --mixcontext --mixalpha 0.1  --seeds 1111   --v 12 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-7  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --mixcontext --mixalpha 0.1  --seeds 1112    --v 12 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-7  --nepochratio .8 --losstype elbo  --priorstdout 1e1 --topk 3  --temppeak 0.9   --mixcontext --mixalpha 0.1  --seeds 1113    --v 12 ;

# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-7  --nepochratio .8 --losstype elbo  --priorstdout 1e0 --topk 3  --temppeak 0.9   --mixcontext --mixalpha 0.1  --seeds 1111    --v 12 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-7  --nepochratio .8 --losstype elbo  --priorstdout 1e0 --topk 3  --temppeak 0.9   --mixcontext --mixalpha 0.1  --seeds 1112    --v 12 &
# CUDA_VISIBLE_DEVICES=0  python3 train_v3_bmafbnn.py      --modelname bmafbresnet20  --configs ./exp_cifar10_cifar100    --nexp  1  --scalelogstd -8  --nfsample 10  --nctset 4 --reglambda 1e-7  --nepochratio .8 --losstype elbo  --priorstdout 1e0 --topk 3  --temppeak 0.9   --mixcontext --mixalpha 0.1  --seeds 1113    --v 12 ;
