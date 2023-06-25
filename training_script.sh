#!/bin/bash

METHOD=$1
DATASET=$2
CONCEPT_SET=$3

echo "Training method: $METHOD"
echo "Dataset: $DATASET"
echo "Concept set: $CONCEPT_SET"

if [ "$METHOD" == "LF-CBM" ]; then
    if [ "$DATASET" == "CIFAR10" ]; then
        python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py \
            --concept_set data/concept_sets/cifar10_filtered.txt
    elif [ "$DATASET" == "CIFAR100" ]; then
        python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py \
            --dataset cifar100 \
            --concept_set data/concept_sets/cifar100_filtered.txt
    elif [ "$DATASET" == "CUB200" ]; then
        python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py \
            --dataset cub \
            --backbone resnet18_cub \
            --concept_set data/concept_sets/cub_filtered.txt \
            --feature_layer features.final_pool \
            --clip_cutoff 0.26 \
            --n_iters 5000 \
            --lam 0.0002
    elif [ "$DATASET" == "doctor_nurse_full" ]; then
        if [ "$CONCEPT_SET" == "with_gender" ]; then
            python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py \
                --dataset doctor_nurse_full \
                --backbone alexnet_doctor_nurse \
                --concept_set data/concept_sets/doctor_nurse_with_gender.txt \
                --clip_cutoff 0.2 \
                --interpretability_cutoff 0.3 \
                --feature_layer avgpool \
                --n_iters 1000 \
                --protected_concepts "a male" "a female"\
                --print
        else
            python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py \
                --dataset doctor_nurse_full \
                --backbone alexnet_doctor_nurse \
                --clip_cutoff 0.2 \
                --interpretability_cutoff 0.3 \
                --concept_set data/concept_sets/doctor_nurse_filtered_new.txt \
                --feature_layer avgpool \
                --n_iters 1000 \
                --print
        fi
    # Note: You should change the level of bias depending on the --dataset value
    elif [ "$DATASET" == "doctor_nurse_gender_biased" ]; then
        if [ "$CONCEPT_SET" == "with_gender" ]; then
            python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py \
                --dataset doctor_nurse_gender_biased \
                --backbone alexnet_doctor_nurse \
                --clip_cutoff 0.25 \
                --interpretability_cutoff 0.35 \
                --concept_set data/concept_sets/doctor_nurse_with_gender.txt \
                --feature_layer avgpool \
                --n_iters 1000 \
                --protected_concepts "a male" "a female"\
                --print 
        else
            python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py \
                --dataset doctor_nurse_gender_biased \
                --backbone alexnet_doctor_nurse \
                --concept_set data/concept_sets/doctor_nurse_filtered_new.txt \
                --feature_layer avgpool \
                --n_iters 1000 \
                --print 
        fi
    elif [ "$DATASET" == "Places365" ]; then
        python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py \
            --dataset places365 \
            --backbone resnet50 \
            --concept_set data/concept_sets/places365_filtered.txt \
            --clip_cutoff 0.28 \
            --n_iters 80 \
            --lam 0.0003
    elif [ "$DATASET" == "ImageNet" ]; then
        python fairness_cv_project/methods/label_free_cbm/src/train_cbm.py \
            --dataset imagenet \
            --backbone resnet50 \
            --concept_set data/concept_sets/imagenet_filtered.txt \
            --clip_cutoff 0.28 \
            --n_iters 80 \
            --lam 0.0001
    else
        echo "Invalid dataset"
    fi
elif [ "$METHOD" == "sparse" ]; then
    if [ "$DATASET" == "CIFAR10" ]; then
        python train_standard.py
    elif [ "$DATASET" == "CIFAR100" ]; then
        python train_standard.py \
            --dataset cifar100 \
            --lam 0.003
    elif [ "$DATASET" == "CUB200" ]; then
        python train_standard.py \
            --dataset cub \
            --backbone resnet18_cub \
            --feature_layer features.final_pool \
            --lam 0.00002 \
            --n_iters 5000
    elif [ "$DATASET" == "Places365" ]; then
        python train_standard.py \
            --dataset places365 \
            --backbone resnet50 \
            --lam 0.0007 \
            --n_iters 80
    elif [ "$DATASET" == "ImageNet" ]; then
        python train_standard.py \
            --dataset imagenet \
            --backbone resnet50 \
            --lam 0.0001 \
            --n_iters 80
    else
        echo "Invalid dataset"
    fi
else
    echo "Invalid training method"
fi

