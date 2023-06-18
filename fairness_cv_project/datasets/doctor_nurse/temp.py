import matplotlib.pyplot as plt
import numpy as np


def compute_number_samples(num_sample_class, ratio):
    def func(x):
        return x / (1-x)

    def func2(x):
        return func(1-x)
    
    if ratio <= 0.5:
        ratio_minority_class = func(ratio)
    else:
        ratio_minority_class = func2(ratio) 

    num_sample_minority_class = int(num_sample_class * ratio_minority_class)
    num_sample_majority_class = num_sample_class
    total_samples = num_sample_minority_class + num_sample_majority_class
    return total_samples

print(compute_number_samples(100, 0))
print(compute_number_samples(100, 1/4))
print(compute_number_samples(100, 1/3))
print(compute_number_samples(100, 1/2))
print(compute_number_samples(100, 2/3))
print(compute_number_samples(100, 3/4))
print(compute_number_samples(100, 1))