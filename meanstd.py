from nn_loader import DiplomaDatasetLoader
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

training_set = DiplomaDatasetLoader(mode='training',
                                        # ratio=0.5,
                                        img_dir="C:/Users/dmalt/diploma/custom_diploma_dataset/",
                                        data_dir="C:/Users/dmalt/diploma/")
    
validation_set = DiplomaDatasetLoader(mode='validation',
                                        # ratio=0.5,
                                        img_dir="C:/Users/dmalt/diploma/custom_diploma_dataset/",
                                        data_dir="C:/Users/dmalt/diploma/")

test_set = DiplomaDatasetLoader(mode='test',
                                            # ratio=0.5,
                                        img_dir="C:/Users/dmalt/diploma/custom_diploma_dataset/",
                                        data_dir="C:/Users/dmalt/diploma/")


# Calculate mean and standard deviation for each channel
# means_training = []
# stds_training = []

# means_validation = []
# stds_validation = []

# means_test = []
# stds_test = []



means_all = []
stds_all = []
all_sets = [training_set] + [validation_set] + [test_set]

print(len(all_sets))

for j in range (3):
    means = []
    stds = []
    for i in range(6):
        channel_values = []
        for sample in all_sets[j]:
            channel_values.append(sample[0][i, :, :].numpy().ravel())
        means.append(np.mean(channel_values))
        stds.append(np.std(channel_values))
    means_all.append(means)
    stds_all.append(stds)

print("Training dataset means = ", means_all[0])
print("Training dataset stds = ", stds_all[0])

print("Validation dataset means = ", means_all[1])
print("Validation dataset means = ", stds_all[1])

print("Testing dataset means = ", means_all[2])
print("Testing dataset means = ", stds_all[2])

total_means = [(x+y+z) / 3 for x, y, z in zip(means_all[0], means_all[1], means_all[2])]
total_stds = [(x+y+z) / 3 for x, y, z in zip(stds_all[0], stds_all[1], stds_all[2])]

print("Total dataset means = ", total_means)
print("Total dataset stds = ", total_stds)