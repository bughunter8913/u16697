import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Split image data
img_train_x, img_test_x, img_train_y, img_test_y = train_test_split(
    train_images_flat, Y, test_size=0.25, random_state=0
)

# Split image data
img_train_x, img_test_x, img_train_y, img_test_y = train_test_split(
    train_images_flat, Y, test_size=0.25, random_state=0
)

# Fix: convert object-type arrays to uniform float32 arrays
img_train_x = np.stack(img_train_x).astype(np.float32)
img_test_x = np.stack(img_test_x).astype(np.float32)

# Convert image arrays from NHWC to NCHW for PyTorch
img_train_x = torch.tensor(img_train_x).permute(0, 3, 1, 2)
img_test_x = torch.tensor(img_test_x).permute(0, 3, 1, 2)

# Convert labels to torch tensors
img_train_y = torch.tensor(img_train_y.to_numpy(), dtype=torch.float32).view(-1, 1)
img_test_y = torch.tensor(img_test_y.to_numpy(), dtype=torch.float32).view(-1, 1)

# Create datasets and dataloaders
train_ds = TensorDataset(img_train_x, img_train_y)
test_ds = TensorDataset(img_test_x, img_test_y)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=batch_size)

