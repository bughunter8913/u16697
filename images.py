import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

train_images_flat = np.load("analysis/pictures_flat.npy", allow_pickle=True)
y = np.load("analysis/labels.npy", allow_pickle=True)
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

# CNN Model
class CNNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(64 * 2 * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

cnn_model = CNNRegressor()

# Train CNN
def train_model(model, train_dl, val_dl=None, epochs=100):
    opt = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    train_loss, val_loss = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_dl:
            preds = model(xb)
            loss = loss_fn(preds, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * xb.size(0)

        train_loss.append(epoch_loss / len(train_dl.dataset))

        if val_dl:
            model.eval()
            val_epoch_loss = 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    val_epoch_loss += loss_fn(model(xb), yb).item() * xb.size(0)
            val_loss.append(val_epoch_loss / len(val_dl.dataset))

        print(f"Epoch {epoch+1}: train loss = {train_loss[-1]:.4f}")

    return train_loss, val_loss

cnn_model = CNNRegressor()
train_losses, _ = train_model(cnn_model, train_dl, epochs=100)

# Evaluate CNN
cnn_model.eval()
with torch.no_grad():
    pred_train = cnn_model(img_train_x).numpy()
    pred_test = cnn_model(img_test_x).numpy()

print("RMSE train:", mean_squared_error(img_train_y, pred_train) ** 0.5)
print("RMSE test:", mean_squared_error(img_test_y, pred_test) ** 0.5)

plt.plot(train_losses, label="train")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


