import pandas as pd
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 1. Daten laden und vorbereiten
df = pd.read_csv("data/listings.csv.gz", nrows=1500)
df = df[["picture_url", "price"]].dropna()
df["price"] = df["price"].str.replace("[$,]", "", regex=True).astype(float)

# 2. Bilder herunterladen und verarbeiten
class AirbnbImageDataset(Dataset):
    def __init__(self, urls, prices, transform=None):
        self.urls = urls
        self.prices = prices
        self.transform = transform or transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.urls)
    
    def __getitem__(self, idx):
        try:
            response = requests.get(self.urls[idx], timeout=5)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            if self.transform:
                img = self.transform(img)
            price = torch.tensor([self.prices[idx]], dtype=torch.float32)
            return img, price
        except:
            return None

# 3. Datensatz erstellen
print("Bilder werden verarbeitet...")
valid_indices = []
images = []
prices = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        response = requests.get(row["picture_url"], timeout=5)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = transforms.Resize((64, 64))(img)
        images.append(np.array(img))
        prices.append(row["price"])
        valid_indices.append(i)
    except:
        continue

# 4. Daten in Training und Test aufteilen (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    images, prices, test_size=0.2, random_state=42
)

# 5. PyTorch Dataset und DataLoader
class TensorDataset(Dataset):
    def __init__(self, images, prices, transform=None):
        self.images = images
        self.prices = prices
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        price = torch.tensor([self.prices[idx]], dtype=torch.float32)
        return img, price

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 6. CNN-Modell definieren
class CNNPricePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNPricePredictor().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. Training
print("Training läuft...")
model.train()
for epoch in range(10):
    total_loss = 0
    for images, prices in train_loader:
        images, prices = images.to(device), prices.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, prices)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
    
    epoch_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/10, Loss: {epoch_loss:.4f}")

# 8. Evaluation
model.eval()
test_loss = 0
with torch.no_grad():
    for images, prices in test_loader:
        images, prices = images.to(device), prices.to(device)
        outputs = model(images)
        test_loss += criterion(outputs, prices).item() * images.size(0)

test_loss = test_loss / len(test_loader.dataset)
print(f"Test RMSE: {test_loss**0.5:.2f} €")

# 9. Beispielvorhersage
def predict_price(url):
    try:
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            prediction = model(img_tensor).item()
        
        return prediction
    except:
        return None

# Beispielvorhersage
example_url = df["picture_url"].iloc[0]
predicted_price = predict_price(example_url)
actual_price = df["price"].iloc[0]
print(f"\nBeispielvorhersage:")
print(f"Tatsächlicher Preis: {actual_price} €")
print(f"Vorhergesagter Preis: {predicted_price:.2f} €")
