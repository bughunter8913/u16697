import pandas as pd
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import os

# 1. Daten laden und vorbereiten
df = pd.read_csv("data/listings.csv.gz", nrows=700)
df = df[["picture_url", "price", "accommodates", "bedrooms"]].dropna()
df["price"] = df["price"].str.replace("[$,]", "", regex=True).astype(float)

# 2. Tabellarische Features vorbereiten
print("Vorbereitung der Features...")
imputer = SimpleImputer(strategy='median')
df[['accommodates', 'bedrooms']] = imputer.fit_transform(df[['accommodates', 'bedrooms']])

scaler = MinMaxScaler()
tab_features = scaler.fit_transform(df[['accommodates', 'bedrooms']])

# 3. Bildverarbeitung mit erweiterter Fehlerbehandlung
print("Bilder werden verarbeitet...")
valid_data = []
drop_positions = []

for pos, (i, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Bilder verarbeiten")):
    try:
        response = requests.get(row["picture_url"], timeout=5)
        response.raise_for_status()  # HTTP-Fehler abfangen
        
        # Content-Type-Validierung
        content_type = response.headers.get('Content-Type', '')
        if 'image' not in content_type:
            raise ValueError(f"Kein Bild-Content (Type: {content_type})")
        
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = img.resize((32, 32))  # Direkte Größenänderung
        
        valid_data.append({
            "image": np.array(img),
            "price": row["price"],
            "tab_features": tab_features[pos]  # Position statt Index
        })
    except Exception as e:
        print(f"Fehler bei Position {pos}: {e}")
        drop_positions.append(pos)

# Synchronisiertes Löschen fehlerhafter Positionen
if drop_positions:
    df = df.drop(df.index[drop_positions]).reset_index(drop=True)
    tab_features = np.delete(tab_features, drop_positions, axis=0)

# 4. Daten in Training und Test aufteilen
if not valid_data:
    raise ValueError("Keine gültigen Bilddaten vorhanden")

images = [item["image"] for item in valid_data]
prices = [item["price"] for item in valid_data]
tab_features_list = [item["tab_features"] for item in valid_data]

X_train_img, X_test_img, X_train_tab, X_test_tab, y_train, y_test = train_test_split(
    images, tab_features_list, prices, 
    test_size=0.2, random_state=42
)

# 5. PyTorch Dataset-Klasse
class AirbnbDataset(Dataset):
    def __init__(self, images, tab_features, prices, transform=None):
        self.images = images
        self.tab_features = tab_features
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
        tab_data = torch.tensor(self.tab_features[idx], dtype=torch.float32)
        price = torch.tensor([self.prices[idx]], dtype=torch.float32)
        return (img, tab_data), price

# Initialisierung der Datasets
train_dataset = AirbnbDataset(X_train_img, X_train_tab, y_train)
test_dataset = AirbnbDataset(X_test_img, X_test_tab, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# 6. Multi-Input Modell
class MultiInputPricePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # Bildverarbeitungszweig
        self.image_branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        # Tabellarischer Zweig
        self.tab_branch = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )
        
        # Kombinierter Regressor
        self.regressor = nn.Sequential(
            nn.Linear(64 * 4 * 4 + 32, 128),  # 32x32 Bild -> 4x4 Feature Map
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        img, tab = x
        img_features = self.image_branch(img)
        tab_features = self.tab_branch(tab)
        combined = torch.cat((img_features, tab_features), dim=1)
        return self.regressor(combined)

# Gerätekonfiguration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende Gerät: {device}")
model = MultiInputPricePredictor().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. Training mit Early Stopping
print("Training läuft...")
best_val_loss = float('inf')
patience = 0
train_losses = []
val_losses = []

for epoch in range(50):
    model.train()
    total_loss = 0
    for (images, tab_data), prices in train_loader:
        images = images.to(device)
        tab_data = tab_data.to(device)
        prices = prices.to(device)
        
        optimizer.zero_grad()
        outputs = model((images, tab_data))
        loss = criterion(outputs, prices)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
    
    epoch_loss = total_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for (images, tab_data), prices in test_loader:
            images = images.to(device)
            tab_data = tab_data.to(device)
            prices = prices.to(device)
            outputs = model((images, tab_data))
            val_loss += criterion(outputs, prices).item() * images.size(0)
    
    val_loss /= len(test_loader.dataset)
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch+1}/50, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print("Neues bestes Modell gespeichert")
    else:
        patience += 1
        if patience >= 5:
            print("Early stopping triggered")
            break

# 8. Evaluation des besten Modells
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
test_loss = 0
with torch.no_grad():
    for (images, tab_data), prices in test_loader:
        images = images.to(device)
        tab_data = tab_data.to(device)
        prices = prices.to(device)
        outputs = model((images, tab_data))
        test_loss += criterion(outputs, prices).item() * images.size(0)

test_loss = test_loss / len(test_loader.dataset)
print(f"Test RMSE: {test_loss**0.5:.2f} €")

# 9. Vorhersagefunktion
def predict_price(url, accommodates, bedrooms):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = img.resize((32, 32))
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Tabellarische Features skalieren
        tab_data = scaler.transform([[accommodates, bedrooms]])
        tab_tensor = torch.tensor(tab_data, dtype=torch.float32).to(device)
        
        model.eval()
        with torch.no_grad():
            prediction = model((img_tensor, tab_tensor)).item()
        
        return prediction
    except Exception as e:
        print(f"Vorhersagefehler: {e}")
        return None

# Beispielvorhersage
if not df.empty:
    example_url = df["picture_url"].iloc[0]
    example_accommodates = df["accommodates"].iloc[0]
    example_bedrooms = df["bedrooms"].iloc[0]
    predicted_price = predict_price(example_url, example_accommodates, example_bedrooms)
    actual_price = df["price"].iloc[0]
    print(f"\nBeispielvorhersage:")
    print(f"Tatsächlicher Preis: {actual_price} €")
    print(f"Vorhergesagter Preis: {predicted_price:.2f} €")
else:
    print("Keine Daten für Beispielvorhersage verfügbar")

# Modell speichern
torch.save(model.state_dict(), 'airbnb_price_predictor.pth')
print("Modell erfolgreich gespeichert")
