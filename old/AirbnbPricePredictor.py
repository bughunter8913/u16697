from matplotlib import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class AirbnbPricePredictor:
    def __init__(self, image_size=32, batch_size=64, learning_rate=0.001):
        self.image_size = image_size
        self.batch_size = batch_size
        self.lr = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = MinMaxScaler()
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.best_val_loss = float('inf')
        self.epoch = 0
        self.train_losses = []
        self.val_losses = []
        
    def prepare_data(self, csv_path, nrows=700):
        # Daten laden und vorbereiten
        df = pd.read_csv(csv_path, nrows=nrows)
        df = df[["picture_url", "price", "accommodates", "bedrooms"]].dropna()
        df["price"] = df["price"].str.replace("[$,]", "", regex=True).astype(float)
        
        # Tabellarische Features verarbeiten
        imputer = SimpleImputer(strategy='median')
        df[['accommodates', 'bedrooms']] = imputer.fit_transform(df[['accommodates', 'bedrooms']])
        tab_features = self.scaler.fit_transform(df[['accommodates', 'bedrooms']])
        
        # Bilder verarbeiten
        images = []
        prices = []
        valid_indices = []
        
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Bilder verarbeiten"):
            try:
                response = requests.get(row["picture_url"], timeout=5)
                img = Image.open(BytesIO(response.content)).convert("RGB")
                img = img.resize((self.image_size, self.image_size))
                images.append(np.array(img))
                prices.append(row["price"])
                valid_indices.append(i)
            except:
                continue
        
        # Gültige Daten speichern
        self.df = df.iloc[valid_indices].reset_index(drop=True)
        self.tab_features = tab_features[valid_indices]
        self.images = np.array(images)
        self.prices = np.array(prices)
        
    def create_datasets(self, test_size=0.2):
        # Datenaufteilung
        (X_train_img, X_test_img, 
         X_train_tab, X_test_tab, 
         y_train, y_test) = train_test_split(
            self.images, self.tab_features, self.prices, 
            test_size=test_size, random_state=42
        )
        
        # PyTorch Datasets
        self.train_dataset = AirbnbDataset(X_train_img, X_train_tab, y_train)
        self.test_dataset = AirbnbDataset(X_test_img, X_test_tab, y_test)
        
        # DataLoader
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size
        )
    
    def build_model(self):
        # Modellarchitektur
        self.model = MultiInputPricePredictor().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    
    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        
        for (images, tab_data), prices in self.train_loader:
            images = images.to(self.device)
            tab_data = tab_data.to(self.device)
            prices = prices.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model((images, tab_data))
            loss = self.criterion(outputs, prices)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * images.size(0)
        
        epoch_loss = total_loss / len(self.train_loader.dataset)
        self.train_losses.append(epoch_loss)
        return epoch_loss
    
    def validate(self):
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for (images, tab_data), prices in self.test_loader:
                images = images.to(self.device)
                tab_data = tab_data.to(self.device)
                prices = prices.to(self.device)
                
                outputs = self.model((images, tab_data))
                loss = self.criterion(outputs, prices)
                val_loss += loss.item() * images.size(0)
        
        val_loss /= len(self.test_loader.dataset)
        self.val_losses.append(val_loss)
        return val_loss
    
    def fit(self, epochs=50, patience=5, checkpoint_dir="checkpoints"):
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_epoch = 0
        
        for epoch in range(epochs):
            self.epoch += 1
            train_loss = self.train_one_epoch()
            val_loss = self.validate()
            
            print(f"Epoch {self.epoch}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f}")
            
            # Checkpoint speichern
            self.save_checkpoint(
                os.path.join(checkpoint_dir, f"checkpoint_epoch_{self.epoch}.pth")
            )
            
            # Bestes Modell speichern
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_epoch = self.epoch
                self.save_checkpoint(
                    os.path.join(checkpoint_dir, "best_model.pth")
                )
                print(f"Neues bestes Modell gespeichert (Val Loss: {val_loss:.4f})")
            
            # Early Stopping
            if self.epoch - best_epoch >= patience:
                print(f"Early Stopping nach {self.epoch} Epochen")
                break
    
    def save_checkpoint(self, path):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'scaler': self.scaler
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        self.scaler = checkpoint['scaler']
        
        print(f"Checkpoint geladen (Epoche {self.epoch}, "
              f"Bester Val Loss: {self.best_val_loss:.4f})")
    
    def predict(self, image_url, accommodates, bedrooms):
        try:
            # Bild verarbeiten
            response = requests.get(image_url, timeout=5)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            
            # Tabellarische Features skalieren
            tab_data = self.scaler.transform([[accommodates, bedrooms]])
            tab_tensor = torch.tensor(tab_data, dtype=torch.float32).to(self.device)
            
            # Vorhersage machen
            self.model.eval()
            with torch.no_grad():
                prediction = self.model((img_tensor, tab_tensor)).item()
            
            return prediction
        except Exception as e:
            print(f"Vorhersagefehler: {e}")
            return None

# Hilfsklassen
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
            nn.Linear(64 * 4 * 4 + 32, 128),
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

# Verwendung der Klasse
if __name__ == "__main__":
    # Initialisierung
    predictor = AirbnbPricePredictor(image_size=32, batch_size=64, learning_rate=0.001)
    
    # Daten vorbereiten
    predictor.prepare_data("data/listings.csv.gz", nrows=700)
    predictor.create_datasets()
    
    # Modell aufbauen
    predictor.build_model()
    
    # Training durchführen
    predictor.fit(epochs=50, patience=5)
    
    # Beispielvorhersage
    example_data = predictor.df.iloc[0]
    predicted = predictor.predict(
        example_data["picture_url"],
        example_data["accommodates"],
        example_data["bedrooms"]
    )
    print(f"Tatsächlicher Preis: {example_data['price']} €")
    print(f"Vorhergesagter Preis: {predicted:.2f} €")
    
    # Checkpoint laden und Training fortsetzen
    predictor.load_checkpoint("checkpoints/best_model.pth")
    predictor.fit(epochs=50)  # Training fortsetzen
