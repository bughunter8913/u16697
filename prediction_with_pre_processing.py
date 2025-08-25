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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class AirbnbPreprocessorAndTrainer:
    def __init__(self, csv_path, nrows=700, image_size=32, batch_size=64, lr=0.001, patience=5):
        self.csv_path = csv_path
        self.nrows = nrows
        self.image_size = image_size
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = MinMaxScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.encoder = None
        self.model = None
        self.feature_columns = [
            "host_is_superhost", "latitude", "longitude", "room_type",
            "accommodates", "bedrooms", "minimum_nights", "number_of_reviews",
            "review_scores_value", "host_identity_verified"
        ]
        self.image_column = "picture_url"
        self.target_column = "price"
        self.extra_columns = ["id"]  # für spätere Zuordnung falls nötig

    def preprocess(self):
        df = pd.read_csv(self.csv_path, nrows=self.nrows)
        # Nur relevante Spalten
        needed = self.feature_columns + [self.image_column, self.target_column] + self.extra_columns
        df = df[[col for col in needed if col in df.columns]].copy()
        # Preis bereinigen
        df = df.dropna(subset=[self.target_column, self.image_column])
        df[self.target_column] = df[self.target_column].astype(str).str.replace("[$,]", "", regex=True).astype(float)
        df = df.dropna(subset=self.feature_columns)
        df = df.reset_index(drop=True)

        # Kategorische Variablen mappen
        df["host_is_superhost"] = df["host_is_superhost"].map({'t': 1, 'f': 0, True: 1, False: 0}).astype(float)
        if "host_identity_verified" in df:
            df["host_identity_verified"] = df["host_identity_verified"].map({'t': 1, 'f': 0, True: 1, False: 0}).astype(float)
        # One-Hot-Encoding für room_type
        if "room_type" in df:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            room_type_encoded = self.encoder.fit_transform(df[["room_type"]])
            room_type_cols = self.encoder.get_feature_names_out(["room_type"])
            room_type_df = pd.DataFrame(room_type_encoded, columns=room_type_cols, index=df.index)
            df = pd.concat([df.drop("room_type", axis=1), room_type_df], axis=1)
        # Imputation numerischer Features
        num_features = ["accommodates", "bedrooms", "number_of_reviews", "review_scores_value", "minimum_nights"]
        df[num_features] = self.imputer.fit_transform(df[num_features])
        # MinMax-Scaling
        scale_cols = [col for col in df.columns if col not in [self.image_column, self.target_column, "id"]]
        df[scale_cols] = self.scaler.fit_transform(df[scale_cols])
        self.df = df

    def process_images(self):
        images = []
        valid_indices = []
        for i, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Bilder verarbeiten"):
            try:
                response = requests.get(row[self.image_column], timeout=5)
                response.raise_for_status()
                content_type = response.headers.get('Content-Type', '')
                if 'image' not in content_type:
                    raise ValueError(f"Kein Bild-Content (Type: {content_type})")
                img = Image.open(BytesIO(response.content)).convert("RGB")
                img = img.resize((self.image_size, self.image_size))
                images.append(np.array(img))
                valid_indices.append(i)
            except Exception as e:
                print(f"Fehler bei Index {i}: {e}")
        self.images = [images[i] for i in range(len(images))]
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)

    def prepare_tensors(self):
        # Features und Ziel extrahieren
        feature_cols = [col for col in self.df.columns if col not in [self.image_column, self.target_column, "id"]]
        X_tab = self.df[feature_cols].values.astype(np.float32)
        y = self.df[self.target_column].values.astype(np.float32)
        images = np.array(self.images)
        # Split
        X_train_tab, X_test_tab, X_train_img, X_test_img, y_train, y_test = train_test_split(
            X_tab, images, y, test_size=0.2, random_state=42
        )
        self.X_train_tab = X_train_tab
        self.X_test_tab = X_test_tab
        self.X_train_img = X_train_img
        self.X_test_img = X_test_img
        self.y_train = y_train
        self.y_test = y_test

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
        def __init__(self, tab_dim):
            super().__init__()
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
            self.tab_branch = nn.Sequential(
                nn.Linear(tab_dim, 32),
                nn.ReLU()
            )
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

    def train_model(self, epochs=50):
        tab_dim = self.X_train_tab.shape[1]
        model = self.MultiInputPricePredictor(tab_dim).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        train_dataset = self.AirbnbDataset(self.X_train_img, self.X_train_tab, self.y_train)
        test_dataset = self.AirbnbDataset(self.X_test_img, self.X_test_tab, self.y_test)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for (images, tab_data), prices in train_loader:
                images, tab_data, prices = images.to(self.device), tab_data.to(self.device), prices.to(self.device)
                optimizer.zero_grad()
                outputs = model((images, tab_data))
                loss = criterion(outputs, prices)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * images.size(0)
            epoch_loss = total_loss / len(train_loader.dataset)
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for (images, tab_data), prices in test_loader:
                    images, tab_data, prices = images.to(self.device), tab_data.to(self.device), prices.to(self.device)
                    outputs = model((images, tab_data))
                    val_loss += criterion(outputs, prices).item() * images.size(0)
            val_loss /= len(test_loader.dataset)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping triggered")
                    break
        self.model = model

    def predict(self, url, tab_features):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img = img.resize((self.image_size, self.image_size))
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            tab_tensor = torch.tensor(tab_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            self.model.eval()
            with torch.no_grad():
                pred = self.model((img_tensor, tab_tensor)).item()
            return pred
        except Exception as e:
            print(f"Vorhersagefehler: {e}")
            return None

# Beispiel für die Nutzung:
if __name__ == "__main__":
    trainer = AirbnbPreprocessorAndTrainer("data/listings.csv.gz", nrows=700)
    trainer.preprocess()
    trainer.process_images()
    trainer.prepare_tensors()
    trainer.train_model(epochs=30)
    # Beispielvorhersage für ersten Datensatz
    example_url = trainer.df["picture_url"].iloc[0]
    example_tab = trainer.df.drop(["picture_url", "price", "id"], axis=1).iloc[0].values
    pred = trainer.predict(example_url, example_tab)
    print(f"Vorhergesagter Preis: {pred:.2f} €")
#     prediction = predict_price(example_url, df["accommodates"].iloc[0], df["bedrooms"].iloc[0])
#     print(f"Vorhergesagter Preis: {prediction:.2f} €")
# else:
#     print("DataFrame ist leer, keine Vorhersage möglich.")
#     prediction = None
#     print("Vorhersagefehler: DataFrame ist leer.")