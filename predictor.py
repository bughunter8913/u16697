import os
import pandas as pd
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, mean_squared_error
from tqdm import tqdm


class AirbnbPreprocessorAndTrainer:
    def __init__(self, csv_path, nrows=1000, image_size=32, batch_size=64, lr=0.001, patience=5, seed=42):
        self.csv_path = csv_path
        self.nrows = nrows
        self.image_size = image_size
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience

        # Reproducibility
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Verwende Gerät:", self.device)

        self.scaler = MinMaxScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.encoder = None
        self.model = None

        # Columns
        self.image_column = "picture_url"
        self.target_column = "price"
        self.optional_id_col = "id"

        # Containers
        self.train_loader = None
        self.test_loader = None
        self.train_dataset = None
        self.test_dataset = None
        self.df = None
        self.images = None
        self.feature_names = None  # nach Encoding/Scaling verwendete Feature-Namen

    def preprocess(self):
        df = pd.read_csv(self.csv_path, nrows=self.nrows)

        # Ensure essential columns
        if self.target_column not in df.columns:
            raise ValueError(f"Zielspalte '{self.target_column}' fehlt im Datensatz.")
        if self.image_column not in df.columns:
            raise ValueError(f"Bildspalte '{self.image_column}' fehlt im Datensatz.")

        # Drop rows without price or image
        df = df.dropna(subset=[self.target_column, self.image_column]).copy()

        # Clean price and coerce numeric (remove $, €, £, ,)
        df[self.target_column] = (
            df[self.target_column]
            .astype(str)
            .str.replace(r"[\$\€\£,]", "", regex=True)
        )
        df[self.target_column] = pd.to_numeric(df[self.target_column], errors="coerce")
        df = df[df[self.target_column].notna()].copy()

        # Outlier filtering (keep as in original preprocessing)
        lower, upper = df[self.target_column].quantile(0.01), df[self.target_column].quantile(0.99)
        df = df[(df[self.target_column] >= lower) & (df[self.target_column] <= upper)]
        print(f"Preisspannweite nach Ausreißerfilter: {lower:.2f} - {upper:.2f}")

        # Log-transform target
        df[self.target_column] = np.log(df[self.target_column])

        # Build feature set: all columns except picture_url, price, optional id, and description
        exclude = {self.image_column, self.target_column, "description"}
        if self.optional_id_col in df.columns:
            exclude.add(self.optional_id_col)
        feature_cols = [c for c in df.columns if c not in exclude]

        # Split numeric vs categorical by dtype
        num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in feature_cols if c not in num_cols]

        # Fill and encode categoricals
        if len(cat_cols) > 0:
            df[cat_cols] = df[cat_cols].astype("object").fillna("Unknown")
            try:
                self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            except TypeError:
                self.encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
            cat_encoded = self.encoder.fit_transform(df[cat_cols])
            cat_feature_names = self.encoder.get_feature_names_out(cat_cols)
            cat_df = pd.DataFrame(cat_encoded, columns=cat_feature_names, index=df.index)
        else:
            cat_df = pd.DataFrame(index=df.index)

        # Impute numeric
        if len(num_cols) > 0:
            df[num_cols] = self.imputer.fit_transform(df[num_cols])
            num_df = df[num_cols].copy()
        else:
            num_df = pd.DataFrame(index=df.index)

        # Final feature matrix
        feats_df = pd.concat([num_df, cat_df], axis=1)
        feats_df = pd.DataFrame(self.scaler.fit_transform(feats_df), columns=feats_df.columns, index=feats_df.index)
        self.feature_names = list(feats_df.columns)

        # Keep target and image (and id if exists)
        keep_cols = [self.image_column, self.target_column] + ([self.optional_id_col] if self.optional_id_col in df.columns else [])
        self.df = pd.concat([feats_df, df[keep_cols]], axis=1)

    def process_images(self):
        images = []
        valid_indices = []

        for i, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Bilder verarbeiten"):
            try:
                url = row[self.image_column]
                response = requests.get(url, timeout=5)
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

        self.images = np.array(images)
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)

    def prepare_tensors(self):
        # All features except image/target/id
        exclude = {self.image_column, self.target_column}
        if self.optional_id_col in self.df.columns:
            exclude.add(self.optional_id_col)
        feature_cols = [c for c in self.df.columns if c not in exclude]
        self.feature_names = feature_cols  # update after any changes

        X_tab = self.df[feature_cols].values.astype(np.float32)
        y = self.df[self.target_column].values.astype(np.float32)

        (X_train_tab, X_test_tab,
         X_train_img, X_test_img,
         y_train, y_test) = train_test_split(
            X_tab, self.images, y, test_size=0.2, random_state=self.seed
        )

        self.train_dataset = self.AirbnbDataset(X_train_img, X_train_tab, y_train)
        self.test_dataset = self.AirbnbDataset(X_test_img, X_test_tab, y_test)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

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
                nn.MaxPool2d(2),      # 16x16 (bei 32x32 Input)
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),      # 8x8
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),      # 4x4
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
        tab_dim = self.train_dataset.tab_features.shape[1]
        model = self.MultiInputPricePredictor(tab_dim).to(self.device)
        criterion = nn.MSELoss()  # Loss auf LOG-Skala
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            model.train()
            total_train_loss = 0.0
            for (images, tab_data), prices in self.train_loader:
                images = images.to(self.device)
                tab_data = tab_data.to(self.device)
                prices = prices.to(self.device)

                optimizer.zero_grad()
                outputs = model((images, tab_data))
                loss = criterion(outputs, prices)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * images.size(0)

            epoch_train_loss = total_train_loss / len(self.train_loader.dataset)
            train_losses.append(epoch_train_loss)

            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for (images, tab_data), prices in self.test_loader:
                    images = images.to(self.device)
                    tab_data = tab_data.to(self.device)
                    prices = prices.to(self.device)

                    outputs = model((images, tab_data))
                    loss = criterion(outputs, prices)
                    total_val_loss += loss.item() * images.size(0)

            epoch_val_loss = total_val_loss / len(self.test_loader.dataset)
            val_losses.append(epoch_val_loss)

            print(f"Epoch {epoch+1}/{epochs}, Train Loss (log): {epoch_train_loss:.4f}, Val Loss (log): {epoch_val_loss:.4f}")

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model.pth')
                print("Neues bestes Modell gespeichert")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping triggered")
                    break

        self.model = model
        return train_losses, val_losses

    def evaluate_loader(self, loader, model, device):
        preds, targets = [], []
        model.eval()
        with torch.no_grad():
            for (imgs, tabs), prices in loader:
                imgs = imgs.to(device)
                tabs = tabs.to(device)
                outputs = model((imgs, tabs))
                preds.extend(outputs.cpu().numpy().flatten())
                targets.extend(prices.cpu().numpy().flatten())
        return np.array(preds), np.array(targets)


def sklearn_metrics_log(preds_log, targets_log):
    rmse = mean_squared_error(targets_log, preds_log, squared=False)
    mae = mean_absolute_error(targets_log, preds_log)
    medae = median_absolute_error(targets_log, preds_log)
    r2 = r2_score(targets_log, preds_log)
    return rmse, mae, medae, r2


if __name__ == "__main__":
    # Pfad anpassen
    csv_path = "data/listings.csv.gz"

    # Initialisieren und Daten verarbeiten
    trainer = AirbnbPreprocessorAndTrainer(csv_path, nrows=500, image_size=32, batch_size=64, lr=0.001, patience=5, seed=42)
    trainer.preprocess()
    trainer.process_images()
    trainer.prepare_tensors()

    # Trainieren
    train_losses, val_losses = trainer.train_model(epochs=30)

    # Bestes Modell laden
    trainer.model.load_state_dict(torch.load('best_model.pth', map_location=trainer.device))

    # Vorhersagen für Train & Test
    preds_train, y_train_log = trainer.evaluate_loader(trainer.train_loader, trainer.model, trainer.device)
    preds_test, y_test_log = trainer.evaluate_loader(trainer.test_loader, trainer.model, trainer.device)

    # Nur scikit-learn Metriken im LOG-Raum ausgeben
    rmse_tr, mae_tr, medae_tr, r2_tr = sklearn_metrics_log(preds_train, y_train_log)
    rmse_te, mae_te, medae_te, r2_te = sklearn_metrics_log(preds_test, y_test_log)

    results = pd.DataFrame([
        {"Split": "Train", "RMSE (log)": rmse_tr, "MAE (log)": mae_tr, "MedAE (log)": medae_tr, "R² (log)": r2_tr},
        {"Split": "Test",  "RMSE (log)": rmse_te, "MAE (log)": mae_te, "MedAE (log)": medae_te, "R² (log)": r2_te},
    ])
    print("\nLeistungsmetriken (nur scikit-learn, LOG-Skala):")
    print(results.to_string(index=False))

    # Beispielvorhersage (LOG)
    if trainer.df is not None and not trainer.df.empty:
        example_row = trainer.df.iloc
        example_url = example_row[trainer.image_column]
        exclude_cols = {trainer.image_column, trainer.target_column}
        if trainer.optional_id_col in trainer.df.columns:
            exclude_cols.add(trainer.optional_id_col)
        example_features = example_row[[c for c in trainer.df.columns if c not in exclude_cols]].values.astype(np.float32)
        pred_log = trainer.predict(example_url, example_features)
        true_log = example_row[trainer.target_column]
        print("\nBeispielvorhersage (LOG):")
        if pred_log is not None:
            print(f"True (log): {true_log:.4f} | Pred (log): {pred_log:.4f}")
        else:
            print("Vorhersage fehlgeschlagen.")
    else:
        print("Keine Daten für Beispielvorhersage verfügbar")
