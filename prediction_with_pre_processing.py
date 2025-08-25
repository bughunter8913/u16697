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
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
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

        self.feature_columns = [
            "host_is_superhost", "latitude", "longitude", "room_type",
            "accommodates", "bedrooms", "minimum_nights", "number_of_reviews",
            "review_scores_value", "host_identity_verified"
        ]
        self.image_column = "picture_url"
        self.target_column = "price"
        self.extra_columns = ["id"]

        self.train_loader = None
        self.test_loader = None
        self.train_dataset = None
        self.test_dataset = None
        self.df = None
        self.images = None

    def preprocess(self):
        df = pd.read_csv(self.csv_path, nrows=self.nrows)

        needed = self.feature_columns + [self.image_column, self.target_column] + self.extra_columns
        df = df[[col for col in needed if col in df.columns]].copy()

        # Drop rows ohne Preis oder Bild
        df = df.dropna(subset=[self.target_column, self.image_column])

        # Preis bereinigen und numerisch machen
        df[self.target_column] = df[self.target_column].astype(str).str.replace("[$,]", "", regex=True)
        df[self.target_column] = pd.to_numeric(df[self.target_column], errors="coerce")
        df = df[df[self.target_column].notna()]

        # Ausreißer filtern
        lower, upper = df[self.target_column].quantile(0.01), df[self.target_column].quantile(0.99)
        df = df[(df[self.target_column] >= lower) & (df[self.target_column] <= upper)]
        print(f"Preisspannweite nach Ausreißerfilter: {lower:.2f} - {upper:.2f} €")

        # Log-Transform des Zielwerts
        df[self.target_column] = np.log(df[self.target_column])

        # Fehlende Features droppen
        df = df.dropna(subset=[c for c in self.feature_columns if c in df.columns])
        df = df.reset_index(drop=True)

        # Feature Engineering: Boolean zu 0/1
        if "host_is_superhost" in df.columns:
            df["host_is_superhost"] = df["host_is_superhost"].map({'t': 1, 'f': 0, True: 1, False: 0}).astype(float)
        if "host_identity_verified" in df.columns:
            df["host_identity_verified"] = df["host_identity_verified"].map({'t': 1, 'f': 0, True: 1, False: 0}).astype(float)

        # One-Hot Encoding für room_type
        if "room_type" in df.columns:
            # Hinweis: je nach sklearn-Version ggf. sparse=False statt sparse_output verwenden
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            room_type_encoded = self.encoder.fit_transform(df[["room_type"]])
            room_type_cols = self.encoder.get_feature_names_out(["room_type"])
            room_type_df = pd.DataFrame(room_type_encoded, columns=room_type_cols, index=df.index)
            df = pd.concat([df.drop("room_type", axis=1), room_type_df], axis=1)

        # Imputation & Scaling
        num_features = [c for c in ["accommodates", "bedrooms", "number_of_reviews", "review_scores_value", "minimum_nights"] if c in df.columns]
        if len(num_features) > 0:
            df[num_features] = self.imputer.fit_transform(df[num_features])

        scale_cols = [col for col in df.columns if col not in [self.image_column, self.target_column, "id"]]
        df[scale_cols] = self.scaler.fit_transform(df[scale_cols])

        self.df = df

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
        feature_cols = [col for col in self.df.columns if col not in [self.image_column, self.target_column, "id"]]
        X_tab = self.df[feature_cols].values.astype(np.float32)
        y = self.df[self.target_column].values.astype(np.float32)

        # Train-Test-Split (tabular, images, target)
        (X_train_tab, X_test_tab,
         X_train_img, X_test_img,
         y_train, y_test) = train_test_split(
            X_tab, self.images, y, test_size=0.2, random_state=self.seed
        )

        # Datasets
        self.train_dataset = self.AirbnbDataset(X_train_img, X_train_tab, y_train)
        self.test_dataset = self.AirbnbDataset(X_test_img, X_test_tab, y_test)

        # DataLoader
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
            # Bild-Zweig (CNN)
            self.image_branch = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),      # 16x16
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),      # 8x8
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),      # 4x4
                nn.Flatten()
            )
            # Tabellarischer Zweig (MLP)
            self.tab_branch = nn.Sequential(
                nn.Linear(tab_dim, 32),
                nn.ReLU()
            )
            # Gemeinsamer Regressor
            self.regressor = nn.Sequential(
                nn.Linear(64 * 4 * 4 + 32, 128),  # 32x32 Input -> 4x4 Featuremap
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
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Training
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

            # Validation
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

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

            # Early Stopping
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

    def calculate_rmse(self, loader):
        # RMSE in der Log-Skala
        self.model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for (images, tab_data), prices in loader:
                images = images.to(self.device)
                tab_data = tab_data.to(self.device)
                prices = prices.to(self.device)
                outputs = self.model((images, tab_data))
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(prices.cpu().numpy().flatten())
        rmse = np.sqrt(np.mean((np.array(all_preds) - np.array(all_targets)) ** 2))
        return rmse

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
                prediction = self.model((img_tensor, tab_tensor)).item()

            return prediction
        except Exception as e:
            print(f"Vorhersagefehler: {e}")
            return None

    def regression_metrics_log(self, preds_log, targets_log):
        mae = mean_absolute_error(targets_log, preds_log)
        rmse = np.sqrt(np.mean((preds_log - targets_log) ** 2))
        medae = median_absolute_error(targets_log, preds_log)
        rme = np.mean((preds_log - targets_log) / targets_log) * 100  # in %
        r2 = r2_score(targets_log, preds_log)
        return mae, rmse, medae, rme, r2

    def regression_metrics_euro(self, preds_log, targets_log):
        preds_eur = np.exp(preds_log)
        targets_eur = np.exp(targets_log)
        mae = mean_absolute_error(targets_eur, preds_eur)
        rmse = np.sqrt(np.mean((preds_eur - targets_eur) ** 2))
        medae = median_absolute_error(targets_eur, preds_eur)
        nonzero = targets_eur != 0
        if np.any(nonzero):
            rme = np.mean((preds_eur[nonzero] - targets_eur[nonzero]) / targets_eur[nonzero]) * 100
        else:
            rme = np.nan
        r2 = r2_score(targets_eur, preds_eur)
        return mae, rmse, medae, rme, r2

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


# ------------------ Zusatz: Tabellenmetriken, Plots & Learning-Curve ------------------ #

def build_results_row(trainer, split_name, loader):
    preds_log, targets_log = trainer.evaluate_loader(loader, trainer.model, trainer.device)
    mae_l, rmse_l, medae_l, rme_l, r2_l = trainer.regression_metrics_log(preds_log, targets_log)
    mae_e, rmse_e, medae_e, rme_e, r2_e = trainer.regression_metrics_euro(preds_log, targets_log)
    return {
        "Modell": "MLP+CNN",
        "Split": split_name,
        "RMSE (log)": rmse_l,
        "RMSE (Preis €)": rmse_e,
        "MSE (Preis €)": rmse_e ** 2,
        "R² (Preis €)": r2_e,
        "MAE (Preis €)": mae_e,
        "MedAE (Preis €)": medae_e
    }


def plot_true_vs_pred_price(trainer, loader, title="True vs. Predicted Price – MLP+CNN", max_price=500):
    preds_log, targets_log = trainer.evaluate_loader(loader, trainer.model, trainer.device)
    y_true_price = np.exp(targets_log)
    preds_price = np.exp(preds_log)

    plt.figure(figsize=(5, 5))
    plt.scatter(y_true_price, preds_price, alpha=0.4)
    plt.plot([0, max_price], [0, max_price], 'r--', label='Ideal')
    plt.xlim(0, max_price)
    plt.ylim(0, max_price)
    plt.xlabel("True Price (€)")
    plt.ylabel("Predicted Price (€)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_learning_curve_pytorch(trainer,
                                train_fracs=np.linspace(0.1, 1.0, 10),
                                epochs_per_frac=6,
                                use_euro_rmse=True,
                                shuffle=True):
    # Volle Trainingsdaten aus dem vorhandenen Dataset
    imgs_full = trainer.train_dataset.images
    tabs_full = trainer.train_dataset.tab_features
    y_full = trainer.train_dataset.prices
    n = len(imgs_full)

    idx_all = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(trainer.seed)
        rng.shuffle(idx_all)

    def make_loader_from_indices(indices):
        subset_ds = trainer.AirbnbDataset(imgs_full[indices], tabs_full[indices], y_full[indices])
        return DataLoader(subset_ds, batch_size=trainer.batch_size, shuffle=True)

    def rmse_on_loader(loader, model):
        preds, targets = [], []
        model.eval()
        with torch.no_grad():
            for (imgs_b, tabs_b), y_b in loader:
                imgs_b = imgs_b.to(trainer.device)
                tabs_b = tabs_b.to(trainer.device)
                out = model((imgs_b, tabs_b)).cpu().numpy().flatten()
                tg = y_b.cpu().numpy().flatten()
                if use_euro_rmse:
                    out = np.exp(out)
                    tg = np.exp(tg)
                preds.extend(out)
                targets.extend(tg)
        preds = np.array(preds)
        targets = np.array(targets)
        return np.sqrt(np.mean((preds - targets) ** 2))

    train_sizes, train_rmse, val_rmse = [], [], []

    for frac in train_fracs:
        m = max(1, int(n * frac))
        indices = idx_all[:m]
        train_loader_frac = make_loader_from_indices(indices)

        # Frisches Modell je Trainingsgröße
        tab_dim = trainer.train_dataset.tab_features.shape[1]
        model = trainer.MultiInputPricePredictor(tab_dim).to(trainer.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=trainer.lr)

        # Kurzes Training pro Größe
        for _ in range(epochs_per_frac):
            model.train()
            for (imgs_b, tabs_b), y_b in train_loader_frac:
                imgs_b = imgs_b.to(trainer.device)
                tabs_b = tabs_b.to(trainer.device)
                y_b = y_b.to(trainer.device)
                optimizer.zero_grad()
                out = model((imgs_b, tabs_b))
                loss = criterion(out, y_b)
                loss.backward()
                optimizer.step()

        train_sizes.append(m)
        train_rmse.append(rmse_on_loader(train_loader_frac, model))
        val_rmse.append(rmse_on_loader(trainer.test_loader, model))

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(train_sizes, train_rmse, 'o-', label=f"Training RMSE {'(€)' if use_euro_rmse else '(log)'}")
    plt.plot(train_sizes, val_rmse, 'o-', label=f"Validation RMSE {'(€)' if use_euro_rmse else '(log)'}")
    plt.xlabel("Trainingsgröße")
    plt.ylabel(f"RMSE {'(€)' if use_euro_rmse else '(log)'}")
    plt.title("Learning Curve – MLP+CNN")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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

    # 1) RMSE auf Log-Skala (wie beim Training)
    train_rmse_log = trainer.calculate_rmse(trainer.train_loader)
    test_rmse_log = trainer.calculate_rmse(trainer.test_loader)
    print(f"\nTrain RMSE (log): {train_rmse_log:.4f}")
    print(f"Test  RMSE (log): {test_rmse_log:.4f}")

    # 2) RMSE im Euro-Raum (Rücktransformation mit exp)
    def calculate_actual_rmse(loader):
        trainer.model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for (images, tab_data), prices in loader:
                images = images.to(trainer.device)
                tab_data = tab_data.to(trainer.device)
                outputs = trainer.model((images, tab_data)).cpu().numpy().flatten()
                prices = prices.cpu().numpy().flatten() if torch.is_tensor(prices) else prices
                all_preds.extend(np.exp(outputs))
                all_targets.extend(np.exp(prices))
        rmse = np.sqrt(np.mean((np.array(all_preds) - np.array(all_targets)) ** 2))
        return rmse

    train_rmse_euro = calculate_actual_rmse(trainer.train_loader)
    test_rmse_euro = calculate_actual_rmse(trainer.test_loader)
    print(f"Train RMSE (€): {train_rmse_euro:.2f}")
    print(f"Test  RMSE (€): {test_rmse_euro:.2f}")

    # Tabellenmetriken analog zum Sklearn-Snippet (für Train & Test)
    results = []
    results.append(build_results_row(trainer, "Train", trainer.train_loader))
    results.append(build_results_row(trainer, "Test", trainer.test_loader))

    results_df = pd.DataFrame(results).sort_values(by="RMSE (Preis €)").reset_index(drop=True)
    print("\nErgebnistabelle (MLP+CNN):")
    print(results_df.to_string(index=False))

    # True-vs-Predicted-Plots im Euro-Raum
    plot_true_vs_pred_price(trainer, trainer.train_loader, title="True vs. Predicted Price – MLP+CNN (Train)", max_price=500)
    plot_true_vs_pred_price(trainer, trainer.test_loader, title="True vs. Predicted Price – MLP+CNN (Test)", max_price=500)

    # Learning-Curve für das PyTorch-Modell
    plot_learning_curve_pytorch(trainer,
                                train_fracs=np.linspace(0.1, 1.0, 10),
                                epochs_per_frac=6,
                                use_euro_rmse=True,
                                shuffle=True)

    # Beispielvorhersage
    if trainer.df is not None and not trainer.df.empty:
        example_row = trainer.df.iloc[0]
        example_url = example_row[trainer.image_column]
        example_features = example_row.drop([trainer.image_column, trainer.target_column] + ([ "id" ] if "id" in trainer.df.columns else [])).values.astype(np.float32)

        predicted_price_log = trainer.predict(example_url, example_features)
        actual_price_log = example_row[trainer.target_column]
        predicted_price_euro = np.exp(predicted_price_log) if predicted_price_log is not None else None
        actual_price_euro = np.exp(actual_price_log)

        print(f"\nBeispielvorhersage:")
        print(f"Tatsächlicher Preis (log): {actual_price_log:.2f}")
        print(f"Tatsächlicher Preis (€): {actual_price_euro:.2f}")
        print(f"Vorhergesagter Preis (log): {predicted_price_log:.2f}" if predicted_price_log is not None else "Vorhersage fehlgeschlagen")
        print(f"Vorhergesagter Preis (€): {predicted_price_euro:.2f}" if predicted_price_euro is not None else "")
    else:
        print("Keine Daten für Beispielvorhersage verfügbar")
