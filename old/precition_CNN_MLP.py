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
    def __init__(self, csv_path, nrows=1000, image_size=64, batch_size=64, lr=0.0001, patience=5):
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
            "accommodates", "bedrooms", "bathrooms", "beds",
            "minimum_nights", "number_of_reviews", "review_scores_rating",
            "review_scores_accuracy", "review_scores_cleanliness",
            "review_scores_checkin", "review_scores_communication",
            "review_scores_location", "review_scores_value",
            "host_identity_verified", "host_listings_count"
        ]
        self.image_column = "picture_url"
        self.target_column = "price"
        self.extra_columns = ["id"]
        self.train_loader = None
        self.test_loader = None

    def preprocess(self):
        df = pd.read_csv(self.csv_path, nrows=self.nrows)
        needed = self.feature_columns + [self.image_column, self.target_column] + self.extra_columns
        df = df[[col for col in needed if col in df.columns]].copy()
        
        # Preprocessing steps
        df = df.dropna(subset=[self.target_column, self.image_column])
        df[self.target_column] = df[self.target_column].astype(str).str.replace("[$,]", "", regex=True).astype(float)
        
        # Keep only rows with valid features
        df = df.dropna(subset=self.feature_columns)
        df = df.reset_index(drop=True)

        # Feature engineering
        df["host_is_superhost"] = df["host_is_superhost"].map({'t': 1, 'f': 0, True: 1, False: 0}).astype(float)
        if "host_identity_verified" in df:
            df["host_identity_verified"] = df["host_identity_verified"].map({'t': 1, 'f': 0, True: 1, False: 0}).astype(float)
        
        # One-Hot Encoding for room_type
        if "room_type" in df:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            room_type_encoded = self.encoder.fit_transform(df[["room_type"]])
            room_type_cols = self.encoder.get_feature_names_out(["room_type"])
            room_type_df = pd.DataFrame(room_type_encoded, columns=room_type_cols, index=df.index)
            df = pd.concat([df.drop("room_type", axis=1), room_type_df], axis=1)
        
        # Imputation and scaling
        num_features = ["accommodates", "bedrooms", "bathrooms", "beds", 
                        "number_of_reviews", "review_scores_rating", "minimum_nights",
                        "review_scores_accuracy", "review_scores_cleanliness",
                        "review_scores_checkin", "review_scores_communication",
                        "review_scores_location", "review_scores_value",
                        "host_listings_count"]
        
        # Only use existing numerical features
        existing_num_features = [f for f in num_features if f in df.columns]
        df[existing_num_features] = self.imputer.fit_transform(df[existing_num_features])
        
        scale_cols = [col for col in df.columns if col not in [self.image_column, self.target_column, "id"]]
        df[scale_cols] = self.scaler.fit_transform(df[scale_cols])
        
        self.df = df

    def process_images(self):
        images = []
        valid_indices = []
        
        for i, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Bilder verarbeiten"):
            try:
                response = requests.get(row[self.image_column], timeout=10)
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
        
        # Train-Test Split
        (X_train_tab, X_test_tab, 
         X_train_img, X_test_img, 
         y_train, y_test) = train_test_split(
            X_tab, self.images, y, 
            test_size=0.2, random_state=42
        )
        
        # Create datasets
        self.train_dataset = self.AirbnbDataset(X_train_img, X_train_tab, y_train)
        self.test_dataset = self.AirbnbDataset(X_test_img, X_test_tab, y_test)
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size
        )

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
            # Bildverarbeitungszweig (CNN)
            self.image_branch = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten()
            )
            
            # Tabellarischer Zweig (MLP)
            self.tab_branch = nn.Sequential(
                nn.Linear(tab_dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64)
            )
            
            # Kombinierter Regressor
            self.regressor = nn.Sequential(
                nn.Linear(128 * 4 * 4 + 64, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
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
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            model.train()
            total_train_loss = 0
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
            
            # Validation
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for (images, tab_data), prices in self.test_loader:
                    images = images.to(self.device)
                    tab_data = tab_data.to(self.device)
                    prices = prices.to(self.device)
                    
                    outputs = model((images, tab_data))
                    loss = criterion(outputs, prices)
                    total_val_loss += loss.item() * images.size(0)
            
            epoch_val_loss = total_val_loss / len(self.test_loader.dataset)
            scheduler.step(epoch_val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
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
        return best_val_loss

    def calculate_rmse(self, loader):
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
            response = requests.get(url, timeout=10)
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

if __name__ == "__main__":
    # Initialize and process data
    trainer = AirbnbPreprocessorAndTrainer("data/listings.csv.gz", nrows=1000)
    trainer.preprocess()
    trainer.process_images()
    trainer.prepare_tensors()
    
    # Train model
    best_val_loss = trainer.train_model(epochs=50)
    
    # Load best model
    trainer.model.load_state_dict(torch.load('best_model.pth'))
    
    # Calculate RMSE
    train_rmse = trainer.calculate_rmse(trainer.train_loader)
    test_rmse = trainer.calculate_rmse(trainer.test_loader)
    print(f"\nTrain RMSE: {train_rmse:.2f} €")
    print(f"Test RMSE: {test_rmse:.2f} €")
    
    # Example prediction
    if not trainer.df.empty:
        example_row = trainer.df.iloc[0]
        example_url = example_row["picture_url"]
        example_features = example_row.drop([trainer.image_column, trainer.target_column, "id"]).values.astype(np.float32)
        
        predicted_price = trainer.predict(example_url, example_features)
        actual_price = example_row["price"]
        
        print(f"\nBeispielvorhersage:")
        print(f"Tatsächlicher Preis: {actual_price} €")
        print(f"Vorhergesagter Preis: {predicted_price:.2f} €")
    else:
        print("Keine Daten für Beispielvorhersage verfügbar")
    
    # Save model
    torch.save(trainer.model.state_dict(), 'airbnb_price_predictor.pth')
    print("Modell erfolgreich gespeichert")
