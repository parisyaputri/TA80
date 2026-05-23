# models/transformer_baseline.py

import warnings
from sklearn.ensemble import RandomForestClassifier

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# =====================================================================
# PYTORCH ACTUAL TRANSFORMER CLASSIFIER IMPLEMENTATION
# Used when torch is installed. Excellent for sequence classification.
# =====================================================================
if HAS_TORCH:
    class TransformerClassifierNet(nn.Module):
        def __init__(self, input_dim, d_model=32, num_heads=2, hidden_dim=64, num_layers=1):
            super(TransformerClassifierNet, self).__init__()
            # Project input dim to d_model (must be divisible by num_heads)
            self.project = nn.Linear(input_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model, 2) # Binary classification output (logits)

        def forward(self, x):
            # Input shape: (batch_size, seq_len, input_dim)
            x_proj = self.project(x)
            out = self.transformer(x_proj)
            # Pool across sequence length
            out = out.mean(dim=1)
            return self.fc(out)


class TransformerBaseline:
    """
    Transformer Baseline Model for Supervised Anomaly Detection.
    
    If PyTorch is installed, this implements a genuine Transformer Encoder 
    classifier that maps sequences to regular/deviant classes.
    If PyTorch is not available, it falls back to a CPU-friendly 
    Random Forest Classifier baseline.
    """

    def __init__(self, epochs=10, batch_size=64, learning_rate=0.001):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.has_torch = HAS_TORCH
        self.fallback_model = None
        self.device = None
        self.net = None

        if not self.has_torch:
            warnings.warn("PyTorch not found. Falling back to RandomForestClassifier CPU baseline.")
            self.fallback_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        # Convert X to float numpy array, y to integer classes
        X_arr = X.to_numpy() if hasattr(X, 'to_numpy') else np.array(X)
        X_arr = X_arr.astype(np.float32)
        
        # If labels are regular/deviant text, convert to binary 0/1
        if hasattr(y, 'to_numpy'):
            y_arr = y.to_numpy()
        else:
            y_arr = np.array(y)
        
        if y_arr.dtype.kind in ['U', 'S', 'O']: # Text labels
            y_arr = np.array([1 if str(val).lower() == 'deviant' else 0 for val in y_arr])
        y_arr = y_arr.astype(np.int64)

        if not self.has_torch:
            self.fallback_model.fit(X_arr, y_arr)
            return self

        # Transformer fitting
        # Prepare inputs as sequences: shape (batch_size, seq_len, input_dim)
        X_tensor = torch.tensor(X_arr, dtype=torch.float32).unsqueeze(1).to(self.device)
        y_tensor = torch.tensor(y_arr, dtype=torch.int64).to(self.device)
        
        input_dim = X_tensor.shape[2]
        self.net = TransformerClassifierNet(input_dim=input_dim).to(self.device)
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.net.train()
        dataset_len = len(X_tensor)
        
        for epoch in range(self.epochs):
            permutation = torch.randperm(dataset_len)
            for i in range(0, dataset_len, self.batch_size):
                indices = permutation[i:i+self.batch_size]
                batch_x = X_tensor[indices]
                batch_y = y_tensor[indices]
                
                optimizer.zero_grad()
                outputs = self.net(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        return self

    def predict(self, X):
        X_arr = X.to_numpy() if hasattr(X, 'to_numpy') else np.array(X)
        X_arr = X_arr.astype(np.float32)

        if not self.has_torch:
            preds = self.fallback_model.predict(X_arr)
            # Map numeric classes back to regular/deviant if required
            # The baseline framework expects binary list or array predictions matching y_train format
            return preds

        # PyTorch Transformer prediction
        self.net.eval()
        X_tensor = torch.tensor(X_arr, dtype=torch.float32).unsqueeze(1).to(self.device)
        with torch.no_grad():
            outputs = self.net(X_tensor)
            _, predicted = torch.max(outputs, 1)
            preds = predicted.cpu().numpy()
            
        return preds