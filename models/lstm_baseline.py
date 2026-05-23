# models/lstm_baseline.py

import warnings
from sklearn.ensemble import IsolationForest

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# =====================================================================
# PYTORCH ACTUAL LSTM AUTOENCODER IMPLEMENTATION
# Used when torch is installed. Excellent for sequence reconstruction.
# =====================================================================
if HAS_TORCH:
    class LSTMAutoencoderNet(nn.Module):
        def __init__(self, input_dim, hidden_dim=16):
            super(LSTMAutoencoderNet, self).__init__()
            # Encoder
            self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            # Decoder
            self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

        def forward(self, x):
            # Input shape: (batch_size, seq_len, input_dim)
            _, (hidden, _) = self.encoder(x)
            # Repeat hidden state for decoder
            # Repeat the hidden state for each time step
            seq_len = x.size(1)
            repeated_hidden = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
            reconstructed, _ = self.decoder(repeated_hidden)
            return reconstructed


class LSTMBaseline:
    """
    LSTM Baseline Model for Unsupervised Anomaly Detection.
    
    If PyTorch is installed, this implements a genuine LSTM Autoencoder 
    that identifies anomalies by reconstruction error.
    If PyTorch is not available, it falls back to a CPU-friendly 
    Isolation Forest baseline.
    """

    def __init__(self, epochs=10, batch_size=64, learning_rate=0.001):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.has_torch = HAS_TORCH
        self.fallback_model = None
        self.device = None
        self.net = None
        self.threshold = None

        if not self.has_torch:
            warnings.warn("PyTorch not found. Falling back to IsolationForest CPU baseline.")
            self.fallback_model = IsolationForest(
                contamination=0.1,
                random_state=42
            )
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X):
        # Convert X to float numpy array
        X_arr = X.to_numpy() if hasattr(X, 'to_numpy') else np.array(X)
        X_arr = X_arr.astype(np.float32)

        if not self.has_torch:
            self.fallback_model.fit(X_arr)
            return self

        # LSTM Autoencoder fitting
        # Prepare inputs as sequences: shape (batch_size, seq_len, input_dim)
        # Here we treat the features of each case as a sequence of length 1 (or multiple if formatted)
        X_tensor = torch.tensor(X_arr, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        input_dim = X_tensor.shape[2]
        self.net = LSTMAutoencoderNet(input_dim=input_dim, hidden_dim=16).to(self.device)
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.net.train()
        dataset_len = len(X_tensor)
        
        for epoch in range(self.epochs):
            permutation = torch.randperm(dataset_len)
            epoch_loss = 0.0
            
            for i in range(0, dataset_len, self.batch_size):
                indices = permutation[i:i+self.batch_size]
                batch_x = X_tensor[indices]
                
                optimizer.zero_grad()
                reconstructed = self.net(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)
                
        # Calculate reconstruction threshold (90th percentile of training error)
        self.net.eval()
        with torch.no_grad():
            reconstructed = self.net(X_tensor)
            errors = torch.mean((reconstructed - X_tensor) ** 2, dim=[1, 2]).cpu().numpy()
            self.threshold = float(np.percentile(errors, 90))
            
        return self

    def predict(self, X):
        X_arr = X.to_numpy() if hasattr(X, 'to_numpy') else np.array(X)
        X_arr = X_arr.astype(np.float32)

        if not self.has_torch:
            preds = self.fallback_model.predict(X_arr)
            return [1 if p == -1 else 0 for p in preds]

        # PyTorch prediction based on reconstruction error
        self.net.eval()
        X_tensor = torch.tensor(X_arr, dtype=torch.float32).unsqueeze(1).to(self.device)
        with torch.no_grad():
            reconstructed = self.net(X_tensor)
            errors = torch.mean((reconstructed - X_tensor) ** 2, dim=[1, 2]).cpu().numpy()
            
        return [1 if err > self.threshold else 0 for err in errors]