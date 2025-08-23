import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from math import atanh

def encoding_model(eeg_data, embedding, split_ratio):
    n_tokens, n_channels, n_timepoints = eeg_data.shape
    assert embedding.shape[0] == n_tokens, "Number of tokens must match between EEG data and embeddings"
    
    X_train, X_test, y_train, y_test = train_test_split(
        embedding, eeg_data, test_size=1-split_ratio, random_state=42
    )
    
    alphas = np.logspace(-3, 3, 7)
    channel_correlations = np.zeros(n_channels)
    
    for channel_idx in range(n_channels):
        y_train_channel = y_train[:, channel_idx, :]
        y_test_channel = y_test[:, channel_idx, :]
        
        y_train_flat = y_train_channel.reshape(y_train_channel.shape[0], -1)
        y_test_flat = y_test_channel.reshape(y_test_channel.shape[0], -1)
        
        ridge = RidgeCV(alphas=alphas, store_cv_results=True)
        ridge.fit(X_train, y_train_flat)
        
        y_pred_flat = ridge.predict(X_test)
        y_pred = y_pred_flat.reshape(y_test_channel.shape)
        time_correlations = []
        for time_idx in range(n_timepoints):
            r, _ = pearsonr(y_test_channel[:, time_idx], y_pred[:, time_idx])
            time_correlations.append(r)
        mean_correlation = np.mean(time_correlations)
        if abs(mean_correlation) >= 0.999:
            mean_correlation = np.sign(mean_correlation) * 0.999
        fisher_z = atanh(mean_correlation)
        
        channel_correlations[channel_idx] = fisher_z
    return channel_correlations
