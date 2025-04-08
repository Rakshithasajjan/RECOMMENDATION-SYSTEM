import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import zipfile
import requests
import os

# Load datasets
df = pd.read_csv(os.path.join(dataset_path, "ratings.csv"))
movies = pd.read_csv(os.path.join(dataset_path, "movies.csv"))

# Pivot table for user-item interactions
user_item_matrix= df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
# Convert the DataFrame to a sparse matrix
from scipy.sparse import csr_matrix  # Import the necessary module
user_item_matrix_sparse = csr_matrix(user_item_matrix.values) # convert to sparse matrix


# Compute similarity for User-Based Collaborative Filtering
user_similarity = cosine_similarity(user_item_matrix)

# Compute similarity for Item-Based Collaborative Filtering
item_similarity = cosine_similarity(user_item_matrix.T)

# Matrix Factorization using SVD
# Use the sparse matrix in svds
U, sigma, Vt = svds(user_item_matrix_sparse, k=50)
sigma = np.diag(sigma)
predictions = np.dot(np.dot(U, sigma), Vt)

# Convert predictions to DataFrame
predicted_ratings = pd.DataFrame(predictions, index=user_item_matrix.index, columns=user_item_matrix.columns)

# Matrix Factorization using SVD
# Convert user_item_matrix to a sparse matrix before passing it to svds
from scipy.sparse import csr_matrix
user_item_matrix_sparse = csr_matrix(user_item_matrix)
U, sigma, Vt = svds(user_item_matrix_sparse, k=50)
sigma = np.diag(sigma)
predictions = np.dot(np.dot(U, sigma), Vt)

# Convert predictions to DataFrame
predicted_ratings = pd.DataFrame(predictions, index=user_item_matrix.index, columns=user_item_matrix.columns)
# Evaluation: RMSE Calculation
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

actual_ratings = user_item_matrix.values.flatten()
predicted_ratings_values = predicted_ratings.values.flatten()
rmse_value = rmse(actual_ratings, predicted_ratings_values)

# Display Results
print(f"RMSE: {rmse_value:.4f}")
