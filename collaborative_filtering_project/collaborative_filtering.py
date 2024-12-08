import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Sample dataset (user-item matrix)
data = {
    "Item1": [5, 3, 0, 1],
    "Item2": [4, 0, 0, 1],
    "Item3": [1, 1, 0, 5],
    "Item4": [0, 1, 5, 4],
    "Item5": [0, 0, 4, 0],
}
user_item_matrix = pd.DataFrame(data, index=["User1", "User2", "User3", "User4"])

# Print the user-item matrix
print("User-Item Matrix:")
print(user_item_matrix)

# Compute user similarity matrix
user_similarity = cosine_similarity(user_item_matrix.fillna(0))
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Print the similarity matrix
print("\nUser Similarity Matrix:")
print(user_similarity_df)

# Prediction function
def predict_ratings(user_item_matrix, user_similarity):
    """
    Predict ratings using User-Based Collaborative Filtering.
    """
    # Normalize user-item matrix by subtracting mean ratings
    mean_user_ratings = user_item_matrix.mean(axis=1).values.reshape(-1, 1)
    ratings_diff = (user_item_matrix - mean_user_ratings).fillna(0).values
    
    # Predicted ratings (dot product of similarity and ratings difference)
    pred = user_similarity @ ratings_diff / np.abs(user_similarity).sum(axis=1).reshape(-1, 1)
    
    # Add back the mean ratings to get the final predicted ratings
    pred += mean_user_ratings
    return pd.DataFrame(pred, index=user_item_matrix.index, columns=user_item_matrix.columns)

# Predict the ratings
predicted_ratings = predict_ratings(user_item_matrix, user_similarity)

# Print the predicted ratings
print("\nPredicted Ratings:")
print(predicted_ratings)

# Recommendation function
def recommend_items(user, user_item_matrix, predicted_ratings, top_n=3):
    """
    Recommend items for a given user.
    """
    # Get unrated items for the user
    unrated_items = user_item_matrix.loc[user][user_item_matrix.loc[user] == 0].index
    
    # Sort predicted ratings for unrated items
    recommendations = predicted_ratings.loc[user, unrated_items].sort_values(ascending=False)
    
    return recommendations.head(top_n)

# Recommend items for User1
user_to_recommend = "User1"
recommended_items = recommend_items(user_to_recommend, user_item_matrix, predicted_ratings)

print(f"\nRecommendations for {user_to_recommend}:")
print(recommended_items)
