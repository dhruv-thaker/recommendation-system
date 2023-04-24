import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

cred = credentials.Certificate('takemain-firebase-adminsdk-37cps-84c4a47f23.json')
firebase_admin.initialize_app(cred)

db = firestore.client()

recipe_ratings_ref = db.collection("recipe_ratings")
recipe_ratings = recipe_ratings_ref.get()

recipe_collection_ref = db.collection("recipe_collection")
recipes = recipe_collection_ref.get()

users_ref = db.collection("users")
users = users_ref.get()

ratings_df = pd.DataFrame(columns=['user_id', 'recipe_id', 'rating'])
for doc in recipe_ratings:
    data = doc.to_dict()
    ratings_df = pd.concat([ratings_df, pd.DataFrame({
        'user_id': [data['uid'].id],
        'recipe_id': [data['recipe_id'].id],
        'rating': [data['recipe_rating']]
    })], ignore_index=True)

user_item_matrix = pd.pivot_table(ratings_df, values='rating', index='user_id', columns='recipe_id')

user_item_matrix = user_item_matrix.fillna(0.0)  # replace NaN with 0.0

user_similarity = cosine_similarity(user_item_matrix)
np.fill_diagonal(user_similarity, 0)

user_similarity = cosine_similarity(user_item_matrix)
item_similarity = cosine_similarity(user_item_matrix.T)

def get_top_n_recommendations(user_id, n=5):
    user_ratings = user_item_matrix.loc[user_id]
    similar_users = user_similarity[user_id]
    similar_user_indices = similar_users.argsort()[::-1][1:]
    similar_user_ratings = user_item_matrix.iloc[similar_user_indices]
    weighted_similar_user_ratings = similar_user_ratings.multiply(similar_users[similar_user_indices].reshape(-1, 1))
    weighted_average_ratings = weighted_similar_user_ratings.sum(axis=0) / similar_users[similar_user_indices].sum()
    user_unrated_recipes = user_ratings.isna()
    top_n_recommendations = weighted_average_ratings[user_unrated_recipes].sort_values(ascending=False)[:n]
    return top_n_recommendations

user_id = 'hSDidkzVOHS9ZW3q3uJxU62FkdI2'
top_n_recommendations = get_top_n_recommendations(user_id)
print(top_n_recommendations)
#recommended_recipes = recommend_recipes('hSDidkzVOHS9ZW3q3uJxU62FkdI2')
#print(recommended_recipes)

