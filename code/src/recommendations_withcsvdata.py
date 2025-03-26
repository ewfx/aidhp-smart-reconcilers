import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import StringLookup
import tensorflow_recommenders as tfrs
from sklearn.model_selection import train_test_split

# Generate random past interactions (which products customers have used)
interactions = pd.read_csv("dataset.csv")

# Convert Interaction Types to Weights
interaction_weights = {"Purchased": 3, "Clicked": 2, "Viewed": 1}
interactions["interaction_score"] = interactions["Interaction_Type"].map(interaction_weights) * interactions["Transaction_Frequency"]

product_mapping = {
    'P01': 'Basic Checking Account',
    'P02': 'Premium Checking Account',
    'P03': 'High-Yield Savings Account',
    'P04': 'Money Market Account',
    'P05': 'Standard Credit Card',
    'P06': 'Gold Credit Card',
    'P07': 'Platinum Credit Card',
    'P08': 'Business Credit Card',
    'P09': 'Personal Loan',
    'P010': 'Home Loan',
    'P011': 'Car Loan',
    'P012': 'Education Loan',
    'P013': 'Mortgage',
    'P014': 'Personal Loan',
    'P015': 'Fixed Deposit',
    'P016': 'Recurring Deposit',
    'P017': 'Investment Fund',
    'P018': 'Car Loan',
    'P019': 'Home Equity Loan',
    'P020': 'Gold Loan',
    'P021': 'Travel Insurance',
    'P022': 'Health Insurance',
    'P023': 'Life Insurance',
    'P024': 'Pet Insurance',
    'P025': 'Business Loan',
    'P026': 'Overdraft Protection',
    'P027': 'Wealth Management Service',
    'P028': 'Retirement Account',
    'P029': 'Savings Account',
    'P030': 'Gold Loan',
    'P031': 'Student Loan',
    'P032': 'Credit Line',
    'P033': 'Investment Advisory',
    'P034': 'Fixed Deposit',
    'P035': 'Trust Services',
    'P036': 'Real Estate Investment',
    'P037': 'Online Savings Account',
    'P038': 'Premium Savings Account',
    'P039': 'Cash Management Account',
    'P040': 'Luxury Credit Card',
    'P041': 'Gold Investment',
    'P042': 'Mutual Fund',
    'P043': 'Bonds',
    'P044': 'Stocks',
    'P045': 'Foreign Exchange Services',
    'P046': 'Financial Planning',
    'P047': 'Estate Planning',
    'P048': 'Long-Term Care Insurance',
    'P049': 'Short-Term Investment',
    'P050': 'Tax Planning'
}

# Ensure Users & Products are Unique
unique_customers = interactions["customer_id"].unique()
unique_products = interactions["product_id"].unique()

# Split data into train and test sets
train_data, test_data = train_test_split(interactions, test_size=0.2, random_state=42)

# Convert to TensorFlow dataset format
train_dataset = tf.data.Dataset.from_tensor_slices(({
    "customer_id": train_data["customer_id"].values,
    "product_id": train_data["product_id"].values
}, train_data["interaction_score"].values)).batch(32)

test_dataset = tf.data.Dataset.from_tensor_slices(({
    "customer_id": test_data["customer_id"].values,
    "product_id": test_data["product_id"].values
}, test_data["interaction_score"].values)).batch(32)

# Unique customer and product IDs for embedding
unique_customer_ids = interactions["customer_id"].unique().astype(str)
unique_product_ids = interactions["product_id"].unique().astype(str)


class BankingRecommendationModel(tfrs.Model):
    def __init__(self):
        super().__init__()
        
        # User & Product Embeddings
        embedding_dim = 32  # Embedding size for customers and products

        # String Lookups (Ensuring Unique Users & Products)
        self.user_lookup = StringLookup(vocabulary=unique_customer_ids)
        self.item_lookup = StringLookup(vocabulary=unique_product_ids)

        self.customer_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_customer_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_customer_ids) + 1, embedding_dim)
        ])

        self.product_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_product_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_product_ids) + 1, embedding_dim)
        ])

        # Rating Prediction
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1)  # Output rating prediction
        ])

        # Loss Function & Metric
        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def compute_loss(self, inputs, training=False):
       # Unpack tuple (features, labels) correctly
       features, labels = inputs  

       customer_embeddings = self.customer_embedding(features["customer_id"])
       product_embeddings = self.product_embedding(features["product_id"])

       # Concatenate embeddings
       combined_features = tf.concat([customer_embeddings, product_embeddings], axis=1)

       # Predict rating
       rating_predictions = self.rating_model(combined_features)

       return self.task(labels=labels, predictions=rating_predictions)

    	


# Compile and train the model
model = BankingRecommendationModel()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

# Train the model
model.fit(train_dataset, epochs=10, validation_data=test_dataset)

# Evaluate on test set
test_loss = model.evaluate(test_dataset)
print("Test Loss:", test_loss)


def recommend_products(customer_id, top_k=10):
    # Convert customer_id to tensor
    customer_tensor = tf.convert_to_tensor([str(customer_id)])

    # Get customer embedding
    customer_embedding = model.customer_embedding(customer_tensor)

    # Compute scores for all products
    product_scores = {}
    for product in unique_product_ids:
        product_tensor = tf.convert_to_tensor([str(product)])
        product_embedding = model.product_embedding(product_tensor)
        
        combined_features = tf.concat([customer_embedding, product_embedding], axis=1)
        predicted_rating = model.rating_model(combined_features).numpy()[0][0]
        
        product_scores[product] = predicted_rating

    # Sort and get top recommendations
    recommended_product_ids = sorted(product_scores, key=product_scores.get, reverse=True)[:top_k]
    
    recommended_products = pd.DataFrame({
        'Product_Id': recommended_product_ids,
        'Product_Name': [product_mapping.get(pid, 'Unknown') for pid in recommended_product_ids]
    })
    
    return recommended_products
