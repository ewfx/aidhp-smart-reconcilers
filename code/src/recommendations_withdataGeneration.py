import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
import pandas as pd
from typing import Dict, Text

# ✅ Generate Simulated Banking Product Data
np.random.seed(42)
num_customers = 500
num_products = 50

customer_ids = [f"CUST_{i}" for i in range(1, num_customers + 1)]
product_ids = [f"P0{i}" for i in range(1, num_products + 1)]
interaction_types = ["viewed", "clicked", "purchased"]

# Simulate Customer-Product Interactions
data = []
for _ in range(5000):  # 5000 interactions
    cust_id = np.random.choice(customer_ids)
    prod_id = np.random.choice(product_ids)
    interaction = np.random.choice(interaction_types, p=[0.6, 0.3, 0.1])  # Most users view first
    data.append([cust_id, prod_id, interaction])

df = pd.DataFrame(data, columns=["customer_id", "product_id", "interaction"])

# Convert interaction type to numerical score (viewed=1, clicked=2, purchased=3)
interaction_mapping = {"viewed": 1, "clicked": 2, "purchased": 3}
df["interaction_score"] = df["interaction"].map(interaction_mapping)

df.to_csv("bankingdata.csv", index=False)

print("✅ Data saved to banking_recommendation_data.csv")

# ✅ Convert to TensorFlow Dataset
dataset = tf.data.Dataset.from_tensor_slices(dict(df))

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


# Split into train and test datasets
train_size = int(0.8 * len(df))
train_dataset = dataset.take(train_size).batch(256)
test_dataset = dataset.skip(train_size).batch(256)

# ✅ Define Improved Embedding Model
class BankingRecommender(tfrs.Model):
    def __init__(self):
        super().__init__()

        # Customer & Product Embeddings
        self.customer_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=customer_ids, mask_token=None),
            tf.keras.layers.Embedding(len(customer_ids) + 1, 32),
        ])
        self.product_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=product_ids, mask_token=None),
            tf.keras.layers.Embedding(len(product_ids) + 1, 32),
        ])

        # Dense Layers for Deep Learning Model
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1)  # Output interaction score
        ])

        # Retrieval Task
        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )

    def compute_loss(self, inputs, training=False):
        features, labels = inputs  # Unpack tuple
        customer_emb = self.customer_embedding(features["customer_id"])
        product_emb = self.product_embedding(features["product_id"])
        
        # Concatenate embeddings
        combined_features = tf.concat([customer_emb, product_emb], axis=1)
        
        # Predict Interaction Score
        predictions = self.rating_model(combined_features)

        return self.task(labels=labels, predictions=predictions)

# ✅ Train the Model
model = BankingRecommender()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

# Train the model
cached_train = train_dataset.map(lambda x: (x, x["interaction_score"])).cache()
cached_test = test_dataset.map(lambda x: (x, x["interaction_score"])).cache()

model.fit(cached_train, epochs=5)

# ✅ Proper Recommendation Function (FIXED)
def recommend_products_for_customer(customer_id, top_k=5):
    """ Recommends top K products for a given customer """
    
    # Embed the customer ID
    customer_embedding = model.customer_embedding(tf.convert_to_tensor([customer_id]))
    
    # Embed all product IDs
    product_embeddings = model.product_embedding(tf.convert_to_tensor(product_ids))
    
    # Compute similarity score
    scores = tf.reduce_sum(customer_embedding * product_embeddings, axis=1)
    
    # Get top-k product indices
    top_indices = tf.argsort(scores, direction='DESCENDING')[:top_k]
    
    # Retrieve top recommended product IDs
    recommended_product_ids = [product_ids[i] for i in top_indices.numpy()]
    
    recommended_products = pd.DataFrame({
        'Product_Id': recommended_product_ids,
        'Product_Name': [product_mapping.get(pid, 'Unknown') for pid in recommended_product_ids]
    })

    return recommended_products

# ✅ Test Personalized Recommendations
cust_id = "CUST_10"
print(f"🔹 Top Products for Customer {cust_id}: {recommend_products_for_customer(cust_id)}")


# ✅ Test Personalized Recommendations
cust_id = "CUST_16"
print(f"🔹 Top Products for Customer {cust_id}: {recommend_products_for_customer(cust_id)}")


