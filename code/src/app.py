from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import sqlite3
import random
import secrets
from textblob import TextBlob  # For sentiment analysis
import matplotlib.pyplot as plt
from io import BytesIO
import plotly.graph_objects as go
import base64
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, T5ForConditionalGeneration, T5Tokenizer, DistilBertForSequenceClassification, DistilBertTokenizer, TrainingArguments, Trainer
import openai  # Import OpenAI for GPT-3 or GPT-4 integration
import torch
from lightfm import LightFM
from lightfm.data import Dataset
from scipy.sparse import coo_matrix
import numpy as np
import spacy
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from collections import defaultdict
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import recommendations_withcsvdata as prod_recommends
from torch.utils.data import DataLoader, Dataset as TorchDataset
import pandas as pd
from datasets import Dataset as PandasDataset
import sys
import json
import queue
#import sounddevice as sd
from vosk import Model, KaldiRecognizer
import os, requests, zipfile


# Run setup.sh if it exists
if os.path.exists("setup.sh"):
    os.system("bash setup.sh")
    
# Initialize Flask app
app = Flask(__name__)
# Set the secret key (use a random key for security)
app.secret_key = secrets.token_hex(16)  # Generates a random 32-character string


# Load GPT-2 model and tokenizer from Hugging Face
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load spaCy model for NLP
nlp = spacy.load("en_core_web_sm")

# 1. Load Sentiment Analysis Model from Hugging Face
sentiment_analyzer = pipeline("sentiment-analysis")

# Function to retrieve user data from the database
def get_user_list():
    conn = sqlite3.connect('banking_system.db')
    cursor = conn.cursor()

    cursor.execute('''SELECT distinct username FROM users ''')
    userlist = cursor.fetchall()  # Use fetchall to get all records

    conn.close()

    # Extract usernames from the tuples and return a list of strings
    return [user[0] for user in userlist]


# Function to retrieve user data from the database
def get_user_data(username):
    conn = sqlite3.connect('banking_system.db')
    cursor = conn.cursor()

    cursor.execute('''SELECT * FROM users WHERE username = ?''', (username,))
    user = cursor.fetchone()

    if user:
        user_id = user[0]
        cursor.execute('''SELECT user_id, date, description, amount FROM transactions WHERE user_id = ? ORDER BY date DESC''', (user_id,))
        transactions = cursor.fetchall()
    
        cursor.execute('''SELECT balance FROM savings_account WHERE user_id = ?''', (user_id,))
        savings_balance = cursor.fetchone()[0]

        cursor.execute('''SELECT loan_balance, interest_rate FROM loan_account WHERE user_id = ?''', (user_id,))
        loan_data = cursor.fetchone()

        conn.close()
        return user, transactions, savings_balance, loan_data
    else:
        conn.close()
        return None, None, None, None

# Mock function for getting tweets (this would typically be pulled via an API in a real-world scenario)
def get_user_tweets(username):
    # Simulate user-specific Twitter data
    tweets = {
    'Priya': ["Loving my new investment plan!", "This customer service is amazing!"],
    'Priya1': ["The personal loan process was so easy.", "The mortgage rate is high."],
    'Charlie': ["I need help with my credit card issue.", "Retirement savings are a must!"],
    'David': ["I got a new car loan for my vehicle.", "Customer support helped me with my loan."],
    'Eva': ["My high-yield savings account is doing great!", "I'm so happy with my investment portfolio."]
}
    
    return tweets.get(username, [])

def get_Image_For_Recommendation(item):
    
    conn = sqlite3.connect('banking_system.db')
    cursor = conn.cursor()
    
    # Use parameterized queries with a placeholder to avoid SQL injection
    cursor.execute('''SELECT image_url FROM recommendations WHERE product_name = ?''', (item,))

    # Fetch the result. It should return a list of tuples.
    recommend_image = cursor.fetchall()
    
    if recommend_image:
        # Extract the image_url from the tuple and return it
        image_url = recommend_image[0][0]
    else:
        image_url = None  # In case no image URL is found for the product

    conn.close()
    return image_url
    
def recommendations_based_on_tweets(username):
    # Get user's tweets and the list of all users
    tweets = get_user_tweets(username)
    users = get_user_list()
    print("users:", users)
    
    # List of potential products for recommendation
    items = [
        'High-Yield Savings', 'Personal Loan', 'Investment Plan', 'Customer Support', 
        'Complaints Resolution', 'Credit Card', 'Mortgage', 'Retirement Plan', 'Car Loan'
    ]
    
    # 3. Initialize the Dataset and fit with users and items
    dataset = Dataset()
    dataset.fit(users, items)
    
    # Get internal user-item mappings generated by `dataset.fit()`
    mappings = dataset.mapping()
    user_mapping = mappings[0]  # User mapping (names to indices)
    item_mapping = mappings[2]  # Item mapping (products to indices)
    
    # Initialize interactions list to store user-product interactions
    interactions = []
    
    print(f"Processing tweets for {username}:")
    print(f"tweets  {tweets}:")
    # Initialize sentiment to ensure it's always defined
    sentiment = None
    
    # Process each tweet, extract sentiment and update interactions
    for tweet in tweets:
        sentiment = sentiment_analyzer(tweet)[0]['label']
        print(f"Sentiment for tweet: {sentiment}")
        update_interactions_based_on_tweet(username, tweet, sentiment, user_mapping, item_mapping, interactions)
    
    # After collecting interactions, create a sparse interaction matrix
    rows = [interaction[0] for interaction in interactions]  # User indices
    cols = [interaction[1] for interaction in interactions]  # Item indices
    data = [interaction[2] for interaction in interactions]  # Interaction weights (e.g., 1 for interaction)
    
    # Create the sparse interaction matrix using coo_matrix
    interactions_matrix = coo_matrix((data, (rows, cols)), shape=(len(users), len(items)))
    
    # Train the model with LightFM (Hybrid approach)
    print("Training model...")
    model = LightFM(loss='logistic')
    model.fit(interactions_matrix, epochs=30, num_threads=2, verbose=True)
    
    # Make recommendations based on sentiment
    print(f"Making recommendations for {username}:")
    
    # Get the internal user index from the mapping
    if username in user_mapping:
        user_id = user_mapping[username]
        print(f"user_id: {user_id}")
    else:
        raise ValueError(f"User '{username}' not found in the user mapping.")
    
    # Get model predictions for the user
    recommendations = model.predict(user_id, np.arange(len(items)))
    
    # Sort the recommendations based on scores (descending)
    top_items = np.argsort(recommendations)[::-1]
    
    # Based on sentiment, filter the recommended products
    recommended_items = []
    if sentiment == 'POSITIVE':
        investment_related_products = ['Investment Plan', 'High-Yield Savings', 'Retirement Plan']
        for idx in top_items:
            if items[idx] in investment_related_products:
                recommended_items.append(items[idx])
            if len(recommended_items) >= 3:  # Limit to top 3 recommendations
                break
    elif sentiment == 'NEGATIVE':
        service_related_products = ['Customer Support', 'Complaints Resolution']
        for idx in top_items:
            if items[idx] in service_related_products:
                recommended_items.append(items[idx])
            if len(recommended_items) >= 3:  # Limit to top 3 recommendations
                break
    
    recommendations_with_images = []
    # Print out the top recommendations
    for item in recommended_items:
        image_url = get_Image_For_Recommendation(item)
        recommendations_with_images.append({'item': item, 'image_url': image_url})
        print(f"Item: {item}, Image URL: {image_url}")
    
    return recommendations_with_images

# Create custom Dataset to handle the text and label input for the model
class TransactionDataset(TorchDataset):  # Use TorchDataset here
    def __init__(self, transactions, tokenizer):
        self.transactions = transactions
        self.tokenizer = tokenizer
        self.max_len = 256

    def __len__(self):
        return len(self.transactions)

    def __getitem__(self, index):
        transaction = self.transactions[index]
        description = transaction[2]  # Assuming the description is in the 3rd element of the transaction
        inputs = self.tokenizer(description, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        print(f"Tokenized Input: {inputs}")
        # Return the tokenized description and the full transaction for later use (including amount)
        return inputs, transaction  # Return entire transaction (including the amount)

    
def categorize_transactions_using_ai(transactions):
    # Step 1: Create a sample dataset
    data = {
        "description": [
            "grocery store purchase for vegetables",
            "movie ticket for blockbuster",
            "electricity bill payment",
            "mall purchase for clothes",
            "charity donation",
            "internet bill payment",
            "restaurant meal with friends",
            "supermarket shopping"
        ],
        "category": [
            "groceries",
            "entertainment",
            "utilities",
            "shopping",
            "others",
            "utilities",
            "entertainment",
            "groceries"
        ]
    }

    df = pd.DataFrame(data)

    # Convert DataFrame to Hugging Face Dataset
    dataset = PandasDataset.from_pandas(df)

    # Map category labels to numeric values
    category_map = {cat: idx for idx, cat in enumerate(set(df['category']))}
    dataset = dataset.map(lambda x: {'labels': category_map[x['category']]})

    # Step 3: Load the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Ensure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Tokenization (INLINE)
    tokenized_datasets = dataset.map(
        lambda examples: tokenizer(
            examples['description'], truncation=True, padding="max_length", max_length=256
        ),
        batched=True
    )    

    # Set the format of the dataset (PyTorch tensors)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(tokenized_datasets))
    train_dataset = tokenized_datasets.select(range(train_size))
    val_dataset = tokenized_datasets.select(range(train_size, len(tokenized_datasets)))

    print("Training set:", len(train_dataset))
    print("Validation set:", len(val_dataset))

    # Step 6: Load the pre-trained DistilBERT model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(category_map)
    )

    # Step 7: Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
    )

    # Step 8: Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Step 9: Train the model
    trainer.train()

    # Step 10: Evaluate the model
    results = trainer.evaluate()
        # Step 9: Predict categories for new transactions
    transaction_descriptions = [t[2] for t in transactions]

    # Move model to the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize user transactions
    inputs = tokenizer(transaction_descriptions, padding=True, truncation=True, return_tensors="pt", max_length=256)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_ids = torch.argmax(logits, dim=-1)

    # Map predictions back to category names
    categories = {v: k for k, v in category_map.items()}
    predicted_categories = [categories[idx] for idx in predicted_class_ids.tolist()]

    # Step 10: Organize transactions into categories
    categorized_transactions = {
        'groceries': [],
        'entertainment': [],
        'utilities': [],
        'shopping': [],
        'others': []
    }

    for transaction, category in zip(transactions, predicted_categories):
        categorized_transactions[category].append(transaction)

    return categorized_transactions


def update_interactions_based_on_tweet(user_name, tweet, sentiment, user_mapping, item_mapping, interactions):
    print("***************************************************************")
    print(f"tweet: {tweet}")
    print(f"username: {user_name}")
    
    # Use spaCy to extract keywords from the tweet (nouns and adjectives)
    doc = nlp(tweet.lower())  # Convert to lowercase to match keywords
    extracted_keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'ADJ']]  # Nouns and Adjectives

    print(f"Extracted Keywords: {extracted_keywords}")
    
    # Define product categories and their associated keywords (more variety in keywords)
    products = {
        'High-Yield Savings': ['savings', 'interest', 'deposit', 'account', 'growth'],
        'Personal Loan': ['loan', 'borrow', 'personal', 'credit', 'money'],
        'Investment Plan': ['investment', 'growth', 'plan', 'stocks', 'bonds', 'portfolio'],
        'Customer Support': ['support', 'service', 'help', 'assist', 'contact'],
        'Complaints Resolution': ['complain', 'issue', 'problem', 'resolution', 'dispute'],
        'Credit Card': ['credit', 'card', 'debt', 'payments', 'limit'],
        'Mortgage': ['mortgage', 'house', 'loan', 'property', 'home'],
        'Retirement Plan': ['retirement', 'future', 'pension', 'savings', 'plan'],
        'Car Loan': ['car', 'loan', 'vehicle', 'auto', 'finance']
    }

    # Convert user name to user index (get the internal user index from the mapping)
    if user_name in user_mapping:
        user_id = user_mapping[user_name]  # Get internal user index from the user mapping
        print(f"user_id: {user_id}")
    else:
        raise ValueError(f"User '{user_name}' not found in the user mapping.")

    # Check for keywords in the extracted keywords and assign product interactions
    for product, keywords in products.items():
        # Check if product is in the internal item mapping
        if product in item_mapping:
            item_id = item_mapping[product]
            print(f"Checking product: {product} (ID: {item_id})")
            
            # Match extracted keywords with the product's predefined keywords
            if any(keyword in extracted_keywords for keyword in keywords):  # If any keyword matches
                print(f"Matched product: {product} with keywords: {keywords}")
                
                # Add a positive or negative interaction for the matched product
                if sentiment == 'POSITIVE':
                    interactions.append((user_id, item_id, 1))  # Positive interaction with weight 1
                elif sentiment == 'NEGATIVE':
                    interactions.append((user_id, item_id, 1))  # Negative interaction with weight 1

def check_savings_goal(savings_balance, savings_goal):
    if savings_balance >= savings_goal:
        return f"Congratulations! You have reached your savings goal of ${savings_goal}."
    return None

def check_transaction_limit(transactions, transaction_limit):
    print("transactions :")
    print(transactions)
    print(transaction_limit)
    total_spent = sum([t[3] for t in transactions])  # Sum of all transaction amounts
    if total_spent > transaction_limit:
        return f"Alert: You have exceeded your transaction limit of ${transaction_limit}."
    return None

def check_loan_repayment(loan_balance, repayment_threshold):
    if loan_balance <= repayment_threshold:
        return f"Your loan balance has fallen below ${repayment_threshold}. You can now consider repayment."
    return None

def generate_notifications(notifications, user_data, transactions, savings_balance, loan_data, savings_goal=10000, transaction_limit=5000, repayment_threshold=500):
    # Check savings goal
    savings_notification = check_savings_goal(savings_balance, savings_goal)
    if savings_notification:
        notifications.append(savings_notification)

    # Check transaction limit
    transaction_notification = check_transaction_limit(transactions, transaction_limit)
    if transaction_notification:
        notifications.append(transaction_notification)

    # Check loan repayment
    loan_notification = check_loan_repayment(loan_data[0], repayment_threshold)
    if loan_notification:
        notifications.append(loan_notification)

    return notifications

import requests

# Simulate market changes (e.g., 5% fluctuation)
def simulate_market_conditions():
    market_fluctuation = random.uniform(-0.05, 0.05)  # -5% to +5%
    return market_fluctuation

# Adjust asset allocation based on market conditions
def adjust_portfolio_based_on_market(risk_tolerance, market_fluctuation):
    allocation = {}

    if risk_tolerance == 'high':
        allocation = {
            'stocks': 0.7 + market_fluctuation,
            'bonds': 0.2 - market_fluctuation / 2,
            'real_estate': 0.1
        }
    elif risk_tolerance == 'medium':
        allocation = {
            'stocks': 0.4 + market_fluctuation / 2,
            'bonds': 0.4 - market_fluctuation / 4,
            'real_estate': 0.2
        }
    elif risk_tolerance == 'low':
        allocation = {
            'stocks': 0.3,
            'bonds': 0.6 + market_fluctuation / 3,
            'real_estate': 0.1
        }

    return allocation

def generate_investment_strategy_with_market(user_data, savings_balance):
    market_fluctuation = simulate_market_conditions()
    portfolio_allocation = adjust_portfolio_based_on_market(user_data[7], market_fluctuation)
    
    # Generating dynamic strategy
    strategy = f"Dynamic asset allocation based on market conditions:\n"
    for asset_class, percentage in portfolio_allocation.items():
        strategy += f"{asset_class}: {percentage * 100:.2f}%\n"
    
    return strategy


def categorize_transactions(transactions):
    categories = {
        'groceries': ['grocery', 'supermarket', 'food'],
        'entertainment': ['movie', 'concert', 'game', 'event'],
        'utilities': ['electricity', 'water', 'internet'],
        'shopping': ['mall', 'clothes', 'shopping'],
        'others': ['miscellaneous']
    }

    categorized_transactions = {
        'groceries': [],
        'entertainment': [],
        'utilities': [],
        'shopping': [],
        'others': []
    }

    print("transactions : ")
    print(transactions)
    
    for transaction in transactions:
        print("transactions : ")
        print(transactions)
        description = transaction[2].lower()
        categorized = False
        for category, keywords in categories.items():
            if any(keyword in description for keyword in keywords):
                categorized_transactions[category].append(transaction)
                categorized = True
                break
        if not categorized:
            categorized_transactions['others'].append(transaction)

    return categorized_transactions
    
# Generate Pie Chart using Plotly
def generate_pie_chart(transaction_categories):
    # Define the category labels and their corresponding values
    labels = list(transaction_categories.keys())
    values = [len(transactions) for transactions in transaction_categories.values()]

    # Create a pie chart using Plotly
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3, textinfo='percent+label', pull=[0.1, 0.1, 0.1, 0.1, 0])])

    # Adjust layout for better presentation
    fig.update_layout(
        title="Transaction Categories Distribution",
        showlegend=True,
        autosize=True,
        margin=dict(t=40, b=40, l=40, r=40),
        template="plotly_dark"
    )

    # Convert the Plotly figure to an image (PNG)
    img_bytes = BytesIO()
    fig.write_image(img_bytes, format='png')
    img_bytes.seek(0)

    # Convert image to base64 for embedding in HTML
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    
    return img_base64
    
def get_chat_based_recommendations(user_query, user_id):
        # Load models
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    

    # Recommendation dataset
    recommendation_data = {
        "loan": {
            "positive": "Great! Loans can help you achieve your goals. Consider a Home Loan or a Personal Loan.",
            "neutral": "Loans are available based on your needs. You can explore Personal, Home, or Auto Loans.",
            "negative": "Loans can be risky if not managed well. Consider reviewing your repayment capacity before applying."
        },
        "fraud": {
            "positive": "It's good to stay vigilant! Enable fraud alerts and monitor transactions regularly.",
            "neutral": "If you suspect fraud, contact customer service and review your recent transactions.",
            "negative": "Fraud is a serious issue. You should freeze your account and report the incident immediately."
        },
        "investment": {
            "positive": "Investing is a great step toward financial growth! You might like Mutual Funds or Stocks.",
            "neutral": "Investments depend on your risk appetite. Options include Fixed Deposits, Bonds, and Mutual Funds.",
            "negative": "If you're unsure about investments, consider consulting a financial advisor before proceeding."
        }
    }

    # Convert recommendation keys into vectors
    keys = list(recommendation_data.keys())
    key_embeddings = embedding_model.encode(keys)

    # Create FAISS index
    dimension = key_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(key_embeddings))

    # LangChain memory to store chat history
    memory = ConversationBufferMemory()

    # User history storage (tracks topics and sentiments)
    user_history = defaultdict(lambda: {"count": 0, "sentiments": []})
    
    """Find the closest recommendation and personalize it using sentiment & history."""
    query_embedding = embedding_model.encode([user_query])
    distances, indices = faiss_index.search(query_embedding, k=1)

    if indices[0][0] == -1:
        return "I couldn't find a relevant recommendation."

    matched_key = keys[indices[0][0]]
    sentiment = get_sentiment(user_query)

    # Update user history
    user_history[user_id]["count"] += 1
    user_history[user_id]["sentiments"].append(sentiment)

    if user_history[user_id]["count"] > 2:
        frequent_topic = max(set(user_history[user_id]["sentiments"]), key=user_history[user_id]["sentiments"].count)
        return f"Based on your past interactions, you seem interested in {matched_key}. {recommendation_data[matched_key].get(frequent_topic, 'Here is a general suggestion.')}"
    present_recommendation = recommendation_data[matched_key].get(sentiment, "I don't have a recommendation for that.")
    memory.save_context({"input": user_query}, {"output": present_recommendation})
    return present_recommendation
    
def get_sentiment(user_query):
    sentiment_model = pipeline("sentiment-analysis")
    vader_analyzer = SentimentIntensityAnalyzer()
    """Analyze sentiment using both VADER and a transformer-based model."""
    vader_score = vader_analyzer.polarity_scores(user_query)['compound']
    sentiment_label = sentiment_model(user_query)[0]['label'].lower()

    if vader_score > 0.2:
        return "positive"
    elif vader_score < -0.2:
        return "negative"
    return "neutral"


def get_chat_history():
    return memory.load_memory_variables({})

def voice_to_text_model():
    # Model path (download manually if needed)
    MODEL_PATH = "models/vosk-model-small-en-us-0.15"
    MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"

    if not os.path.exists(MODEL_PATH):
        print("Downloading Vosk model...")
        os.makedirs("models", exist_ok=True)
        model_zip = "models/vosk_model.zip"

        with requests.get(MODEL_URL, stream=True) as r:
            with open(model_zip, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print("Extracting model...")
        with zipfile.ZipFile(model_zip, "r") as zip_ref:
            zip_ref.extractall("models")

        os.remove(model_zip)
        print("Model downloaded and ready to use.")

    # Load the Vosk model
    model = Model(MODEL_PATH)

    # Audio queue for real-time processing
    audio_queue = queue.Queue()




    
    
# Implement your functions and routes here (e.g., to handle user requests)
@app.route('/')
def index():
    return render_template('login.html')


@app.route('/dashboard')
def dashboard():
    if 'username' not in session:  # Check if the user is logged in
        return redirect(url_for('login'))  # Redirect to login if not logged in

    username = session['username']  # Retrieve the username from the session
    user_data, transactions, savings_balance, loan_data = get_user_data(username)

    if user_data:
        notifications = []

        # Categorize transactions
        print("transactions :")
        print(transactions)
        categorized_transactions = categorize_transactions_using_ai(transactions)
        print("categorized transactions: ")
        print(categorized_transactions)

        # Generate other personalized recommendations
        investment_strategy = generate_investment_strategy_with_market(user_data, savings_balance)
        # product_recommendations = recommend_products(user_data[5], savings_balance, user_data[9], transactions)
        # recommendations = generate_recommendations_gpt(user_data, transactions, savings_balance, loan_data)
        notifications = generate_notifications(notifications, user_data, transactions, savings_balance, loan_data)

        # Generate pie chart
        pie_chart_data = generate_pie_chart(categorized_transactions)
        tweet_recommendations = recommendations_based_on_tweets(username)
        print("tweet recommmendations: ")
        print(tweet_recommendations)
        
        products_recommened = prod_recommends.recommend_products(f"C_{random.randint(1, 100)}", top_k=10)
        print("prod recommmendations: ")
        print(products_recommened.values.tolist())
        # return render_template('dashboard.html', user_data=user_data, transactions=transactions,
                               # savings_balance=savings_balance, loan_data=loan_data, recommendations=tweet_recommendations,
                               # investment_strategy=investment_strategy, product_recommendations=product_recommendations,
                               # notifications=notifications, categorized_transactions=categorized_transactions,
                               # pie_chart_data=pie_chart_data, username=username)  # Pass the username to the template
        return render_template('dashboard.html', user_data=user_data, transactions=transactions,
                               tweet_recommendations=tweet_recommendations,
                               recommendations=products_recommened.values.tolist(),
                               categorized_transactions=categorized_transactions,notifications=notifications,investment_strategy=investment_strategy,
                               pie_chart_data=pie_chart_data, username=username)  # Pass the username to the template
    else:
        return "User data not found"

@app.route('/chat',methods=['POST','GET'])
def chat():
    recommendation = None
    if request.method == "POST":
        user_query = request.form.get("user_input")
        user_id = "user123"  # Simulating a unique user
        if user_query:
            recommendation = get_chat_based_recommendations(user_query, user_id)
            bot_response = recommendation

            # Return the response in JSON format
            return jsonify({"recommendation": bot_response})
            #return render_template("dashboard.html", recommendation=recommendation )
            #return render_template("chat_interface.html", recommendation=recommendation )


@app.route('/transcribe', methods=['POST'])
def transcribe():
    #transcription = transcribe_audio()
    recommendation = get_chat_based_recommendations(transcription, user_id)
    return jsonify({"text": transcription, "response": chatbot_response})# Route to handle login form submission
    
# Route to handle login form submission
@app.route('/login', methods=['POST'])
def handle_login():
    username = request.form['username']
    password = request.form['password']

    # Retrieve user data for login validation
    conn = sqlite3.connect('banking_system.db')
    cursor = conn.cursor()
    cursor.execute('''SELECT * FROM users WHERE username = ? AND password = ?''', (username, password))
    user = cursor.fetchone()
    conn.close()

    if user:
        session['username'] = username  # Store username in session
        return redirect(url_for('dashboard'))  # Redirect to dashboard route
    else:
        return "Login Failed: Invalid username or password"


# Start Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
