# 🚀 Project Name

## 📌 Table of Contents
- [Introduction](#introduction)
- [Demo](#demo)
- [Inspiration](#inspiration)
- [What It Does](#what-it-does)
- [How We Built It](#how-we-built-it)
- [Challenges We Faced](#challenges-we-faced)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Team](#team)

---

## 🎯 Introduction
 A brief overview of your project and its purpose. Mention which problem statement are your attempting to solve. Keep it concise and engaging.
 Modern customers expect highly personalized experiences that cater to their unique preferences

## 🎥 Demo
🔗 [Live Demo](#demo) (if applicable)  
📹 [Video Demo](#demo) (if applicable)  
🖼️ Screenshots:

![Screenshot 1](link-to-image)

## 💡 Inspiration
Our approach to the problem statement is to consider bank customers, who will be availing one or more of the bank products and how we can personalize their experience and provide recommendations which optimizes their engagement. 

## ⚙️ What It Does
 - 🔹 Categorize based on users transaction history and provide recommendations using AI
 - 🔹 Alert notification based on transaction limit
 - 🔹 Provide product recommendation to the user using clustering model
 - 🔹 Provide recommendation based on social media data for the user
 - 🔹 A chat bot which provided recommendations based on key in text or uploaded voice message
 - 🔹 Investment Strategy using market trends

## 🛠️ How We Built It
Flask-based web application with functionalities for financial transaction management, sentiment analysis, AI-driven recommendations, and user interaction handling. Key functionalities include:
User Authentication: Login and session management.
Database Operations: Fetching user data, transactions, savings, and loan details from DB.
Sentiment Analysis: Extracting sentiments from tweets and using them for financial product recommendations.
AI-Powered Recommendations:
Personalized product recommendations based on tweets (using LightFM).
Chat-based recommendations using Sentence-BERT and FAISS.
Transaction categorization using DistilBERT.
Financial Alerts & Strategy: Notifications based on savings goals, transaction limits, and loan repayments.
Visualization: Transaction category pie chart with Plotly.![image](https://github.com/user-attachments/assets/df51a46f-c475-4e62-b4e9-3795dd3ce90d)


## 🚧 Challenges We Faced
- Open AI provided relevant recommendations but its services can be availed through paid subscription
- Dependency resolution for different models while creating docker image


## 🏃 How to Run
1. Clone the repository  
   ```sh
   git clone  https://github.com/ewfx/aidhp-smart-reconcilers.git
   ```
2. Install dependencies  
   ```sh
    pip install -r requirements.txt (for Python)
   ```
3. Run the project  
   ```sh
    python app.py
   ```

## 🏗️ Tech Stack
- 🔹 Frontend: HTML
- 🔹 Backend: Python
- 🔹 Database: SqlLightDB
- 🔹 Other: Flask / Docker

## 👥 Team
- **aidhp-smart-reconcilers** - [GitHub](#) | [LinkedIn](#)
