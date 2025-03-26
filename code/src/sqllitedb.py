import sqlite3

# Create a connection to the SQLite database (it will create the database if it doesn't exist)
conn = sqlite3.connect('banking_system.db')
cursor = conn.cursor()

# Create tables for users, transactions, savings account, and loan account
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        password TEXT NOT NULL,
        email TEXT,
        income REAL,
        expenses REAL,
        savings_goal REAL,
        risk_tolerance TEXT,
        investment_goals TEXT,
        credit_score INTEGER
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        date TEXT,
        description TEXT,
        amount REAL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS savings_account (
        user_id INTEGER,
        balance REAL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS loan_account (
        user_id INTEGER,
        loan_balance REAL,
        interest_rate REAL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
''')

cursor.execute('''
    CREATE TABLE recommendations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_name TEXT NOT NULL,
        image_url TEXT NOT NULL,
        description TEXT NOT NULL
    )
''')

# Insert test data into the 'users' table
cursor.execute('''
    INSERT INTO users (username, password, email, income, expenses, savings_goal, risk_tolerance, investment_goals, credit_score)
    VALUES ('Priya', 'password123', 'priya@example.com', 6000, 2000, 10000, 'medium', 'Retirement planning', 720),
           ('Priya1', 'password456', 'priya1@example.com', 4500, 1500, 5000, 'high', 'Real estate investment', 690)
''')

# Insert test data into the 'transactions' table for  (user_id = 1)
cursor.execute('''
    INSERT INTO transactions (user_id, date, description, amount)
    VALUES (1, '2025-03-01', 'Grocery Store', -150),
           (1, '2025-03-02', 'Restaurant', -50),
           (1, '2025-03-03', 'Amazon Shopping', -200),
           (1, '2025-03-04', 'Salary', 6000),
           (1, '2025-03-05', 'Subscription Service', -20)
''')

# Insert test data into the 'transactions' table for  (user_id = 2)
cursor.execute('''
    INSERT INTO transactions (user_id, date, description, amount)
    VALUES (2, '2025-03-01', 'Movie Tickets', -30),
           (2, '2025-03-02', 'Dining Out', -60),
           (2, '2025-03-03', 'Salary', 4500),
           (2, '2025-03-04', 'Internet Bill', -40),
           (2, '2025-03-05', 'Shopping Mall', -120)
''')

# Insert test data into the 'savings_account' table
cursor.execute('''
    INSERT INTO savings_account (user_id, balance)
    VALUES (1, 8000),
           (2, 3000)
''')

# Insert test data into the 'loan_account' table
cursor.execute('''
    INSERT INTO loan_account (user_id, loan_balance, interest_rate)
    VALUES (1, 2500, 5.5),
           (2, 1500, 7.0)
''')

cursor.execute('''
        INSERT INTO recommendations (product_name, image_url, description)
        VALUES
        ('High-Yield Savings', 'https://cdn-icons-png.flaticon.com/512/586/586121.png', 'Earn higher interest rates with a high-yield savings account.'),
        ('Personal Loan', 'https://cdn-icons-png.flaticon.com/512/4630/4630893.png', 'Get funds quickly with a personal loan to cover your needs.'),
        ('Investment Plan', 'https://cdn-icons-png.flaticon.com/512/2974/2974977.png', 'Grow your wealth with a personalized investment plan.'),
        ('Customer Support', 'https://cdn-icons-png.flaticon.com/512/3515/3515192.png', 'Our customer service team is here to help you with all your queries.'),
        ('Complaints Resolution', 'https://cdn-icons-png.flaticon.com/512/1142/1142494.png', 'Resolve your issues with our efficient complaints resolution service.'),
        ('Credit Card', 'https://cdn-icons-png.flaticon.com/512/5968/5968764.png', 'Manage your expenses with a flexible and reliable credit card.'),
        ('Mortgage', 'https://cdn-icons-png.flaticon.com/512/2762/2762470.png', 'Get the best rates on mortgages for your dream home.'),
        ('Retirement Plan', 'https://cdn-icons-png.flaticon.com/512/1077/1077012.png', 'Save for a secure and comfortable retirement with our plans.'),
        ('Car Loan', 'https://cdn-icons-png.flaticon.com/512/1654/1654027.png', 'Drive your dream car with our flexible car loan options.');
''')

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Database and test data created successfully!")
