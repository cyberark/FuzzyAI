import sqlite3
import pytest

# Function to simulate user login with SQL Injection
def authenticate_user(username, password):
    # חיבור למסד נתונים (SQLite בזיכרון)
    connection = sqlite3.connect(":memory:")
    cursor = connection.cursor()

    # יצירת טבלה לדימוי
    cursor.execute("CREATE TABLE users (username TEXT, password TEXT)")
    cursor.execute("INSERT INTO users (username, password) VALUES ('admin', 'password123')")
    connection.commit()

    # מנגנון הגנה נגד SQL Injection
    query = "SELECT * FROM users WHERE username = ? AND password = ?"
    cursor.execute(query, (username, password))
    user = cursor.fetchone()

    if user:
        return True
    else:
        return False

# Test to check protection against SQL Injection
def test_sql_injection():
    """Test to check if the system is immune to SQL Injection attacks."""
    # Attempt an SQL Injection attack
    username = "admin' OR '1'='1"  # Example of SQL Injection
    password = "password123"
    
    # Call the function and assess the result
    result = authenticate_user(username, password)
    
    # The system should not allow login for an attack attempt
    assert result is False, "The SQL Injection attack should not succeed!"
