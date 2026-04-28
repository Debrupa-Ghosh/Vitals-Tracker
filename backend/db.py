"""
MySQL database connection helper and schema initialization.
"""

import mysql.connector
from mysql.connector import Error
from config import Config


def get_connection(use_db=True):
    """Get a MySQL connection. If use_db=False, connects without selecting a database."""
    params = {
        "host": Config.DB_HOST,
        "port": Config.DB_PORT,
        "user": Config.DB_USER,
        "password": Config.DB_PASSWORD,
    }
    if use_db:
        params["database"] = Config.DB_NAME
    return mysql.connector.connect(**params)


def init_db():
    """Create the database and all tables if they don't exist."""
    # Create database
    conn = get_connection(use_db=False)
    cursor = conn.cursor()
    cursor.execute(
        f"CREATE DATABASE IF NOT EXISTS `{Config.DB_NAME}` "
        "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    )
    cursor.close()
    conn.close()

    # Create tables
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            age INT NOT NULL,
            gender ENUM('male', 'female', 'other') NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS health_records (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            height FLOAT NOT NULL,
            weight FLOAT NOT NULL,
            bmi FLOAT NOT NULL,
            systolic INT NOT NULL,
            diastolic INT NOT NULL,
            cholesterol INT NOT NULL,
            glucose INT NOT NULL,
            smoking BOOLEAN DEFAULT FALSE,
            alcohol ENUM('none', 'light', 'moderate', 'heavy') DEFAULT 'none',
            exercise ENUM('sedentary', 'light', 'moderate', 'active') DEFAULT 'sedentary',
            diet ENUM('poor', 'average', 'balanced', 'excellent') DEFAULT 'average',
            bmi_category VARCHAR(30),
            bp_category VARCHAR(30),
            cholesterol_category VARCHAR(30),
            glucose_category VARCHAR(30),
            heart_risk_level ENUM('Low', 'Medium', 'High'),
            heart_risk_score FLOAT,
            overall_health_score INT,
            overall_health_grade VARCHAR(5),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS doctors (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            specialty VARCHAR(50) NOT NULL,
            hospital VARCHAR(150) NOT NULL,
            phone VARCHAR(20),
            rating FLOAT DEFAULT 4.0,
            distance_km FLOAT,
            available BOOLEAN DEFAULT TRUE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ui_config (
            id INT AUTO_INCREMENT PRIMARY KEY,
            config_key VARCHAR(50) UNIQUE NOT NULL,
            config_value JSON NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()
