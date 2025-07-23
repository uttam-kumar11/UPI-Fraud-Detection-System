UPI Sentinel: Real-Time UPI Fraud Detection System
A real-time desktop dashboard that simulates a stream of UPI transactions and uses an anomaly detection model to identify and flag potentially fraudulent activity. 

🌟 Key Features
Live Simulation: Simulates a real-time feed of UPI transactions, providing a dynamic environment for fraud detection.

Machine Learning Model: Utilizes an IsolationForest anomaly detection model trained on user spending patterns to identify suspicious transactions.

Interactive Dashboard: A user-friendly interface built with PyQt5 that allows for real-time monitoring and analysis.

Explainable AI (XAI): When a fraudulent transaction is detected, the dashboard provides a simple, human-readable explanation for the alert (e.g., "Amount is 18810% higher than user's average").

Detailed Analytics: Click on any transaction to view detailed graphs of the user's historical spending and hourly activity patterns.

User Controls: Includes controls to pause, resume, reset the simulation, and adjust its speed for easier demonstration and analysis.

Fraud Log: A dedicated log that provides a clean, running list of all detected fraudulent transactions for a quick summary of suspicious activity.

🛠️ Technologies Used
Backend & Modeling:

Python

Pandas & NumPy (for data manipulation and synthetic data generation)

Scikit-learn (for the IsolationForest machine learning model)

Desktop Application (GUI):

PyQt5

Data Visualization:

Matplotlib (for embedding dynamic graphs in the application)

🚀 How to Run This Project
Clone the repository:

git clone https://github.com/uttam-kumar11/UPI-Fraud-Detection-System.git

Navigate to the project directory:

cd UPI-Fraud-Detection-System

Create and activate a Conda environment:

conda create --name upi_sentinel python=3.9
conda activate upi_sentinel

Install the required libraries:

pip install pandas numpy scikit-learn joblib matplotlib pyqt5

Run the application:

python app.py
