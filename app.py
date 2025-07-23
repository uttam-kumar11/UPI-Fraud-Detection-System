# --- Imports for the Application ---
import sys
import pandas as pd
import joblib
import numpy as np
import traceback # Import traceback for detailed error printing
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QListWidget, 
                             QListWidgetItem, QWidget, QPushButton, QSlider, 
                             QHBoxLayout, QVBoxLayout, QGridLayout, QFrame,
                             QStyle, QStatusBar)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor, QFont, QIcon

# --- Matplotlib Imports for Graphing ---
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# This class defines our main application window
class UpiSentinelApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UPI Sentinel")
        self.setGeometry(100, 100, 1200, 800)
        
        self.is_paused = False
        self.simulation_speed = 1000

        self.load_assets()
        self.init_ui()
        self.apply_stylesheet()
        self.start_simulation()

    def load_assets(self):
        """Loads the dataset and the trained model."""
        try:
            self.model = joblib.load('fraud_model.joblib')
            self.data = pd.read_csv('upi_transactions.csv')
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            self.data['hour_of_day'] = self.data['timestamp'].dt.hour
            self.data['original_index'] = self.data.index
        except FileNotFoundError as e:
            print(f"Error loading assets: {e}.")
            sys.exit()

    def init_ui(self):
        """Initializes all the user interface elements."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15) # Add spacing between main sections
        main_layout.setContentsMargins(15, 15, 15, 15)

        # --- Header ---
        header_label = QLabel("UPI Sentinel – Fraud Detection Dashboard")
        header_label.setObjectName("HeaderLabel")
        main_layout.addWidget(header_label)

        # --- Controls Panel (Top) ---
        controls_frame = QFrame()
        controls_frame.setObjectName("CardFrame")
        controls_layout = QHBoxLayout(controls_frame)
        
        play_icon = self.style().standardIcon(QStyle.SP_MediaPlay)
        pause_icon = self.style().standardIcon(QStyle.SP_MediaPause)
        reset_icon = self.style().standardIcon(QStyle.SP_MediaSeekBackward)

        self.pause_button = QPushButton("Pause")
        self.pause_button.setIcon(pause_icon)
        self.pause_button.clicked.connect(self.toggle_pause)
        
        self.reset_button = QPushButton("Reset")
        self.reset_button.setIcon(reset_icon)
        self.reset_button.clicked.connect(self.reset_simulation)
        
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(0, 2); self.speed_slider.setValue(1); self.speed_slider.setFixedWidth(200)
        self.speed_slider.valueChanged.connect(self.change_speed)
        
        controls_layout.addWidget(self.pause_button); controls_layout.addWidget(self.reset_button)
        controls_layout.addWidget(QLabel("Speed:")); controls_layout.addWidget(self.speed_slider)
        controls_layout.addStretch()
        main_layout.addWidget(controls_frame)

        # --- Main Content Area (Split Left/Right) ---
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)
        main_layout.addLayout(content_layout)

        # --- Left Side: Live Feed ---
        left_frame = QFrame()
        left_frame.setObjectName("CardFrame")
        left_panel = QVBoxLayout(left_frame)
        self.live_feed_label = QLabel("Live Transaction Feed")
        left_panel.addWidget(self.live_feed_label)
        self.transaction_list = QListWidget()
        self.transaction_list.itemClicked.connect(self.update_dashboard)
        left_panel.addWidget(self.transaction_list)
        content_layout.addWidget(left_frame, 1)

        # --- Right Side: Dashboard Panel ---
        right_frame = QFrame()
        right_frame.setObjectName("CardFrame")
        right_panel = QVBoxLayout(right_frame)
        self.dashboard_label = QLabel("User Analytics Dashboard")
        right_panel.addWidget(self.dashboard_label)
        
        self.fraud_alert_frame = QFrame()
        self.fraud_alert_frame.setObjectName("FraudAlertFrame")
        self.fraud_alert_frame.setVisible(False)
        fraud_alert_layout = QVBoxLayout(self.fraud_alert_frame)
        self.fraud_alert_label = QLabel("!! FRAUD ALERT !!")
        self.fraud_reason_label = QLabel("Reason: N/A")
        fraud_alert_layout.addWidget(self.fraud_alert_label); fraud_alert_layout.addWidget(self.fraud_reason_label)
        right_panel.addWidget(self.fraud_alert_frame)

        info_frame = QFrame()
        info_frame.setObjectName("InfoFrame")
        info_grid = QGridLayout(info_frame) # Set the layout for the frame
        self.user_id_label = QLabel("User ID: N/A")
        self.avg_txn_label = QLabel("Avg. Normal Txn: N/A")
        self.avg_txn_label.setToolTip("Average of transactions not flagged as fraud.")
        self.total_txn_label = QLabel("Total Transactions: N/A")
        info_grid.addWidget(self.user_id_label, 0, 0); info_grid.addWidget(self.avg_txn_label, 0, 1)
        info_grid.addWidget(self.total_txn_label, 1, 0)
        right_panel.addWidget(info_frame) # BUG FIX: Add the frame, not the layout

        self.history_canvas = FigureCanvas(Figure(figsize=(5, 2.5)))
        self.hourly_canvas = FigureCanvas(Figure(figsize=(5, 2.5)))
        right_panel.addWidget(self.history_canvas); right_panel.addWidget(self.hourly_canvas)
        
        right_panel.addWidget(QLabel("Detected Fraud Log"))
        self.fraud_log_list = QListWidget()
        self.fraud_log_list.setMaximumHeight(120)
        right_panel.addWidget(self.fraud_log_list)
        
        content_layout.addWidget(right_frame, 1)

        self.setStatusBar(QStatusBar(self))

    def apply_stylesheet(self):
        """Applies a dark theme stylesheet to the application."""
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1E1E2F; }
            QLabel { font-size: 14px; color: #A0A4B8; background: transparent; }
            QLabel#HeaderLabel { font-size: 24px; font-weight: bold; color: #FFFFFF; padding-bottom: 10px; }
            QFrame#CardFrame { background-color: #2C2F48; border: 1px solid #3A3D5A; border-radius: 8px; padding: 10px; }
            QPushButton { background-color: #3B82F6; color: white; border: none; padding: 8px 16px; border-radius: 4px; font-size: 14px; font-weight: bold; }
            QPushButton:hover { background-color: #2563EB; }
            QListWidget { background-color: #1E1E2F; border: 1px solid #3A3D5A; border-radius: 4px; font-size: 14px; color: #A0A4B8; }
            QFrame#InfoFrame { background-color: transparent; border: none; }
            QFrame#FraudAlertFrame { background-color: #FF4C4C; border-radius: 4px; }
            QFrame#FraudAlertFrame QLabel { color: white; font-size: 16px; font-weight: bold; background: transparent; }
            QSlider::groove:horizontal { border: 1px solid #3A3D5A; background: #1E1E2F; height: 8px; border-radius: 4px; }
            QSlider::handle:horizontal { background: #3B82F6; border: 1px solid #3B82F6; width: 18px; margin: -2px 0; border-radius: 9px; }
            QStatusBar { background-color: #2C2F48; color: #A0A4B8; font-weight: bold; }
            QToolTip { background-color: #1E1E2F; color: #FFFFFF; border: 1px solid #3A3D5A; }
        """)
        bold_font = QFont(); bold_font.setBold(True); bold_font.setPointSize(16)
        title_color = "#FFFFFF"
        self.live_feed_label.setFont(bold_font); self.live_feed_label.setStyleSheet(f"color: {title_color}; background: transparent;")
        self.dashboard_label.setFont(bold_font); self.dashboard_label.setStyleSheet(f"color: {title_color}; background: transparent;")

    def start_simulation(self):
        self.transaction_index = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_next_transaction)
        self.statusBar().showMessage("Simulation Started.", 3000)
        self.timer.start(self.simulation_speed)

    def process_next_transaction(self):
        if self.transaction_index >= len(self.data):
            self.timer.stop(); self.statusBar().showMessage("End of Simulation.", 5000); return
            
        transaction = self.data.iloc[self.transaction_index]
        features_for_model = transaction[['amount', 'hour_of_day']].values.reshape(1, -1)
        prediction = self.model.predict(features_for_model)
        is_predicted_fraud = (prediction[0] == -1)
        
        display_text = f"Time: {transaction['timestamp'].strftime('%Y-%m-%d %H:%M')} | User: {transaction['user_id']} | Amount: ₹{transaction['amount']:.2f}"
        list_item = QListWidgetItem(display_text)
        list_item.setData(Qt.UserRole, transaction['original_index']); list_item.setData(Qt.UserRole + 1, is_predicted_fraud)
        
        if is_predicted_fraud:
            list_item.setBackground(QColor('#FF4C4C')); list_item.setForeground(QColor('white'))
            self.fraud_log_list.insertItem(0, display_text)
            self.statusBar().showMessage(f"Fraud Alert! User: {transaction['user_id']}, Amount: ₹{transaction['amount']:.2f}", 5000)
            
        self.transaction_list.insertItem(0, list_item)
        self.transaction_index += 1

    def update_dashboard(self, item):
        try:
            original_idx = item.data(Qt.UserRole); is_predicted_fraud = item.data(Qt.UserRole + 1)
            selected_transaction = self.data.loc[original_idx]; user_id = selected_transaction['user_id']
            user_data = self.data[self.data['user_id'] == user_id]
            
            if is_predicted_fraud:
                self.fraud_alert_frame.setVisible(True); self.explain_fraud(selected_transaction, user_data)
            else:
                self.fraud_alert_frame.setVisible(False)

            avg_amount = user_data[user_data['is_fraud'] == 0]['amount'].mean()
            total_txns = len(user_data)
            self.user_id_label.setText(f"<b>User ID:</b> {user_id}")
            
            if pd.isna(avg_amount):
                self.avg_txn_label.setText("<b>Avg. Normal Txn:</b> N/A")
            else:
                self.avg_txn_label.setText(f"<b>Avg. Normal Txn:</b> ₹{avg_amount:.2f}")

            self.total_txn_label.setText(f"<b>Total Transactions:</b> {total_txns}")
            self.statusBar().showMessage(f"Displaying analytics for User: {user_id}", 3000)

            self.plot_history(user_data, selected_transaction, is_predicted_fraud)
            self.plot_hourly_activity(user_data)
        
        except Exception as e:
            print("--- AN ERROR OCCURRED IN DASHBOARD UPDATE ---")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {e}")
            traceback.print_exc()
            print("-------------------------------------------")


    def explain_fraud(self, fraud_txn, user_data):
        normal_avg = user_data[user_data['is_fraud'] == 0]['amount'].mean()
        
        if pd.isna(normal_avg):
            self.fraud_reason_label.setText("Reason: No normal transaction history exists for this user.")
            return

        if fraud_txn['amount'] > normal_avg * 5:
            percentage_increase = (fraud_txn['amount'] / normal_avg - 1) * 100
            self.fraud_reason_label.setText(f"Reason: Amount is {percentage_increase:.0f}% higher than user's average.")
            return
        if 1 <= fraud_txn['hour_of_day'] <= 6:
            self.fraud_reason_label.setText(f"Reason: Transaction occurred at an unusual time ({fraud_txn['hour_of_day']}:00).")
            return
        self.fraud_reason_label.setText("Reason: Transaction pattern deviates from user's normal behavior.")

    def plot_history(self, user_data, selected_transaction, is_predicted_fraud):
        fig = self.history_canvas.figure
        fig.clear(); fig.patch.set_facecolor('#2C2F48')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#2C2F48')
        ax.plot(user_data['timestamp'], user_data['amount'], marker='o', linestyle='-', color='#3B82F6', label='Normal Txns')
        if is_predicted_fraud:
            ax.plot(selected_transaction['timestamp'], selected_transaction['amount'], 'o', color='#FF4C4C', markersize=10, label='Predicted Fraud')
        else:
            ax.plot(selected_transaction['timestamp'], selected_transaction['amount'], 'o', color='#10B981', markersize=10, label='Selected Txn')
        ax.set_title("Transaction History", color='#FFFFFF'); ax.set_xlabel("Date", color='#A0A4B8'); ax.set_ylabel("Amount (₹)", color='#A0A4B8')
        ax.tick_params(axis='x', colors='#A0A4B8', rotation=45); ax.tick_params(axis='y', colors='#A0A4B8')
        ax.spines['bottom'].set_color('#3A3D5A'); ax.spines['left'].set_color('#3A3D5A')
        ax.spines['top'].set_color('none'); ax.spines['right'].set_color('none')
        ax.legend(facecolor='#2C2F48', edgecolor='#3A3D5A', labelcolor='#A0A4B8')
        fig.tight_layout(); self.history_canvas.draw()

    def plot_hourly_activity(self, user_data):
        fig = self.hourly_canvas.figure
        fig.clear(); fig.patch.set_facecolor('#2C2F48')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#2C2F48')
        hourly_counts = user_data['hour_of_day'].value_counts().sort_index()
        ax.bar(hourly_counts.index, hourly_counts.values, color='#3B82F6')
        ax.set_title("Hourly Activity Pattern", color='#FFFFFF')
        ax.set_xlabel("Hour of Day (24h)", color='#A0A4B8'); ax.set_ylabel("Number of Transactions", color='#A0A4B8')
        ax.tick_params(axis='x', colors='#A0A4B8'); ax.tick_params(axis='y', colors='#A0A4B8')
        ax.spines['bottom'].set_color('#3A3D5A'); ax.spines['left'].set_color('#3A3D5A')
        ax.spines['top'].set_color('none'); ax.spines['right'].set_color('none')
        ax.grid(axis='y', color='#3A3D5A', linestyle='--', alpha=0.7)
        fig.tight_layout(); self.hourly_canvas.draw()

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        play_icon = self.style().standardIcon(QStyle.SP_MediaPlay)
        pause_icon = self.style().standardIcon(QStyle.SP_MediaPause)
        if self.is_paused:
            self.timer.stop(); self.pause_button.setText("Resume"); self.pause_button.setIcon(play_icon)
            self.statusBar().showMessage("Simulation Paused.", 3000)
        else:
            self.timer.start(self.simulation_speed); self.pause_button.setText("Pause"); self.pause_button.setIcon(pause_icon)
            self.statusBar().showMessage("Simulation Resumed.", 3000)

    def reset_simulation(self):
        self.timer.stop(); self.transaction_list.clear(); self.fraud_log_list.clear()
        self.transaction_index = 0; self.is_paused = False
        self.pause_button.setText("Pause"); self.pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.statusBar().showMessage("Simulation Reset.", 3000)
        self.timer.start(self.simulation_speed)

    def change_speed(self):
        speeds = [200, 1000, 2500]
        self.simulation_speed = speeds[self.speed_slider.value()]
        if not self.is_paused: self.timer.start(self.simulation_speed)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = UpiSentinelApp()
    mainWindow.show()
    sys.exit(app.exec_())
