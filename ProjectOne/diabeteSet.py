from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, send
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import seaborn as sns
from io import StringIO
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import requests
import json

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config["UPLOAD_FOLDER"] = "uploads"
socketio = SocketIO(app)

if not os.path.exists("uploads"):
    os.makedirs("uploads")

#if not os.path.exists("static"):
    os.makedirs("static")


def ask_ollama(prompt, model="mistral"):
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "stream": True
        },
        stream=True
    )

    full_response = ""
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                full_response += data.get("message", {}).get("content", "")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return full_response
    

@socketio.on("message")
def handle_message(msg):
    print(f"Message: {msg}")
    upload_folder = app.config["UPLOAD_FOLDER"]
    uploaded_files = sorted(
        [f for f in os.listdir(upload_folder) if f.endswith('.csv')],
        key=lambda f: os.path.getmtime(os.path.join(upload_folder, f)),
        reverse=True
    )
    if uploaded_files:
        latest_file_path = os.path.join(upload_folder, uploaded_files[0])
        try:
            df = pd.read_csv(latest_file_path)
            sample_data = df.head(3).to_string(index=False)
            context = f"The user uploaded this diabetes data sample:\n{sample_data}\n\nUser says: {msg}"
        except Exception as e:
            print(f"Error reading file: {e}")
            context = f"(Could not read uploaded file)\nUser says: {msg}"
    else:
        context = f"(No uploaded file found)\nUser says: {msg}"

    reply = ask_ollama(context)
    print(f"Bot: {reply}")
    send({"from": "user", "text": msg})
    send({"from": "bot", "text": reply})


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        print("No file selected")
        return jsonify({"error": "No file selected"}), 400
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)
    print(f"File saved to {file_path}")
    analyzer = DiabetesAnalyzer(data_source=file_path)
    analyzer.full_analysis()
    file_size = round(os.path.getsize(file_path) / 1024, 2)  # KB
    file_type = file.content_type
    return jsonify({
        "filename": file.filename,
        "size": f"{file_size} KB",
        "type": file_type
    })

@app.route("/")
def index():
    correlation_exists = os.path.exists('ProjectOne/static/correlation_matrix.png')
    distributions_exists = os.path.exists('ProjectOne/static/distributions.png')
    outliers_exists = os.path.exists('ProjectOne/static/outliers.png')
    missing_exists = os.path.exists('ProjectOne/static/missing.png')
    pca_exists = os.path.exists('ProjectOne/static/pca.png')
    return render_template("index.html",
                         correlation_exists=correlation_exists,
                         distributions_exists=distributions_exists,
                         outliers_exists=outliers_exists,
                         missing_exists=missing_exists, pca_exists=pca_exists)

class DiabetesAnalyzer:
    def __init__(self, data_source=None, user_data=None):
        if data_source is not None:
            self.df = pd.read_csv(data_source)
        elif user_data is not None:
            if isinstance(user_data, pd.DataFrame):
                self.df = user_data
            else:
                self.df = pd.read_csv(StringIO(user_data))
        else:
            raise ValueError("Either data_source or user_data must be provided")   
        self.clean_data()
    
    def clean_data(self):
        zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        zero_columns = [col for col in zero_columns if col in self.df.columns]
        self.df[zero_columns] = self.df[zero_columns].replace(0, np.nan)
        cat_columns = self.df.select_dtypes(include=['object']).columns  
        for col in cat_columns:
            
            if len(self.df[col].unique()) <= 10: 
                self.df[col] = LabelEncoder().fit_transform(self.df[col])
            else:
                self.df = pd.get_dummies(self.df, columns=[col])  

    
    def analyze_missing_values(self):
        print("\nMissing values analysis:")
        print("="*40) 
        missing_values = self.df.isnull().sum()
        total_missing = missing_values.sum()   
        print("Missing values per column:")
        print(missing_values)
        print("\nTotal missing values:", total_missing)
        
        if total_missing > 0:
            plt.figure(figsize=(10, 6))
            sns.heatmap(self.df.isnull(), cbar=False, cmap='viridis')
            plt.title("Missing Values Heatmap")
            plt.savefig('ProjectOne/static/missing.png')
            plt.close()
            missing_percent = (self.df.isnull().mean() * 100).round(2)
            print("\nMissing values percentage:")
            print(missing_percent)
        else:
            print("\nNo missing values found in the dataset.")
        return missing_values
    
    def show_basic_info(self):
        print("\nDataset information:")
        print("="*40)
        print(self.df.info())
        print("\nDescriptive statistics:")
        print(self.df.describe())
    
    def plot_correlation_matrix(self):
        plt.figure(figsize=(12, 8))
        correlation_matrix = self.df.corr().round(2)
        sns.heatmap(data=correlation_matrix, annot=True, cmap="YlGnBu", 
                   vmin=-1, vmax=1, center=0)
        plt.title("Feature Correlation Matrix")
        plt.savefig('ProjectOne/static/correlation_matrix.png')
        plt.close()
    
    def plot_distributions(self):
        print("\nPlotting feature distributions...")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        n_cols = len(numeric_cols)
        n_rows = int(np.ceil(n_cols / 3))
        plt.figure(figsize=(15, 5 * n_rows))
        for i, col in enumerate(numeric_cols, 1):
            plt.subplot(n_rows, 3, i)
            sns.histplot(self.df[col], kde=True, bins=20)
            plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.savefig('ProjectOne/static/distributions.png')
        plt.close()
    
    def plot_outliers(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        n_cols = len(numeric_cols)
        n_rows = int(np.ceil(n_cols / 4))
        plt.figure(figsize=(15, 4 * n_rows))
        for i, col in enumerate(numeric_cols, 1):
            plt.subplot(n_rows, 4, i)
            sns.boxplot(y=self.df[col])
            plt.title(f'Boxplot of {col}')
        plt.tight_layout()
        plt.savefig('ProjectOne/static/outliers.png')
        plt.close()
    
    def full_analysis(self):
        self.show_basic_info()
        self.analyze_missing_values()
        self.plot_distributions()
        self.plot_outliers()
        self.plot_correlation_matrix()

    def preprocess_data(self):
        print("\nDropping rows with missing values...")
        self.df = self.df.dropna()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df.select_dtypes(include=[np.number]))
        return scaled_data

    def classification_logistic_regression(self):
        print("\nPerforming Logistic Regression...")
        X = self.df.drop("Outcome", axis=1)
        y = self.df["Outcome"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        print(f"Logistic Regression Accuracy: {accuracy}")
    
    def classification_random_forest(self):
        print("\nPerforming Random Forest Classification...")
        X = self.df.drop("Outcome", axis=1)
        y = self.df["Outcome"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        print(f"Random Forest Accuracy: {accuracy}")
    
    def classification_svm(self):
        print("\nPerforming SVM Classification...")
        X = self.df.drop("Outcome", axis=1)
        y = self.df["Outcome"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = SVC()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        print(f"SVM Accuracy: {accuracy}")
    
    def clustering_kmeans(self):
        print("\nPerforming K-Means Clustering...")
        X = self.df.drop("Outcome", axis=1)
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(X)
        print(f"K-Means Cluster labels: {kmeans.labels_}")
    
    def pca_reduction(self):
        print("\nPerforming PCA...")
        X = self.df.drop("Outcome", axis=1)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        print(f"New shape after PCA: {X_pca.shape}")
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.df["Outcome"], cmap='viridis')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA of Diabetes dataset')
        plt.colorbar()
        plt.savefig('ProjectOne/static/pca.png')
        plt.close()

if __name__ == "__main__":
    analyzer = DiabetesAnalyzer(data_source='C:/ai_data_analysis/py/ProjectOne/diabetes.csv')
    analyzer.preprocess_data()
    analyzer.classification_logistic_regression()
    analyzer.classification_random_forest()
    analyzer.classification_svm()
    analyzer.clustering_kmeans()
    analyzer.pca_reduction()
    

    socketio.run(app, debug=False)

