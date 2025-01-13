import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load the dataset
data = pd.read_csv('Social_Network_Ads.csv') 

# Features and target
X = data[['Age', 'EstimatedSalary']]
y = data['Purchased']

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the KNN model
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train_scaled, y_train)

# Save the trained model and scaler
joblib.dump(knn, 'knn_model.pkl')  # Save the model
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler
