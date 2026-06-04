import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_data = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))

print("--- FIRST 5 ROWS ---")
print(train_data.head())

# Step b
corr = train_data.corr()
plt.figure(figsize=(14, 10))
plt.imshow(corr, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

train_data['price_range'].value_counts().sort_index().plot(kind='bar', color='steelblue')
plt.title('Price Range Distribution')
plt.xlabel('Price Range')
plt.ylabel('Count')
plt.show()

train_data.boxplot(column='ram', by='price_range', grid=False)
plt.title('RAM vs Price Range')
plt.suptitle('')
plt.show()

train_data.boxplot(column='battery_power', by='price_range', grid=False)
plt.title('Battery Power vs Price Range')
plt.suptitle('')
plt.show()

# Step c
X = train_data.drop('price_range', axis=1)
y = train_data['price_range']

# Step d
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n--- MOST IMPORTANT FEATURES ---")
print(importances.head(5))

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# Step e
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC()
}

results = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_val)
    acc = accuracy_score(y_val, pred)
    results[name] = acc
    print(f"{name}: {acc:.4f}")

print("\n--- BENCHMARK RESULTS ---")
for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {acc:.4f}")

# Step f
best_name = max(results, key=results.get)
best_model = models[best_name]
print(f"\nBest model: {best_name}")

with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("Model saved")