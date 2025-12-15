import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
df = pd.read_csv("c:/Users/hello/Downloads/Mobile Reviews Sentiment (1).csv")
print(df.head())
print("Shape:", df.shape)

# 3. EDA (basic)
print("\n HEAD ")
print(df.head())
print("\n INFO ")
print(df.info())
print("\n DESCRIBE (all) ")
print(df.describe(include="all"))

print("\n NULL VALUES ")
print(df.isnull().sum())

print("\n UNIQUE VALUES (sample) ")
print(df.nunique().head(20))

# handling null values
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
print("\nNulls After Cleaning:\n", df.isnull().sum())

# 5. QUICK VISUALIZATIONS (NO IF CONDITIONS)
sns.set(style="whitegrid")
# 1. Rating distribution
plt.figure(figsize=(6,4))
sns.countplot(x=df['rating'])
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

# 2. Average rating of top 12 brands
plt.figure(figsize=(10,4))
brand_avg = df.groupby('brand')['rating'].mean().sort_values(ascending=False).head(12)
sns.barplot(x=brand_avg.index, y=brand_avg.values)
plt.xticks(rotation=45)
plt.title("Top 12 Brands by Average Rating")
plt.ylabel("Average Rating")
plt.show()

# 3. Numeric correlations heatmap
plt.figure(figsize=(10,8))
num_df = df.select_dtypes(include=np.number)
sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap (Numeric Columns)")
plt.show()

# 4. Review length vs Rating
plt.figure(figsize=(6,4))
sns.scatterplot(x=df['review_length'], y=df['rating'], alpha=0.4)
plt.title("Review Length vs Rating")
plt.xlabel("Review Length")
plt.ylabel("Rating")
plt.show()

# 5. Price vs Rating
plt.figure(figsize=(6,4))
sns.scatterplot(x=df['price_usd'], y=df['rating'], alpha=0.4)
plt.title("Price (USD) vs Rating")
plt.xlabel("Price (USD)")
plt.ylabel("Rating")
plt.show()
# label encoding
le = LabelEncoder()
for c in cat_cols:
    if c != "review_text":
        df[c] = le.fit_transform(df[c].astype(str))
# standardization
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
print(df[num_cols])
# Define features manually (columns that exist in your dataset)
features = ["age", "price_usd", "review_length", "word_count", "helpful_votes"]

# Features (X) and target (y)
X = df[features]
y = df["rating"]
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Linear Regression
lin = LinearRegression()
lin.fit(X_train, y_train)

# Prediction
y_pred = lin.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Linear Regression RMSE:", rmse)
print("Linear Regression R2:", r2)
print("predicted value: ",y_pred)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted")
plt.show()

# polynomial regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train_p, y_train_p)
y_pred_poly = model.predict(X_test_p)
mse_poly = mean_squared_error(y_test_p, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
r2_poly = r2_score(y_test_p, y_pred_poly)
print("Polynomial RMSE:", rmse_poly)
print("Polynomial R2:", r2_poly)
print("polynomial predicted value: ",y_pred_poly)
plt.scatter(y_test_p, y_pred_poly)
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted (Polynomial Regression)")
plt.show()

# classification features (use same X)# Create helpfulness classes (Low, Medium, High)
# 1. Prepare price_usd target
df['price_usd'] = df['price_usd'].fillna(df['price_usd'].median())

q1 = df['price_usd'].quantile(0.33)
q2 = df['price_usd'].quantile(0.66)

price_class = []
for p in df['price_usd']:
    if p <= q1:
        price_class.append(0)
    elif p <= q2:
        price_class.append(1)
    else:
        price_class.append(2)

df['price_class'] = price_class

# 2. Choose features (use price_usd instead of helpful_votes; edit if needed)
features = ["age", "review_length", "word_count", "price_usd"]
features = [f for f in features if f in df.columns]

# 3. Prepare X and y
Xc = df[features].fillna(0).astype(float)
yc = df['price_class']

# 4. Scale features (simple)
scaler = StandardScaler()
Xc_scaled = scaler.fit_transform(Xc)

# 5. Train-test split
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc_scaled, yc, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(Xc_train, yc_train)
pred_dt = dt.predict(Xc_test)

print("Decision Tree Accuracy:", round(accuracy_score(yc_test, pred_dt), 4))
print("Decision Tree Confusion Matrix:\n", confusion_matrix(yc_test, pred_dt))
print("Decision Tree Classification Report:\n", classification_report(yc_test, pred_dt))

cm = confusion_matrix(yc_test, pred_dt)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 7. KNN (k=5)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(Xc_train, yc_train)
pred_knn = knn.predict(Xc_test)

print("KNN Accuracy:", round(accuracy_score(yc_test, pred_knn), 4))
print("KNN Confusion Matrix:\n", confusion_matrix(yc_test, pred_knn))
print("KNN Classification Report:\n", classification_report(yc_test, pred_knn))

cm = confusion_matrix(yc_test, pred_knn)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - KNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Simple GaussianNB + Confusion Matrix visualization

nb = GaussianNB()
nb.fit(Xc_train, yc_train)

pred_nb = nb.predict(Xc_test)

acc_nb = accuracy_score(yc_test, pred_nb)
print("GaussianNB Accuracy:", round(acc_nb, 4))
print(confusion_matrix(yc_test, pred_nb))
print(classification_report(yc_test, pred_nb))

# Simple confusion matrix plot

cm = confusion_matrix(yc_test, pred_nb)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - GaussianNB")
plt.show()


# PLOT CONFUSION MATRIX FOR BEST CLASSIFIER (choose highest accuracy)
# Accuracies of all classifiers

acc_dt = accuracy_score(yc_test, pred_dt)
acc_knn = accuracy_score(yc_test, pred_knn)
acc_nb = accuracy_score(yc_test, pred_nb)

print("DecisionTree Accuracy:", acc_dt)
print("KNN Accuracy:", acc_knn)
print("GaussianNB Accuracy:", acc_nb)

#  BEST CLASSIFIER
accs = {
    'DecisionTree': acc_dt,
    'KNN': acc_knn,
    'GaussianNB': acc_nb
}

best_name = max(accs, key=accs.get)
print("\nBest Classifier:", best_name)

# predictions of best classifier
preds = {
    'DecisionTree': pred_dt,
    'KNN': pred_knn,
    'GaussianNB': pred_nb
}[best_name]

# Visualization (Confusion Matrix)

cm = confusion_matrix(yc_test, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Best Classifier: " + best_name)
plt.show()

# Pick the best trained model
best_model = {
'DecisionTree': dt,
'KNN': knn,
'GaussianNB': nb}[best_name]
# Take real input from user
a = float(input("Enter age: "))
rl = float(input("Enter review length: "))
wc = float(input("Enter word count: "))
p = float(input("Enter price in USD: "))

# Create sample

sample = [[a, rl, wc, p]]

# Predict on real input 
pred = best_model.predict(sample)
print("Predicted class:", pred[0])
labels = ["Low Price", "Mid Price", "High Price"]
print("Meaning:", labels[pred[0]])
# 0 → Low Price
# 1 → Mid Price
# 2 → High Price
# print("Meaning:", labels[pred[0]])