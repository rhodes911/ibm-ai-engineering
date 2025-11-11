# Scikit-learn Machine Learning Ecosystem

**Date:** November 10, 2025  
**Module:** 1 - Introduction to Machine Learning  
**Topic:** The ML Ecosystem and Scikit-learn Library Deep Dive  

---

## Overview

This lesson explores the **machine learning ecosystem**—the interconnected tools, frameworks, libraries, platforms, and processes that support ML development—with a deep dive into **scikit-learn**, the most popular library for classical machine learning in Python.

### Learning Objectives
After this lesson, you will be able to:
- ✅ Describe the machine learning ecosystem and its components
- ✅ Explain the features of the scikit-learn library
- ✅ Understand how scikit-learn works within the Python ML ecosystem
- ✅ Implement a basic ML workflow using scikit-learn

---

## Motivating Example: Music Streaming App

### The Business Problem

**Scenario:** You developed a music streaming app with features like:
- Play and download music
- Share music files
- Create playlists

**Goal:** Increase user base by understanding and predicting user behavior

### The Data Collection

You collect information on users' **listening habits:**
- 🎵 What songs they play
- ⏱️ How long they listen to songs
- ⏭️ Which songs they skip
- 📋 Playlist creation patterns
- 🔁 Repeat listening behavior
- 📱 Device and time-of-day patterns

### The Challenge

Once you've collected this data, you need to:
1. **Normalize** the data
2. Find **inconsistent data**
3. Identify **missing values**
4. Detect **outliers**
5. **Build models** to predict user preferences
6. **Deploy** recommendations in production

> "Machine learning tools can generate this type of information."

This is where the ML ecosystem comes in!

---

## What is the Machine Learning Ecosystem?

### Definition

> "The machine learning ecosystem refers to the interconnected tools, frameworks, libraries, platforms, and processes that support developing, deploying, and managing machine learning models."

### Key Components

The ML ecosystem covers the **entire pipeline:**

```
1. Data Collection
   ↓
2. Preprocessing (cleaning, transformation)
   ↓
3. Model Training (algorithm selection, fitting)
   ↓
4. Model Evaluation (testing, validation)
   ↓
5. Model Deployment (production integration)
   ↓
6. Monitoring (performance tracking, retraining)
```

### Why It Matters

**Without the ecosystem:**
- You'd build everything from scratch
- Months of development for basic tasks
- Error-prone implementations
- No standard practices

**With the ecosystem:**
- ✅ Pre-built, tested components
- ✅ Hours instead of months
- ✅ Industry-standard implementations
- ✅ Community support and documentation

---

## The Python Machine Learning Ecosystem

### Why Python?

Python offers a **wide variety of tools and libraries** for machine learning, forming one of the most widely used ecosystems.

### The Foundational Stack

Several **open-source Python libraries** comprise the core ML ecosystem:

```
┌─────────────────────────────────────┐
│          Scikit-learn               │
│    (ML Models & Pipelines)          │
└─────────────┬───────────────────────┘
              │
    ┌─────────┴──────────┬─────────────┐
    │                    │             │
┌───▼────┐         ┌─────▼────┐   ┌───▼────────┐
│ SciPy  │         │Matplotlib│   │  Pandas    │
│Scientific        │Visualization   │Data Analysis
└───┬────┘         └─────┬────┘   └───┬────────┘
    │                    │            │
    └────────────┬───────┴────────────┘
                 │
            ┌────▼─────┐
            │  NumPy   │
            │Foundation│
            └──────────┘
```

---

## The Five Core Libraries

### 1. NumPy (Foundation)

**Purpose:** Foundational ML support

**Key Features:**
- Efficient numerical computations
- Large, multidimensional arrays
- Mathematical operations
- Linear algebra

**Example:**
```python
import numpy as np

# Create arrays (much faster than Python lists)
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Efficient operations
mean = np.mean(data)          # 5.0
std = np.std(data)            # 2.58
transposed = data.T           # Transpose matrix
dot_product = np.dot(data, data.T)  # Matrix multiplication
```

**Why it's foundational:**
- All other libraries build on NumPy
- Provides the array data structure
- C-optimized for performance

---

### 2. Pandas (Data Handling)

**Purpose:** Data analysis, visualization, cleaning, and preparation

**Built on:** NumPy and Matplotlib

**Key Innovation:** **DataFrames** - versatile arrays for handling data

**Example:**
```python
import pandas as pd

# Load music streaming data
df = pd.read_csv('user_listening_data.csv')

# Quick exploration
print(df.head())
print(df.describe())
print(df.info())

# Data cleaning
df = df.dropna()  # Remove missing values
df = df.drop_duplicates()  # Remove duplicates

# Find inconsistent data
outliers = df[df['listen_duration'] > 10000]  # Songs over 10k seconds?

# Feature engineering
df['skip_rate'] = df['songs_skipped'] / df['songs_played']
df['avg_session_length'] = df['total_listen_time'] / df['sessions']

# Aggregation
top_songs = df.groupby('song_id')['play_count'].sum().sort_values(ascending=False).head(10)
```

**Why it matters:**
- Makes data manipulation intuitive
- Handles missing data gracefully
- Perfect for exploratory data analysis

---

### 3. SciPy (Scientific Computing)

**Purpose:** Advanced scientific and technical computing

**Built on:** NumPy

**Key Modules:**
- **Optimization** - Minimize/maximize functions
- **Integration** - Numerical integration
- **Linear regression** - Statistical modeling
- **Interpolation** - Estimate between points
- **Signal processing** - FFT, filters
- **Statistics** - Distributions, hypothesis tests

**Example:**
```python
from scipy import stats, optimize

# Statistical test - Are heavy listeners and light listeners different?
heavy_listeners = df[df['hours_per_week'] > 10]['satisfaction_score']
light_listeners = df[df['hours_per_week'] <= 10]['satisfaction_score']

t_stat, p_value = stats.ttest_ind(heavy_listeners, light_listeners)
print(f"P-value: {p_value:.4f}")  # If < 0.05, significant difference

# Optimization - Find optimal recommendation threshold
def user_retention(threshold):
    # Return negative retention (minimize = maximize retention)
    predicted_retention = model.predict_retention(threshold)
    return -predicted_retention

result = optimize.minimize(user_retention, x0=0.5)
optimal_threshold = result.x[0]
```

---

### 4. Matplotlib (Visualization)

**Purpose:** Creating plots and visualizations

**Built on:** NumPy

**Key Features:**
- Extensive visualization types
- Highly customizable
- Publication-quality figures
- Interactive plots

**Example:**
```python
import matplotlib.pyplot as plt

# Visualize listening patterns
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Listen duration distribution
axes[0, 0].hist(df['listen_duration'], bins=50, edgecolor='black')
axes[0, 0].set_title('Distribution of Listen Duration')
axes[0, 0].set_xlabel('Seconds')

# Skip rate over time
axes[0, 1].plot(df['date'], df['skip_rate'])
axes[0, 1].set_title('Skip Rate Over Time')

# Genre popularity
genre_counts = df['genre'].value_counts()
axes[1, 0].bar(genre_counts.index, genre_counts.values)
axes[1, 0].set_title('Songs by Genre')
axes[1, 0].tick_params(axis='x', rotation=45)

# User engagement
axes[1, 1].scatter(df['sessions'], df['total_listen_time'], alpha=0.5)
axes[1, 1].set_title('Sessions vs Total Listen Time')
axes[1, 1].set_xlabel('Number of Sessions')
axes[1, 1].set_ylabel('Total Listen Time (minutes)')

plt.tight_layout()
plt.show()
```

---

### 5. Scikit-learn (Machine Learning)

**Purpose:** Building classical machine learning models

**Built on:** NumPy, SciPy, and Matplotlib

**The Star of the Show!** Let's dive deep...

---

## Scikit-learn: Deep Dive

### What is Scikit-learn?

> "Scikit-learn is a free machine learning library for the Python programming language."

### Key Characteristics

**1. Comprehensive Algorithm Selection**
- ✅ **Classification** - Categorize data
- ✅ **Regression** - Predict continuous values
- ✅ **Clustering** - Group similar data
- ✅ **Dimensionality Reduction** - Reduce features

**2. Production-Ready**
- Wide, up-to-date selection of algorithms
- Designed to integrate with NumPy and SciPy
- Battle-tested in industry

**3. Well-Supported**
- ✅ Excellent documentation
- ✅ Large community support network
- ✅ Constantly evolving
- ✅ Contributions from thousands of developers
- ✅ Second only to Pandas in popularity

**4. Easy to Use**
> "Implementing machine learning models with scikit-learn is easy, with just a few lines of Python code."

---

### What's Built Into Scikit-learn?

> "Most of the tasks that need to be done in a machine learning pipeline are already implemented in scikit-learn."

#### Data Preprocessing
- ✅ **Data cleaning** - Handle missing values
- ✅ **Scaling** - Standardize/normalize features
- ✅ **Feature selection** - Choose best features
- ✅ **Feature extraction** - Create new features
- ✅ **Train/test splitting** - Separate data for evaluation

#### Modeling
- ✅ **Model setup** - Initialize algorithms
- ✅ **Model fitting** - Train on data
- ✅ **Hyperparameter tuning** - Optimize parameters
- ✅ **Cross-validation** - Robust evaluation

#### Evaluation & Deployment
- ✅ **Prediction** - Generate outputs
- ✅ **Evaluation** - Measure performance
- ✅ **Exporting** - Save models for production

---

## Complete Scikit-learn Workflow Example

Let's walk through a **complete machine learning workflow** for our music streaming app, predicting whether users will churn (stop using the app).

### Step 0: Import and Load Data

```python
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, svm, metrics
import pickle

# Load data
df = pd.read_csv('user_data.csv')

# Separate features and target
X = df[['listen_hours', 'skip_rate', 'playlist_count', 'days_since_signup']].values
y = df['churned'].values  # 1 = churned, 0 = stayed

print(f"Dataset shape: {X.shape}")
print(f"Churn rate: {y.mean():.2%}")
```

---

### Step 1: Data Preprocessing - Scaling

**Why scale?** Different features have different ranges:
- `listen_hours`: 0-100
- `skip_rate`: 0-1
- `playlist_count`: 0-50
- `days_since_signup`: 0-1000

**Problem:** Models can be biased toward features with larger values.

**Solution:** Standardize (scale to mean=0, std=1)

```python
from sklearn.preprocessing import StandardScaler

# Create scaler
scaler = StandardScaler()

# Fit and transform training data
X_scaled = scaler.fit_transform(X)

print("Original data:")
print(X[:3])
print("\nScaled data:")
print(X_scaled[:3])
print("\nMean per feature:", X_scaled.mean(axis=0))  # ~[0, 0, 0, 0]
print("Std per feature:", X_scaled.std(axis=0))      # ~[1, 1, 1, 1]
```

**Output:**
```
Original data:
[[45.2  0.15  12  365]
 [12.5  0.65   3   45]
 [78.3  0.08  28  890]]

Scaled data:
[[ 0.23 -0.45  0.15  0.12]
 [-1.45  1.23 -0.89 -1.34]
 [ 1.67 -0.78  1.45  1.89]]
```

---

### Step 2: Train/Test Split

**Purpose:** Separate data for training and unbiased evaluation

> "In supervised learning, you want to split your dataset into train and test sets to train your model and then test the model's accuracy separately."

```python
from sklearn.model_selection import train_test_split

# Split: 67% training, 33% testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, 
    y, 
    test_size=0.33,  # 33% for testing
    random_state=42,  # Reproducibility
    stratify=y        # Maintain churn rate in both sets
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training churn rate: {y_train.mean():.2%}")
print(f"Test churn rate: {y_test.mean():.2%}")
```

**Why 33%?** Common practice:
- 70/30 split
- 80/20 split
- 67/33 split (shown here)

**Key:** Test set must remain unseen during training!

---

### Step 3: Model Instantiation

**Choose an algorithm:** Support Vector Classification (SVC)

> "You can instantiate a classifier model using a support vector classification algorithm."

```python
from sklearn.svm import SVC

# Create model object
clf = SVC(
    gamma=0.001,    # Kernel coefficient (hyperparameter)
    C=100.0         # Regularization parameter (hyperparameter)
)

print(f"Model: {clf}")
print(f"Hyperparameters: gamma={clf.gamma}, C={clf.C}")
```

**What just happened?**
- Created a model object called `clf`
- Initialized hyperparameters (`gamma` and `C`)
- Model is **not trained yet**—just configured

---

### Step 4: Model Training (Fitting)

**Train the model on training data:**

> "After initializing your model clf, you can train your model on the training data. The clf model learns to predict the classes for unknown cases by passing the training set to the fit method."

```python
# Train the model
clf.fit(X_train, y_train)

print("Model training complete!")
print(f"Number of support vectors: {clf.n_support_}")
```

**What happens during `.fit()`?**
1. Algorithm analyzes patterns in `X_train` and `y_train`
2. Adjusts internal parameters to minimize errors
3. Learns decision boundary between churned/stayed
4. Returns the trained model

**The Magic:** Complex math happens automatically!

---

### Step 5: Prediction

**Use the trained model on test data:**

> "Then you can use the test data to generate predictions. The result tells you the predicted class for each observation in the test set."

```python
# Generate predictions
y_pred = clf.predict(X_test)

print(f"Predictions: {y_pred[:20]}")  # First 20 predictions
print(f"Actual:      {y_test[:20]}")  # First 20 actual values

# Compare a few examples
comparison = pd.DataFrame({
    'Actual': y_test[:10],
    'Predicted': y_pred[:10],
    'Correct': y_test[:10] == y_pred[:10]
})
print(comparison)
```

**Output:**
```
   Actual  Predicted  Correct
0       1          1     True
1       0          0     True
2       0          1    False  ← Error
3       1          1     True
4       0          0     True
5       1          0    False  ← Error
6       0          0     True
7       1          1     True
8       0          0     True
9       1          1     True
```

---

### Step 6: Model Evaluation

**Measure performance with multiple metrics:**

> "You can also use different metrics to evaluate your model accuracy, such as a confusion matrix to compare the predicted and actual labels for the test set."

#### 6a. Accuracy Score

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")
# Example: Accuracy: 87.5%
```

#### 6b. Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Visualize
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stayed', 'Churned'])
disp.plot(cmap='Blues')
plt.title('Churn Prediction Confusion Matrix')
plt.show()
```

**Confusion Matrix Interpretation:**
```
                Predicted
               Stayed  Churned
Actual Stayed    [[850     50]
       Churned     [75    125]]

True Negatives (TN) = 850   ✓ Correctly predicted "stayed"
False Positives (FP) = 50   ✗ Predicted "churned" but stayed
False Negatives (FN) = 75   ✗ Predicted "stayed" but churned
True Positives (TP) = 125   ✓ Correctly predicted "churned"

Accuracy = (TN + TP) / Total = (850 + 125) / 1100 = 88.6%
```

#### 6c. Classification Report

```python
from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred, target_names=['Stayed', 'Churned'])
print(report)
```

**Output:**
```
              precision    recall  f1-score   support

      Stayed       0.92      0.94      0.93       900
     Churned       0.71      0.63      0.67       200

    accuracy                           0.89      1100
   macro avg       0.82      0.79      0.80      1100
weighted avg       0.88      0.89      0.88      1100
```

**Metrics Explained:**
- **Precision:** Of predicted churns, how many were correct?
- **Recall:** Of actual churns, how many did we catch?
- **F1-Score:** Harmonic mean of precision and recall

---

### Step 7: Save Model for Production

**Export the trained model:**

> "And finally, you can save your model as a pickle file and retrieve it whenever you like."

```python
import pickle

# Save model to file
with open('churn_model.pkl', 'wb') as file:
    pickle.dump(clf, file)

print("Model saved to churn_model.pkl")

# Later: Load model from file
with open('churn_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Use loaded model
new_user_data = scaler.transform([[30.5, 0.25, 8, 120]])
prediction = loaded_model.predict(new_user_data)
print(f"New user prediction: {'Churned' if prediction[0] == 1 else 'Stayed'}")
```

**Why save models?**
- ✅ Training takes time (minutes to hours)
- ✅ Production systems need pre-trained models
- ✅ Can version and track models
- ✅ Share with team or deploy to servers

---

## Complete Workflow Summary

```python
# 0. IMPORT & LOAD
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

X, y = load_data()  # Your features and target

# 1. PREPROCESSING - Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. SPLIT - 67% train, 33% test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.33, random_state=42
)

# 3. INSTANTIATE - Create model
clf = SVC(gamma=0.001, C=100.0)

# 4. TRAIN - Fit on training data
clf.fit(X_train, y_train)

# 5. PREDICT - Generate predictions
y_pred = clf.predict(X_test)

# 6. EVALUATE - Check accuracy
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")
print(f"Confusion Matrix:\n{cm}")

# 7. SAVE - Export for production
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)
```

**That's it!** A complete ML pipeline in ~20 lines of code.

---

## Scikit-learn Algorithm Categories

### 1. Classification Algorithms

**Purpose:** Predict categorical labels

**Available in scikit-learn:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
```

**Example Use Cases:**
- Email spam detection (spam/not spam)
- Customer churn prediction (churn/stay)
- Image classification (cat/dog/bird)
- Disease diagnosis (positive/negative)

---

### 2. Regression Algorithms

**Purpose:** Predict continuous numerical values

**Available in scikit-learn:**
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
```

**Example Use Cases:**
- House price prediction ($450,000)
- Stock price forecasting ($157.32)
- Temperature prediction (72.5°F)
- Sales forecasting (15,234 units)

---

### 3. Clustering Algorithms

**Purpose:** Group similar data points (unsupervised)

**Available in scikit-learn:**
```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
```

**Example Use Cases:**
- Customer segmentation (premium/budget/occasional)
- Document topic grouping
- Image compression
- Anomaly detection

---

### 4. Dimensionality Reduction

**Purpose:** Reduce number of features while preserving information

**Available in scikit-learn:**
```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
```

**Example Use Cases:**
- Reduce 1000 features to 50 (faster training)
- Visualize high-dimensional data in 2D/3D
- Remove correlated features
- Noise reduction

---

## Advanced Scikit-learn Features

### 1. Cross-Validation

**Problem:** Single train/test split might be lucky or unlucky

**Solution:** Multiple splits for robust evaluation

```python
from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
scores = cross_val_score(clf, X_scaled, y, cv=5)
print(f"Scores: {scores}")
print(f"Average: {scores.mean():.2%} (+/- {scores.std():.2%})")

# Output: Average: 87.3% (+/- 2.1%)
```

---

### 2. Hyperparameter Tuning

**Problem:** How do we know `gamma=0.001` and `C=100` are best?

**Solution:** Grid search to test many combinations

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}

# Grid search with cross-validation
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.2%}")

# Use best model
best_model = grid_search.best_estimator_
```

---

### 3. Pipelines

**Problem:** Many steps to remember (scale → split → train → predict)

**Solution:** Chain steps into a pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),      # Step 1: Scale
    ('classifier', SVC(gamma=0.001))   # Step 2: Classify
])

# Fit pipeline (automatically scales, then trains)
pipeline.fit(X_train, y_train)

# Predict (automatically scales new data, then predicts)
y_pred = pipeline.predict(X_test)
```

**Benefits:**
- ✅ Fewer lines of code
- ✅ Less error-prone
- ✅ Easy to save entire workflow
- ✅ Prevents data leakage

---

## Why Scikit-learn is So Popular

### 1. Consistency

**All models have the same interface:**
```python
model.fit(X_train, y_train)      # Train
model.predict(X_test)             # Predict
model.score(X_test, y_test)       # Evaluate
```

Easy to swap algorithms:
```python
# Try Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Try Random Forest
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Try SVM
clf = SVC()
clf.fit(X_train, y_train)
```

---

### 2. Documentation

**Excellent resources:**
- User guide with theory and examples
- API reference for every function
- Tutorials for common tasks
- Examples gallery with working code

**Example:** https://scikit-learn.org/stable/modules/svm.html

---

### 3. Community

**Large support network:**
- Thousands of contributors
- Active Stack Overflow community
- Regular updates and bug fixes
- Exceeded only by Pandas in popularity

---

### 4. Integration

**Works seamlessly with ecosystem:**
```python
import numpy as np              # Arrays
import pandas as pd             # DataFrames
from scipy import stats         # Statistics
import matplotlib.pyplot as plt # Visualization
from sklearn.ensemble import RandomForestClassifier  # ML
```

All libraries designed to work together!

---

## The Scikit-learn Cheat Sheet

### Common Tasks

| Task | Code |
|------|------|
| **Load data** | `X, y = load_data()` |
| **Split data** | `X_train, X_test, y_train, y_test = train_test_split(X, y)` |
| **Scale features** | `scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)` |
| **Train model** | `model.fit(X_train, y_train)` |
| **Predict** | `y_pred = model.predict(X_test)` |
| **Evaluate** | `accuracy_score(y_test, y_pred)` |
| **Cross-validate** | `cross_val_score(model, X, y, cv=5)` |
| **Tune parameters** | `GridSearchCV(model, param_grid, cv=5).fit(X, y)` |
| **Save model** | `pickle.dump(model, open('model.pkl', 'wb'))` |

---

## Key Takeaways

### 1. The ML Ecosystem is Interconnected
> "The machine learning ecosystem refers to the interconnected tools, frameworks, libraries, platforms, and processes that support developing, deploying, and managing machine learning models."

Everything works together: NumPy → SciPy → Pandas → Matplotlib → Scikit-learn

---

### 2. Python's ML Stack is Powerful

**Five core libraries:**
- **NumPy** - Numerical foundation
- **Pandas** - Data manipulation
- **SciPy** - Scientific computing
- **Matplotlib** - Visualization
- **Scikit-learn** - Machine learning

---

### 3. Scikit-learn is Comprehensive

> "Most tasks required in a machine learning pipeline are already implemented in scikit-learn."

**Includes:**
- Data preprocessing
- Model training
- Evaluation
- Hyperparameter tuning
- Cross-validation
- Model export

---

### 4. Easy to Use

> "Implementing machine learning models with scikit-learn is easy, with just a few lines of Python code."

**Basic workflow:**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

---

### 5. Production-Ready

- ✅ Battle-tested algorithms
- ✅ Efficient implementations
- ✅ Easy to save and deploy
- ✅ Well-documented
- ✅ Large community

---

## Study Questions

1. What is the machine learning ecosystem?
2. What are the five core Python libraries for ML, and how do they relate to each other?
3. What is the purpose of NumPy in the ML ecosystem?
4. What data structure does Pandas use to handle data?
5. What types of algorithms does scikit-learn provide?
6. What does the `.fit()` method do in scikit-learn?
7. What is the purpose of train/test splitting?
8. Why do we scale features before training?
9. What is a confusion matrix and what does it show?
10. How do you save a trained scikit-learn model?

---

## Practical Exercise

**Challenge:** Build a complete ML pipeline for the music streaming churn prediction

**Dataset features:**
- `listen_hours` - Hours listened per week
- `skip_rate` - Percentage of songs skipped
- `playlist_count` - Number of playlists created
- `days_since_signup` - Account age
- `churned` - Target variable (0=stayed, 1=churned)

**Tasks:**
1. Load and explore the data with Pandas
2. Visualize feature distributions with Matplotlib
3. Scale features with StandardScaler
4. Split into train/test (70/30)
5. Train three different models:
   - Logistic Regression
   - Random Forest
   - SVM
6. Compare accuracies
7. For the best model, show:
   - Confusion matrix
   - Classification report
8. Save the best model as a pickle file

**Bonus:**
- Use cross-validation for each model
- Try GridSearchCV to tune hyperparameters
- Create a Pipeline combining scaling and modeling

---

## Code Template

```python
# Import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pickle

# 1. Load data
df = pd.read_csv('music_users.csv')
X = df[['listen_hours', 'skip_rate', 'playlist_count', 'days_since_signup']].values
y = df['churned'].values

# 2. Explore
print(df.describe())
df.hist(bins=20, figsize=(12, 8))
plt.show()

# 3. Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# 5. Train multiple models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'SVM': SVC(gamma='auto')
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name}: {accuracy:.2%}")

# 6. Best model details
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print(f"\nBest model: {best_model_name}")
y_pred = best_model.predict(X_test)

# 7. Detailed evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Save
with open('best_churn_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("\nModel saved!")
```

---

*These notes are based on "Scikit-learn Machine Learning Ecosystem" from Module 1 of the IBM AI Engineering Professional Certificate course.*
