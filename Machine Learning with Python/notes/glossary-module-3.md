# Module 3 Glossary: Building Supervised Learning Models

## Supervised Learning Overview

**Supervised Learning**
A type of machine learning where models are trained on labeled data, meaning each training example has an input and a known output (target).

**Label**
The known output or target value associated with each training example in supervised learning.

**Labeled Data**
A dataset where each observation includes both input features and the corresponding correct output.

**Training Set**
The portion of labeled data used to train (fit) the model.

**Test Set**
The portion of labeled data held out to evaluate the trained model's performance on unseen data.

## Classification Concepts

**Binary Classification**
Classification tasks with exactly two possible output classes (e.g., spam/not spam, malignant/benign).

**Multiclass Classification**
Classification tasks with more than two possible output classes (e.g., classifying images into dog, cat, bird, horse).

**One-vs-Rest (OvR) / One-vs-All (OvA)**
A strategy for multiclass classification where one binary classifier is trained for each class versus all other classes.

**One-vs-One (OvO)**
A strategy for multiclass classification where a binary classifier is trained for every pair of classes.

**Multi-label Classification**
Classification where each instance can be assigned multiple labels simultaneously (e.g., a movie can be both "action" and "comedy").

**Class Imbalance**
A situation where the number of observations in different classes is significantly unequal, which can bias model training.

## Decision Trees

**Decision Tree**
A tree-structured model that makes predictions by learning a series of if-then-else decision rules from the training data.

**Node**
A point in a decision tree representing a decision or a leaf (terminal node).

**Root Node**
The topmost node in a decision tree, representing the entire dataset before any splits.

**Internal Node**
A node in a decision tree that represents a decision based on a feature, leading to further branches.

**Leaf Node (Terminal Node)**
An end node in a decision tree that provides the final prediction (class label or numerical value).

**Branch**
A connection between nodes in a decision tree, representing the outcome of a decision.

**Split**
The process of dividing a node into two or more sub-nodes based on a feature and threshold.

**Splitting Criterion**
The rule or metric used to decide how to split a node (e.g., Gini impurity, entropy, information gain).

**Gini Impurity**
A metric measuring the probability of incorrectly classifying a randomly chosen element, used as a splitting criterion. Lower is better. Formula: $\text{Gini} = 1 - \sum_{i=1}^{C} p_i^2$

**Entropy**
A measure of impurity or disorder in a dataset, used in decision trees to quantify information content. Formula: $H = -\sum_{i=1}^{C} p_i \log_2(p_i)$

**Information Gain**
The reduction in entropy achieved by splitting on a particular feature, used to select the best split.

**Pruning**
The process of removing branches from a decision tree to reduce complexity and prevent overfitting.

**Pre-pruning (Early Stopping)**
Stopping tree growth early based on criteria like maximum depth or minimum samples per leaf.

**Post-pruning**
Growing a full tree and then removing branches that provide little improvement in performance.

**Tree Depth**
The length of the longest path from the root node to a leaf node.

**Maximum Depth**
A hyperparameter limiting how deep a decision tree can grow.

**Minimum Samples Split**
The minimum number of samples required to split an internal node.

**Minimum Samples Leaf**
The minimum number of samples required to be at a leaf node.

## Regression Trees

**Regression Tree**
A decision tree used for predicting continuous numerical values instead of class labels.

**Mean Squared Error (for Trees)**
The splitting criterion commonly used in regression trees, measuring the variance of target values in each node.

**Variance Reduction**
The decrease in variance achieved by a split in a regression tree, used to determine the best split.

## K-Nearest Neighbors (KNN)

**K-Nearest Neighbors (KNN)**
A non-parametric, instance-based learning algorithm that classifies data points based on the classes of their k nearest neighbors.

**Instance-Based Learning**
Learning approaches that store training examples and make predictions by comparing new instances to stored examples.

**Lazy Learning**
Algorithms that delay processing until prediction time rather than learning a model during training (e.g., KNN).

**k (in KNN)**
The hyperparameter specifying the number of nearest neighbors to consider when making a prediction.

**Distance Metric**
A function measuring the similarity or dissimilarity between data points (e.g., Euclidean distance, Manhattan distance).

**Euclidean Distance**
The straight-line distance between two points in space. Formula: $d = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$

**Manhattan Distance (L1 Distance)**
The sum of absolute differences between coordinates. Formula: $d = \sum_{i=1}^{n}|x_i - y_i|$

**Minkowski Distance**
A generalized distance metric that includes Euclidean and Manhattan distances as special cases.

**Majority Voting**
The process in KNN classification where the predicted class is the most common class among the k nearest neighbors.

**Weighted Voting**
A variant of majority voting where closer neighbors have more influence on the prediction.

**Curse of Dimensionality**
The phenomenon where algorithm performance degrades as the number of features increases, particularly affecting distance-based methods like KNN.

## Support Vector Machines (SVM)

**Support Vector Machine (SVM)**
A powerful supervised learning algorithm that finds the optimal hyperplane to separate classes or predict values.

**Hyperplane**
A decision boundary that separates classes in feature space. In 2D it's a line; in 3D it's a plane; in higher dimensions it's a hyperplane.

**Support Vectors**
The data points that lie closest to the decision boundary and influence the position and orientation of the hyperplane.

**Margin**
The distance between the hyperplane and the nearest data points from each class. SVM aims to maximize this margin.

**Hard Margin**
An SVM variant that requires all training points to be correctly classified with no exceptions (only works for linearly separable data).

**Soft Margin**
An SVM variant that allows some misclassifications, controlled by a penalty parameter C, to handle non-linearly separable data.

**C Parameter**
A regularization hyperparameter in SVM controlling the trade-off between maximizing margin and minimizing classification errors.

**Kernel**
A function that transforms data into a higher-dimensional space, allowing SVM to find non-linear decision boundaries.

**Kernel Trick**
A mathematical technique allowing SVM to operate in high-dimensional spaces without explicitly computing the transformation.

**Linear Kernel**
A kernel function that results in a linear decision boundary, equivalent to standard SVM.

**Polynomial Kernel**
A kernel function that can model polynomial relationships of a specified degree.

**Radial Basis Function (RBF) Kernel / Gaussian Kernel**
A popular kernel that can model complex non-linear relationships, controlled by a gamma parameter.

**Gamma Parameter**
A hyperparameter in RBF kernel controlling the influence of individual training examples. High gamma means closer points have more influence.

## Ensemble Methods (Introduction)

**Ensemble Learning**
Combining multiple models to create a more powerful predictive model than any individual model alone.

**Bagging (Bootstrap Aggregating)**
An ensemble method that trains multiple models on different random subsets of the training data and aggregates their predictions.

**Boosting**
An ensemble method that trains models sequentially, with each new model focusing on correcting errors made by previous models.

**Random Forest**
An ensemble of decision trees trained using bagging and random feature selection, typically providing better generalization than single trees.

**Voting**
The process of combining predictions from multiple models, either by majority vote (classification) or averaging (regression).

## Bias-Variance Tradeoff

**Bias**
The error introduced by approximating a complex real-world problem with a simplified model. High bias leads to underfitting.

**Variance**
The amount by which model predictions would change if trained on different data. High variance leads to overfitting.

**Bias-Variance Tradeoff**
The fundamental tension in machine learning: reducing bias typically increases variance and vice versa. The goal is finding the right balance.

**Underfitting**
When a model is too simple (high bias) to capture the underlying patterns in the data, performing poorly on both training and test data.

**Overfitting**
When a model is too complex (high variance) and learns noise in the training data, performing well on training data but poorly on test data.

**Model Complexity**
The capacity of a model to learn intricate patterns. More complex models can capture detailed relationships but risk overfitting.

**Generalization Error**
The expected error on new, unseen data, decomposed into bias, variance, and irreducible error.

**Irreducible Error**
The noise inherent in the data that cannot be reduced by any model, representing the lower bound on prediction error.

## Model Selection and Complexity

**Model Selection**
The process of choosing between different types of models or different configurations of the same model type.

**Hyperparameter Tuning**
The process of finding optimal hyperparameter values to improve model performance (covered more in Module 5).

**Training Error**
The error a model makes on the data it was trained on, typically lower than test error.

**Test Error**
The error a model makes on held-out test data, providing an estimate of generalization performance.

**Validation Set**
A separate dataset used during model development to tune hyperparameters without using the test set.

**Learning Curve**
A plot showing model performance (error) as a function of training set size or training iterations, useful for diagnosing bias/variance issues.

**Capacity**
A model's ability to fit a variety of functions, related to its complexity.

## Feature Engineering (Revisited)

**Feature Scaling**
Transforming features to a similar scale, important for distance-based algorithms like KNN and SVM.

**Standardization (Z-score Normalization)**
Scaling features to have zero mean and unit variance. Formula: $z = \frac{x - \mu}{\sigma}$

**Normalization (Min-Max Scaling)**
Scaling features to a fixed range, typically [0, 1]. Formula: $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$

**Categorical Encoding**
Converting categorical variables into numerical format for use in machine learning models.

**One-Hot Encoding**
Creating binary dummy variables for each category in a categorical feature.

**Label Encoding**
Assigning a unique integer to each category in a categorical feature.

## Implementation Terms

**DecisionTreeClassifier**
The scikit-learn class for building decision tree classifiers.

**DecisionTreeRegressor**
The scikit-learn class for building decision tree regressors.

**KNeighborsClassifier**
The scikit-learn class implementing K-Nearest Neighbors for classification.

**KNeighborsRegressor**
The scikit-learn class implementing K-Nearest Neighbors for regression.

**SVC (Support Vector Classifier)**
The scikit-learn class implementing Support Vector Machine for classification.

**SVR (Support Vector Regressor)**
The scikit-learn class implementing Support Vector Machine for regression.

**RandomForestClassifier**
The scikit-learn class implementing Random Forest for classification.

**RandomForestRegressor**
The scikit-learn class implementing Random Forest for regression.

**fit() method**
Trains the model on the provided training data.

**predict() method**
Makes predictions using the trained model.

**score() method**
Returns a performance metric (accuracy for classification, R² for regression) on given data.

---

*Last updated: November 10, 2025*
