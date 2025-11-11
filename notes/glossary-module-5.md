# Module 5 Glossary: Evaluating and Validating Machine Learning Models

## Model Evaluation Overview

**Model Evaluation**
The process of assessing how well a trained model performs on data, using various metrics and techniques.

**Performance Metric**
A quantitative measure used to evaluate how well a model is performing on a task.

**Evaluation Metric**
A specific formula or calculation used to measure model performance (e.g., accuracy, precision, F1-score).

**Unseen Data**
Data that was not used during model training, used to test generalization performance.

**Hold-out Set**
A portion of data set aside and not used for training, reserved for evaluation.

**Training Performance**
How well a model performs on the data it was trained on.

**Test Performance**
How well a model performs on held-out test data, indicating generalization ability.

**Generalization**
A model's ability to perform well on new, unseen data beyond the training set.

## Classification Metrics

**Accuracy**
The proportion of correct predictions (both positive and negative) out of all predictions. Formula: $\frac{TP + TN}{TP + TN + FP + FN}$

**Error Rate**
The proportion of incorrect predictions. Formula: $1 - \text{Accuracy}$ or $\frac{FP + FN}{TP + TN + FP + FN}$

**Confusion Matrix**
A table showing true positives, true negatives, false positives, and false negatives for classification problems.

**True Positive (TP)**
Instances correctly predicted as positive.

**True Negative (TN)**
Instances correctly predicted as negative.

**False Positive (FP)**
Instances incorrectly predicted as positive (Type I error).

**False Negative (FN)**
Instances incorrectly predicted as negative (Type II error).

**Precision (Positive Predictive Value)**
The proportion of positive predictions that are actually correct. Formula: $\frac{TP}{TP + FP}$

**Recall (Sensitivity, True Positive Rate)**
The proportion of actual positive instances that are correctly identified. Formula: $\frac{TP}{TP + FN}$

**Specificity (True Negative Rate)**
The proportion of actual negative instances that are correctly identified. Formula: $\frac{TN}{TN + FP}$

**F1-Score**
The harmonic mean of precision and recall, providing a single metric that balances both. Formula: $F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

**F-Beta Score**
A generalized F-score that allows weighting precision and recall differently using parameter β.

**Macro-Average**
Computing a metric independently for each class and then taking the unweighted mean (treats all classes equally).

**Micro-Average**
Computing a metric globally by counting total true positives, false negatives, and false positives across all classes.

**Weighted Average**
Computing a metric for each class and taking the average weighted by the number of instances in each class.

**ROC Curve (Receiver Operating Characteristic)**
A plot showing the trade-off between true positive rate (recall) and false positive rate at various classification thresholds.

**AUC (Area Under the ROC Curve)**
A single number summarizing ROC curve performance, ranging from 0 to 1 (higher is better). AUC = 0.5 means random guessing.

**Precision-Recall Curve**
A plot showing the trade-off between precision and recall at various classification thresholds, useful for imbalanced datasets.

**Average Precision**
The weighted mean of precisions at each threshold in the precision-recall curve, summarizing the curve with a single value.

**Classification Report**
A summary showing precision, recall, F1-score, and support for each class in a classification problem.

**Support**
The number of actual occurrences of each class in the dataset.

**Matthews Correlation Coefficient (MCC)**
A balanced metric for binary classification that accounts for all four confusion matrix values, ranging from -1 to 1.

**Cohen's Kappa**
A statistic measuring inter-rater agreement for categorical items, adjusted for chance agreement.

**Log Loss (Logarithmic Loss)**
A metric that penalizes confident wrong predictions more severely, measuring the probability estimates' quality.

## Regression Metrics

**Mean Absolute Error (MAE)**
The average of absolute differences between predicted and actual values. Formula: $\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$

**Mean Squared Error (MSE)**
The average of squared differences between predicted and actual values. Formula: $\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$

**Root Mean Squared Error (RMSE)**
The square root of MSE, in the same units as the target variable. Formula: $\text{RMSE} = \sqrt{\text{MSE}}$

**Mean Absolute Percentage Error (MAPE)**
The average of absolute percentage errors, useful when you want to understand relative error magnitude.

**R-squared (R² / Coefficient of Determination)**
The proportion of variance in the dependent variable explained by the model. Formula: $R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$

**Adjusted R-squared**
R² adjusted for the number of predictors, penalizing unnecessary complexity.

**Residual**
The difference between actual and predicted values for a single observation.

**Residual Plot**
A scatter plot of residuals vs. predicted values or features, used to diagnose model problems.

## Cross-Validation

**Cross-Validation**
A resampling technique that uses different portions of data to train and test a model on different iterations, providing a more robust performance estimate.

**K-Fold Cross-Validation**
Dividing data into k equal folds, training on k-1 folds and testing on the remaining fold, repeated k times.

**Fold**
One of the k equal-sized partitions in k-fold cross-validation.

**Stratified K-Fold Cross-Validation**
K-fold cross-validation that preserves the percentage of samples for each class in each fold.

**Leave-One-Out Cross-Validation (LOOCV)**
An extreme case of k-fold where k equals the number of samples (each fold contains one sample).

**Holdout Validation**
Simply splitting data into train and test sets once (not technically cross-validation).

**Validation Curve**
A plot showing model performance vs. a hyperparameter value, used to assess impact of hyperparameter choices.

**Cross-Validation Score**
The performance metric averaged across all folds in cross-validation.

**Cross-Validation Standard Deviation**
The variation in performance across different folds, indicating model stability.

## Hyperparameter Tuning

**Hyperparameter**
A configuration setting for a learning algorithm that is set before training and controls the learning process.

**Hyperparameter Tuning (Hyperparameter Optimization)**
The process of finding optimal hyperparameter values to maximize model performance.

**Parameter**
An internal variable learned from training data (e.g., weights), as opposed to hyperparameters.

**Grid Search**
An exhaustive search over a manually specified subset of hyperparameter values, trying all combinations.

**Random Search**
Sampling hyperparameter values randomly from specified distributions, often more efficient than grid search.

**Search Space**
The range or set of possible values for each hyperparameter being tuned.

**GridSearchCV**
A scikit-learn class that performs grid search with cross-validation.

**RandomizedSearchCV**
A scikit-learn class that performs random search with cross-validation.

**Best Estimator**
The model configuration (hyperparameters) that achieved the best cross-validation score during tuning.

**Best Parameters**
The specific hyperparameter values of the best estimator.

**Best Score**
The best cross-validation score achieved during hyperparameter tuning.

**Nested Cross-Validation**
Using cross-validation both for hyperparameter tuning (inner loop) and for model evaluation (outer loop) to get unbiased performance estimates.

**Bayesian Optimization**
An advanced hyperparameter tuning method that builds a probabilistic model of the objective function to guide the search efficiently.

## Overfitting Prevention

**Overfitting**
When a model learns training data too well, including noise, resulting in poor generalization to new data.

**Underfitting**
When a model is too simple to capture the underlying patterns in the data.

**Train-Test Split**
Dividing data into separate training and testing subsets to evaluate generalization.

**Regularization**
Techniques that constrain or penalize model complexity to prevent overfitting.

**Early Stopping**
Stopping training when performance on validation data starts to degrade, preventing overfitting in iterative algorithms.

**Data Augmentation**
Creating additional training examples through transformations, helping models generalize better.

**Dropout**
A regularization technique (mainly for neural networks) that randomly drops units during training.

**Ensemble Methods**
Combining multiple models to reduce overfitting and improve generalization.

## Regularization Techniques

**Regularization**
Adding a penalty term to the loss function to discourage complex models and prevent overfitting.

**L1 Regularization (Lasso)**
Regularization that adds a penalty proportional to the absolute value of coefficients, can shrink some coefficients to exactly zero (feature selection).

**L2 Regularization (Ridge)**
Regularization that adds a penalty proportional to the square of coefficients, shrinks all coefficients but rarely to zero.

**Elastic Net**
A combination of L1 and L2 regularization, controlled by two hyperparameters.

**Regularization Parameter (λ or α)**
The hyperparameter controlling the strength of the regularization penalty. Higher values mean more regularization.

**Ridge Regression**
Linear regression with L2 regularization, useful when features are correlated.

**Lasso Regression**
Linear regression with L1 regularization, performs feature selection by shrinking some coefficients to zero.

**Alpha (α)**
The regularization strength parameter in scikit-learn's Ridge, Lasso, and ElasticNet.

**Feature Selection (via Regularization)**
Using L1 regularization to automatically select important features by shrinking unimportant ones to zero.

**Coefficient Shrinkage**
The effect of regularization reducing the magnitude of model coefficients.

## Handling Special Cases

**Imbalanced Classes**
When the number of instances in different classes is highly unequal, potentially biasing the model toward the majority class.

**Class Weights**
Assigning different importance to different classes during training to handle imbalanced data.

**SMOTE (Synthetic Minority Over-sampling Technique)**
A technique for handling imbalanced data by creating synthetic examples of the minority class.

**Undersampling**
Reducing the number of majority class examples to balance class distribution.

**Oversampling**
Increasing the number of minority class examples to balance class distribution.

**Outliers**
Data points that differ significantly from other observations, which can negatively impact model performance.

**Robust Regression**
Regression techniques less sensitive to outliers than ordinary least squares.

**Winsorization**
Limiting extreme values in data by replacing outliers with less extreme values.

**Anomaly Detection**
Identifying data points that deviate significantly from the norm, useful for outlier detection.

## Model Selection

**Model Comparison**
Evaluating multiple models using the same metrics to choose the best performer.

**Baseline Model**
A simple model used as a reference point to evaluate whether more complex models provide meaningful improvements.

**Statistical Significance**
Determining whether differences in model performance are meaningful or due to random chance.

**Occam's Razor**
The principle that simpler models should be preferred when they perform similarly to complex models.

**No Free Lunch Theorem**
The principle that no single machine learning algorithm works best for all problems.

**Learning Curve**
A plot showing model performance (e.g., error) vs. training set size, useful for diagnosing underfitting/overfitting.

**Validation Curve**
A plot showing model performance vs. a hyperparameter value, used to diagnose optimal hyperparameter settings.

## Implementation Terms

**train_test_split()**
A scikit-learn function to split data into training and testing sets.

**cross_val_score()**
A scikit-learn function to evaluate a model using cross-validation.

**cross_validate()**
A scikit-learn function that performs cross-validation and returns multiple metrics.

**GridSearchCV**
Exhaustive search over specified parameter values with cross-validation.

**RandomizedSearchCV**
Randomized search over parameter distributions with cross-validation.

**scoring parameter**
Specifies which metric to use for evaluation in scikit-learn functions.

**cv parameter**
Specifies the cross-validation splitting strategy (e.g., cv=5 for 5-fold).

**classification_report()**
A scikit-learn function generating a text report of classification metrics.

**confusion_matrix()**
A scikit-learn function computing the confusion matrix.

**roc_curve()**
A scikit-learn function computing ROC curve values.

**roc_auc_score()**
A scikit-learn function computing the AUC score.

**mean_squared_error()**
A scikit-learn function computing MSE.

**mean_absolute_error()**
A scikit-learn function computing MAE.

**r2_score()**
A scikit-learn function computing R² score.

**Ridge class**
Scikit-learn implementation of ridge regression.

**Lasso class**
Scikit-learn implementation of lasso regression.

**ElasticNet class**
Scikit-learn implementation of elastic net regression.

---

*Last updated: November 10, 2025*
