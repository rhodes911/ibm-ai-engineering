# Introduction to Logistic Regression

**Date:** November 13, 2025  
**Module:** 2 - Linear and Logistic Regression  
**Topic:** Introduction to Logistic Regression

---

## Overview

Logistic regression bridges statistics and machine learning by transforming linear combinations of features into calibrated probabilities for binary outcomes. Unlike linear regression, which predicts unbounded continuous values, logistic regression constrains predictions to the 0-1 range using the sigmoid (logit) function. This lesson distills why, when, and how to apply logistic regression to real-world problems ranging from telecom churn prediction to medical risk scoring.

---

## Learning Objectives

After studying this lesson you will be able to:

1. Describe how logistic regression converts linear scores into probabilities and decisions.
2. Identify scenarios where a probabilistic binary classifier is preferable to pure regression.
3. Interpret coefficients, odds, and log-odds to reason about feature impact.
4. Visualize logistic decision boundaries in 1D, 2D, and high-dimensional feature spaces.
5. Build, evaluate, and deploy logistic regression models in scikit-learn.
6. Explain practical considerations such as threshold tuning, feature scaling, and class imbalance.

---

## 1. Why Logistic Regression?

| Business Question | Need Binary Target? | Need Probability? | Logistic Regression Fit? | Real-World Example |
| --- | --- | --- | --- | --- |
| Which telecom customers will churn next month? | Yes (Churn = 1) | Yes (Retention team prioritizes by risk) | Excellent | Telco analyzing 75K consumer accounts with contract + usage data |
| Which online payment is fraudulent? | Yes | Yes (Risk engine sets review thresholds) | Excellent | FinTech platform scoring 1M transactions/day |
| How many units will we sell next quarter? | No (continuous target) | Maybe | Poor | Use linear or Poisson regression instead |
| Which machine will fail in the next 24 hours? | Yes | Yes (Maintenance scheduling) | Excellent | Manufacturing plant with IoT telemetry |

Key reasons teams reach for logistic regression:

- **Binary outcomes** such as churn vs retain, disease vs healthy, pass vs fail.
- **Probabilities** needed for ranking, risk tolerance, and business policy.
- **Interpretable coefficients** show how odds change per unit increase in a feature.
- **Linearly separable data** where a weighted sum with an activation can separate classes.

### Rapid Checklist

- Target encoded as 0/1? [x]
- Need interpreted probabilities, not only labels? [x]
- Features roughly independent or low multicollinearity? [x]
- Decision boundary can be approximated by a plane or soft curve? [x]

If most boxes check out, logistic regression is a strong baseline before trying more complex models.

---

## 2. Vocabulary Refresher

| Term | Intuition | Example |
| --- | --- | --- |
| **Logit / Log-Odds** | Natural log of odds of class 1; logistic regression fits this linearly. | `log(p/(1-p)) = -4 + 0.08 * Age + 0.6 * HasContract` |
| **Sigmoid** | Smooth S-shaped function mapping any number into (0,1). | An internal score of 3.1 becomes probability 0.957. |
| **Decision Boundary** | Geometric surface where predicted probability equals threshold. | In 2D, a straight line splitting churn vs retain customers. |
| **Weight Vector (θ)** | Coefficients multiplying features; direction defines separating hyperplane. | `[θ0, θ1, θ2]` for intercept, age, contract length. |
| **Binary Target** | Dependent variable limited to 0 or 1. | `Churn` column with `0 = stayed`, `1 = left`. |

---

## 3. Visualizing Probability vs Class Outputs

```
Linear Regression Output (Unbounded)
|
|        /
|       /
|      /
|_____/____________

Step Function Classifier (Harsh Threshold)
|
|_____|```````````````` (0 or 1 only)
        0.5

Sigmoid Curve (Logistic Regression)
|
|          ~~~~~~~1
|       ~~~
|    ~~~
| ~~~
|~ 0                      1
|______________________________
        Linear Score z
```

- **Linear regression** provides continuous predictions that can exceed 1 or drop below 0.
- **Step functions** classify directly but discard probabilistic nuance, treating 0.51 and 0.99 identically.
- **Sigmoid** curves retain smooth probability transitions, enabling calibrated risk scores.

---

## 4. Mathematical Foundation

### 4.1 Linear Score

A linear combination of features produces score *z*:

$$z = θ_0 + θ_1 x_1 + θ_2 x_2 + \ldots + θ_n x_n$$

### 4.2 Sigmoid Transformation

$$σ(z) = \frac{1}{1 + e^{-z}} = \hat{p}(y = 1|x)$$

- When *z = 0*, `σ(z) = 0.5` (most uncertain).
- As *z* -> +inf, probability approaches 1.
- As *z* -> -inf, probability approaches 0.

### 4.3 Log-Odds Interpretation

$$\log\left(\frac{p}{1-p}\right) = θ_0 + θ_1 x_1 + ... + θ_n x_n$$

Each coefficient `θ_i` adjusts the log-odds linearly. Exponentiating `θ_i` yields the **odds ratio** for a one-unit change in the corresponding feature.

---

## 5. Decision Boundaries Across Dimensions

| Dimensionality | Boundary Shape | Example |
| --- | --- | --- |
| 1D | Threshold point on a number line | Age > 42.7 predicts churn |
| 2D | Line | `0.4 * ContractLength + 0.8 * MonthlyCharge = 30` |
| 3D | Plane | `θ0 + θ1 Age + θ2 Calls + θ3 Tenure = 0` |
| nD | Hyperplane | Same as above but in feature space beyond human visualization |

In datasets that are **linearly separable**, logistic regression finds a hyperplane that maximizes likelihood. For partially overlapping classes, it still yields the best-fitting probabilistic boundary under its assumptions.

---

## 6. Probability vs Class Decisions

1. **Predict Probability (y_hat)**
   - Use sigmoid to convert the linear score to `p_hat`.
   - Example: Customer A with features yields `p_hat = 0.78`.

2. **Apply Threshold (tau)**
   - Default tau = 0.5 but should be tuned to business goals.
   - Example: Retention team contacts anyone with `p_hat >= 0.35` because losing a customer is expensive.

3. **Produce Class Label**
   - `y_hat_class = 1` if `p_hat >= tau`, else `0`.

4. **Calibrate and Rank**
   - Sort customers by probability to allocate marketing incentives efficiently.

---

## 7. Feature Impact and Coefficient Interpretation

- **Positive coefficient** increases log-odds of class 1.
- **Negative coefficient** decreases log-odds.
- **Odds ratio = e^{θ_i}**
  - Example: `θ_contract = 0.9 -> e^{0.9} ~ 2.46`. Customers with annual contracts are 2.46x more likely to stay.

| Feature | Coefficient θ | Odds Ratio | Interpretation |
| --- | --- | --- | --- |
| MonthlySpend (per $10) | 0.35 | 1.42 | Each $10 increase in spend raises churn odds by 42% (possible sign of price fatigue). |
| Tenure (years) | -0.62 | 0.54 | Each year of loyalty halves churn odds. |
| SupportTickets (3-month total) | 0.18 | 1.20 | Each extra support ticket raises churn odds by 20%. |

Interpretation tips:

- Standardize numeric features to compare coefficient magnitudes meaningfully.
- Use domain knowledge to explain counterintuitive signs before acting on them.

---

## 8. Case Study: Telecom Churn Mini-Workflow

1. **Data Snapshot**
   - 10,000 consumer accounts, 18 features (demographics, contracts, services).
   - Target column `Churn` (1 = left last billing cycle).

2. **Exploratory Signals**
   - Scatter plot Age vs Churn shows weak linear separation.
   - Contract type (monthly vs annual) strongly differentiates classes.

3. **Modeling Steps**
   - Encode categorical features (DSL vs Fiber, etc.).
   - Scale numeric features (tenure, monthly charge) for stable optimization.
   - Fit `LogisticRegression(penalty="l2", class_weight="balanced")` using scikit-learn.
   - Evaluate with accuracy, ROC-AUC, precision-recall (since churn rate is only 27%).

4. **Business Utilization**
   - Send retention offers to top 5% highest risk customers each month.
   - Provide dashboards showing probability trends per segment (Millennials on month-to-month contracts, etc.).

---

## 9. Comparing Model Choices

| Criteria | Logistic Regression | Linear Regression | Decision Tree |
| --- | --- | --- | --- |
| Target Type | Binary (0/1) | Continuous | Binary or multiclass |
| Output | Probability + class label | Continuous number | Class label (probabilities with smoothing) |
| Interpretability | High (coefficients show impact) | High but wrong target type | Medium (rules) |
| Handles Nonlinearities | Limited (needs feature engineering) | N/A | Good |
| Training Speed | Very fast | Very fast | Fast but slower |
| When Preferable | Risk scoring, explainable AI | Forecasting amounts | Capturing complex interactions |

---

## 10. Practical Implementation Walkthrough

1. **Data Preparation**
   - Handle missing values (median for numeric, most frequent for categorical).
   - Encode categories via one-hot encoding.
   - Scale numeric inputs when penalty-based solvers are used.

2. **Training (scikit-learn)**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Features and target
data = pd.read_csv("telecom_churn.csv")
X = data.drop(columns=["Churn"])
y = data["Churn"]

numeric_features = ["tenure_months", "monthly_charges", "support_calls"]
categorical_features = ["contract_type", "payment_method", "device_protection"]

numeric_pipe = Pipeline([
    ("scaler", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features),
])

model = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model.fit(X_train, y_train)
print("Test accuracy:", model.score(X_test, y_test))
```

3. **Evaluation Beyond Accuracy**
   - Plot ROC and Precision-Recall curves to diagnose threshold trade-offs.
   - Inspect confusion matrix to count false alarms vs misses.
   - Use calibration curves to verify probability quality.

4. **Deployment**
   - Serialize pipeline with `joblib.dump(model, "telecom_churn_lr.joblib")`.
   - Serve through API that outputs both class label and probability.
   - Track drift by monitoring average predicted probabilities vs actual churn each quarter.

---

## 11. Handling Special Scenarios

### 11.1 Class Imbalance

- Use `class_weight="balanced"` or custom weights based on inverse class frequency.
- Oversample minority class (SMOTE) or undersample majority class.
- Evaluate with metrics sensitive to imbalance (recall, F1, ROC-AUC, PR-AUC).

### 11.2 Nonlinear Relationships

- Engineer interaction or polynomial terms (e.g., `Tenure * MonthlyCharge`).
- Use spline basis functions or binning to approximate nonlinearity.
- Switch to tree-based models if linear separability fails.

### 11.3 Feature Scaling Necessity

- Required for gradient-based solvers (lbfgs, saga) to converge quickly.
- Helps interpret coefficient magnitudes when features share comparable scales.

---

## 12. Visual Decision Flow (ASCII Diagram)

```
Start
 |
 |-- Is target binary? ---- No --> Choose regression/classification variant
 |
 Yes
 |
 |-- Need calibrated probabilities? --- No --> Consider SVM or tree
 |
 Yes
 |
 |-- Features roughly linearly separable? --- No --> Add nonlinear features or try tree/NN
 |
 Yes
 |
 --> Logistic Regression Baseline
```

---

## 13. Key Formulas and Concepts

- **Sigmoid**: `σ(z) = 1 / (1 + e^{-z})`
- **Log-Odds**: `log(p/(1-p)) = θ^T x`
- **Odds Ratio Change**: `e^{θ_i}` per unit increase in feature `x_i`.
- **Binary Cross-Entropy Loss**: `L = -[y log(p_hat) + (1-y) log(1-p_hat)]`
- **Gradient for θ_j**: `dL/dθ_j = (p_hat - y) * x_j`
- **Decision Boundary**: `θ^T x + θ_0 = 0`

---

## 14. Real-World Illustrations

1. **Healthcare**: Predict probability of hospital readmission within 30 days.
   - Features: comorbidity count, length of stay, discharge disposition.
   - Action: High-probability patients receive follow-up telehealth calls.

2. **E-commerce**: Probability a visitor converts during current session.
   - Features: pages viewed, dwell time, referral channel, cart value.
   - Action: Trigger personalized discount if probability dips below 0.35.

3. **Manufacturing**: Equipment failure likelihood in next shift.
   - Features: vibration amplitude, lubricant temperature, runtime hours.
   - Action: Maintenance crew inspects machines above 0.6 probability threshold.

4. **Banking**: Mortgage default probability.
   - Features: credit score, debt-to-income ratio, property type, loan-to-value.
   - Action: Underwriting adjusts interest rate or requires co-signer for high-risk applicants.

---

## 15. Workflow Checklist

1. **Define Outcome**: Convert to binary (e.g., churn = Yes/No -> 1/0).
2. **Audit Data Quality**: Missing values, inconsistent categories.
3. **EDA**: Understand feature distributions, correlations, separability.
4. **Feature Engineering**: Interactions, scaling, categorical encoding.
5. **Model Training**: Choose solver (`lbfgs`, `liblinear`, `saga`).
6. **Validation**: Stratified train/test split or k-fold cross-validation.
7. **Threshold Selection**: Optimize for cost-sensitive metric (e.g., maximize expected lifetime value saved).
8. **Monitoring**: Track precision, recall, calibration drift after deployment.

---

## 16. Key Takeaways

- Logistic regression is a probabilistic classifier best suited for binary targets requiring interpretability.
- The sigmoid function ensures outputs remain between 0 and 1, enabling calibrated decisions.
- Coefficients translate directly into odds ratios, offering intuitive explanations for stakeholders.
- Threshold tuning tailors predictions to business costs (false positives vs false negatives).
- Despite its simplicity, logistic regression remains a powerful baseline for many production systems.

---

## 17. Study Questions

1. Why is logistic regression preferred over linear regression for churn prediction?
2. How do you interpret a coefficient of -0.75 on the tenure feature?
3. What business considerations influence threshold selection beyond pure accuracy?
4. Describe how class imbalance affects model training and evaluation.
5. When would you add polynomial or interaction features to a logistic model?
6. Explain the difference between odds, probability, and log-odds in plain language.
7. How can you verify that predicted probabilities are well calibrated?
8. Provide an example where a low threshold (e.g., 0.2) is preferable to 0.5 and justify it.
9. What symptoms indicate that logistic regression is underfitting your dataset?
10. How would you communicate odds ratios to non-technical stakeholders?

---

## 18. Practical Exercises

1. **Threshold Tuning Exercise**
   - Using a churn dataset, compute precision, recall, F1, and business cost for thresholds from 0.1 to 0.9. Plot the trade-offs and decide on an operating point for a retention campaign with fixed budget.

2. **Feature Impact Analysis**
   - Train a logistic regression model on a healthcare readmission dataset. Create a table of coefficients, odds ratios, and domain interpretations. Highlight features whose impact contradicts clinical expectations and propose validation steps.

3. **Probability Calibration Drill**
   - Fit logistic regression and random forest to the same dataset. Use calibration curves and Brier scores to compare probability quality. Explain why logistic regression often produces better-calibrated probabilities out of the box.

4. **Linearly Separable Thought Experiment**
   - Sketch two 2D datasets: one linearly separable, one overlapping. Discuss how logistic regression decision boundaries would differ and what additional features could improve separability in the overlapping case.

5. **Real-World Storytelling**
   - Pick an industry (finance, retail, healthcare) and craft a one-page briefing explaining how logistic regression drives a specific decision, the inputs required, and how teams act on the resulting probabilities.

---

## 19. Additional Resources

- IBM: *Machine Learning with Python* Module 2 video playlist.
- scikit-learn documentation: [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
- Richard McElreath, *Statistical Rethinking*, Chapter on GLMs for deeper Bayesian perspective.
- Google Developers, *Machine Learning Crash Course*, logistic regression module.

---

## 20. Summary

Logistic regression remains a cornerstone of applied machine learning because it offers the rare combination of speed, interpretability, and probabilistic rigor. By linking linear feature combinations to the sigmoid function, teams can rank risks, justify interventions, and iterate quickly. Mastering the nuances covered in this lesson-from log-odds intuition to threshold calibration-prepares you to deploy reliable, explainable classifiers in high-stakes business environments.
