# Training a Logistic Regression Model

**Date:** November 13, 2025  
**Module:** 2 - Linear and Logistic Regression  
**Topic:** Training a Logistic Regression Model

---

<!-- Plan: short checklist per repo guidance -->
<!--
- Define model and cost (log loss)
- Derive gradient and update rule
- Explain GD, mini-batch GD, and SGD
- Cover learning rate, schedules, convergence, stopping
- Show scikit-learn solvers and configs
- Provide code snippets + exercises
- Key takeaways and study questions
-->

## Overview

Training logistic regression means estimating parameters (theta) that map feature vectors to probabilities of the positive class while minimizing a convex cost—the binary cross-entropy (log loss). We’ll connect the dots from the sigmoid link to log loss, derive the gradient used for optimization, and explain practical optimization methods: batch gradient descent (GD), mini-batch GD, and stochastic gradient descent (SGD). You’ll also learn how learning rate and stopping criteria drive convergence, when to favor different solvers (lbfgs, liblinear, saga), and how to evaluate progress via loss curves and calibration.

---

## Learning Objectives

After this lesson you will be able to:
- Write the logistic regression hypothesis, log loss objective, and gradient.
- Explain the difference between GD, mini-batch GD, and SGD—and why it matters.
- Select a learning rate and schedule; describe effects of too-small or too-large values.
- Implement early stopping and other stopping criteria in practice.
- Choose and configure scikit-learn solvers for different data regimes.
- Interpret convergence signals and diagnose optimization failures.

---

## 1. Model and Cost

### 1.1 Hypothesis

For features x ∈ R^d and parameters θ ∈ R^d, compute the linear score z and probability p̂:

- Linear score: z = θ₀ + θᵀx
- Sigmoid (logistic function): σ(z) = 1 / (1 + exp(-z))
- Predicted probability: p̂ = P(y=1|x; θ) = σ(z)

### 1.2 Log Loss (Binary Cross-Entropy)

For dataset D = {(xᵢ, yᵢ)} with yᵢ ∈ {0,1}, the average log loss is:

L(θ) = - (1/n) Σᵢ [ yᵢ log(p̂ᵢ) + (1 - yᵢ) log(1 - p̂ᵢ) ]

Why it works:
- Encourages confident, correct predictions: small loss when y=1 and p̂→1, or y=0 and p̂→0.
- Penalizes confident, incorrect predictions harshly (stability via small ε in implementations).
- Convex in θ for logistic regression, ensuring a unique global minimum (no local minima traps).

### 1.3 Gradient of Log Loss

Let X be the feature matrix with bias handled explicitly or via intercept term.
For a single example (x, y):
- z = θᵀx, p̂ = σ(z)
- ∂L/∂θ = (p̂ - y) x

For the full dataset (mean gradient):
- ∇L(θ) = (1/n) Σᵢ (p̂ᵢ - yᵢ) xᵢ

With L2 regularization (Ridge), λ>0:
- L_reg(θ) = L(θ) + (λ/2n) ||θ||²
- ∇L_reg(θ) = ∇L(θ) + (λ/n) θ (excluding the bias by convention)

---

## 2. Optimization Methods

### 2.1 Batch Gradient Descent (GD)
- Uses the full dataset each iteration to compute the exact gradient.
- Update: θ ← θ - η ∇L(θ)
- Pros: Stable, accurate gradient estimates, smooth convergence on convex problems.
- Cons: Expensive per step on large datasets (must sweep all n rows each iteration).

### 2.2 Mini-Batch Gradient Descent
- Uses batches of size b (e.g., 32, 64, 128) to estimate gradient.
- Update per batch: θ ← θ - η (1/b) Σ (p̂ - y) x over the batch.
- Pros: Excellent trade-off—vectorized, fast updates, smoother than pure SGD, widely used.
- Cons: Slight gradient noise; need to choose batch size.

### 2.3 Stochastic Gradient Descent (SGD)
- Uses a single sample (b=1) or very small batch to update θ.
- Pros: Very fast per update; can escape shallow plateaus; scales to streams.
- Cons: Noisy updates; may “wander” around the minimum; requires schedules to converge.

### 2.4 Learning Rate (η)
- Too small: slow convergence; may appear “stuck.”
- Too large: overshoots minimum; divergence or oscillation.
- Practical defaults: 1e-3 to 1e-1, depending on feature scaling and problem scale.
- Always scale features for stability (e.g., StandardScaler).

### 2.5 Learning Rate Schedules
- Step decay: ηₖ = η₀ · γ^⌊k/s⌋ (drop every s steps by factor γ).
- Time decay: ηₖ = η₀ / (1 + αk).
- Cosine decay, exponential decay, or adaptive methods (AdaGrad, RMSProp, Adam) in broader contexts.

### 2.6 Stopping Criteria
- Max iterations/epochs reached.
- Early stopping: stop if validation loss fails to improve for N checks.
- Gradient norm below tolerance.
- Parameter change below tolerance.

---

## 3. Practical Training Loop (NumPy Pseudocode)

```python
# Given X (n x d), y (n,), learning rate eta, max_iters, batch_size
# Assume X includes bias column or manage θ0 separately

# Initialize
theta = np.zeros(d)                 # start simple; random small values also OK
for epoch in range(max_epochs):     # multiple passes over data
    # Optionally shuffle
    idx = np.random.permutation(n)
    X_shuf, y_shuf = X[idx], y[idx]

    # Mini-batch updates
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        Xb, yb = X_shuf[start:end], y_shuf[start:end]
        z = Xb @ theta
        p_hat = 1 / (1 + np.exp(-z))
        grad = (Xb.T @ (p_hat - yb)) / (end - start)
        theta -= eta * grad

    # Optional: compute train/val loss, apply early stopping
```

Implementation notes:
- Always add small ε when computing log for stability.
- Standardize numeric features; one-hot encode categorical features.
- Monitor both loss and target metrics (e.g., ROC-AUC, F1) over time.

---

## 4. scikit-learn Solvers and Settings

`sklearn.linear_model.LogisticRegression` supports multiple solvers:

- lbfgs: quasi-Newton, robust for L2, supports multinomial; great default for medium/large dense data.
- liblinear: coordinate descent via LIBLINEAR; good for smaller datasets and L1 or L2; one-vs-rest.
- saga: stochastic variance-reduced gradient; supports L1/L2/ElasticNet and large sparse data.
- newton-cg: second-order method; works for multinomial with L2.

Key parameters:
- `max_iter`: iteration cap (increase if convergence warnings appear).
- `penalty`: 'l2' (default), 'l1', 'elasticnet' (with `l1_ratio`).
- `C`: inverse regularization strength (smaller C = stronger regularization).
- `class_weight`: set to 'balanced' to adjust for class imbalance.
- `fit_intercept`: usually True; ensure no duplicated bias term in X.
- `n_jobs`: parallelism where supported.

Example pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric_columns),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
])

model = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced")),
])

model.fit(X_train, y_train)
```

---

## 5. Gradient Descent vs SGD: When To Use What

| Aspect | Batch GD | Mini-batch GD | SGD |
| --- | --- | --- | --- |
| Gradient accuracy | Highest | High | Lowest |
| Update speed | Slowest | Fast | Fastest per update |
| Memory use | Highest | Moderate | Lowest |
| Convergence | Smoothest | Smooth | Noisy |
| Dataset size | Small/Medium | Medium/Large | Very Large/Streaming |

Rules of thumb:
- Start with mini-batch GD + standardized features.
- Use SGD for large, streaming, or online settings.
- Tune batch size and learning rate jointly; monitor loss curve.

---

## 6. Diagnostics and Troubleshooting

Symptoms and likely causes:
- Loss diverges or oscillates → Learning rate too high; scale features; try different solver.
- Convergence warning from sklearn → Increase `max_iter`; scale features; switch solver.
- Overfitting (train loss << val loss) → Increase regularization (decrease C); gather more data; add constraints.
- Underfitting (high bias) → Reduce regularization (increase C); add interactions; try richer features.

Loss curves:
- Compute and plot loss per epoch/iteration to visualize progress.
- For SGD, expect noisy curves; consider moving average or larger batches.

---

## 7. Worked Example: Manual Step Updates (Conceptual)

Given a single sample (x, y):
- z = θᵀx, p̂ = σ(z)
- Gradient g = (p̂ - y) x
- Update: θ ← θ - η g

With classes imbalanced, consider:
- `class_weight` to reweight gradient contributions.
- Stratified sampling in mini-batches.

---

## 8. Regularization and Generalization

- L2 (Ridge): shrinks coefficients uniformly; stabilizes optimization.
- L1 (Lasso): promotes sparsity; useful for feature selection.
- Elastic Net: combination of L1 and L2; balance via `l1_ratio`.
- Tune `C` on validation via grid search or cross-validation.

---

## 9. Summary of Key Formulas

- Sigmoid: σ(z) = 1 / (1 + exp(-z))
- Log loss: L(θ) = - (1/n) Σ [ y log p̂ + (1-y) log(1-p̂) ]
- Gradient: ∇L(θ) = (1/n) Σ (p̂ - y) x
- Update (GD): θ ← θ - η ∇L(θ)
- With L2: ∇L_reg(θ) = ∇L(θ) + (λ/n) θ

---

## 10. Study Questions

1. Why does log loss penalize confident incorrect predictions more severely than mild ones?
2. Describe the trade-offs between batch GD, mini-batch GD, and SGD.
3. What are signs your learning rate is too high? Too low?
4. How does feature scaling impact convergence and solver choice?
5. Why might `class_weight='balanced'` help in imbalanced classification?
6. When would you prefer `saga` over `lbfgs` in scikit-learn?
7. Explain early stopping and how you would implement it with a validation set.
8. How does L1 differ from L2 regularization in effect and use cases?
9. What is the practical meaning of `C` in sklearn’s LogisticRegression?
10. How would you plot and interpret a noisy SGD loss curve?

---

## 11. Practical Exercises

1) Implement mini-batch GD for logistic regression in NumPy: plot training loss vs epochs for batch sizes 1, 32, 128. Interpret differences.

2) Learning rate sweep: for η ∈ {1e-4, 1e-3, 1e-2, 1e-1}, run the same training loop and compare loss curves. Identify the stable region.

3) scikit-learn solver comparison: run lbfgs, liblinear, saga with/without `class_weight='balanced'`. Compare convergence warnings, accuracy, F1, and training time.

4) Early stopping: implement patience=5 on a validation split; show best epoch and final metrics.

5) Calibration check: compare Brier scores and calibration curves for logistic regression vs a tree-based model trained on the same features.

---

## Key Takeaways

- Logistic regression training minimizes log loss; gradient is (p̂ - y) x.
- Mini-batch GD is the practical default; SGD scales to massive/streaming data.
- Learning rate and feature scaling dominate convergence behavior.
- scikit-learn solvers offer robust, battle-tested optimization—choose based on data size and sparsity.
- Use early stopping, regularization, and diagnostics to keep models both accurate and stable.
