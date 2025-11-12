# Multiple Linear Regression

**Date:** November 11, 2025  
**Module:** 2 - Linear and Logistic Regression  
**Topic:** Multiple Linear Regression - Theory, Applications, and Pitfalls  

---

## Overview

This lesson extends simple linear regression to **multiple linear regression**, which uses two or more independent variables to predict a continuous target variable. Multiple linear regression provides more accurate predictions by considering multiple factors simultaneously, but requires careful variable selection to avoid overfitting and multicollinearity issues.

### Learning Objectives

After completing this lesson, you will be able to:
- ✅ Describe multiple linear regression and its mathematical formulation
- ✅ Compare multiple linear regression with simple linear regression
- ✅ Understand how to incorporate multiple features into predictions
- ✅ Handle categorical variables in regression models
- ✅ Apply multiple linear regression to real-world scenarios
- ✅ Identify and avoid common pitfalls (overfitting, multicollinearity)
- ✅ Understand what-if scenario analysis
- ✅ Choose appropriate methods for estimating coefficients

---

## 1. What is Multiple Linear Regression?

### Definition

> **Multiple Linear Regression:** An extension of simple linear regression that uses **two or more** independent variables to estimate a dependent variable.

### Key Characteristics

**"Multiple" means:**
- Uses **multiple** predictors (features)
- Considers combined effects of several variables
- More realistic for real-world problems

**Still "Linear":**
- Creates a **linear combination** of features
- Relationship is additive
- Each feature contributes independently (assuming no interactions)

---

### Mathematical Formulation

**General Equation:**
$$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + ... + \theta_n x_n$$

**Where:**
- $\hat{y}$ = Predicted value (dependent variable)
- $x_1, x_2, ..., x_n$ = Independent variables (features)
- $\theta_0$ = Intercept (bias term)
- $\theta_1, \theta_2, ..., \theta_n$ = Coefficients (weights)

---

### Matrix Representation

**Feature Matrix (X):**
$$X = \begin{bmatrix} 1 & x_{1,1} & x_{1,2} & ... & x_{1,n} \\ 1 & x_{2,1} & x_{2,2} & ... & x_{2,n} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_{m,1} & x_{m,2} & ... & x_{m,n} \end{bmatrix}$$

**Note:** First column of 1's accounts for the bias term $\theta_0$

**Weight Vector (θ):**
$$\theta = \begin{bmatrix} \theta_0 \\ \theta_1 \\ \theta_2 \\ \vdots \\ \theta_n \end{bmatrix}$$

**Matrix Form:**
$$\hat{y} = X\theta$$

---

## 2. Example: CO2 Emissions Prediction

### Dataset

Extending our automobile dataset to use **multiple features**:

| Car ID | Engine Size (L) | Cylinders | Fuel Consumption (L/100km) | CO2 Emissions (g/km) |
|--------|----------------|-----------|----------------------------|----------------------|
| 1      | 2.0            | 4         | 8.5                        | 196                  |
| 2      | 2.4            | 4         | 9.6                        | 221                  |
| 3      | 1.5            | 4         | 5.9                        | 136                  |
| 4      | 3.5            | 6         | 11.0                       | 255                  |
| 5      | 3.5            | 6         | 10.1                       | 244                  |
| ...    | ...            | ...       | ...                        | ...                  |
| 9      | 2.4            | 4         | 9.5                        | 214                  |

---

### Simple vs Multiple Regression

**Simple Linear Regression (from previous lesson):**
$$\text{CO2} = 125.7 + 39 \times \text{EngineSize}$$

Uses **only one** feature (Engine Size)

**Multiple Linear Regression:**
$$\text{CO2} = 125 + 6.2 \times \text{EngineSize} + 14 \times \text{Cylinders} + ... + \theta_n \times \text{FuelConsumption}$$

Uses **multiple** features (Engine Size, Cylinders, Fuel Consumption, etc.)

---

### Why Multiple Regression is Better

**Advantage:** Captures more information
- Engine size alone doesn't tell the whole story
- Cylinders, fuel consumption, and other factors also matter
- Combining features → more accurate predictions

**Example:**
```
Two cars with 2.4L engines:

Car A: 2.4L, 4 cylinders, 9.5 L/100km fuel
Car B: 2.4L, 6 cylinders, 12.0 L/100km fuel

Simple regression: Both predicted same (219 g/km)
❌ Ignores differences in cylinders and fuel consumption

Multiple regression: Different predictions
✅ Car A: 214 g/km
✅ Car B: 242 g/km (accounts for extra cylinders and fuel)
```

---

## 3. Building a Multiple Linear Regression Model

### Example Coefficients

**Given trained model parameters:**
- $\theta_0 = 125$ (intercept)
- $\theta_1 = 6.2$ (Engine Size coefficient)
- $\theta_2 = 14$ (Cylinders coefficient)
- $\theta_3 = 10$ (Fuel Consumption coefficient)

**Model Equation:**
$$\text{CO2} = 125 + 6.2 \times \text{EngineSize} + 14 \times \text{Cylinders} + 10 \times \text{FuelConsumption}$$

---

### Making a Prediction (Record #9)

**Given Car Features:**
- Engine Size = 2.4 L
- Cylinders = 4
- Fuel Consumption = 9.5 L/100km

**Calculation:**
$$\text{CO2} = 125 + 6.2(2.4) + 14(4) + 10(9.5)$$
$$\text{CO2} = 125 + 14.88 + 56 + 95$$
$$\text{CO2} = 290.88 \text{ g/km}$$

**Video states:** Predicted CO2 = 214.1 g/km (using different coefficients)

**Interpretation:** Each feature contributes to the final prediction based on its weight.

---

### Understanding Relative Importance

**From the coefficients:**
- **Fuel Consumption (10):** Largest impact per unit
- **Cylinders (14):** Significant impact
- **Engine Size (6.2):** Moderate impact

**Practical Meaning:**
- Adding 1 cylinder → +14 g/km CO2
- Increasing fuel consumption by 1 L/100km → +10 g/km CO2
- Adding 1L engine size → +6.2 g/km CO2

---

## 4. Geometric Interpretation

### Dimensions Matter

**Simple Linear Regression (1 feature):**
```
Solution: A LINE in 2D space

CO2 |     /
    |    /
    |   /
    |  /
    | /
    |_____________ Engine Size
```

---

**Multiple Linear Regression (2 features):**
```
Solution: A PLANE in 3D space

      CO2
       |
       |    /‾‾‾\
       |   /     \  (plane)
       |  /       \
       | /________\_____ Cylinders
       |/
      Engine Size
```

---

**Multiple Linear Regression (3+ features):**
```
Solution: A HYPERPLANE in n-dimensional space

Cannot visualize directly!
But mathematically, it's a flat surface in high-dimensional space
```

**Key Insight:** As dimensions increase, we move from line → plane → hyperplane.

---

## 5. Handling Categorical Variables

### The Challenge

**Problem:** Regression requires numerical inputs, but some variables are categorical.

**Examples of Categorical Variables:**
- Car Type: Manual, Automatic
- Color: Red, Blue, Green, Black
- Fuel Type: Gasoline, Diesel, Electric, Hybrid

---

### Solution 1: Binary Variables (Two Categories)

**Example: Transmission Type**

Original: "Manual" or "Automatic"

**Encoding:**
- Manual → 0
- Automatic → 1

**In Model:**
$$\text{CO2} = \theta_0 + \theta_1 \times \text{EngineSize} + \theta_2 \times \text{IsAutomatic}$$

**Prediction:**
```
Manual car:    IsAutomatic = 0
               CO2 = θ₀ + θ₁×EngineSize + θ₂×(0)
               CO2 = θ₀ + θ₁×EngineSize

Automatic car: IsAutomatic = 1
               CO2 = θ₀ + θ₁×EngineSize + θ₂×(1)
               CO2 = θ₀ + θ₁×EngineSize + θ₂

Difference: θ₂ (coefficient shows effect of automatic transmission)
```

---

### Solution 2: One-Hot Encoding (Multiple Categories)

**Example: Fuel Type**

Original: "Gasoline", "Diesel", "Electric", "Hybrid"

**One-Hot Encoding:** Create binary feature for each category

| Original | Is_Gasoline | Is_Diesel | Is_Electric | Is_Hybrid |
|----------|-------------|-----------|-------------|-----------|
| Gasoline | 1           | 0         | 0           | 0         |
| Diesel   | 0           | 1         | 0           | 0         |
| Electric | 0           | 0         | 1           | 0         |
| Hybrid   | 0           | 0         | 0           | 1         |

**In Model:**
$$\text{CO2} = \theta_0 + ... + \theta_k \times \text{IsDiesel} + \theta_{k+1} \times \text{IsElectric} + \theta_{k+2} \times \text{IsHybrid}$$

**Note:** Drop one category (e.g., Gasoline) to avoid multicollinearity (dummy variable trap).

---

## 6. Real-World Applications

### Application 1: Education

**Research Question:**
"What factors affect student exam performance?"

**Independent Variables:**
- Revision time (hours)
- Test anxiety (score 1-10)
- Lecture attendance (percentage)
- Gender (0=male, 1=female)
- Previous GPA

**Dependent Variable:**
- Exam score (0-100)

**Model:**
$$\text{ExamScore} = \theta_0 + \theta_1 \times \text{RevisionHours} + \theta_2 \times \text{Anxiety} + \theta_3 \times \text{Attendance} + \theta_4 \times \text{Gender} + \theta_5 \times \text{PreviousGPA}$$

**Example Result:**
$$\text{ExamScore} = 30 + 5 \times \text{RevisionHours} - 3 \times \text{Anxiety} + 0.4 \times \text{Attendance} + 2 \times \text{Gender} + 15 \times \text{PreviousGPA}$$

**Interpretation:**
- Each hour of revision: +5 points
- Each point of anxiety: -3 points
- Each 1% attendance: +0.4 points
- Being female: +2 points
- Each GPA point: +15 points

---

### Application 2: Healthcare (What-If Analysis)

**Research Question:**
"How does BMI affect blood pressure?"

**Model:**
$$\text{BloodPressure} = \theta_0 + \theta_1 \times \text{BMI} + \theta_2 \times \text{Age} + \theta_3 \times \text{Exercise} + \theta_4 \times \text{Smoking}$$

**What-If Scenario:**
"If a patient reduces BMI from 30 to 28, how much will blood pressure drop?"

**Calculation:**
```
Current: BP = θ₀ + θ₁(30) + θ₂(Age) + θ₃(Exercise) + θ₄(Smoking)
After:   BP = θ₀ + θ₁(28) + θ₂(Age) + θ₃(Exercise) + θ₄(Smoking)

Difference: Δ BP = θ₁(30) - θ₁(28) = 2θ₁

If θ₁ = 1.5: ΔBP = 2(1.5) = 3 mmHg reduction
```

**Interpretation:** Losing 2 BMI points reduces blood pressure by 3 mmHg (holding all else constant).

---

## 7. What-If Scenarios

### Definition

> **What-If Scenario:** Hypothetical changes to one or more input features to see predicted outcome changes.

### How It Works

**Process:**
1. Start with current feature values
2. Change one or more features
3. Keep other features constant
4. Calculate new prediction
5. Compare to baseline

---

### Example: CO2 Emissions What-If

**Baseline Car:**
- Engine Size: 3.0L
- Cylinders: 6
- Fuel Consumption: 11.0 L/100km
- **Predicted CO2: 250 g/km**

**What-If #1:** "What if we reduce engine size to 2.5L?"
```
New prediction:
CO2 = 125 + 6.2(2.5) + 14(6) + 10(11.0)
    = 125 + 15.5 + 84 + 110
    = 334.5 g/km

Wait, this doesn't make sense! 
Smaller engine should reduce emissions.
```

**Problem:** Engine size is correlated with cylinders and fuel consumption. We can't change just one variable realistically.

---

## 8. Pitfalls of Multiple Linear Regression

### Pitfall 1: Overfitting

**Problem:** Adding too many variables causes model to memorize training data.

**Example:**
```
Dataset: 100 cars
Features: 95 variables (engine size, color, owner's birthday, ...)

Model: Perfect fit on training data (100% accuracy)
Problem: Captures noise and irrelevant patterns
Result: Poor predictions on new cars
```

**Visual:**
```
Training Data:
Points: • • • • •
Model:  \_/‾\_/‾\  (fits every wiggle)
✅ Training accuracy: 99%

Test Data:
Points: × × × × ×
Model:  \_/‾\_/‾\  (wrong curves)
❌ Test accuracy: 45%
```

**Solution:**
- Use fewer, more relevant variables
- Apply regularization techniques
- Cross-validation
- Keep it simple

---

### Pitfall 2: Multicollinearity (Collinear Variables)

**Problem:** Two or more variables are highly correlated with each other.

**Definition:**
> **Collinearity:** When variables are correlated, they are no longer independent because they predict each other.

**Example:**
```
Predicting House Price:

Variable 1: Square footage (2,000 sq ft)
Variable 2: Number of rooms (8 rooms)

Correlation: +0.95 (highly correlated!)

Problem: 
- Bigger houses have more rooms
- These variables measure similar things
- Model can't distinguish their individual effects
- Coefficients become unstable
```

---

### Why Multicollinearity is Bad

**Issues:**
1. **Unreliable coefficients:** Small data changes → large coefficient changes
2. **Can't isolate effects:** Which variable truly matters?
3. **Wrong what-if scenarios:** Can't change one without the other
4. **Statistical significance:** Standard errors inflate

**Example with Correlation:**
```
Model: Price = θ₀ + θ₁(SqFt) + θ₂(Rooms)

Problem: If SqFt increases, Rooms also increases
Can't say: "What if we add 500 sq ft but keep rooms same?"
Unrealistic scenario!

Result: Coefficients don't reflect true relationships
```

---

### Detecting Multicollinearity

**Methods:**
1. **Correlation Matrix:** Look for high correlations (>0.7 or >0.8)
2. **Variance Inflation Factor (VIF):** VIF > 10 indicates problem
3. **Visual inspection:** Scatter plot matrix

**Example Correlation Matrix:**
```
                 EngineSize  Cylinders  FuelConsumption
EngineSize            1.00       0.95             0.88
Cylinders             0.95       1.00             0.85
FuelConsumption       0.88       0.85             1.00

High correlations (0.95, 0.88, 0.85) → Multicollinearity present!
```

---

### Pitfall 3: Impossible What-If Scenarios

**Problem:** Considering scenarios that violate physical/logical constraints.

**Examples:**
```
❌ "What if car has 2.0L engine but 8 cylinders?"
   (Physically impossible for most engines)

❌ "What if fuel consumption is 5 L/100km but engine is 5.0L?"
   (Unrealistic - large engines can't be that efficient)

❌ "What if person is 30 years old but has 40 years experience?"
   (Logically impossible)
```

**Solution:** Understand domain constraints before running scenarios.

---

### Pitfall 4: Extrapolation Beyond Training Range

**Problem:** Making predictions outside the range of training data.

**Example:**
```
Training Data:
Engine sizes: 1.5L to 4.5L
Model trained on this range

What-If Scenario:
"Predict CO2 for 10.0L engine"

Problem: No data for engines this large!
Model behavior unknown in this region
Prediction unreliable
```

**Visual:**
```
CO2
 |
 |        [Training Data Range]
 |        ↓________________↓
 |  . . . • • • • • • • • •  ? ? ? ? 
 |__________________________________ Engine Size
     1.5L                4.5L        10.0L
                                      ↑
                                 Extrapolation
                                 (unreliable!)
```

---

## 9. Building a Good Multiple Regression Model

### Variable Selection Criteria

**Choose variables that are:**

1. **✅ Most Correlated with Target**
   - Strong relationship with dependent variable
   - High predictive power
   - Example: Fuel consumption → CO2 emissions (r = 0.95)

2. **✅ Uncorrelated with Each Other**
   - Minimal multicollinearity
   - Independent contributions
   - Example: Engine size and transmission type (r = 0.1)

3. **✅ Most Understood**
   - Clear interpretation
   - Domain knowledge supports relationship
   - Example: "Larger engines produce more emissions" (makes sense)

4. **✅ Most Controllable**
   - Can be measured accurately
   - Can be changed in what-if scenarios
   - Example: Engine size (concrete, measurable)

---

### Balanced Approach

**Don't include:**
- ❌ Too many variables (overfitting)
- ❌ Highly correlated variables (multicollinearity)
- ❌ Irrelevant variables (noise)
- ❌ Variables with poor data quality

**Do include:**
- ✅ Strong predictors
- ✅ Independent variables
- ✅ Meaningful variables
- ✅ Reliable measurements

---

### Example: Variable Selection for CO2 Prediction

**Available Variables:**
| Variable | Correlation with CO2 | Correlation with Others | Include? |
|----------|---------------------|------------------------|----------|
| **Fuel Consumption** | 0.95 | High with engine/cylinders | ✅ Strong predictor |
| **Engine Size** | 0.82 | High with cylinders | ❌ Remove (use fuel instead) |
| **Cylinders** | 0.79 | High with engine | ❌ Remove (redundant) |
| **Weight** | 0.75 | Medium | ✅ Independent factor |
| **Transmission** | 0.35 | Low | ✅ Independent, controllable |
| **Color** | 0.02 | Low | ❌ Not relevant |

**Final Model:**
$$\text{CO2} = \theta_0 + \theta_1 \times \text{FuelConsumption} + \theta_2 \times \text{Weight} + \theta_3 \times \text{IsAutomatic}$$

---

## 10. Estimating Coefficients

### Method 1: Ordinary Least Squares (OLS)

**Approach:** Matrix-based calculation using linear algebra.

**Formula:**
$$\theta = (X^T X)^{-1} X^T y$$

**Where:**
- $X$ = Feature matrix (m samples × n features)
- $y$ = Target vector
- $X^T$ = Transpose of X
- $(X^T X)^{-1}$ = Inverse of $X^T X$

**Characteristics:**
- ✅ **Closed-form solution** (direct calculation)
- ✅ **Exact answer** (no approximation)
- ✅ **Fast for small/medium datasets** (< 10,000 samples)
- ❌ **Computationally expensive for large data** (matrix inversion)
- ❌ **Requires matrix invertibility** (fails if multicollinearity severe)

---

### Method 2: Gradient Descent

**Approach:** Iterative optimization starting from random values.

**Process:**
```
1. Initialize: θ = [random values]
2. Repeat:
   a. Calculate predictions: ŷ = Xθ
   b. Calculate error: MSE = (1/n)Σ(y - ŷ)²
   c. Calculate gradient: ∇MSE
   d. Update weights: θ = θ - α∇MSE
3. Stop when: MSE stops decreasing
```

**Learning Rate (α):**
- Controls step size
- Too large → overshooting
- Too small → slow convergence

**Characteristics:**
- ✅ **Scales to large datasets** (millions of samples)
- ✅ **Works with singular matrices** (handles multicollinearity better)
- ✅ **Memory efficient** (no large matrix inversion)
- ❌ **Approximate solution** (stops near minimum)
- ❌ **Requires tuning** (learning rate, iterations)
- ❌ **Slower for small data** (many iterations)

---

### Method Comparison

| Aspect | OLS | Gradient Descent |
|--------|-----|------------------|
| **Solution Type** | Exact | Approximate |
| **Computation** | Direct calculation | Iterative |
| **Speed (small data)** | Fast | Slower |
| **Speed (large data)** | Very slow/impossible | Fast |
| **Memory** | High (matrix ops) | Low |
| **Hyperparameters** | None | Learning rate, iterations |
| **Multicollinearity** | Can fail | More robust |
| **Best for** | <10K samples | >10K samples |

---

### When to Use Each

**Use OLS when:**
- Dataset is small to medium (<10,000 samples)
- Want exact solution
- No severe multicollinearity
- Have sufficient memory

**Use Gradient Descent when:**
- Dataset is large (>10,000 samples)
- Memory constrained
- Matrix inversion fails
- Using regularization (Ridge, Lasso)

---

## 11. Summary of Key Concepts

### Simple vs Multiple Regression

| Aspect | Simple | Multiple |
|--------|--------|----------|
| **Predictors** | One (1) | Multiple (2+) |
| **Equation** | $\hat{y} = \theta_0 + \theta_1 x$ | $\hat{y} = \theta_0 + \sum_{i=1}^{n} \theta_i x_i$ |
| **Geometry** | Line (2D) | Plane (3D) / Hyperplane (n-D) |
| **Accuracy** | Lower | Higher (usually) |
| **Interpretation** | Very easy | More complex |
| **Overfitting Risk** | Low | Higher |

---

### Key Takeaways

✅ **Multiple Linear Regression**
- Extension of simple linear regression
- Uses 2+ independent variables
- Forms linear combination of features
- Results in better predictions (usually)

✅ **Applications**
- Education: Predict student performance
- Healthcare: What-if scenario analysis
- Business: Multifactor predictions
- Science: Understand relationships

✅ **Handling Categorical Variables**
- Binary: Encode as 0/1
- Multiple categories: One-hot encoding
- Drop one category to avoid collinearity

✅ **Pitfalls to Avoid**
- Overfitting (too many variables)
- Multicollinearity (correlated predictors)
- Impossible scenarios (violate constraints)
- Extrapolation (beyond training range)

✅ **Variable Selection**
- Choose correlated with target
- Choose uncorrelated with each other
- Choose understood and controllable
- Balance complexity with performance

✅ **Estimation Methods**
- OLS: Exact, fast for small data
- Gradient Descent: Approximate, scales to large data

---

## 12. Study Questions

1. **What is the main advantage of multiple linear regression over simple linear regression?**
   <details>
   <summary>Answer</summary>
   Multiple regression considers multiple factors simultaneously, capturing more information and typically providing more accurate predictions than using a single predictor alone.
   </details>

2. **Write the equation for a multiple linear regression model predicting salary from years of experience, education level, and age.**
   <details>
   <summary>Answer</summary>
   Salary = θ₀ + θ₁×Experience + θ₂×Education + θ₃×Age
   </details>

3. **How would you encode a categorical variable "Vehicle Type" with values "Sedan", "SUV", "Truck" for use in regression?**
   <details>
   <summary>Answer</summary>
   Use one-hot encoding: Create three binary variables (IsSeduan, IsSUV, IsTruck), then drop one to avoid multicollinearity. For example, keep IsSUV and IsTruck, where Sedan is represented by both being 0.
   </details>

4. **What is multicollinearity and why is it a problem?**
   <details>
   <summary>Answer</summary>
   Multicollinearity occurs when independent variables are highly correlated with each other. It's problematic because: (1) coefficients become unstable, (2) can't isolate individual effects, (3) what-if scenarios become unrealistic, (4) standard errors inflate.
   </details>

5. **If a model has coefficients θ₀=100, θ₁=50 (age), θ₂=20 (income in $1000s), predict output for age=30, income=$60k.**
   <details>
   <summary>Answer</summary>
   ŷ = 100 + 50(30) + 20(60) = 100 + 1,500 + 1,200 = 2,800
   </details>

6. **What's the difference between OLS and Gradient Descent for estimating coefficients?**
   <details>
   <summary>Answer</summary>
   OLS provides an exact solution using matrix operations (fast for small data), while Gradient Descent iteratively approximates the solution (better for large datasets, more scalable).
   </details>

7. **Name three pitfalls of multiple linear regression.**
   <details>
   <summary>Answer</summary>
   (1) Overfitting from too many variables, (2) Multicollinearity from correlated predictors, (3) Unreliable extrapolation beyond training data range.
   </details>

8. **What criteria should guide variable selection for multiple regression?**
   <details>
   <summary>Answer</summary>
   Choose variables that are: (1) highly correlated with target, (2) uncorrelated with each other, (3) well-understood with clear interpretation, (4) controllable and measurable.
   </details>

---

## 13. Practical Exercise

### Exercise: Student Performance Prediction

**Scenario:** Build a multiple regression model to predict final exam scores.

**Available Variables:**
- Study Hours (0-40)
- Sleep Hours (4-10)
- Previous Test Score (0-100)
- Attendance % (60-100)
- Stress Level (1-10)
- Hours on Social Media (0-8)

**Given Correlations:**

| Variable | Correlation with Exam Score | Correlation with Study Hours |
|----------|----------------------------|----------------------------|
| Study Hours | +0.75 | 1.00 |
| Sleep Hours | +0.45 | -0.30 |
| Previous Test | +0.82 | +0.40 |
| Attendance | +0.68 | +0.55 |
| Stress Level | -0.52 | +0.25 |
| Social Media | -0.61 | -0.70 |

**Tasks:**

1. Which variables would you include in your model and why?
2. Are any variables likely to cause multicollinearity?
3. Write the regression equation using your selected variables
4. If coefficients are: θ₀=20, θ₁=1.5 (study), θ₂=15 (previous test), θ₃=0.3 (attendance), predict score for: Study=25 hrs, Previous=75, Attendance=90%

---

### Solution

<details>
<summary>Click to see solution</summary>

**1. Variable Selection:**

**Include:**
- ✅ **Previous Test Score** (r=+0.82, strongest predictor, independent measure)
- ✅ **Study Hours** (r=+0.75, strong predictor, actionable)
- ✅ **Attendance** (r=+0.68, good predictor, relatively independent)
- ✅ **Sleep Hours** (r=+0.45, moderate predictor, independent of study hours)

**Exclude:**
- ❌ **Stress Level** (r=-0.52, moderate but may correlate with study hours)
- ❌ **Social Media** (r=-0.70, strongly correlated with study hours r=-0.70, causes multicollinearity)

**2. Multicollinearity Concerns:**

**Problem pair:** Study Hours and Social Media (r=-0.70)
- High negative correlation
- Essentially measure opposite sides of same behavior
- Including both would cause unstable coefficients
- **Solution:** Keep Study Hours (more direct predictor), drop Social Media

**3. Regression Equation:**

ExamScore = θ₀ + θ₁×StudyHours + θ₂×PreviousTest + θ₃×Attendance + θ₄×SleepHours

**4. Prediction:**

Given: θ₀=20, θ₁=1.5, θ₂=15, θ₃=0.3
(Assuming θ₄ not provided, or model doesn't include sleep)

```
ExamScore = 20 + 1.5(25) + 15(75) + 0.3(90)
          = 20 + 37.5 + 1,125 + 27
          = 1,209.5

Wait, this doesn't make sense! Score > 100!
```

**Issue:** Coefficient θ₂=15 is too large. Let's use realistic coefficients:

If θ₂=0.15 (not 15):
```
ExamScore = 20 + 1.5(25) + 0.15(75) + 0.3(90)
          = 20 + 37.5 + 11.25 + 27
          = 95.75

Predicted Score: 96% ✅ (realistic!)
```

**Interpretation:**
- Base score: 20 points
- Each study hour: +1.5 points (25 hrs → +37.5)
- Each previous test point: +0.15 points (75 → +11.25)
- Each attendance %: +0.3 points (90 → +27)

</details>

---

## Conclusion

Multiple linear regression extends simple regression by incorporating multiple predictors, enabling more accurate and realistic models. Success requires careful variable selection, awareness of pitfalls like overfitting and multicollinearity, and choosing appropriate estimation methods. When done correctly, multiple regression provides powerful insights into complex relationships and enables sophisticated what-if scenario analysis.

**Next Lesson:** Polynomial and Non-Linear Regression - handling curved relationships.

---

*These notes are part of Module 2 of the IBM AI Engineering Professional Certificate course.*
