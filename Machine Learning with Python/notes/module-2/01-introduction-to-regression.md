# Introduction to Regression

**Date:** November 11, 2025  
**Module:** 2 - Linear and Logistic Regression  
**Topic:** Introduction to Regression - Fundamentals and Applications  

---

## Overview

This lesson introduces **regression**, a fundamental supervised learning technique for predicting continuous numerical values. Regression models the relationship between a target variable and explanatory features, enabling predictions for new data based on learned patterns.

### Learning Objectives

After completing this lesson, you will be able to:
- ✅ Define regression and explain its purpose in machine learning
- ✅ Compare simple regression and multiple regression
- ✅ Distinguish between linear and nonlinear regression
- ✅ Explain real-world applications of regression across multiple domains
- ✅ Identify appropriate use cases for regression analysis

---

## 1. What is Regression?

### Definition

> **Regression** is a type of supervised learning model that models a relationship between a continuous target variable and explanatory features (predictors).

### Key Characteristics

**Type:** Supervised Learning
- Requires labeled training data
- Learns from input-output pairs
- Makes predictions on new, unseen data

**Output:** Continuous Values
- Predicts numerical values (not categories)
- Examples: prices, temperatures, emissions, rainfall amounts
- Range: Any real number within reasonable bounds

**Purpose:** Modeling Relationships
- Identifies patterns between features and target
- Quantifies strength and direction of relationships
- Enables prediction based on learned patterns

---

## 2. Motivating Example: CO2 Emissions

### The Dataset

Consider a dataset of automobile CO2 emissions with the following features:

| Feature | Description | Example Values |
|---------|-------------|----------------|
| **Engine Size** | Engine displacement in liters | 2.0L, 3.5L, 5.0L |
| **Number of Cylinders** | Count of engine cylinders | 4, 6, 8 |
| **Fuel Consumption** | Liters per 100 km | 8.5, 11.2, 14.7 |
| **CO2 Emissions** | Grams per kilometer (target) | 180g, 240g, 320g |

### The Question

**Can we predict the CO2 emission of a new car from the listed features?**

**Answer: Yes!** ✅

---

### The Regression Approach

```
Step 1: Historical Data
┌─────────────────────────────────────────────────┐
│ Past Cars Dataset                                │
│ - 1000 cars with known features and emissions  │
│ - Engine size, cylinders, fuel consumption      │
│ - Measured CO2 emissions                         │
└─────────────────────────────────────────────────┘
                    ↓
Step 2: Model Training
┌─────────────────────────────────────────────────┐
│ Regression Algorithm                             │
│ - Learns relationship between features/emissions│
│ - Identifies patterns and correlations          │
│ - Builds predictive model                       │
└─────────────────────────────────────────────────┘
                    ↓
Step 3: Prediction
┌─────────────────────────────────────────────────┐
│ New/Hypothetical Car                             │
│ Input: Engine=2.5L, Cylinders=4, Fuel=9.0L/100km│
│ Output: Predicted CO2 = 195g/km                 │
└─────────────────────────────────────────────────┘
```

**Key Insight:** Using potentially predictive features, regression can predict continuous values like CO2 emissions for cars never seen during training.

---

## 3. Types of Regression

### Overview

```
Regression
├── Simple Regression (1 independent variable)
│   ├── Simple Linear Regression
│   └── Simple Nonlinear Regression
│
└── Multiple Regression (2+ independent variables)
    ├── Multiple Linear Regression
    └── Multiple Nonlinear Regression
```

**Choosing the Right Type:**
- Depends on the data available for dependent variable
- Depends on the type of model that provides best fit
- Consider: number of predictors and nature of relationships

---

## 4. Simple Regression

### Definition

> **Simple Regression:** A single independent variable estimates a dependent variable.

### Characteristics

**Structure:**
- One predictor → One target
- Simplest form of regression
- Easy to visualize and interpret

**Types:**
1. **Simple Linear Regression** - Straight line relationship
2. **Simple Nonlinear Regression** - Curved relationship

---

### Simple Linear Regression

**Relationship:** Straight line (linear)

**Equation:** 
$$y = \beta_0 + \beta_1x$$

or equivalently:
$$y = mx + b$$

Where:
- $y$ = dependent variable (target)
- $x$ = independent variable (predictor)
- $\beta_0$ (or $b$) = intercept
- $\beta_1$ (or $m$) = slope

**Example: CO2 Emissions**

```
Predicting: CO2 Emissions
Using: Engine Size only

Model: CO2 = 50 + 40 × EngineSize

Interpretation:
- Base emissions: 50 g/km (intercept)
- Each additional liter: +40 g/km (slope)

Predictions:
- 2.0L engine: CO2 = 50 + 40(2.0) = 130 g/km
- 3.5L engine: CO2 = 50 + 40(3.5) = 190 g/km
- 5.0L engine: CO2 = 50 + 40(5.0) = 250 g/km
```

**Visual:**
```
CO2 Emissions (g/km)
    |
300 |                    •
    |                •
250 |            •
    |        •
200 |    •
    | •
150 |____________________________________________
    1.0   2.0   3.0   4.0   5.0   6.0
                Engine Size (L)

Note: Points fall on/near straight line
```

---

### Simple Nonlinear Regression

**Relationship:** Curved (nonlinear)

**Characteristics:**
- Relationship cannot be captured by straight line
- Requires polynomial, exponential, or other nonlinear functions
- More flexible but also more complex

**Example: CO2 Emissions (Nonlinear)**

```
Model: CO2 = 20 + 30×EngineSize + 5×(EngineSize)²

Predictions:
- 2.0L: CO2 = 20 + 30(2) + 5(4) = 100 g/km
- 3.0L: CO2 = 20 + 30(3) + 5(9) = 155 g/km
- 5.0L: CO2 = 20 + 30(5) + 5(25) = 295 g/km

Note: Emissions increase faster at larger engine sizes
      (curved relationship)
```

---

## 5. Multiple Regression

### Definition

> **Multiple Regression:** More than one independent variable is used to estimate the dependent variable.

### Characteristics

**Structure:**
- Multiple predictors → One target
- More realistic for complex real-world problems
- Can capture combined effects of multiple factors

**Types:**
1. **Multiple Linear Regression** - Linear combination of predictors
2. **Multiple Nonlinear Regression** - Nonlinear relationships

---

### Multiple Linear Regression

**Relationship:** Linear combination of multiple variables

**Equation:**
$$y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n$$

Where:
- $y$ = dependent variable
- $x_1, x_2, ..., x_n$ = independent variables
- $\beta_0$ = intercept
- $\beta_1, \beta_2, ..., \beta_n$ = coefficients

**Example: CO2 Emissions (Multiple Predictors)**

```
Predicting: CO2 Emissions
Using: Engine Size AND Number of Cylinders

Model: CO2 = 30 + 25×EngineSize + 15×Cylinders

Interpretation:
- Base emissions: 30 g/km
- Each liter of engine: +25 g/km
- Each cylinder: +15 g/km

Predictions:
┌─────────┬──────────┬──────────────┐
│ Engine  │ Cylinders│ Predicted CO2│
├─────────┼──────────┼──────────────┤
│ 2.0L    │ 4        │ 30+50+60=140 │
│ 3.5L    │ 6        │ 30+87.5+90=207.5│
│ 5.0L    │ 8        │ 30+125+120=275│
└─────────┴──────────┴──────────────┘

Note: Model considers both factors simultaneously
```

**Advantage:** More accurate predictions by combining multiple relevant features.

---

### Multiple Nonlinear Regression

**Relationship:** Nonlinear relationships between variables

**Characteristics:**
- Can capture complex interactions
- May include polynomial terms, interactions, or nonlinear functions
- Most flexible but requires more data

**Example:**
```
Model: CO2 = 20 + 30×EngineSize + 15×Cylinders + 
             5×(EngineSize)² + 2×(EngineSize×Cylinders)

Terms:
- Linear: EngineSize, Cylinders
- Polynomial: (EngineSize)²
- Interaction: EngineSize × Cylinders
```

---

## 6. Comparison: Simple vs Multiple Regression

| Aspect | Simple Regression | Multiple Regression |
|--------|-------------------|---------------------|
| **Predictors** | One (1) | Multiple (2+) |
| **Equation** | $y = \beta_0 + \beta_1x$ | $y = \beta_0 + \beta_1x_1 + ... + \beta_nx_n$ |
| **Complexity** | Low | Medium to High |
| **Visualization** | 2D line/curve | Multi-dimensional space |
| **Accuracy** | Lower (limited info) | Higher (more info) |
| **Interpretability** | Very easy | More complex |
| **Use Case** | Initial exploration, simple relationships | Real-world predictions, complex relationships |

### Visual Comparison

```
Simple Linear Regression (2D)
     y |     •
       |   •   •
       | •   /
       |   / •
       | /   •
       |____________ x

Multiple Linear Regression (3D plane)
       z
       |
       |    •    •
       |  •   /  •
       | •  /   •
       |  /  •
       |/________ y
      /
     x

Note: With 4+ variables, cannot visualize directly
```

---

## 7. Real-World Applications of Regression

### When to Use Regression

**Core Principle:** Use regression when you want to **estimate a continuous value**.

**Not for:** Classification (categories), Clustering (groups), or Anomaly detection

---

### Business Applications

#### 1. Sales Forecasting

**Problem:** Predict future sales revenue

**Independent Variables:**
- Number of customers
- Number of leads generated
- Order history
- Marketing spend
- Seasonal factors

**Example:**
```
Model: Annual Sales = 50,000 + 500×Customers + 
                      100×Leads + 0.3×OrderHistory

Prediction for salesperson:
- 200 customers
- 500 leads
- $100,000 order history

Sales = 50,000 + 500(200) + 100(500) + 0.3(100,000)
      = 50,000 + 100,000 + 50,000 + 30,000
      = $230,000 predicted annual sales
```

---

#### 2. House Price Prediction

**Problem:** Estimate real estate property values

**Independent Variables:**
- Size (square footage)
- Number of bedrooms
- Number of bathrooms
- Location (zip code, neighborhood)
- Age of house
- Lot size
- Amenities (garage, pool, etc.)

**Example:**
```
Model: Price = 100,000 + 150×SqFt + 20,000×Bedrooms + 
               15,000×Bathrooms + 50,000×HasGarage

Prediction for house:
- 2,000 sq ft
- 3 bedrooms
- 2 bathrooms
- Has garage

Price = 100,000 + 150(2000) + 20,000(3) + 15,000(2) + 50,000(1)
      = 100,000 + 300,000 + 60,000 + 30,000 + 50,000
      = $540,000
```

---

#### 3. Predictive Maintenance

**Problem:** Predict when equipment will need maintenance

**Independent Variables:**
- Operating hours
- Temperature readings
- Vibration levels
- Age of equipment
- Previous maintenance history
- Load/stress levels

**Example:**
```
Automobile Maintenance Prediction:
Model: DaysUntilMaintenance = 10,000 - 0.5×Mileage - 
                               50×Age_Years - 100×AvgSpeed

Prediction:
- 50,000 miles driven
- 5 years old
- Average speed 65 mph

Days = 10,000 - 0.5(50,000) - 50(5) - 100(65)
     = 10,000 - 25,000 - 250 - 6,500
     = -21,750 (maintenance overdue!)

Industrial Machine Prediction:
Predict hours until bearing failure based on vibration,
temperature, and runtime hours.
```

**Benefit:** Proactive maintenance rather than reactive (waiting for failure)

---

#### 4. Employment Income Prediction

**Problem:** Estimate salary/income levels

**Independent Variables:**
- Hours of work per week
- Education level
- Occupation category
- Sex/gender
- Age
- Years of experience
- Geographic location
- Industry

**Example:**
```
Model: Income = 20,000 + 5,000×EducationYears + 
                2,000×Experience + 500×HoursPerWeek

Prediction for worker:
- 16 years education (Bachelor's)
- 10 years experience
- 40 hours per week

Income = 20,000 + 5,000(16) + 2,000(10) + 500(40)
       = 20,000 + 80,000 + 20,000 + 20,000
       = $140,000 per year
```

---

### Environmental Applications

#### 5. Rainfall Estimation

**Problem:** Predict precipitation amounts

**Independent Variables (Meteorological Factors):**
- Temperature
- Humidity
- Wind speed
- Air pressure (barometric)
- Cloud cover
- Dew point

**Example:**
```
Model: Rainfall_mm = -50 + 2×Humidity + 0.5×CloudCover - 
                     0.1×AirPressure + 0.3×Temperature

Prediction for tomorrow:
- 80% humidity
- 70% cloud cover
- 1010 mb air pressure
- 25°C temperature

Rainfall = -50 + 2(80) + 0.5(70) - 0.1(1010) + 0.3(25)
         = -50 + 160 + 35 - 101 + 7.5
         = 51.5 mm (moderate rain expected)
```

---

#### 6. Wildfire Risk Assessment

**Problem:** Determine probability and severity of wildfires

**Independent Variables:**
- Temperature
- Humidity
- Wind speed
- Vegetation density
- Drought conditions
- Historical fire data
- Topography

**Example:**
```
Severity Score (0-100):
Model: Severity = 10 + 2×Temperature - 1.5×Humidity + 
                  3×WindSpeed + 0.5×DroughtDays

Prediction:
- 35°C (hot)
- 15% humidity (dry)
- 30 km/h wind
- 60 days since rain

Severity = 10 + 2(35) - 1.5(15) + 3(30) + 0.5(60)
         = 10 + 70 - 22.5 + 90 + 30
         = 177.5 → Capped at 100 (Extreme Risk)
```

---

### Healthcare Applications

#### 7. Disease Spread Prediction

**Problem:** Predict spread of infectious diseases

**Independent Variables:**
- Population density
- Contact rate
- Vaccination rate
- Previous infection rate
- Seasonality
- Travel patterns

**Example:**
```
Model: NewCases = 50 + 0.1×CurrentCases + 0.5×PopulationDensity -
                  10×VaccinationRate + 20×ContactRate

Prediction for next week:
- 1,000 current cases
- 10,000 people per sq km
- 60% vaccination rate
- Contact rate: 5 people/day

NewCases = 50 + 0.1(1000) + 0.5(10000) - 10(60) + 20(5)
         = 50 + 100 + 5000 - 600 + 100
         = 4,650 predicted new cases
```

---

#### 8. Disease Risk Estimation

**Problem:** Estimate likelihood of developing diseases

**Independent Variables:**
- Age
- BMI (Body Mass Index)
- Blood pressure
- Cholesterol levels
- Family history
- Lifestyle factors (smoking, exercise)
- Blood glucose levels

**Example: Diabetes Risk**
```
Model: DiabetesRiskScore = -10 + 0.5×Age + 2×BMI + 
                           0.1×BloodGlucose + 20×FamilyHistory

Prediction for patient:
- 55 years old
- BMI 32 (obese)
- Blood glucose 110 mg/dL
- Family history: Yes (1)

Risk = -10 + 0.5(55) + 2(32) + 0.1(110) + 20(1)
     = -10 + 27.5 + 64 + 11 + 20
     = 112.5 (High risk - needs intervention)
```

---

### Industries Using Regression

**Summary Table:**

| Industry | Application | Predicted Value |
|----------|-------------|-----------------|
| **Finance** | Credit scoring | Default probability, loan amount |
| **Healthcare** | Patient outcomes | Recovery time, readmission risk |
| **Retail** | Demand forecasting | Product sales, inventory needs |
| **Real Estate** | Property valuation | House prices, rental rates |
| **Manufacturing** | Quality control | Defect rates, production output |
| **Transportation** | Route optimization | Travel time, fuel consumption |
| **Energy** | Load forecasting | Power demand, consumption patterns |
| **Agriculture** | Crop yield | Harvest quantity, optimal planting time |

---

## 8. Regression Algorithms

### Overview of Algorithms

Regression encompasses many different algorithms, each suited to specific conditions and contexts.

---

### Classical Statistical Methods

**Linear Regression**
- Simple and interpretable
- Assumes linear relationships
- Fast training and prediction

**Polynomial Regression**
- Captures nonlinear patterns
- Uses polynomial terms
- Can overfit with high degrees

**Characteristics:**
- Well-established mathematical theory
- Decades of use
- Strong assumptions about data
- Excellent for inference and interpretation

---

### Modern Machine Learning Methods

**Random Forest Regressor**
- Ensemble of decision trees
- Handles nonlinear relationships
- Robust to outliers
- Works well with many features

**XGBoost (Extreme Gradient Boosting)**
- State-of-the-art performance
- Gradient boosting framework
- Efficient and scalable
- Handles missing data

**K-Nearest Neighbors (KNN)**
- Non-parametric method
- Predicts based on similar instances
- No training phase
- Sensitive to feature scaling

**Support Vector Machines (SVM)**
- Can use kernel tricks
- Works in high-dimensional spaces
- Effective for small to medium datasets

**Neural Networks**
- Deep learning approach
- Extremely flexible
- Requires large datasets
- Can capture very complex patterns

---

### Algorithm Selection Guide

| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| **Linear Regression** | Simple relationships, interpretability | Fast, interpretable | Assumes linearity |
| **Polynomial** | Curved relationships | Flexible | Can overfit |
| **Random Forest** | Complex nonlinear data | Robust, accurate | Less interpretable |
| **XGBoost** | Competitions, high accuracy | State-of-art performance | Requires tuning |
| **KNN** | Small datasets, no assumptions | Simple, no training | Slow prediction |
| **SVM** | Medium data, high dimensions | Effective with kernels | Expensive for large data |
| **Neural Networks** | Very large datasets, complex patterns | Extremely flexible | Requires lots of data |

---

## 9. Key Takeaways

### Core Concepts

✅ **Regression Definition**
- Supervised learning technique
- Models relationships between continuous target and features
- Predicts numerical values (not categories)

✅ **Simple vs Multiple Regression**
- **Simple:** One independent variable → One dependent variable
- **Multiple:** Multiple independent variables → One dependent variable
- Multiple regression generally more accurate (more information)

✅ **Linear vs Nonlinear**
- **Linear:** Straight-line relationships
- **Nonlinear:** Curved, complex relationships
- Applies to both simple and multiple regression

✅ **Wide Applications**
- **Business:** Sales forecasting, price prediction, maintenance scheduling
- **Environment:** Rainfall estimation, wildfire risk
- **Healthcare:** Disease spread, risk assessment
- **Many others:** Finance, retail, manufacturing, transportation

✅ **Many Algorithms Available**
- **Classical:** Linear, polynomial regression
- **Modern ML:** Random Forest, XGBoost, KNN, SVM, Neural Networks
- Choose based on data characteristics and requirements

---

## 10. Study Questions

Test your understanding of regression concepts:

1. **What is the primary difference between regression and classification?**
   <details>
   <summary>Answer</summary>
   Regression predicts continuous numerical values (e.g., price, temperature), while classification predicts discrete categories (e.g., spam/not spam, yes/no).
   </details>

2. **If you want to predict house prices using only square footage, which type of regression would you use?**
   <details>
   <summary>Answer</summary>
   Simple linear regression (one predictor, assuming linear relationship between size and price).
   </details>

3. **Why might multiple regression be more accurate than simple regression?**
   <details>
   <summary>Answer</summary>
   Multiple regression considers multiple factors simultaneously, capturing more information about what influences the target variable, leading to more accurate predictions.
   </details>

4. **Give three real-world applications of regression from different industries.**
   <details>
   <summary>Answer</summary>
   Examples: (1) Finance - credit risk scoring, (2) Healthcare - predicting patient recovery time, (3) Retail - demand forecasting for inventory management.
   </details>

5. **When would you choose nonlinear regression over linear regression?**
   <details>
   <summary>Answer</summary>
   When the relationship between variables is curved rather than straight-line, such as diminishing returns in marketing spend or accelerating growth patterns.
   </details>

6. **What are the key differences between classical regression methods (linear/polynomial) and modern ML methods (Random Forest/XGBoost)?**
   <details>
   <summary>Answer</summary>
   Classical methods are interpretable with strong mathematical theory but assume linear relationships. Modern ML methods are more flexible, handle nonlinear patterns better, and often more accurate, but less interpretable.
   </details>

7. **In the CO2 emissions example, what are the independent variables and what is the dependent variable?**
   <details>
   <summary>Answer</summary>
   Independent (predictors): Engine size, number of cylinders, fuel consumption. Dependent (target): CO2 emissions.
   </details>

8. **Why is regression useful for predictive maintenance?**
   <details>
   <summary>Answer</summary>
   Regression can predict when equipment will fail based on operating conditions, allowing proactive maintenance before failure occurs, reducing downtime and costs.
   </details>

---

## 11. Practical Exercise

### Exercise: Building Your First Regression Model (Conceptual)

**Scenario:** You work for a real estate company and want to predict apartment rental prices.

**Available Data:**
- 1,000 apartments with known rental prices
- Features: Square footage, number of bedrooms, floor level, distance to subway (km), age of building (years)

**Tasks:**

1. **Identify the problem type:**
   - Is this regression or classification? Why?
   - What is the dependent variable?
   - What are the independent variables?

2. **Choose regression type:**
   - Should you use simple or multiple regression? Why?
   - Would you expect linear or nonlinear relationships?

3. **Formulate the model:**
   - Write a conceptual equation for your regression model
   - Example format: Price = β₀ + β₁(Feature1) + β₂(Feature2) + ...

4. **Make predictions:**
   - If your model is: Price = 500 + 1.5×SqFt + 200×Bedrooms - 50×DistanceSubway
   - Predict rent for: 800 sq ft, 2 bedrooms, 0.5 km from subway

5. **Identify algorithm:**
   - Would you start with linear regression or a more complex algorithm?
   - What would you consider when choosing?

---

### Solution

<details>
<summary>Click to see solution</summary>

1. **Problem Type:**
   - **Regression** (predicting continuous price value, not a category)
   - **Dependent variable:** Rental price (dollars per month)
   - **Independent variables:** Square footage, bedrooms, floor level, distance to subway, building age

2. **Regression Type:**
   - **Multiple regression** (multiple predictors available)
   - **Likely linear initially** (reasonable starting assumption that price increases with size, decreases with distance to subway)
   - May add nonlinear terms later if needed

3. **Conceptual Model:**
   ```
   Price = β₀ + β₁(SqFt) + β₂(Bedrooms) + β₃(Floor) + 
           β₄(DistanceSubway) + β₅(Age)
   
   Expected signs:
   - SqFt: Positive (bigger = more expensive)
   - Bedrooms: Positive (more rooms = more expensive)
   - Floor: Positive (higher floors often preferred)
   - DistanceSubway: Negative (farther = less desirable)
   - Age: Negative (older = less expensive)
   ```

4. **Prediction:**
   ```
   Price = 500 + 1.5(800) + 200(2) - 50(0.5)
        = 500 + 1,200 + 400 - 25
        = $2,075 per month
   ```

5. **Algorithm Choice:**
   - **Start with Linear Regression:**
     * Simple and interpretable
     * Fast to train
     * Good baseline
     * Easy to explain to stakeholders
   
   - **Consider upgrading if:**
     * Linear model has poor accuracy
     * Relationships appear nonlinear in exploratory analysis
     * Have sufficient data (>10,000 apartments)
     * Willing to trade interpretability for accuracy
   
   - **Progression:**
     * Linear Regression → Polynomial Regression → Random Forest → XGBoost

</details>

---

## Summary

This lesson introduced **regression** as a fundamental machine learning technique for predicting continuous values. You learned:

- **Regression** models relationships between continuous targets and explanatory features
- **Simple regression** uses one predictor; **multiple regression** uses multiple predictors
- Both can be **linear** (straight-line) or **nonlinear** (curved)
- **Applications** span business (sales, pricing), environment (weather, wildfires), healthcare (disease prediction), and many other domains
- **Many algorithms** exist, from classical (linear, polynomial) to modern ML (Random Forest, XGBoost, neural networks)

Regression is one of the most widely used machine learning techniques due to its versatility and the prevalence of continuous prediction problems in the real world.

---

**Next Lesson:** Introduction to Simple Linear Regression - diving deeper into the mathematics and implementation of linear regression with one predictor.

---

*These notes are part of Module 2 of the IBM AI Engineering Professional Certificate course.*
