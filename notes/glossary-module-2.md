# Module 2 Glossary: Linear and Logistic Regression

## Regression Fundamentals

**Regression**
A type of supervised learning where the goal is to predict a continuous numerical value (e.g., price, temperature, sales).
*Example: Zillow predicting a house will sell for $487,350 based on square footage, location, and bedrooms—outputting a specific dollar amount rather than a category like "expensive" or "cheap."*

**Linear Regression**
A statistical method that models the relationship between a dependent variable and one or more independent variables using a linear equation (straight line).
*Example: Modeling the relationship between advertising spend (independent variable) and sales revenue (dependent variable) with the equation Sales = 50,000 + 2.5 × AdSpend, meaning every additional $1 in advertising generates $2.50 in sales.*

**Simple Linear Regression**
Linear regression with a single independent variable (one predictor). Equation: $y = mx + b$ or $y = \beta_0 + \beta_1x$
*Example: Predicting a person's weight based solely on their height using Weight = -100 + 2.5 × Height, where height is the only predictor and the model draws a single straight line through the data.*

**Multiple Linear Regression**
Linear regression with two or more independent variables. Equation: $y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n$
*Example: Predicting house prices using Price = 50,000 + 150 × SquareFeet + 20,000 × Bedrooms + 30,000 × HasGarage - 5,000 × Age, combining multiple factors to make more accurate predictions than using square footage alone.*

**Dependent Variable**
The variable being predicted or explained in a regression model. Also called the response variable, target, or output variable.
*Example: In a model predicting customer lifetime value, the dependent variable is the total dollar amount each customer will spend—this is what you're trying to predict based on factors like age, income, and purchase history.*

**Independent Variable**
A variable used to predict the dependent variable. Also called predictor, feature, or input variable.
*Example: In predicting exam scores, independent variables might include hours_studied (8 hours), previous_GPA (3.5), attendance_rate (95%), and hours_slept (7 hours)—all used to predict the final exam score.*

**Coefficient**
The numerical weight assigned to each independent variable in a regression equation, indicating the strength and direction of the relationship.
*Example: In Sales = 10,000 + 300 × WebsiteVisitors + 0.05 × EmailsSent, the coefficient 300 means each additional website visitor increases sales by $300, while the coefficient 0.05 means each email sent adds only $0.05 to sales.*

**Intercept**
The value of the dependent variable when all independent variables are zero. Represented as $\beta_0$ or $b$ in the regression equation.
*Example: In the equation Salary = 45,000 + 2,500 × YearsExperience, the intercept 45,000 represents the predicted starting salary for someone with zero years of experience—their baseline salary before experience adds value.*

**Slope**
The coefficient that describes the rate of change in the dependent variable for a one-unit change in the independent variable.
*Example: In Temperature_F = 32 + 1.8 × Temperature_C, the slope 1.8 means for every 1-degree increase in Celsius, Fahrenheit increases by 1.8 degrees—so going from 10°C to 11°C means going from 50°F to 51.8°F.*

**Continuous Variable**
A variable that can take any numerical value within a range, including decimals, as opposed to discrete categories.
*Example: House prices ($347,253.50), temperature (72.3°F), CO2 emissions (195.7 g/km) are continuous—they can be any number with decimal precision, unlike categories like "expensive/cheap" or "hot/cold."*

**Target Variable**
The variable being predicted in a supervised learning model. Also called dependent variable, response variable, or output variable.
*Example: In predicting customer lifetime value, the target is total spending ($5,247.83)—this is what you're trying to predict based on age, income, and purchase history.*

**Explanatory Features**
The input variables used to explain or predict the target variable. Also called independent variables, predictors, or features.
*Example: To predict salary, explanatory features include years_of_experience (8), education_level (Masters), industry (Technology), and location (San Francisco)—all used to explain why someone earns their salary.*

**Nonlinear Regression**
Regression where the relationship between variables is curved rather than straight, requiring polynomial, exponential, or other nonlinear functions.
*Example: Website traffic growth follows exponential curve—first month 100 visitors, second 300, third 900, fourth 2,700 (×3 growth each month)—cannot be captured by straight line, requires exponential model like Traffic = 100 × 3^months.*

**Polynomial Regression**
A type of nonlinear regression that uses polynomial terms (squared, cubed, etc.) to model curved relationships between variables.
*Example: Crop yield vs fertilizer follows polynomial: Yield = 50 + 30×Fertilizer - 2×(Fertilizer)²—initially each kg of fertilizer adds 30 bushels, but excessive fertilizer (above 7.5kg) actually reduces yield due to over-fertilization, creating a curved relationship.*

**Best-Fit Line**
The single straight line that passes closest to all data points in a scatter plot, minimizing the total distance (sum of squared residuals) to all points.
*Example: Plotting 100 houses' size vs price shows scattered points; the best-fit line might be Price = 50,000 + 150×SqFt, positioned so the total squared distance from all 100 points to this line is smaller than any other possible line.*

**Predicted Value (ŷ)**
The output of a regression model for a given input, represented as ŷ (y-hat), as opposed to the actual observed value y.
*Example: Model predicts ŷ = $450,000 for a house, but actual sale price y = $465,000—the predicted value ($450k) differs from actual value ($465k) by $15k residual error.*

**Ordinary Least Squares (OLS)**
A regression method that finds coefficients by minimizing the sum of squared residuals, providing a closed-form mathematical solution without iteration.
*Example: OLS regression on 1,000 house sales calculates θ₀ = 50,000 and θ₁ = 150 directly using formulas from Gauss and Legendre—no trial-and-error, no hyperparameters, just pure calculation taking milliseconds.*

**Bias Coefficient**
Another name for the intercept (θ₀) in a regression equation, representing the predicted value when all independent variables equal zero.
*Example: In Salary = 45,000 + 2,500×Experience, the bias coefficient 45,000 is the starting salary (when experience = 0 years), also called the y-intercept or constant term.*

**Scatter Plot**
A graph displaying the relationship between two variables as points, with the independent variable on the x-axis and dependent variable on the y-axis, used to visualize correlation patterns.
*Example: Plotting engine size (x-axis) vs CO2 emissions (y-axis) for 500 cars shows points trending upward from bottom-left to top-right, visually confirming that larger engines produce more emissions before fitting any regression line.*

**Correlation**
A statistical measure describing the strength and direction of a linear relationship between two variables, ranging from -1 to +1.
*Example: Ice cream sales and temperature have correlation +0.92 (strong positive—hotter days mean more sales), while temperature and heating bills have -0.88 (strong negative—hotter days mean lower heating), and shoe size and IQ have 0.02 (no relationship).*

**Hyperplane**
In multiple linear regression with more than two features, the decision boundary that separates or predicts values—a flat surface in high-dimensional space.
*Example: Predicting house prices using size, bedrooms, and age creates a 3D plane; adding location, garage, and condition creates a hyperplane in 6D space that we can't visualize but mathematically defines predictions like Price = 50k + 150×Size + 20k×Bedrooms - 5k×Age + 30k×Location + 15k×Garage - 2k×Condition.*

**One-Hot Encoding**
Converting categorical variables into binary (0/1) columns, one for each category, to make them usable in regression models.
*Example: Converting FuelType (Gasoline, Diesel, Electric) into three columns: Is_Gasoline=[1,0,0], Is_Diesel=[0,1,0], Is_Electric=[0,0,1]—then drop one column (say Gasoline) to avoid dummy variable trap, so Diesel car becomes [1,0] and Electric becomes [0,1].*

**What-If Scenario**
Hypothetical changes to input features to predict outcomes, used for sensitivity analysis and understanding variable impacts.
*Example: Healthcare model shows patient's blood pressure = 140 mmHg at BMI=30; what-if scenario asks "If patient reduces BMI to 28, what's new BP?"—answer: BP = 136 mmHg, showing 2-point BMI reduction lowers pressure by 4 mmHg.*

**Feature Matrix (X)**
The matrix containing all independent variables for all samples, typically with rows as samples and columns as features, often including a column of 1's for the bias term.
*Example: For 1,000 houses with 3 features (size, bedrooms, age), X is a 1000×4 matrix—first column all 1's (for intercept), then size, bedrooms, age—allowing matrix operation X×θ to compute all 1,000 predictions simultaneously.*

**Weight Vector (θ)**
The column vector containing all coefficients including the intercept (θ₀, θ₁, θ₂, ..., θₙ), used in matrix form of regression equation ŷ = Xθ.
*Example: For model Price = 50,000 + 150×Size + 20,000×Bedrooms, weight vector θ = [50000, 150, 20000]ᵀ—multiplying feature matrix X by this vector produces all predictions in one operation.*

**Collinearity / Collinear Variables**
When two or more independent variables are highly correlated with each other, making them no longer truly independent and causing instability in coefficient estimates.
*Example: Including both "Square Feet" (2,400 sq ft) and "Square Meters" (223 sq m) in house price model—they measure the same thing with perfect correlation (r=1.0), making coefficients unstable and what-if scenarios impossible (can't change one without changing the other).*

**Variance Inflation Factor (VIF)**
A quantitative measure of multicollinearity for each variable, calculated as VIF = 1/(1-R²) where R² is from regressing that variable on all other variables. VIF > 10 indicates problematic collinearity.
*Example: Predicting CO2 from EngineSize and Cylinders—regressing EngineSize on Cylinders gives R²=0.92, so VIF = 1/(1-0.92) = 12.5—exceeding threshold of 10 means these variables are too correlated and one should be removed.*

**Dummy Variable Trap**
The error of including all categories from one-hot encoding, creating perfect multicollinearity because all dummy variables sum to 1, making the matrix non-invertible.
*Example: Encoding Color (Red, Blue, Green) creates Is_Red, Is_Blue, Is_Green—but including all three causes trap because Is_Red + Is_Blue + Is_Green = 1 always. Solution: drop one (say Green), so Red=[1,0], Blue=[0,1], Green=[0,0].*

## Classification Fundamentals

**Classification**
A type of supervised learning where the goal is to predict a categorical label or class (e.g., spam/not spam, positive/negative).
*Example: Gmail automatically categorizing incoming emails as "spam" or "not spam," or a bank's fraud detection system labeling transactions as "fraudulent" or "legitimate"—outputting categories rather than numbers.*

**Binary Classification**
Classification problems with exactly two possible classes or categories (e.g., yes/no, 0/1, true/false).
*Example: A medical test predicting whether a patient has diabetes (Yes) or not (No), or a loan approval system deciding to "Approve" or "Deny" an application—only two possible outcomes.*

**Multiclass Classification**
Classification problems with more than two possible classes (e.g., classifying animals into dog, cat, bird, etc.).
*Example: Amazon automatically categorizing products into departments like "Electronics," "Clothing," "Books," "Home & Kitchen," or "Sports"—choosing from dozens of possible categories rather than just two.*

**Classifier**
A model or algorithm that performs classification tasks by assigning input data to predefined categories.
*Example: A spam filter classifier that examines email features (sender, subject, content) and outputs "spam" or "not spam," or an image classifier that looks at pixels and outputs "cat," "dog," or "bird."*

**Class Label**
The category or group to which an observation belongs in a classification problem.
*Example: In a customer churn dataset, each customer has a class label of either "Churned" or "Retained," or in medical imaging, each scan is labeled "Normal," "Benign," or "Malignant."*

**Positive Class**
In binary classification, the class of primary interest (e.g., "disease present," "spam email").
*Example: In cancer detection, "cancer present" is the positive class because that's what doctors are trying to detect, or in fraud detection, "fraudulent transaction" is positive because that's the important event to catch.*

**Negative Class**
In binary classification, the class representing the absence of the condition of interest (e.g., "no disease," "legitimate email").
*Example: In cancer screening, "no cancer detected" is the negative class representing healthy patients, or in email filtering, "legitimate email" is negative because it's normal, expected mail.*

## Logistic Regression

**Logistic Regression**
A statistical method used for binary classification that estimates the probability of an observation belonging to a particular class.
*Example: Predicting whether a customer will click on an ad—the model outputs probability 0.73 (73% chance of clicking), then converts to decision "Yes, will click" if above 0.5 threshold, or "No, won't click" if below.*

**Sigmoid Function (Logistic Function)**
A mathematical function that maps any real-valued number to a value between 0 and 1, used in logistic regression to convert linear outputs to probabilities. Formula: $\sigma(z) = \frac{1}{1 + e^{-z}}$
*Example: A linear combination produces z = 2.5, the sigmoid converts this to probability: 1/(1+e^-2.5) = 0.924 or 92.4% probability—squashing any number (even -100 or +100) into valid 0-1 probability range.*

**Odds**
The ratio of the probability of an event occurring to the probability of it not occurring. Odds = $\frac{p}{1-p}$
*Example: If probability of winning is 0.75 (75%), then odds = 0.75/0.25 = 3, meaning "3 to 1 odds" or "3 times more likely to win than lose"—same as saying 3 wins for every 1 loss.*

**Log Odds (Logit)**
The natural logarithm of the odds, which forms the linear component in logistic regression. $\text{logit}(p) = \ln\left(\frac{p}{1-p}\right)$
*Example: If probability of disease is 0.9, odds = 0.9/0.1 = 9, and log odds = ln(9) = 2.197—logistic regression models this log odds as a linear equation like 2.197 = -5 + 0.3×Age + 2×Smoker.*

**Probability**
A numerical value between 0 and 1 representing the likelihood of an event occurring.
*Example: Weather forecast showing 70% (0.70) chance of rain means if conditions are repeated 100 times, it would rain about 70 times—or a model predicting 0.15 probability a customer churns (15% chance of leaving).*

**Decision Boundary**
The threshold (typically 0.5 for binary classification) that separates one class from another in the output space.
*Example: In fraud detection, if predicted probability > 0.5, classify as "fraud," otherwise "legitimate"—so a transaction with 0.51 probability is flagged as fraud, but 0.49 passes as legitimate, even though they're very close.*

**Threshold**
The probability cutoff used to convert continuous probability predictions into discrete class labels.
*Example: In cancer screening, setting threshold at 0.3 (30%) means flag any case with ≥30% probability as "needs further testing" to catch more cases (high recall), while 0.8 threshold means only flag very confident cases to reduce false alarms (high precision).*

## Model Training and Optimization

**Cost Function (Loss Function)**
A mathematical function that measures how well a model's predictions match the actual values. The goal of training is to minimize this function.
*Example: For house price predictions, if model predicts $400k but actual is $450k (error of $50k), and another predicts $500k (error of $50k), the cost function aggregates all these errors into a single number like MSE = $2,500,000,000 that training tries to minimize.*

**Mean Squared Error (MSE)**
A common cost function for regression that calculates the average of the squared differences between predicted and actual values. $\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
*Example: Predicting 3 house prices: errors are $10k, -$20k, $5k; squared errors are 100M, 400M, 25M; MSE = (100M + 400M + 25M)/3 = 175M—squaring penalizes large errors more heavily than small ones.*

**Root Mean Squared Error (RMSE)**
The square root of MSE, providing an error metric in the same units as the target variable. $\text{RMSE} = \sqrt{\text{MSE}}$
*Example: If MSE = 10,000 (dollars squared), RMSE = √10,000 = $100, meaning on average predictions are off by about $100—easier to interpret than "10,000 squared dollars."*

**Log Loss (Binary Cross-Entropy)**
A cost function used in logistic regression that measures the performance of a classification model whose output is a probability between 0 and 1.
*Example: Predicting spam (actual label = 1) with probability 0.9 gives low log loss ≈0.105, but predicting 0.1 probability gives high log loss ≈2.303—heavily penalizing confident wrong predictions.*

**Gradient Descent**
An optimization algorithm used to minimize the cost function by iteratively adjusting model parameters in the direction of steepest descent.
*Example: Imagine standing on a mountain (cost function) blindfolded—gradient descent takes steps downhill (opposite of gradient) repeatedly: step→measure slope→step downhill→repeat until reaching the valley bottom (minimum cost).*

**Learning Rate**
A hyperparameter that controls the size of steps taken during gradient descent. Too large can cause overshooting; too small can make training very slow.
*Example: With learning rate 0.001, model takes tiny careful steps (10,000 iterations to converge), but with 0.1 takes big steps (100 iterations)—however, 1.0 might overshoot and bounce around the minimum forever without converging.*

**Convergence**
The point at which the training algorithm stops improving the model because it has found a minimum of the cost function.
*Example: Training loss drops from 1000→500→280→150→85→43→22→11→10.8→10.7→10.65—when changes become tiny (< 0.01 per iteration), the model has converged and further training won't help much.*

**Epoch**
One complete pass through the entire training dataset during the training process.
*Example: With 10,000 training examples, one epoch means the model has seen all 10,000 examples once—training for 50 epochs means each example is used 50 times to update the model, typically improving accuracy each epoch.*

**Iteration**
A single update of the model parameters during training, typically processing one batch of data.
*Example: With 10,000 examples and batch size 100, one epoch = 100 iterations (10,000/100)—each iteration processes 100 examples, calculates gradient, and updates weights once.*

**Mean Absolute Error (MAE)**
The average of the absolute values of prediction errors, providing an easy-to-interpret metric in the same units as the target variable.
*Example: Predicting house prices with errors of +$15k, -$10k, +$25k, -$5k gives MAE = (15k+10k+25k+5k)/4 = $13.75k—meaning on average, predictions are off by $13,750, easier to understand than squared errors.*

**R-squared (R² or Coefficient of Determination)**
A statistical measure representing the proportion of variance in the dependent variable explained by the independent variable(s). Values range from 0 to 1.
*Example: R² = 0.85 means 85% of variance in house prices is explained by square footage, bedrooms, and location—the model captures most but not all factors (missing 15% from things like renovations or market timing).*

**Adjusted R-squared**
A modified version of R² that adjusts for the number of predictors in the model, penalizing unnecessary complexity.
*Example: Adding 50 random variables to a model increases R² from 0.80 to 0.82 but adjusted R² drops from 0.80 to 0.65—revealing those variables don't truly help and just overfit the training data.*

**Train/Test Split**
The practice of dividing a dataset into separate training and testing subsets to evaluate model generalization on unseen data.
*Example: Splitting 1,000 car records into 800 for training (80%) and 200 for testing (20%)—model learns from 800 cars and proves it can generalize by accurately predicting emissions for the 200 unseen test cars.*

**Out-of-Sample Accuracy**
The performance of a model on data it has never seen during training, providing an honest estimate of generalization ability.
*Example: Model achieves 95% accuracy on training data but only 78% on test data—the 78% out-of-sample accuracy is the true measure of how well it will perform on new real-world data, revealing 17% overfitting.*

**Random State**
A seed value that ensures reproducible random splitting of data, allowing consistent train/test splits across multiple runs.
*Example: Using `random_state=42` in train_test_split ensures every time you run the code, the same 800 records go to training and same 200 to testing—critical for comparing different models fairly and debugging.*

**Reshape**
A NumPy operation that changes the dimensions of an array without changing its data, often needed to convert 1D arrays to 2D for sklearn models.
*Example: X_train with shape (800,) is a 1D array, but sklearn expects 2D input with shape (n_samples, n_features), so `X_train.reshape(-1, 1)` converts to (800, 1)—800 rows, 1 column, meeting sklearn's requirements.*

## Model Performance

**Residual**
The difference between the actual value and the predicted value for a single observation. $\text{residual} = y_{\text{actual}} - y_{\text{predicted}}$
*Example: Actual house price is $500k, model predicts $480k, residual = $500k - $480k = $20k (model underestimated by $20k)—positive residuals mean underestimation, negative means overestimation.*

**Sum of Squared Residuals (SSR)**
The sum of the squared differences between actual and predicted values, used as a measure of model error.
*Example: Three predictions with residuals $10k, -$15k, $8k give SSR = (10k)² + (-15k)² + (8k)² = 100M + 225M + 64M = 389M—lower SSR means better model fit to the data.*

**R-squared (R² or Coefficient of Determination)**
A statistical measure representing the proportion of variance in the dependent variable explained by the independent variable(s). Values range from 0 to 1.
*Example: R² = 0.85 means 85% of variance in house prices is explained by square footage, bedrooms, and location—the model captures most but not all factors (missing 15% from things like renovations or market timing).*

**Adjusted R-squared**
A modified version of R² that adjusts for the number of predictors in the model, penalizing unnecessary complexity.
*Example: Adding 50 random variables to a model increases R² from 0.80 to 0.82 but adjusted R² drops from 0.80 to 0.65—revealing those variables don't truly help and just overfit the training data.*

**Accuracy**
The proportion of correct predictions (both true positives and true negatives) among all predictions made.
*Example: Model makes 1000 predictions: 450 spam correctly identified, 500 legitimate correctly identified, 30 false positives, 20 false negatives—accuracy = (450+500)/1000 = 95%.*

**Confusion Matrix**
A table used to evaluate classification performance, showing true positives, true negatives, false positives, and false negatives.
*Example: Cancer screening of 1000 patients shows matrix: TN=900 (healthy correctly identified), TP=80 (cancer caught), FP=10 (false alarms), FN=10 (cancer missed)—revealing high accuracy but 10 dangerous misses.*

**True Positive (TP)**
A prediction where the model correctly predicts the positive class.
*Example: Model predicts "fraud" and transaction is actually fraudulent—correctly catching a bad actor, or predicting "disease present" when patient truly has the disease.*

**True Negative (TN)**
A prediction where the model correctly predicts the negative class.
*Example: Model predicts "not spam" and email is legitimately from your bank—correctly allowing good email through, or predicting "no disease" for a healthy patient.*

**False Positive (FP)**
A prediction where the model incorrectly predicts the positive class (Type I error).
*Example: Spam filter marks your boss's important email as spam (predicted spam, actually legitimate)—causing you to miss critical messages, or airport security falsely flagging an innocent person.*

**False Negative (FN)**
A prediction where the model incorrectly predicts the negative class (Type II error).
*Example: Spam filter lets obvious spam through (predicted not spam, actually spam)—your inbox fills with junk, or cancer screening missing actual cancer (predicted no cancer, actually has cancer) with serious consequences.*

## Statistical Concepts

**Correlation**
A statistical measure describing the strength and direction of a linear relationship between two variables, ranging from -1 to +1.
*Example: Ice cream sales and temperature have correlation +0.92 (strong positive—hotter days mean more sales), while temperature and heating bills have -0.88 (strong negative—hotter days mean lower heating), and shoe size and IQ have 0.02 (no relationship).*

**Causation**
A relationship where one variable directly influences or causes changes in another variable. Correlation does not imply causation.
*Example: Ice cream sales correlate with drowning deaths (both peak in summer), but ice cream doesn't cause drowning—the hidden cause is warm weather. However, smoking does cause lung cancer (true causation proven through controlled studies).*

**Multicollinearity**
A situation in multiple regression where two or more independent variables are highly correlated, which can cause problems in estimating coefficients.
*Example: Including both "square feet" and "number of rooms" to predict house price—they're highly correlated (bigger houses have more rooms), making it unclear which variable truly drives price, and causing unstable coefficient estimates.*

**Outlier**
A data point that differs significantly from other observations, potentially having a large impact on the regression model.
*Example: Predicting salaries: most data ranges $40k-$120k, but one CEO makes $5M—this outlier pulls the regression line upward, making predictions inaccurate for typical employees; removing it improves model for 99% of cases.*

**Homoscedasticity**
The assumption that the variance of residuals is constant across all levels of the independent variable(s).
*Example: Predicting income from years of education—residuals (prediction errors) should be roughly ±$10k whether predicting for 12 years or 20 years of education, not ±$5k at low education but ±$50k at high education.*

**Heteroscedasticity**
A violation of the homoscedasticity assumption, where residuals have non-constant variance.
*Example: Predicting stock returns—small companies show huge prediction errors (±50% return variance) while large stable companies show tiny errors (±5% variance), making model less reliable and confidence intervals incorrect.*

**Linearity**
The assumption that the relationship between independent and dependent variables can be adequately represented by a linear equation.
*Example: Doubling advertising spend from $10k to $20k doubles sales from $100k to $200k, and tripling to $30k triples sales to $300k—a perfectly linear relationship that fits y = mx + b.*

**Normal Distribution**
A bell-shaped probability distribution that is symmetric around the mean, often assumed for residuals in linear regression.
*Example: In a well-fit model, prediction errors should cluster around zero (most predictions close to actual) with fewer extreme errors—like 68% of errors within ±$10k, 95% within ±$20k, forming the classic bell curve.*

## Model Assumptions and Limitations

**Assumptions of Linear Regression**
Key requirements for valid linear regression: linearity, independence of errors, homoscedasticity, normality of residuals, and no multicollinearity.
*Example: Before trusting house price predictions, verify: (1) price increases linearly with size, (2) one house's error doesn't affect another's, (3) prediction errors are similar for small and large houses, (4) errors form bell curve, (5) size and rooms aren't too correlated.*

**Limitations**
Constraints or weaknesses of a model, such as inability to capture non-linear relationships or sensitivity to outliers.
*Example: Linear regression can't capture that car value drops 20% immediately upon purchase then gradually thereafter (non-linear), or that one $10M mansion skews all predictions upward (outlier sensitivity)—requiring different models.*

**Non-linear Relationships**
Relationships between variables that cannot be adequately represented by a straight line, requiring more complex models.
*Example: Marketing returns follow diminishing returns—first $10k in ads generates $100k sales, next $10k generates $50k, next $10k only $25k (logarithmic curve), not the constant $100k per $10k that linear regression would predict.*

**Extrapolation**
Making predictions outside the range of the training data, which can be unreliable and should be done with caution.
*Example: Model trained on houses 1,000-3,000 sq ft ($200k-$600k) predicting a 10,000 sq ft mansion will cost $2M—but luxury market follows different rules (location, amenities), making this extrapolation dangerously inaccurate.*

**Interpolation**
Making predictions within the range of the training data, generally more reliable than extrapolation.
*Example: Model trained on houses 1,000-3,000 sq ft reliably predicts a 2,000 sq ft house because it has seen similar examples—like predicting between known data points 1,500 sq ft ($350k) and 2,500 sq ft ($500k).*

**Feature Scaling / Standardization**
Transforming features to similar scales to prevent features with larger magnitudes from dominating the model. Includes standardization (z-score) and normalization (min-max).
*Example: House dataset has sqft (1,000-4,000) and bedrooms (2-6)—without scaling, sqft dominates because its values are 100x larger. Standardization converts both to mean=0, std=1, so sqft becomes [-1.2, 0.5, 1.8] and bedrooms [-0.9, 0.1, 1.3], equally weighted.*

**StandardScaler**
A scikit-learn preprocessing tool that standardizes features by removing the mean and scaling to unit variance: z = (x - μ) / σ.
*Example: Income feature has mean=$75,000 and std=$25,000; StandardScaler transforms $100,000 to z=(100k-75k)/25k=+1.0 and $50,000 to z=(50k-75k)/25k=-1.0, centering data at 0 with spread of 1.*

**Z-Score Normalization**
Standardization technique that transforms data to have mean=0 and standard deviation=1 using formula: z = (x - mean) / std_dev.
*Example: Test scores have mean=75, std=10; student scoring 85 gets z-score=(85-75)/10=+1.0 (one standard deviation above average), while 60 gets z=(60-75)/10=-1.5 (1.5 std devs below average).*

## Implementation Terms

**fit() method**
The scikit-learn method used to train a model on training data.
*Example: `model.fit(X_train, y_train)` trains the model on features X_train and labels y_train—like showing a student 1,000 practice problems with answers so they learn the patterns before taking the test.*

**predict() method**
The scikit-learn method used to make predictions using a trained model.
*Example: `predictions = model.predict(X_test)` generates predictions for new data—after training on house features, predict prices for 200 new houses like [450000, 380000, 520000, ...] without needing actual prices.*

**predict_proba() method**
A method in classification models that returns probability estimates for each class.
*Example: `probabilities = model.predict_proba(X_test)` returns [[0.23, 0.77], [0.91, 0.09]] meaning first customer has 77% probability of churning, second has 9% probability—giving confidence levels instead of just Yes/No predictions.*

**coef_ attribute**
In scikit-learn, the learned coefficients (weights) of a fitted linear model.
*Example: After training, `model.coef_` returns [150, 20000, -5000] meaning price increases $150 per square foot, $20k per bedroom, and decreases $5k per year of age—revealing what the model learned about each feature's importance.*

**intercept_ attribute**
In scikit-learn, the learned intercept (bias term) of a fitted linear model.
*Example: `model.intercept_` returns 50000, meaning the baseline house price is $50k before considering size, bedrooms, or age—the starting point where the regression line crosses the y-axis.*

**LinearRegression class**
The scikit-learn class implementing ordinary least squares linear regression.
*Example: `from sklearn.linear_model import LinearRegression; model = LinearRegression(); model.fit(X, y)` creates and trains a model to find the best-fit line minimizing squared errors for continuous predictions like prices or temperatures.*

**LogisticRegression class**
The scikit-learn class implementing logistic regression for classification.
*Example: `from sklearn.linear_model import LogisticRegression; model = LogisticRegression(); model.fit(X, y)` trains a classifier for binary outcomes like spam/not spam or fraud/legitimate, outputting probabilities and class predictions.*

---

*Last updated: November 11, 2025*
