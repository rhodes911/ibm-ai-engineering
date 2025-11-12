# A Day in the Life of a Machine Learning Engineer

**Date:** November 10, 2025  
**Module:** 1 - Introduction to Machine Learning  
**Topic:** Practical Walkthrough of ML Model Lifecycle  
**Presenter:** Isioma (ML Engineer)

---

## Overview

This lesson provides a **real-world walkthrough** of the ML model lifecycle through the eyes of a working ML engineer building a product recommendation system for a beauty products company.

### Project Goal
Create and deploy a model that recommends similar beauty products based on customer purchase history to increase business revenue.

---

## The End-User Pain Point

**User Story:**
> "As a beauty product customer, I would like to receive recommendations for other products based on my purchase history so that I will be able to address my skincare needs and improve the overall health of my skin."

### Why This Matters
- Clearly defines **who** (beauty product customer)
- States **what** they need (product recommendations)
- Explains **why** (address skincare needs, improve skin health)
- Aligns ML solution with client's business needs (increase revenue)

**Key Learning:** Always start with understanding the user's problem before building anything!

---

## Stage 1: Problem Definition

### Activities Performed
✓ Work with client to understand business goals  
✓ Define end-user pain point  
✓ Create user story  
✓ Ensure ML solution aligns with client needs  

### Why This Is Critical
> **"Defining the problem or stating the situation is very important, because I want to make sure the machine learning solution I am providing is aligned with the client's needs."**

### In This Project
- **Business Goal:** Increase revenue through personalized recommendations
- **User Need:** Discover products for skincare needs
- **ML Solution:** Similarity-based product recommendation system
- **Success Metrics:** (implied) Click-through rate, purchase conversion, user satisfaction ratings

### Time Investment
⏱️ **Low to Medium** — But absolutely critical! A misaligned solution wastes all subsequent effort.

---

## Stage 2: Data Collection

### Key Questions to Answer
1. What kind of data does the company have?
2. Where will the data come from (sources)?
3. What data is relevant to the problem?

### Data Sources Identified

#### User Data
- **Demographics:** Age, gender, location, skin type
- **Purchase History:** Products bought, quantities, dates, prices
- **Transaction Data:** Completed purchases, payment info, order details

#### Product Data
- **Inventory:** Available products
- **Product Attributes:** 
  - What the product does (cleanser, moisturizer, serum)
  - Ingredients list
  - Target skin concerns (acne, dryness, aging)
- **Popularity Metrics:** Sales volume, trending products
- **Customer Ratings:** Star ratings, number of reviews

#### Behavioral Data (Other Sources)
- **Saved Products:** Items in wishlist
- **Liked Products:** Favorites, thumbs up
- **Search History:** Keywords searched, products viewed
- **Most Visited Products:** Time spent on product pages
- **Browsing Patterns:** Click sequences, category navigation

### Data Transformation Activities
After collection, perform major transformations:
- **Wrangling:** Cleaning and organizing messy data
- **Aggregating:** Combining data at different levels (daily → monthly sales)
- **Joining:** Combining related data from different tables
- **Merging:** Bringing together user and product data
- **Mapping:** Aligning data from different sources onto one schema

### Central Data Repository
> **Goal:** "This reduces the need to deal with multiple databases every time we need to pull data."

Creating a **single source of truth** makes subsequent work much easier.

### Time Investment
⏱️ **HIGH** — Gathering from multiple sources is time-consuming!

---

## Stage 3: Data Preparation

### The Messy Reality
> "Most of the time, data from multiple sources will contain errors, different formatting, and missing data."

### Overlap with Data Collection
**Important Note:** Data preparation overlaps with data collection—they can be done in tandem (simultaneously).

### Focus of This Stage
Preparing a **"somewhat final version of the data"** ready for modeling.

---

### Data Cleaning Activities

#### 1. Filter Out Irrelevant Data
**Example:** Remove test accounts, admin users, cancelled orders

#### 2. Remove Extreme Values (Outliers)
**Why:** Avoid influencing the dataset unfairly
**Example:** 
- Customer who bought 1,000 products (likely bulk reseller, not typical user)
- Product priced at $0.01 (data entry error)

#### 3. Handle Missing Values
Two approaches:
- **Remove:** Delete rows/columns with too many missing values
- **Impute:** Fill in missing values
  - Random generation (careful with this!)
  - Mean/median imputation
  - Forward fill / backward fill for time series

**Example:** If "skin type" is missing, might impute based on products purchased

#### 4. Proper Data Formatting
Ensure each column has correct data type:
- **Dates:** Convert strings like "2024-11-10" to datetime format
- **Numbers:** Ensure prices are float, not string "$45.99"
- **Strings:** Text fields properly identified
- **Categories:** Categorical variables encoded properly

---

### Feature Engineering

Creating additional features to improve model performance.

#### Example Features Created

**1. Average Duration Between Transactions**
```
For User #123:
- Purchase 1: Jan 1, 2024
- Purchase 2: Jan 15, 2024 (14 days later)
- Purchase 3: Feb 10, 2024 (26 days later)
→ Average duration: 20 days
```

**2. Most Purchased Products**
```
User #123 bought:
- Cleanser: 5 times
- Moisturizer: 3 times
- Serum: 1 time
→ Top product category: Cleansers
```

**3. Skin Issue Targeting Feature**
```
Products targeting:
- Acne: [Product A, Product C]
- Dryness: [Product B, Product D]
- Aging: [Product E]

User #123 purchased Products A, B, C
→ User concerns: Acne (2 products), Dryness (1 product)
```

---

### Exploratory Data Analysis (EDA)

#### Visual Pattern Identification
- **Plots:** Histograms, scatter plots, heat maps
- **Purpose:** Visually identify patterns and relationships

**Example Visualizations:**
- Product ratings distribution
- Purchase frequency by age group
- Seasonal buying patterns
- Price vs. purchase volume

#### Expert Validation
> "Validate the data based on information that the beauty product subject matter expert has given me."

Work with domain experts to ensure data makes sense!

**Example:** Expert confirms that users with dry skin typically buy products with hyaluronic acid

#### Correlation Analysis
Identify important variables/features for users' buying habits

**Example Findings:**
- Strong correlation between "skin type: dry" and "moisturizer purchases"
- Users searching for "fragrance-free" avoid products with perfumes
- High ratings correlate with repeat purchases

---

### Data Splitting Strategy

#### Key Decision: How to Split?

**Option 1: Random Split**
- Randomly assign 70% training, 15% validation, 15% test
- Pros: Simple, unbiased
- Cons: May not reflect real-world scenario

**Option 2: Temporal Split (Chosen for this project)**
- **Training Set:** All transactions except most recent per user
- **Test Set:** Most recent transaction per user
- **Constraint:** Ensure at least one transaction by same user in training set

**Why This Approach?**
```
User #123's transactions:
1. Jan 2024: Cleanser → Training
2. Mar 2024: Moisturizer → Training  
3. Oct 2024: Serum → Testing

Model learns from past purchases (Jan, Mar)
Tests on most recent (Oct) to simulate real recommendations
```

This mimics production: recommending products based on history to predict future purchases!

### Time Investment
⏱️ **VERY HIGH** — Typically 60-80% of total project time!

---

## Stage 4: Model Development

### Practical Approach
> "Realistically, I try to leverage as many pre-existing frameworks and resources as possible, so I don't create anything from scratch."

**Key Lesson:** Don't reinvent the wheel! Use established algorithms and libraries.

---

### Technique 1: Content-Based Filtering

#### What It Is
Finds similarity between products based on **product content/attributes**.

#### How It Works
1. Analyze product features (ingredients, purpose, skin type target)
2. Calculate similarity scores between products
3. Recommend products similar to what user already bought

#### Example Logic
```
User bought: "Gentle Cleanser for Dry Skin"
Product attributes:
- Contains: Water, Glycerin, Hyaluronic Acid
- For: Dry skin
- Type: Cleanser
- Property: Gentle, Moisturizing

Similar product: "Hydrating Moisturizer for Dry Skin"
- Contains: Glycerin, Hyaluronic Acid, Ceramides
- For: Dry skin
- Type: Moisturizer
- Property: Moisturizing

Recommendation Logic: User has dry skin and likes moisture-focused 
products → Recommend moisturizer with similar ingredients
```

#### Implementation Steps
1. **Create similarity scores** between purchased products and all products
2. **Rank products** by similarity score
3. **Recommend top-ranked** most similar products

#### Additional Considerations
Factor in user preferences:
- **Negative signals:** User searched "paraben-free" → Exclude products with parabens
- **Ingredient avoidance:** Don't recommend products user explicitly doesn't want

---

### Technique 2: Collaborative Filtering

#### What It Is
Uses **user data** and behavior to find similarities between users.

#### Core Idea
> "If User A and User B have similar preferences, products that User A likes will likely appeal to User B too."

#### How It Works

**Step 1: Create User Similarity**
Based on how users view/rate products
```
User A ratings:          User B ratings:
Product 1: 5 stars       Product 1: 5 stars
Product 2: 4 stars       Product 2: 4 stars  
Product 3: 2 stars       Product 3: 3 stars
→ Users A and B are similar (similar rating patterns)
```

**Step 2: Group Users**
Create user buckets based on characteristics:
- **Demographics:** Age group (20-30), region (urban), skin type (oily)
- **Behavior:** Products rated, products purchased
- **Preferences:** Favorite brands, ingredient preferences

**Step 3: Make Predictions**
For new user with limited history:
- Find which bucket they belong to
- Take **average ratings** from existing members
- Assume new user will rate similarly to group average
- **Recommend products** highly rated by similar users

#### Example Scenario
```
New User Profile:
- Age: 25
- Skin type: Combination
- Purchased: 1 cleanser (rated 5 stars)

Similar User Group (25-30 age, combination skin):
- Average rating for "Vitamin C Serum": 4.8 stars
- 85% purchased it after buying cleanser
→ Recommend Vitamin C Serum to new user
```

---

### Final Model: Hybrid Approach

> "The final model will be a combination of the two techniques."

#### Why Combine?
- **Content-Based:** Good for recommending similar products, works for new products
- **Collaborative:** Good for discovering unexpected preferences, works for popular items
- **Hybrid:** Gets best of both worlds!

#### Example Combined Recommendation
```
User bought: Cleanser for acne-prone skin

Content-Based suggests:
1. Acne treatment serum (similar purpose)
2. Oil-free moisturizer (similar skin type)

Collaborative Filtering suggests:
1. Vitamin C serum (users like you bought this)
2. Clay mask (highly rated by similar users)

Final Top 3 Recommendations:
1. Acne treatment serum (both methods agree - highest confidence)
2. Vitamin C serum (collaborative + complementary)
3. Oil-free moisturizer (content-based match)
```

### Time Investment
⏱️ **MEDIUM to HIGH** — Depends on complexity and experimentation needed.

---

## Stage 5: Model Evaluation

### Two-Phase Evaluation

---

### Phase 1: Initial Testing (Offline Evaluation)

#### Activities
✓ **Tune the model:** Adjust hyperparameters for better performance  
✓ **Test on held-out data:** Use the test set created earlier  
✓ **Validate recommendations:** Ensure they make sense  

#### Evaluation Questions
- Are recommendations relevant to user's purchase history?
- Do recommendations match user's skincare needs?
- Are products diverse yet related?
- Are we avoiding products user explicitly dislikes?

#### Metrics to Check
- **Precision@K:** Of top 10 recommendations, how many are relevant?
- **Recall@K:** Of all relevant products, how many are in top 10?
- **NDCG:** Normalized Discounted Cumulative Gain (ranking quality)
- **Coverage:** Are we recommending diverse products, not just popular ones?

#### Example Results
```
Test User #123:
Actual next purchase: Moisturizer for dry skin

Top 5 Recommendations:
1. Hydrating Face Cream (✓ Correct category!)
2. Hyaluronic Acid Serum (✓ Related)
3. Night Repair Moisturizer (✓ Correct category!)
4. Vitamin C Serum (✓ Complementary)
5. Eye Cream (✓ Related)

4 out of 5 are relevant → 80% Precision@5
```

---

### Phase 2: User Feedback Testing (Online Evaluation)

> "Once I am satisfied with the results, I will further evaluate the model by experimenting with the recommendations on a group of users and asking for their feedback."

#### A/B Testing Setup
- **Group A (Control):** Random product recommendations
- **Group B (Test):** ML model recommendations
- Compare performance between groups

#### Data Collection Methods

**1. User Ratings**
Ask users to rate recommendations:
- "How relevant were these product suggestions?" (1-5 stars)
- "Did these recommendations meet your needs?" (Yes/No)

**2. Behavioral Metrics**
Track actual user actions:
- **Click-Through Rate (CTR):** % of users who click recommendations
- **Purchase Conversion:** % of users who buy recommended products
- **Average Order Value:** Do recommendations increase spending?
- **Return Rate:** Do users keep products they bought?

**3. Additional Metrics**
- Time spent viewing recommendations
- Number of products added to wishlist
- User session length (do recommendations keep users engaged?)

#### Example Feedback Analysis
```
Test Group Results (1,000 users, 2 weeks):
- Click-Through Rate: 15% (vs 8% control)
- Purchase Conversion: 6% (vs 3% control)
- Average Rating: 4.2/5 stars
- Revenue Impact: +$12,000 from recommendations

Qualitative Feedback:
- "Loved the moisturizer suggestion!" (User #445)
- "Finally found products for my skin type!" (User #892)
- "Would prefer cruelty-free options" (User #223) → Feature idea!
```

### Time Investment
⏱️ **MEDIUM** — Initial testing is quick; user testing takes weeks for meaningful data.

---

## Stage 6: Model Deployment

### Going to Production

> "Now that I am done with building and testing, the model is ready to go to production."

#### Deployment Targets
For this project:
- **Beauty product mobile app:** iOS and Android
- **E-commerce website:** Desktop and mobile web
- **Email campaigns:** Personalized product suggestions (potential)

#### Implementation Details

**Real-Time Recommendations:**
```
User opens app → Model API called → Recommendations displayed
Response time: < 100ms
```

**Batch Recommendations:**
```
Nightly batch process:
1. Update all user profiles
2. Pre-compute recommendations
3. Store in database
4. Serve instantly when user logs in
```

#### Deployment Considerations
- **Scalability:** Handle thousands of concurrent users
- **Latency:** Respond within milliseconds
- **Reliability:** 99.9% uptime requirement
- **Monitoring:** Track model performance in real-time

---

### Continuous Monitoring

> "While this is the last step, I still need to track the deployed model's performance to make sure it continues to do the job that the business requires."

#### What to Monitor

**Performance Metrics:**
- Recommendation accuracy
- Click-through rates over time
- Purchase conversion rates
- User engagement metrics

**System Health:**
- API response times
- Error rates
- System uptime
- Server resource usage

**Business Impact:**
- Revenue from recommendations
- Customer satisfaction scores
- Return on investment (ROI)

#### Alert Triggers

Set up alerts for:
- CTR drops below 10% (was 15%)
- API response time > 200ms
- Error rate > 1%
- User complaints increase

---

### Future Iterations

> "Future iterations may include retraining the model based on new information in order to expand its capabilities."

#### Retraining Schedule
- **Weekly:** Update with latest purchase data
- **Monthly:** Full model retraining
- **Quarterly:** Evaluate new algorithms and techniques

#### Capability Expansions

**Potential Improvements:**
1. **Seasonal Recommendations:** Sunscreen in summer, heavy creams in winter
2. **Ingredient Preferences:** Track which ingredients users avoid/prefer
3. **Multi-Product Routines:** Suggest complete skincare routines, not just single products
4. **Price Sensitivity:** Consider user's price range preferences
5. **Brand Loyalty:** Factor in favorite brands
6. **Social Proof:** "Users with similar skin bought this"
7. **Trending Products:** Highlight what's popular now

**New Data Sources:**
- Social media mentions
- Influencer partnerships
- Clinical study results
- Competitor product data

### Time Investment
⏱️ **ONGOING** — Deployment is not the end; it's the beginning of continuous improvement!

---

## Time Distribution Summary

Based on this real-world project:

| Stage | Time Investment | % of Project |
|-------|----------------|--------------|
| **Problem Definition** | Low-Medium | 5-10% |
| **Data Collection** | High | 15-20% |
| **Data Preparation** | **VERY HIGH** | **40-50%** |
| **Model Development** | Medium-High | 15-20% |
| **Model Evaluation** | Medium | 10-15% |
| **Deployment & Monitoring** | Ongoing | Continuous |

### Key Takeaway
> **Data Collection and Preparation consume 60-70% of project time!**

This is consistent across most ML projects—the majority of work is in getting quality data ready, not in fancy algorithms.

---

## Critical Success Factors

### 1. Clear Problem Definition
✓ Align with business needs  
✓ Understand user pain points  
✓ Define measurable success criteria  

### 2. Quality Data
✓ Gather from multiple relevant sources  
✓ Clean thoroughly  
✓ Validate with domain experts  

### 3. Appropriate Techniques
✓ Use proven methods (don't reinvent)  
✓ Combine techniques when beneficial  
✓ Consider both product and user perspectives  

### 4. Rigorous Evaluation
✓ Test offline first  
✓ Validate with real users  
✓ Collect both quantitative and qualitative feedback  

### 5. Continuous Improvement
✓ Monitor performance post-deployment  
✓ Plan for regular retraining  
✓ Expand capabilities based on learnings  

---

## Key Learnings

### From Isioma's Experience

**1. Alignment is Everything**
> "I want to make sure the machine learning solution I am providing is aligned with the client's needs."

**2. Data Work Dominates**
Most time is spent on data collection and preparation, not modeling.

**3. Use Existing Tools**
> "I try to leverage as many pre-existing frameworks and resources as possible."

**4. Hybrid Approaches Work**
Combining techniques (content-based + collaborative) often beats single methods.

**5. Deployment Isn't The End**
Continuous monitoring and improvement is required for long-term success.

**6. User Feedback Matters**
Real user testing reveals issues offline evaluation misses.

**7. Iteration is Normal**
> "Future iterations may include retraining the model based on new information."

---

## Practical Takeaways for Aspiring ML Engineers

### Skills You Need

**Technical Skills:**
- Python programming
- Data wrangling (pandas, NumPy)
- ML frameworks (scikit-learn, TensorFlow)
- SQL for data querying
- API development
- Cloud platforms (AWS, GCP, Azure)

**Domain Knowledge:**
- Understand the business you're working in
- Collaborate with subject matter experts
- Translate business problems to ML problems

**Soft Skills:**
- Communication with stakeholders
- Project management
- Iterative thinking
- Problem-solving

### Daily Activities Mix

An ML Engineer's day includes:
- 30-40%: Data work (cleaning, exploration, feature engineering)
- 20-30%: Model experimentation and tuning
- 15-20%: Meetings (stakeholders, team, domain experts)
- 10-15%: Deployment and monitoring
- 10%: Documentation and knowledge sharing

### Career Advice

**Start Simple:** Master data fundamentals before complex algorithms  
**Learn by Doing:** Build end-to-end projects like this one  
**Stay Current:** ML evolves rapidly; continuous learning essential  
**Collaborate:** Best solutions come from team efforts  
**Focus on Impact:** It's not about fancy models; it's about solving real problems  

---

## Study Questions

1. Why is the problem definition stage critical despite taking relatively less time?
2. What are the three main categories of data collected for the beauty product recommendation system?
3. Why does data preparation typically take 40-50% of project time?
4. What's the difference between content-based and collaborative filtering?
5. Why use a hybrid approach combining both techniques?
6. What are the two phases of model evaluation, and what's different about them?
7. Why is monitoring necessary after deployment?
8. How did the ML engineer decide to split data for this project, and why?
9. What are some potential future improvements mentioned for the recommendation system?
10. Why leverage existing frameworks rather than building from scratch?

---

## Practical Application Exercise

**Scenario:** You're building a movie recommendation system for a streaming service.

Map out your approach for each stage:

1. **Problem Definition:** What's the user pain point? Business goal?
2. **Data Collection:** What data sources would you use?
3. **Data Preparation:** What cleaning and feature engineering would you do?
4. **Model Development:** Which techniques would you use? Why?
5. **Evaluation:** How would you test if recommendations are good?
6. **Deployment:** Where would recommendations appear?
7. **Monitoring:** What metrics would you track?

---

*These notes are based on "A Day in the Life of a Machine Learning Engineer" from Module 1 of the IBM AI Engineering Professional Certificate course, presented by Isioma.*
