# Data Scientist vs AI Engineer

**Date:** November 10, 2025  
**Module:** 1 - Introduction to Machine Learning  
**Topic:** Comparing Data Science and AI Engineering Roles  
**Presenter:** Isaac Key (Former Data Scientist, now AI Engineer at IBM)

---

## Overview

This lesson explores the **four key areas** where Data Scientists and AI Engineers (specifically Generative AI Engineers) differ, highlighting how the advent of generative AI has created a new distinct field called AI Engineering.

### The Big Question
Are AI engineers just data scientists in disguise? **No**—they're complementary but distinct roles with different focuses.

---

## Industry Context: What Changed?

### Traditional Landscape
- Data scientists have **always used AI models** for analysis
- Machine learning has been core to data science for years

### The Generative AI Revolution
> "With the advent of generative AI, the boundaries of what AI can do are being pushed in ways that we've never seen before."

**Key Shift:** Generative AI breakthroughs were so groundbreaking that it **split off into its own distinct field** → **AI Engineering**

### Why This Matters
- Generative AI can create new content (text, images, code, etc.)
- Foundation models generalize across tasks without retraining
- New tools and workflows emerged specifically for gen AI
- Different skill sets and approaches needed

---

## The Four Key Differences

```
1. Use Cases (What problems they solve)
2. Data (What they work with)
3. Models (What algorithms they use)
4. Processes (How they build solutions)
```

---

## Difference 1: Use Cases

### Data Scientist: "Data Storyteller"
> "They take massive amounts of messy real-world data, and they use mathematical models to translate this data into insights."

### AI Engineer: "AI System Builder"
> "They use foundation models to build generative AI systems that help to transform business processes."

---

### Data Scientist Use Cases

#### A. Descriptive Analytics (Describing the Past)
Understanding what has happened.

**Exploratory Data Analysis (EDA)**
- Graphing data
- Statistical inference
- Understanding distributions and patterns

*Example:* Analyzing last quarter's sales data to understand which products sold best and why.

**Clustering**
- Group similar data points based on characteristics
- Customer segmentation

*Example:* Segmenting customers into "budget shoppers," "premium buyers," "occasional purchasers" based on purchase behavior.

---

#### B. Predictive Analytics (Predicting the Future)
> "Every good story has a reader trying to figure out what's going to come next, and that's where predictive use cases come in."

**Regression Models**
- Predict continuous numeric values
- Temperature, revenue, house prices

*Example:* Predicting next quarter's revenue based on historical trends, marketing spend, and seasonality.

**Classification Models**
- Predict categorical values
- Success/failure, yes/no, categories

*Example:* Predicting whether a customer will churn (yes/no) or classifying transactions as fraudulent or legitimate.

---

### AI Engineer Use Cases

#### A. Prescriptive Analytics (Choosing Best Actions)
Making recommendations about what should be done.

**Decision Optimization**
- Assess a set of possible actions
- Choose the most optimal path based on requirements/standards

*Example:* Determining the optimal distribution route for delivery trucks considering traffic, fuel costs, delivery windows, and customer priorities.

**Recommendation Engines**
- Suggest targeted actions or items

*Example:* Suggesting targeted marketing campaigns for specific customer segments, recommending when to send emails and what offers to include.

---

#### B. Generative Use Cases (Creating New Content)
Creating content that didn't exist before.

**Intelligent Assistants**
- Coding assistants (GitHub Copilot)
- Digital advisors (financial, health)

*Example:* A coding assistant that helps developers write code by suggesting completions, generating functions, and explaining complex code.

**Chatbots**
- Conversational search
- Information retrieval
- Content summarization

*Example:* A customer service chatbot that understands natural language questions, searches documentation, and provides personalized answers with sources.

---

### Use Case Comparison Table

| Type | Data Scientist | AI Engineer |
|------|---------------|-------------|
| **Descriptive** | ✓ Primary (EDA, clustering) | ○ Less common |
| **Predictive** | ✓ Primary (regression, classification) | ○ Sometimes |
| **Prescriptive** | ○ Sometimes | ✓ Primary (optimization, recommendations) |
| **Generative** | ✗ Rarely | ✓ Primary (assistants, chatbots) |

---

## Difference 2: Data

> "People say that data is a new oil because like oil, you have to search for and find the right data and then use the right processes to transform it into various products, which then power various processes."

---

### Data Scientist: Structured Data (Tabular)

**Primary Data Type:** Structured/Tabular data
- Like Excel spreadsheets or SQL databases
- Rows and columns
- Well-organized

**Scale:** 
- Hundreds to hundreds of thousands of observations
- Typical: 10,000 - 500,000 rows
- Dozens to hundreds of features/columns

**Data Processing:**
```
Raw Data → Heavy Cleaning & Preprocessing → Modeling
```

#### Cleaning Activities:
1. **Remove outliers** - Data points that are abnormally high/low
2. **Join tables** - Combine data from multiple sources
3. **Filter data** - Remove irrelevant records
4. **Create new features** - Engineer features from existing data
5. **Handle missing values** - Impute or remove
6. **Normalize/standardize** - Scale numeric features

**Example Dataset:**
```
Customer Purchase History (50,000 rows × 25 columns)
- Customer_ID, Age, Income, Purchase_Date, Product_Category
- Purchase_Amount, Payment_Method, Location, Previous_Purchases
- Customer_Since_Date, Email_Opened, Website_Visits, etc.
```

**Note:** Data scientists **still work with unstructured data**, but not as much as AI engineers.

---

### AI Engineer: Unstructured Data

**Primary Data Types:**
- **Text** (documents, articles, conversations)
- **Images** (photos, diagrams, screenshots)
- **Videos** (clips, movies, surveillance footage)
- **Audio** (speech, music, sound effects)
- **Code** (programming languages)

**Scale:** MASSIVE
- **Billions to trillions** of tokens for training LLMs
- Much larger than traditional ML datasets

**Example: Large Language Model (LLM)**
```
Training Data Scale:
- GPT-3: ~300 billion tokens
- GPT-4: Estimated 1+ trillion tokens
- Token ≈ 3/4 of a word on average

For context: 
- Harry Potter series ≈ 1 million words
- English Wikipedia ≈ 4 billion words
```

**Processing Focus:**
- Less emphasis on traditional "cleaning"
- More focus on data quality and diversity
- Prompt engineering for interaction
- Fine-tuning on domain-specific data

---

### Data Comparison

| Aspect | Data Scientist | AI Engineer |
|--------|---------------|-------------|
| **Primary Type** | Structured (tabular) | Unstructured (text, images, video) |
| **Scale** | Hundreds to hundreds of thousands | Billions to trillions (tokens) |
| **Preparation** | Heavy cleaning, feature engineering | Quality filtering, prompt design |
| **Storage** | Databases, data warehouses | Data lakes, object storage |
| **Format** | CSV, SQL tables, Parquet | Raw files, embeddings, vectors |

---

## Difference 3: Underlying Models

### Data Scientist Toolbox: Hundreds of Models

**Characteristics:**
- **Narrow scope** - Each model specialized for specific task
- **Smaller size** - Thousands to millions of parameters
- **Less compute** - Can train on CPU or single GPU
- **Fast training** - Seconds to hours
- **Task-specific** - Each use case = new model

**Model Variety:**
```
Classification:
├── Logistic Regression
├── Decision Trees
├── Random Forest
├── Support Vector Machines (SVM)
├── K-Nearest Neighbors (KNN)
└── Gradient Boosting (XGBoost, LightGBM)

Regression:
├── Linear Regression
├── Polynomial Regression
├── Ridge/Lasso Regression
└── Regression Trees

Clustering:
├── K-Means
├── Hierarchical Clustering
├── DBSCAN
└── Gaussian Mixture Models

And hundreds more...
```

**Key Limitation:** Hard to generalize beyond training domain
*Example:* A model trained on English text reviews can't suddenly analyze French reviews or predict house prices—it's specialized.

---

### AI Engineer Toolbox: Foundation Models

**One Model Type to Rule Them All**
> "Foundation models are revolutionary because they allow for one single type of model to generalize to a wide range of tasks without having to be retrained."

**Characteristics:**
- **Wide scope** - Generalizes across many tasks
- **Massive size** - Billions of parameters
  - GPT-3: 175 billion parameters
  - GPT-4: Estimated 1+ trillion parameters
- **High compute** - Hundreds to thousands of GPUs
- **Long training** - Weeks to months
- **Multi-purpose** - Same model, many tasks

**Foundation Model Examples:**

**Language Models (LLMs):**
- GPT-4 (OpenAI)
- Claude (Anthropic)
- LLaMA (Meta)
- Gemini (Google)

**Vision Models:**
- DALL-E (OpenAI)
- Stable Diffusion
- Midjourney

**Multimodal Models:**
- GPT-4V (vision + text)
- Gemini (text, image, video, audio)

**Code Models:**
- GitHub Copilot (powered by Codex)
- CodeLLaMA

---

### Model Comparison

| Aspect | Traditional ML (Data Scientist) | Foundation Models (AI Engineer) |
|--------|----------------------------------|--------------------------------|
| **Number of model types** | Hundreds | One (foundation model) |
| **Scope** | Narrow (task-specific) | Wide (multi-task) |
| **Parameters** | Thousands to millions | Billions |
| **Training compute** | CPU or single GPU | Hundreds to thousands of GPUs |
| **Training time** | Seconds to hours | Weeks to months |
| **Generalization** | Limited to training domain | Generalizes across domains |
| **Retraining needed** | For each new task | No (use pre-trained) |

---

## Difference 4: Processes & Workflows

---

### Data Scientist Process: Train from Scratch

```
1. Define Use Case
   ↓
2. Identify & Collect Data (task-specific)
   ↓
3. Prepare Data
   - Clean
   - Feature engineering
   - Split (train/validation/test)
   ↓
4. Train Model
   - Select algorithm
   - Train on data
   ↓
5. Validate & Tune
   - Cross-validation
   - Hyperparameter tuning
   - Feature selection
   ↓
6. Deploy Model
   - Cloud endpoint
   - Real-time inference
   ↓
7. Monitor & Maintain
```

**Key Techniques:**
- **Feature Engineering** - Creating informative features
- **Cross-Validation** - Testing model robustness
- **Hyperparameter Tuning** - Optimizing model settings
- **A/B Testing** - Comparing model versions

**Example Workflow:**
```
Goal: Predict customer churn

1. Define: Predict if customer cancels within 30 days
2. Collect: Customer demographics, usage patterns, support tickets
3. Prepare: Clean missing values, create "days_since_last_login" feature
4. Train: Try Logistic Regression, Random Forest, XGBoost
5. Validate: Cross-validate, tune XGBoost parameters
6. Deploy: REST API endpoint for real-time predictions
7. Monitor: Track accuracy weekly, retrain monthly
```

---

### AI Engineer Process: Use Pre-Trained Models

```
1. Define Use Case
   ↓
2. Select Pre-Trained Foundation Model ⭐ SKIP DATA COLLECTION!
   ↓
3. Prompt Engineering
   - Design instructions
   - Few-shot examples
   ↓
4. Advanced Techniques (Optional)
   - Prompt chaining
   - RAG (Retrieval-Augmented Generation)
   - Fine-tuning (PEFT)
   - Autonomous agents
   ↓
5. Embed in System/Workflow
   - Assistants
   - Applications with UI
   - Automation
   ↓
6. Monitor & Improve
```

**Key Enabler: AI Democratization**
> "A big fancy word that simply means making AI more widely accessible to everyday users."

**Where to Find Models:**
- **Hugging Face** - Open-source model hub
- **OpenAI API** - GPT models via API
- **Google Cloud** - Gemini, PaLM
- **Azure OpenAI** - Microsoft's hosting
- **AWS Bedrock** - Multiple foundation models

---

### Key AI Engineering Techniques

#### 1. Prompt Engineering
Interacting with models via natural language instructions.

**Example:**
```
Simple Prompt:
"Summarize this customer review."

Advanced Prompt:
"You are a customer service analyst. Read the following review 
and provide: 1) Overall sentiment (positive/negative/neutral), 
2) Main issues mentioned, 3) Actionable recommendations. 
Keep it under 100 words.

Review: [customer text here]"
```

#### 2. Prompt Chaining
Linking multiple prompts together for complex tasks.

**Example:**
```
Chain for Blog Post Creation:
Prompt 1: "Generate 5 blog post ideas about AI in healthcare"
    ↓
Prompt 2: "Pick the most interesting idea and create an outline"
    ↓
Prompt 3: "Write the introduction section from the outline"
    ↓
Prompt 4: "Write the next section..."
```

#### 3. RAG (Retrieval-Augmented Generation)
Grounding answers in truth by retrieving relevant information.

**How it works:**
```
User Question
    ↓
1. Retrieve relevant documents from knowledge base
    ↓
2. Add documents to prompt as context
    ↓
3. Generate answer based on provided context
    ↓
Answer grounded in actual documents (with sources!)
```

**Example:**
```
Without RAG:
Q: "What's our company's return policy?"
A: [Model might hallucinate or give generic answer]

With RAG:
Q: "What's our company's return policy?"
→ Retrieves actual policy document
→ Provides context to model
A: "According to your policy document, customers have 30 days 
   for returns with receipt, and 15 days for exchanges..."
```

#### 4. PEFT (Parameter-Efficient Fine-Tuning)
Adapting foundation models to domain-specific data without full retraining.

**Why it matters:**
- Full fine-tuning is expensive (requires retraining billions of parameters)
- PEFT trains only small adapters (< 1% of parameters)
- Much faster and cheaper

**Example:**
```
General LLM → PEFT on medical data → Medical-specialized LLM
- Only trains ~10M adapter parameters instead of 175B
- Takes hours instead of weeks
- Much lower cost
```

#### 5. Autonomous Agents
Systems that reason through complex multi-step problems.

**Example:**
```
User: "Plan a 3-day trip to Paris for under $2000"

Agent Process:
1. Break down task (flights, hotels, activities, budget)
2. Search for flight options
3. Compare prices
4. Search for hotels within remaining budget
5. Find activities and restaurants
6. Create itinerary
7. Verify total cost < $2000
8. Present complete plan
```

---

### Process Comparison

| Stage | Data Scientist | AI Engineer |
|-------|---------------|-------------|
| **Start** | Define use case | Define use case |
| **Data** | Collect task-specific data | Use model's training data |
| **Model** | Train from scratch | Use pre-trained foundation model |
| **Customization** | Feature engineering, hyperparameter tuning | Prompt engineering, RAG, fine-tuning |
| **Time to first working solution** | Days to weeks | Hours to days |
| **Primary skill** | Statistical modeling, algorithm selection | Prompt design, system integration |
| **Deployment** | Model endpoint (API) | Embedded in application/workflow |

---

## High-Level Summary: The Four Differences

### Visual Comparison

```
┌─────────────────────┬──────────────────────┬──────────────────────┐
│                     │   DATA SCIENTIST     │    AI ENGINEER       │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ 1. USE CASES        │                      │                      │
│   Role              │ Data Storyteller     │ AI System Builder    │
│   Focus             │ Descriptive          │ Prescriptive         │
│                     │ Predictive           │ Generative           │
│   Examples          │ - EDA, clustering    │ - Decision opt       │
│                     │ - Regression         │ - Recommendations    │
│                     │ - Classification     │ - Assistants         │
│                     │                      │ - Chatbots           │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ 2. DATA             │                      │                      │
│   Type              │ Structured/Tabular   │ Unstructured         │
│   Format            │ Tables, databases    │ Text, images, video  │
│   Scale             │ 100s to 100,000s     │ Billions to trillions│
│   Preparation       │ Heavy cleaning       │ Quality filtering    │
│                     │ Feature engineering  │ Prompt design        │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ 3. MODELS           │                      │                      │
│   Types             │ Hundreds of models   │ Foundation models    │
│   Scope             │ Narrow (specific)    │ Wide (general)       │
│   Size              │ Millions params      │ Billions params      │
│   Training time     │ Seconds to hours     │ Weeks to months      │
│   Compute           │ CPU/single GPU       │ 100s-1000s of GPUs   │
│   Generalization    │ Limited              │ Extensive            │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ 4. PROCESSES        │                      │                      │
│   Starting point    │ Collect data         │ Pre-trained model    │
│   Workflow          │ Train from scratch   │ Prompt & customize   │
│   Key techniques    │ - Feature eng        │ - Prompt eng         │
│                     │ - Cross-validation   │ - RAG                │
│                     │ - Hyperparameter     │ - PEFT               │
│                     │   tuning             │ - Agents             │
│   Time to solution  │ Days to weeks        │ Hours to days        │
└─────────────────────┴──────────────────────┴──────────────────────┘
```

---

## Important Notes on Overlap

> "It's important to note that there is still overlap between the two fields."

### Where They Intersect

**Data Scientists may also:**
- Work on prescriptive use cases (optimization, recommendations)
- Use unstructured data (text analysis, image classification)
- Deploy chatbots or assistants
- Use pre-trained models (transfer learning)

**AI Engineers may also:**
- Work with structured data (for fine-tuning, RAG databases)
- Do data analysis and EDA
- Build traditional ML models for specific components
- Perform statistical analysis

### The Reality
These roles exist on a **spectrum**, not as completely separate silos.

```
Data Science ←──────── Overlap Zone ────────→ AI Engineering
           [Traditional ML] ← → [Gen AI Applications]
```

---

## The Rapidly Evolving Landscape

> "Both of these fields are continuing to evolve at a blazing fast pace with new research papers, new models, new tools coming out every single day."

### What This Means for You

**Stay Current:**
- New models released monthly
- New techniques emerge constantly
- Tools and frameworks evolve rapidly
- Best practices change

**Be Flexible:**
- Don't get locked into one approach
- Learn both traditional ML and gen AI
- Understand when to use each
- Combine techniques when beneficial

**Keep Learning:**
- Follow research papers (ArXiv)
- Try new models (Hugging Face)
- Join communities (GitHub, Discord)
- Experiment with tools

---

## Career Implications

### Skills for Data Scientists

**Core Technical:**
- Statistics and probability
- Python (pandas, NumPy, scikit-learn)
- SQL and databases
- Data visualization
- Machine learning algorithms
- Feature engineering
- Model evaluation

**Domain Knowledge:**
- Business acumen
- Domain-specific expertise
- Data storytelling
- Statistical inference

### Skills for AI Engineers

**Core Technical:**
- Python (LangChain, Hugging Face)
- API integration
- Prompt engineering
- RAG implementation
- Vector databases
- Fine-tuning techniques
- System architecture

**Design Skills:**
- User experience
- Conversation design
- System integration
- Workflow automation

### The Hybrid Professional

Many professionals will need skills from **both domains**:
- Understanding when to use traditional ML vs gen AI
- Combining techniques for best results
- Communicating across teams
- Adapting to new tools and methods

---

## Deciding Which Path (or Both!)

### Choose Data Science if you love:
- Statistical analysis and mathematics
- Finding insights in structured data
- Building predictive models
- A/B testing and experimentation
- Working with tabular datasets
- Traditional software development

### Choose AI Engineering if you love:
- Building conversational AI
- Working with text, images, video
- System design and integration
- Creative applications of AI
- Rapid prototyping
- Cutting-edge technology

### Do Both if you want:
- Maximum flexibility
- Complete understanding of AI landscape
- Ability to choose best tool for job
- Competitive advantage in job market

---

## Key Takeaways

### The Big Picture
1. ✅ **Generative AI created a new field** - AI Engineering is distinct from traditional Data Science
2. ✅ **Four key differences** - Use cases, data, models, processes all differ
3. ✅ **Both are valuable** - Different problems require different approaches
4. ✅ **Overlap exists** - Not completely separate; skills complement each other
5. ✅ **Rapid evolution** - Both fields changing constantly; continuous learning essential

### Philosophical Insight
> "With data, AI, and a creative mind, really anything is possible."

The future belongs to those who can:
- **Understand both traditional ML and gen AI**
- **Choose the right tool for the problem**
- **Combine techniques creatively**
- **Keep learning and adapting**

---

## Study Questions

1. What are the four key areas where Data Scientists and AI Engineers differ?
2. How would you describe the role of a Data Scientist in one phrase? AI Engineer?
3. What's the difference between descriptive, predictive, prescriptive, and generative use cases?
4. Why is structured data often called "tabular" data?
5. What scale of data do LLMs typically train on compared to traditional ML models?
6. What does "narrow scope" vs "wide scope" mean for models?
7. What is AI democratization and why does it matter?
8. What is RAG and what problem does it solve?
9. Name three advanced AI engineering techniques beyond basic prompting.
10. Where do Data Science and AI Engineering overlap?

---

## Practical Exercise

**Scenario:** You work at an e-commerce company. Determine whether each problem is better suited for a Data Scientist or AI Engineer (or both):

1. Predicting which customers will make a purchase next month
2. Building a customer service chatbot for common questions
3. Analyzing which products are frequently bought together
4. Creating product descriptions from images and specifications
5. Optimizing warehouse inventory levels across locations
6. Generating personalized marketing email content
7. Predicting delivery times based on historical data
8. Building a voice-based shopping assistant

**Answers:**
1. Data Scientist (predictive classification)
2. AI Engineer (chatbot, gen AI)
3. Data Scientist (association rules, clustering)
4. AI Engineer (generative, vision + text)
5. Could be either (DS: predictive optimization, AIE: decision optimization)
6. AI Engineer (generative content)
7. Data Scientist (regression model)
8. AI Engineer (conversational AI, multimodal)

---

*These notes are based on "Data Scientist vs AI Engineer" from Module 1 of the IBM AI Engineering Professional Certificate course, presented by Isaac Key, IBM AI Engineer.*
