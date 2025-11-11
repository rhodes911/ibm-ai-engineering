# Copilot Instructions (Repo-wide)

## General
- Always begin by outlining a short plan as a commented checklist before coding.
- Prefer clear, idiomatic Python 3.10+ with type hints and docstrings.
- Use Black formatting and Pylint-friendly code; keep functions small and pure.
- Explain non-trivial steps in comments (why, not just what).

## Course Lesson Transcripts
When user pastes a lesson transcript from the IBM AI Engineering course:

### Step 1: Create Comprehensive Notes
- Create a new markdown file in `/notes/module-X/` (e.g., `notes/module-1/06-lesson-title.md`)
- Include the following sections:
  - Title with date, module, and topic
  - Overview and learning objectives
  - Detailed content with examples, code snippets, and tables
  - Visual diagrams and comparison charts where helpful
  - Key takeaways section
  - Study questions (5-10)
  - Practical exercises
- Aim for 300-500+ lines of comprehensive, well-structured notes
- Use code examples, tables, and clear hierarchical organization
- Add real-world examples for every major concept

### Step 2: Update Module Glossary
- Open the appropriate `glossary-module-X.md` file
- Identify NEW terms introduced in the lesson that aren't already defined
- Add new terms to the relevant section of the glossary
- Every term must include:
  - **Term Name** in bold
  - Clear definition (1-2 sentences)
  - Real-world example in italics starting with "*Example:*"
- Maintain alphabetical order within sections where possible
- Common sections: Core Concepts, ML Lifecycle, Data Terminology, Python and Tools, AI Subfields, Types of ML, etc.

### Step 3: Create Practice Lab Notebook (If No Official Lab Exists)
- Check if an official course lab exists for this lesson in `/labs/module-X/`
- If NO official lab exists, create a practice notebook in `/practice-labs/module-X/`
- Name format: `##-lesson-topic-practice.ipynb` (e.g., `01-introduction-to-regression-practice.ipynb`)
- Include the following cells:
  - **Title & Overview**: Markdown cell with lesson objectives and what will be practiced
  - **Setup Cell**: "Run Me First" with imports, random seeds, matplotlib settings
  - **Real-World Dataset Generation**: Create synthetic but realistic dataset (50-200 rows)
  - **Exploratory Data Analysis**: Visualizations (scatter plots, histograms, correlation matrices)
  - **Concept Implementation**: Step-by-step code implementing key concepts from lesson
  - **Practice Exercises**: 3-5 exercises with clear instructions and expected outputs
  - **Solutions**: Expandable solution cells for self-checking
- Dataset requirements:
  - Use realistic industry context (e-commerce, healthcare, real estate, etc.)
  - Include 3-6 meaningful features with appropriate correlations
  - Add some noise/outliers for realism
  - Include data dictionary explaining each column
- Code quality:
  - Use NumPy for synthetic data generation with seeds
  - Use Pandas for data manipulation
  - Use Matplotlib/Seaborn for visualizations
  - Include comments explaining each major step
  - Show both "manual" calculations and scikit-learn implementations when applicable

### Example Glossary Entry Format:
```markdown
**Term Name**
A clear, concise definition of the term explaining what it is and why it matters in ML.
*Example: A specific, concrete real-world scenario showing the term in action, including numbers, companies, or practical details that make it relatable.*
```

### Quality Standards:
- Notes should be study-ready with no further editing needed
- Examples should span multiple industries (healthcare, finance, e-commerce, etc.)
- Include specific companies/technologies when relevant (Netflix, Tesla, AWS, etc.)
- Code examples should be complete and runnable
- Comparisons should use tables for clarity
- Glossary examples must be practical and illustrative, not generic

## Notebooks
- For every .ipynb file, start with a "Run Me First" cell that installs/imports dependencies and sets seeds.
- Use deterministic seeds and note library versions at the end of each notebook.
- Show minimal, reproducible examples; avoid long cells—split into logical steps.

## ML / DL
- Structure training scripts as: config → data → model → train → eval → save.
- Log metrics clearly; prefer confusion matrix / ROC for classifiers, MAE/MSE for regressors.
- In TensorFlow/Keras and PyTorch, separate model definition from training loops.
- Provide a minimal training/evaluation example after defining any new model.

## Tests & Quality
- Generate basic unit tests when adding utilities using pytest style.
- Include a quick usage example in the docstring of every public function.

## Files & Project Hygiene
- Never commit credentials or large data; respect .gitignore.
- If adding dependencies, update requirements.txt and pin exact versions when possible.

## Responses in Chat
- When asked for code, return a single, complete file unless told otherwise.
- If unsure, ask one concise clarifying question, then propose a sensible default.
