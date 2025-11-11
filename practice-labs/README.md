# Practice Labs

This directory contains **custom practice notebooks** created to reinforce learning for each lesson in the IBM AI Engineering course. These notebooks complement the official course labs with additional hands-on exercises using realistic synthetic datasets.

---

## 🎯 Purpose

**Why Practice Labs?**
- 🧠 **Reinforce Concepts**: Practice makes perfect - code through each concept
- 📊 **Real-World Scenarios**: Work with realistic industry examples
- 🔬 **Hands-On Learning**: Learn by doing, not just reading
- 💪 **Build Confidence**: Master techniques before moving to next lesson
- 🎓 **Self-Paced**: Complete exercises at your own speed

---

## 📁 Structure

```
practice-labs/
├── module-1/          # Module 1 practice notebooks
├── module-2/          # Module 2 practice notebooks
│   ├── 01-introduction-to-regression-practice.ipynb
│   ├── 03-multiple-linear-regression-practice.ipynb
│   └── ...
├── module-3/          # Module 3 practice notebooks
├── module-4/          # Module 4 practice notebooks
├── module-5/          # Module 5 practice notebooks
├── module-6/          # Module 6 practice notebooks
└── README.md          # This file
```

**Naming Convention:** `##-lesson-topic-practice.ipynb`
- Example: `01-introduction-to-regression-practice.ipynb`
- Numbers match the lesson order in each module

---

## 🆚 Practice Labs vs Official Labs

| Aspect | Practice Labs (This Folder) | Official Labs (`/labs/`) |
|--------|----------------------------|-------------------------|
| **Purpose** | Reinforce learning, extra practice | Course-required assignments |
| **Dataset** | Synthetic, realistic, generated | Real datasets from course |
| **Difficulty** | Progressive, guided | Matches course requirements |
| **Exercises** | 5-10 practice problems | Specific course objectives |
| **Solutions** | Included in notebook | Sometimes provided |
| **When to Use** | After watching lesson, before official lab | As required by course |

**Recommended Workflow:**
1. 📺 Watch course lesson video
2. 📝 Review comprehensive notes (`/notes/module-X/`)
3. 💻 Complete practice lab (this folder)
4. 🎓 Complete official lab (`/labs/module-X/`)
5. ✅ Quiz/assessment

---

## 📚 Module 2: Linear and Logistic Regression

### Available Practice Labs

#### 01. Introduction to Regression Practice
**File:** `01-introduction-to-regression-practice.ipynb`  
**Lesson:** Introduction to Regression  
**Official Lab:** ❌ No official lab (only this practice lab)

**Scenario:** E-Commerce Product Pricing  
**Dataset:** 150 electronic products (phones/tablets)  
**Features:** brand_score, screen_size, storage_gb, ram_gb, battery_hours, camera_mp, rating → price_usd

**Topics Covered:**
- ✅ Understanding continuous vs discrete variables
- ✅ Regression vs classification comparison
- ✅ Simple linear regression (1 predictor)
- ✅ Multiple linear regression (7 predictors)
- ✅ Model evaluation (R², RMSE, MAE)
- ✅ Feature importance analysis
- ✅ Identifying overpriced/underpriced products
- ✅ What-if scenario analysis

**Exercises:** 5 practice problems with solutions

---

#### 02. Simple Linear Regression Lab (Official)
**File:** See `/labs/module-2/01-simple-linear-regression-lab.ipynb`  
**Official Lab:** ✅ Yes  
**Practice Lab:** ❌ Not needed (official lab exists)

---

#### 03. Multiple Linear Regression Practice
**File:** `03-multiple-linear-regression-practice.ipynb`  
**Lesson:** Multiple Linear Regression  
**Official Lab:** ✅ Yes (but practice first!)

**Scenario:** Real Estate House Valuation  
**Dataset:** 200 houses across urban/suburban/rural areas  
**Features:** sqft, bedrooms, bathrooms, age_years, garage_spaces, location, has_pool, condition → price_usd

**Topics Covered:**
- ✅ Handling categorical variables (one-hot encoding)
- ✅ Detecting multicollinearity (correlation matrix, VIF)
- ✅ Building multiple regression models
- ✅ Matrix representation (X, θ)
- ✅ OLS vs Gradient Descent comparison
- ✅ 3D visualization (regression plane)
- ✅ What-if scenario analysis (renovation ROI)
- ✅ Feature importance and coefficients interpretation
- ✅ Identifying best value houses

**Exercises:** 3 practice problems with solutions

---

## 🚀 How to Use These Notebooks

### Prerequisites
```bash
# Ensure you have required libraries
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Running Notebooks

1. **Open in VS Code:**
   ```
   File → Open File → Select practice lab notebook
   ```

2. **Select Python Kernel:**
   - Click kernel selector in top-right
   - Choose Python environment with required libraries

3. **Run "Setup" Cell First:**
   - Every notebook has a "Run Me First!" setup cell
   - Installs dependencies and sets random seeds
   - Ensures reproducible results

4. **Work Through Sequentially:**
   - Read explanations in markdown cells
   - Run code cells in order
   - Complete practice exercises
   - Compare with provided solutions

5. **Experiment:**
   - Modify parameters and re-run
   - Try different feature combinations
   - Generate your own datasets
   - Break things and fix them (best learning!)

---

## 🎓 Learning Tips

### For Maximum Learning:

1. **📝 Type, Don't Copy:**
   - Type out code manually instead of copy-paste
   - Builds muscle memory and understanding

2. **🔍 Understand Output:**
   - Don't just run cells - read the output
   - Understand what each number/chart means
   - Ask: "Does this result make sense?"

3. **🎨 Visualize Everything:**
   - Plots help intuition
   - Try different visualization types
   - Understand what patterns to look for

4. **💪 Do Exercises First:**
   - Try exercises without looking at solutions
   - Struggle is where learning happens
   - Check solution only after attempting

5. **🔄 Repeat with Variations:**
   - Change dataset sizes
   - Add/remove features
   - Try different parameter values
   - See how results change

6. **📊 Compare Methods:**
   - When multiple approaches shown, compare results
   - Understand trade-offs
   - Know when to use each method

---

## 🐛 Troubleshooting

### Common Issues:

**ImportError: No module named 'XXX'**
```bash
# Install missing library
pip install XXX
```

**Cells don't run in order:**
- Restart kernel: Click "Restart" in notebook toolbar
- Run all cells from top: "Run All" button

**Random seed doesn't work:**
- Ensure setup cell was run first
- Restart kernel and run from beginning

**Visualizations don't appear:**
- Check if `%matplotlib inline` is in setup cell
- Try restarting kernel

**Memory error:**
- Reduce dataset size (change `n_samples`)
- Restart VS Code
- Close other applications

---

## 📦 Dataset Information

All practice labs use **synthetic datasets** generated with NumPy. These datasets are:

✅ **Realistic** - Based on real-world patterns and correlations  
✅ **Reproducible** - Same seed = same data every time  
✅ **Documented** - Data dictionary included in each notebook  
✅ **Appropriate Size** - 100-200 samples (runs fast, easy to understand)  
✅ **Noisy** - Includes realistic variance and outliers  

**Why Synthetic?**
- Learn without downloading external files
- Control correlations and patterns
- Modify easily for experimentation
- No privacy/license concerns

---

## 🤝 Contributing

**Found an issue or have improvements?**
- Add comments in notebooks
- Create additional exercises
- Share interesting variations
- Document your experiments

---

## 📚 Related Resources

**Official Course Materials:**
- 📝 Lesson Notes: `/notes/module-X/`
- 🔬 Official Labs: `/labs/module-X/`
- 📖 Glossaries: `/notes/glossary-module-X.md`

**External Learning:**
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [NumPy Documentation](https://numpy.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)

---

## ✅ Completion Checklist

Track your progress through Module 2:

- [ ] Introduction to Regression (Practice Lab)
- [ ] Simple Linear Regression (Official Lab)
- [ ] Multiple Linear Regression (Practice Lab)
- [ ] Multiple Linear Regression (Official Lab)
- [ ] Polynomial Regression (Coming soon)
- [ ] Logistic Regression (Coming soon)

---

**Happy Learning! 🚀**

*These practice labs are designed to reinforce your understanding through hands-on coding. Take your time, experiment, and most importantly - have fun!*
