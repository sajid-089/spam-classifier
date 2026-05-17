# Task 1: Email Spam Classification

## Objective
Build a machine learning model that reads an email or SMS message and
automatically classifies it as spam or not spam (ham).

## Dataset
- **Name:** SMS Spam Collection Dataset
- **Source:** UCI Machine Learning Repository / Kaggle
- **Size:** 5,572 messages
- **Classes:** Ham - 4,825 (86.6%) | Spam - 747 (13.4%)
- **Format:** CSV file with label and message columns

## Approach

### Step 1: Data Loading
- Loaded CSV dataset using pandas
- Cleaned column names and dropped unnecessary columns

### Step 2: Exploratory Data Analysis
- Checked dataset shape, null values, and duplicates
- Analyzed class distribution (spam vs ham ratio)
- Measured average message length and word count per class
- Found that spam messages are generally longer than ham messages
- Created bar charts, pie charts, and length histograms

### Step 3: Text Preprocessing
Applied these cleaning steps to every message:
1. Converted to lowercase
2. Removed special characters, numbers, and punctuation
3. Removed common English stopwords
4. Applied Porter Stemming (running -> run, playing -> play)
5. Removed very short words (2 characters or less)

### Step 4: Feature Extraction
- Used TF-IDF to convert text into numerical feature vectors
- TF-IDF assigns higher scores to words that are important in a
  specific message but rare across the whole dataset
- Used top 3000 most relevant words as features

### Step 5: Train-Test Split
- 80% training, 20% testing
- Stratified split to maintain spam/ham ratio in both sets

### Step 6: Model Training
Trained two models for comparison:

**Naive Bayes:** Classic text classification algorithm that uses
word probabilities to determine message category. Fast and effective.

**Logistic Regression:** Linear classifier that draws a decision
boundary between spam and ham. Strong baseline model.

### Step 7: Evaluation
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix for detailed error analysis
- Compared both models side by side

### Step 8: Custom Testing
Tested the best model on hand-written messages to verify
real-world performance on completely unseen inputs.

## Results

| Model               | Accuracy | Precision | Recall |
|---------------------|----------|-----------|--------|
| Naive Bayes         | ~97%     | ~100%     | ~80%   |
| Logistic Regression | ~96%     | ~98%      | ~78%   |

**Best Model:** Naive Bayes

## Output Files
| File | Description |
|------|-------------|
| eda_analysis.png | Data exploration charts |
| confusion_matrices.png | Confusion matrices for both models |
| model_comparison.png | Accuracy comparison bar chart |

## How to Run
```bash
pip install -r requirements.txt
python spam_classifier.py

##Key Learnings
1. Text data needs heavy preprocessing before it becomes useful for ML
2. TF-IDF effectively converts text into meaningful numerical features
3. Naive Bayes works surprisingly well for text classification
4. Comparing multiple models helps select the best one
5. Multiple evaluation metrics give a complete performance picture

##Tools Used
Python, pandas, numpy, scikit-learn, NLTK, matplotlib, seaborn