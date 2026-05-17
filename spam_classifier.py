# ============================================================
# EMAIL SPAM CLASSIFIER
# Machine Learning Internship Project
# ============================================================

# --- importing required libraries ---
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # save plots without GUI window
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_score,
                             recall_score, f1_score)

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import warnings
warnings.filterwarnings('ignore')


# ============================================================
# STEP 1: DATA LOADING
# ============================================================

def load_data():
    """
    Load the SMS Spam Collection dataset from local CSV file.
    The dataset contains SMS messages labeled as 'spam' or 'ham'.
    """
    local_path = os.path.join('data', 'spam.csv')

    if os.path.exists(local_path):
        print("[INFO] Loading dataset from local file...")
        df = pd.read_csv(local_path, encoding='latin-1')

        # the kaggle CSV has extra unnamed columns, drop them
        df = df[['v1', 'v2']]
        df.columns = ['label', 'message']
    else:
        # fallback: load from URL if local file not found
        print("[INFO] Local file not found, downloading from URL...")
        url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
        df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

    print(f"[INFO] Dataset loaded successfully! Shape: {df.shape}")
    return df


# ============================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

def explore_data(df):
    """
    Understand the dataset - check class distribution,
    message lengths, null values, and sample messages.
    """
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)

    # basic info
    print(f"\nTotal messages: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nNull values:\n{df.isnull().sum()}")

    # check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")

    # class distribution - how many spam vs ham
    print(f"\nLabel Distribution:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        pct = count / len(df) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")

    # add message length and word count columns for analysis
    df['msg_length'] = df['message'].apply(len)
    df['word_count'] = df['message'].apply(lambda x: len(x.split()))

    print(f"\nAverage Message Length (characters):")
    print(df.groupby('label')['msg_length'].mean().to_string())

    print(f"\nAverage Word Count:")
    print(df.groupby('label')['word_count'].mean().to_string())

    # show sample messages from each class
    print(f"\n--- Sample HAM messages ---")
    ham_samples = df[df['label'] == 'ham']['message'].head(3)
    for i, msg in enumerate(ham_samples, 1):
        print(f"  {i}. {msg[:80]}...")

    print(f"\n--- Sample SPAM messages ---")
    spam_samples = df[df['label'] == 'spam']['message'].head(3)
    for i, msg in enumerate(spam_samples, 1):
        print(f"  {i}. {msg[:80]}...")

    return df


def create_eda_plots(df):
    """
    Create visualizations for EDA:
    - Bar chart of spam vs ham counts
    - Pie chart of percentage distribution
    - Message length histograms for both classes
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = ['#27ae60', '#e74c3c']

    # Plot 1: count bar chart
    df['label'].value_counts().plot(kind='bar', ax=axes[0, 0], color=colors)
    axes[0, 0].set_title('Spam vs Ham - Count', fontsize=13)
    axes[0, 0].set_xlabel('Label')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=0)

    # Plot 2: pie chart
    df['label'].value_counts().plot(kind='pie', ax=axes[0, 1],
                                     autopct='%1.1f%%', colors=colors,
                                     startangle=90)
    axes[0, 1].set_title('Spam vs Ham - Percentage', fontsize=13)
    axes[0, 1].set_ylabel('')

    # Plot 3: ham message length distribution
    df[df['label'] == 'ham']['msg_length'].hist(bins=50, ax=axes[1, 0],
                                                 color='#27ae60', alpha=0.7)
    axes[1, 0].set_title('Ham Messages - Length Distribution', fontsize=13)
    axes[1, 0].set_xlabel('Message Length (characters)')
    axes[1, 0].set_ylabel('Frequency')

    # Plot 4: spam message length distribution
    df[df['label'] == 'spam']['msg_length'].hist(bins=50, ax=axes[1, 1],
                                                  color='#e74c3c', alpha=0.7)
    axes[1, 1].set_title('Spam Messages - Length Distribution', fontsize=13)
    axes[1, 1].set_xlabel('Message Length (characters)')
    axes[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('eda_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[INFO] EDA plots saved as 'eda_analysis.png'")


# ============================================================
# STEP 3: TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):
    """
    Clean a single message by applying these steps:
    1. Convert to lowercase
    2. Remove special characters and numbers
    3. Remove extra whitespace
    4. Remove stopwords (common words like 'the', 'is', 'a')
    5. Apply stemming (reduce words to root form: running -> run)
    """
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    # convert to lowercase
    text = text.lower()

    # remove special characters and numbers, keep only letters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # split into individual words
    words = text.split()

    # remove stopwords and apply stemming
    # also skip very short words (length <= 2) as they add noise
    cleaned_words = []
    for word in words:
        if word not in stop_words and len(word) > 2:
            cleaned_words.append(stemmer.stem(word))

    return ' '.join(cleaned_words)


def apply_preprocessing(df):
    """
    Apply text preprocessing to all messages in the dataset.
    Also convert labels to numeric: ham=0, spam=1.
    """
    print("\n[INFO] Text preprocessing started...")

    df['cleaned'] = df['message'].apply(preprocess_text)

    # show before vs after cleaning examples
    print("\nBefore vs After Cleaning (3 examples):")
    for i in range(3):
        print(f"\n  BEFORE: {df['message'].iloc[i][:70]}...")
        print(f"  AFTER:  {df['cleaned'].iloc[i][:70]}...")

    # convert labels to numbers
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

    print(f"\n[INFO] Preprocessing complete!")
    print(f"  Label mapping: ham -> 0, spam -> 1")
    return df


# ============================================================
# STEP 4: FEATURE EXTRACTION (TF-IDF)
# ============================================================

def extract_features(df):
    """
    Convert text data into numerical features using TF-IDF.

    TF-IDF (Term Frequency - Inverse Document Frequency):
    - Assigns a score to each word based on how important it is
    - If a word appears frequently in one message but rarely in others,
      it gets a higher score (meaning it's a distinguishing word)
    - max_features=3000 means we keep the top 3000 most important words
    """
    print("\n[INFO] Extracting features using TF-IDF...")

    tfidf = TfidfVectorizer(max_features=3000)

    X = tfidf.fit_transform(df['cleaned'])
    y = df['label_num']

    print(f"  Feature matrix: {X.shape[0]} messages x {X.shape[1]} features")
    print(f"  This means {X.shape[1]} unique words are being used as features")

    # show some sample feature words
    feature_names = tfidf.get_feature_names_out()
    print(f"\n  Sample features (words): {list(feature_names[:10])}")

    return X, y, tfidf


# ============================================================
# STEP 5: TRAIN-TEST SPLIT
# ============================================================

def split_data(X, y):
    """
    Split data into 80% training and 20% testing.
    stratify=y ensures the spam/ham ratio stays the same in both sets.
    random_state=42 makes the split reproducible.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n[INFO] Data split completed:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Testing set:  {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


# ============================================================
# STEP 6: MODEL TRAINING
# ============================================================

def train_naive_bayes(X_train, y_train):
    """
    Train a Multinomial Naive Bayes classifier.
    This is a classic choice for text classification because:
    - It works well with word frequency features (TF-IDF)
    - It's fast to train even on large datasets
    - It performs surprisingly well despite its simplicity
    """
    print("\n[INFO] Training Naive Bayes model...")
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)
    print("  Training complete!")
    return model


def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression classifier.
    Training a second model for comparison to see which one
    performs better on this specific dataset.
    """
    print("[INFO] Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print("  Training complete!")
    return model


# ============================================================
# STEP 7: EVALUATION
# ============================================================

def evaluate_model(model, model_name, X_test, y_test):
    """
    Evaluate a trained model on the test set.
    Metrics used:
    - Accuracy: overall percentage of correct predictions
    - Precision: out of all predicted spam, how many were actually spam
    - Recall: out of all actual spam, how many did we catch
    - F1 Score: harmonic mean of precision and recall
    """
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"\n{'='*50}")
    print(f"  {model_name} - Results")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"\n  Detailed Classification Report:")
    print(classification_report(y_test, predictions,
                                target_names=['Ham', 'Spam']))

    return predictions, acc


def plot_confusion_matrices(y_test, nb_pred, lr_pred, nb_acc, lr_acc):
    """
    Plot confusion matrices for both models side by side.
    A confusion matrix shows:
    - True Positives: correctly identified spam
    - True Negatives: correctly identified ham
    - False Positives: ham incorrectly labeled as spam
    - False Negatives: spam that was missed (labeled as ham)
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Naive Bayes confusion matrix
    cm_nb = confusion_matrix(y_test, nb_pred)
    sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Greens', ax=axes[0],
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    axes[0].set_title(f'Naive Bayes (Accuracy: {nb_acc:.2%})', fontsize=13)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    # Logistic Regression confusion matrix
    cm_lr = confusion_matrix(y_test, lr_pred)
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    axes[1].set_title(f'Logistic Regression (Accuracy: {lr_acc:.2%})', fontsize=13)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[INFO] Confusion matrices saved as 'confusion_matrices.png'")


def plot_model_comparison(nb_acc, lr_acc):
    """
    Create a bar chart comparing the accuracy of both models.
    This makes it easy to visually see which model performed better.
    """
    models = ['Naive Bayes', 'Logistic Regression']
    accuracies = [nb_acc, lr_acc]
    colors = ['#27ae60', '#2980b9']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, accuracies, color=colors, width=0.5)

    # add accuracy labels on top of each bar
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.2%}', ha='center', fontsize=13, fontweight='bold')

    ax.set_ylim(0.9, 1.0)
    ax.set_title('Model Comparison - Accuracy', fontsize=14)
    ax.set_ylabel('Accuracy')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[INFO] Model comparison plot saved as 'model_comparison.png'")


# ============================================================
# STEP 8: TEST WITH CUSTOM MESSAGES
# ============================================================

def test_custom_messages(model, tfidf):
    """
    Test the trained model with custom email/SMS messages.
    This demonstrates that the model works on unseen, real-world text.
    """
    test_messages = [
        "Congratulations! You won a free iPhone! Click here to claim now!",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT: Your bank account has been compromised. Click link to verify.",
        "Can you send me the notes from yesterday's class?",
        "Win cash prizes worth lakhs!!! Send FREE to 80882",
        "Mom said dinner is ready, come home soon",
        "FREE entry in a weekly competition to win prizes! Text WIN to 80085",
        "I'll be late for the meeting, start without me"
    ]

    print(f"\n{'='*60}")
    print("  TESTING WITH CUSTOM MESSAGES")
    print(f"{'='*60}")

    for msg in test_messages:
        # apply the same preprocessing used during training
        cleaned = preprocess_text(msg)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        label = "SPAM" if prediction == 1 else "HAM"
        icon = "[X]" if prediction == 1 else "[OK]"

        print(f"\n  {icon} [{label}]")
        print(f"      \"{msg[:65]}{'...' if len(msg) > 65 else ''}\"")


# ============================================================
# MAIN FUNCTION - ENTIRE PIPELINE RUNS FROM HERE
# ============================================================

def main():
    print("=" * 60)
    print("  EMAIL SPAM CLASSIFIER")
    print("  ML Internship Project")
    print("=" * 60)

    # step 1: load data
    df = load_data()

    # step 2: explore the dataset
    df = explore_data(df)
    create_eda_plots(df)

    # step 3: preprocess text
    df = apply_preprocessing(df)

    # step 4: extract features using TF-IDF
    X, y, tfidf = extract_features(df)

    # step 5: split into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # step 6: train both models
    nb_model = train_naive_bayes(X_train, y_train)
    lr_model = train_logistic_regression(X_train, y_train)

    # step 7: evaluate both models
    nb_pred, nb_acc = evaluate_model(nb_model, "Naive Bayes", X_test, y_test)
    lr_pred, lr_acc = evaluate_model(lr_model, "Logistic Regression", X_test, y_test)

    # create evaluation plots
    plot_confusion_matrices(y_test, nb_pred, lr_pred, nb_acc, lr_acc)
    plot_model_comparison(nb_acc, lr_acc)

    # step 8: test with custom messages using the best model
    best_model = nb_model if nb_acc >= lr_acc else lr_model
    best_name = "Naive Bayes" if nb_acc >= lr_acc else "Logistic Regression"
    print(f"\n[INFO] Best performing model: {best_name}")

    test_custom_messages(best_model, tfidf)

    # print final summary
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Naive Bayes Accuracy:         {nb_acc:.2%}")
    print(f"  Logistic Regression Accuracy: {lr_acc:.2%}")
    print(f"  Best Model: {best_name}")
    print(f"\n  Generated Files:")
    print(f"    - eda_analysis.png")
    print(f"    - confusion_matrices.png")
    print(f"    - model_comparison.png")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()