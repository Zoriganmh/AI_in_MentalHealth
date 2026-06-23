import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt_tab')
import os
import matplotlib.pyplot as plt
import seaborn as sns
from config import OUTPUT_DIR, TRAIN_PATH, VAL_PATH, TEST_PATH, TEXT_COL


def load_dataset(data_path):
    """Load dataset from a given CSV file path."""
    return pd.read_csv(data_path)

def basic_statistics(df):
    print("Basic Information:")
    print(df.info())  
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
 
def word_count(text):
    return len(str(text).split())

def sentence_count(text):
    sents = sent_tokenize(text)
    return len(sents)

def token_count(text):
    tokens = word_tokenize(text)
    return len(tokens)

def save_plot(plot_func, file_name, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plot_func()
    plt.savefig(os.path.join(folder_path, file_name))
    plt.close()

def class_distribution(df, folder_path):
    print("\nClass Distribution:")
    class_dist = df['class_name'].value_counts()
    print(class_dist)
    
    plt.figure(figsize=(8, 6))
    sns.countplot(x='class_name', data=df)
    plt.title('Class Distribution')
    plt.xticks(rotation=45)
    save_plot(plt.show, 'class_distribution.png', folder_path)

def visualize_token_count(df, folder_path):
    plt.figure(figsize=(8, 6))
    sns.histplot(df['token_count'], bins=30, kde=True)
    plt.title('Token Count Distribution')
    plt.xlabel('Token Count')
    plt.ylabel('Frequency')
    save_plot(plt.show, 'token_count_distribution.png', folder_path)

def visualize_word_count(df, folder_path):
    plt.figure(figsize=(8, 6))
    sns.histplot(df['word_count'], bins=30, kde=True)
    plt.title('Word Count Distribution')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    save_plot(plt.show, 'word_count_distribution.png', folder_path)

def visualize_sentence_count(df, folder_path):
    plt.figure(figsize=(8, 6))
    sns.histplot(df['sentence_count'], bins=30, kde=True)
    plt.title('Sentence Count Distribution')
    plt.xlabel('Sentence Count')
    plt.ylabel('Frequency')
    save_plot(plt.show, 'sentence_count_distribution.png', folder_path)

def analyze_data():
    train_df = load_dataset(TRAIN_PATH)
    val_df = load_dataset(VAL_PATH)
    test_df = load_dataset(TEST_PATH)

    for df in [train_df, val_df, test_df]:
        df['word_count'] = df[TEXT_COL].apply(word_count)
        df['token_count'] = df[TEXT_COL].apply(token_count)
        df['sentence_count'] = df[TEXT_COL].apply(sentence_count)
    
    for dataset_name, df in zip(['train', 'val', 'test'], [train_df, val_df, test_df]):
        
        folder_path = f"{OUTPUT_DIR}/analysis/{dataset_name}"

        print(f"\nAnalyzing {dataset_name} dataset...")    

        basic_statistics(df)
    
        class_distribution(df, folder_path)
            
        visualize_word_count(df, folder_path)
        visualize_token_count(df, folder_path)
        visualize_sentence_count(df, folder_path)

if __name__ == "__main__":
    analyze_data()