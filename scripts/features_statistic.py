import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from config import FEATURES_DIR, TRAIN_PATH, OUTPUT_DIR

def load_and_merge_all(split='train'):
    """Merges all available feature sets with their labels."""
    labels_df = pd.read_csv(TRAIN_PATH)[['class_id', 'class_name']]
    
    graph_df = pd.read_csv(os.path.join(FEATURES_DIR, 'graph', f'{split}_graph_features.csv')).drop(columns=['class_id'])
    emotion_df = pd.read_csv(os.path.join(FEATURES_DIR, 'emotion', f'emotion_{split}_features.csv'))
    syntax_df = pd.read_csv(os.path.join(FEATURES_DIR, 'syntax', f'syntax_{split}_features.csv')).drop(columns=['class_id'])
    
    return pd.concat([labels_df, graph_df, emotion_df, syntax_df], axis=1)

def plot_individual_aspects(df):
    """Generates separate heatmaps for clearer analysis."""
    profile = df.groupby('class_name').mean().drop(columns=['class_id'])
    # Normalize for visual comparison (0-1)
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min())
    
    aspects = {
        "Graph_Cohesion": ['avg_nodes', 'avg_edges', 'avg_density', 'avg_loops', 'avg_weighted_edges'],
        "Syntactic_Complexity": ['noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio', 'pron_ratio', 'avg_dep_distance'],
        "Emotional_State": [col for col in profile_norm.columns if col.startswith('emo_')]
    }

    analysis_dir = os.path.join(OUTPUT_DIR, 'analysis', 'aspects')
    os.makedirs(analysis_dir, exist_ok=True)

    for aspect_name, cols in aspects.items():
        plt.figure(figsize=(12, len(cols) * 0.4 + 2))
        sns.heatmap(profile_norm[cols].T, annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title(f"{aspect_name} Fingerprint")
        
        save_path = os.path.join(analysis_dir, f"{aspect_name.lower()}_heatmap.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    data = load_and_merge_all('train')
    plot_individual_aspects(data)