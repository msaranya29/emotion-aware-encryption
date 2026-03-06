import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os
import urllib.request
import io

def get_data():
    """
    Attempts to download a real emotion dataset CSV.
    Falls back to a rich synthetic dataset to guarantee the model trains 
    independently and reliably in any environment.
    """
    try:
        # First try to download a public dataset (e.g., HuggingFace datasets exported to raw github)
        url = "https://raw.githubusercontent.com/dair-ai/emotion_dataset/master/experiments/data/training.csv"
        # We will wrap it in try-except in case URL goes dead or network blocks
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            csv_data = response.read().decode('utf-8')
            df = pd.read_csv(io.StringIO(csv_data))
            # Rename columns if needed depending on dataset schema
            if 'text' in df.columns and 'label' in df.columns:
                return df
    except Exception as e:
        print("Note: Could not download remote dataset. Falling back to built-in fallback dataset.")
    
    # Fallback to local offline representation of multi-class emotions
    data = {"text": [
        "I am feeling very happy today!", "This is the best day of my life", "I feel so joyful and energetic.",
        "I am extremely sad and depressed", "I can't stop crying", "Everything is falling apart around me",
        "I am so angry right now!", "This makes me furious", "I want to break something out of rage",
        "I am feeling quite anxious", "I'm worried about the test tomorrow", "I have panic and tightness in my chest",
        "This is a pleasant surprise", "Wow, I didn't expect this at all!", "I am absolutely amazed",
        "I feel completely numb and empty", "This whole situation is devastating", "I am ecstatic about the news"
    ] * 20, 
    "emotion": [
        "joy", "joy", "joy",
        "sadness", "sadness", "sadness",
        "anger", "anger", "anger",
        "anxiety", "anxiety", "anxiety",
        "surprise", "surprise", "surprise",
        "sadness", "sadness", "joy"
    ] * 20}
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("Fetching dataset...")
    df = get_data()
    print(f"Training TF-IDF + Logistic Regression on {len(df)} samples...")
    
    # TF-IDF Features
    vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
    X = vectorizer.fit_transform(df['text'])
    y = df['emotion']
    
    # Train Logistic Regression
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X, y)
    
    # Output model to .pkl files
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vec_path = os.path.join(base_dir, 'vectorizer.pkl')
    model_path = os.path.join(base_dir, 'saved_model.pkl')
    
    with open(vec_path, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Training complete! Model and vectorizer saved to:\n- {model_path}\n- {vec_path}")
