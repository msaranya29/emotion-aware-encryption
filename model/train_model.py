"""
mini Emotion Cipher — Emotion Detection Model Training
=======================================================
Uses TF-IDF + Logistic Regression to classify text into emotions.
Supports: joy, sadness, anger, anxiety/fear, surprise, disgust, excitement, love, neutral

Run: python -m model.train_model  (from project root)
  OR cd model && python train_model.py
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os

def get_data():
    """
    Rich synthetic emotion dataset with diverse, realistic phrases.
    Covers: joy, sadness, anger, anxiety, fear, surprise, disgust, excitement, love, neutral
    """
    data = {
        "text": [
            # ── joy ──────────────────────────────────────────────────────────────
            "I am feeling very happy today!",
            "This is the best day of my life",
            "I feel so joyful and energetic",
            "Life is beautiful and I love every moment",
            "I just got the news and I'm overjoyed",
            "Today was absolutely wonderful, I couldn't be happier",
            "My heart is full of happiness right now",
            "Everything feels perfect and I'm so grateful",
            "I feel a deep sense of contentment and peace",
            "Laughing with friends makes everything better",
            "That made me smile from ear to ear",
            "Pure bliss — I am so thankful for this moment",
            "I woke up feeling radiant and full of hope",
            "The sun is shining and so is my mood",

            # ── sadness ──────────────────────────────────────────────────────────
            "I am extremely sad and depressed",
            "I can't stop crying and I don't know why",
            "Everything is falling apart around me",
            "I feel a deep hollow emptiness inside",
            "Nothing matters anymore, I'm so down",
            "I miss them so much, my heart aches",
            "I feel so lonely and abandoned",
            "This loss is unbearable and I'm grieving",
            "I have no energy and everything feels hopeless",
            "Tears keep rolling down my face",
            "I feel like nobody understands me",
            "I cried myself to sleep last night",
            "The world feels grey and cold today",
            "I am heartbroken after what happened",

            # ── anger ────────────────────────────────────────────────────────────
            "I am so angry right now!",
            "This makes me furious beyond measure",
            "I want to scream out of frustration",
            "How dare they do that to me?",
            "I am boiling with rage",
            "I can't believe this injustice — it infuriates me",
            "I'm seething and need to calm down",
            "They crossed a line and I am livid",
            "Stop being so unfair, it makes me furious",
            "I lost my temper and I regret nothing",
            "This is absolutely outrageous",
            "I feel a fire of anger burning inside me",
            "I am beyond annoyed and completely fed up",
            "Nobody respects me and that makes me so mad",

            # ── anxiety ──────────────────────────────────────────────────────────
            "I am feeling quite anxious about everything",
            "I'm worried about the test tomorrow",
            "I have this constant knot of worry in my stomach",
            "I keep overthinking everything and can't stop",
            "My mind is racing and I can't calm down",
            "I'm so stressed about the deadline ahead",
            "The uncertainty is making me nervous",
            "I've been tense and on edge all day",
            "I can't sleep because of all the anxious thoughts",
            "I feel a tightness in my chest from worry",
            "I'm dreading what might happen next",
            "Everything feels overwhelming and I can't breathe",
            "I spiraled into a panic and couldn't ground myself",
            "The pressure is immense and I'm struggling",

            # ── fear ─────────────────────────────────────────────────────────────
            "I am absolutely terrified",
            "This situation scares me deeply",
            "I had a nightmare that felt so real",
            "I'm paralyzed by fear and can't move",
            "The darkness makes me so afraid",
            "I fear the worst is about to happen",
            "I feel a creeping dread I can't shake",
            "My hands are shaking from fright",
            "That was genuinely frightening and I'm still scared",
            "I'm afraid of losing everything I have",
            "The threat feels real and I'm terrified",
            "Fear grips me every single night",

            # ── surprise ─────────────────────────────────────────────────────────
            "This is a pleasant surprise I didn't expect",
            "Wow, I didn't expect this at all!",
            "I am absolutely amazed by what just happened",
            "I couldn't believe my eyes when I saw it",
            "What a twist! I was totally caught off guard",
            "This news came out of nowhere and shocked me",
            "I'm stunned — I had no idea",
            "I gasped when I saw the result",
            "That revelation left me speechless",
            "I was blindsided by this completely",
            "Unbelievable! Nobody saw that coming",

            # ── disgust ──────────────────────────────────────────────────────────
            "That is absolutely disgusting",
            "I feel repulsed by what I saw",
            "This makes me feel sick to my stomach",
            "I can't believe how revolting that was",
            "I am nauseated by this behavior",
            "The thought of it makes me cringe",
            "Gross, I want nothing to do with that",
            "I feel deeply offended and grossed out",
            "That smell was absolutely repulsive",
            "Seeing that left me feeling dirty and disgusted",

            # ── excitement ───────────────────────────────────────────────────────
            "I am so excited for what comes next!",
            "The anticipation is killing me in the best way",
            "I can't wait — this is going to be amazing",
            "My energy is through the roof right now",
            "I'm buzzing with excitement and enthusiasm",
            "This project has me thrilled beyond words",
            "I jumped out of bed so eager to start",
            "Finally got the job offer! I'm thrilled!",
            "The countdown begins — I'm pumped!",
            "I feel on top of the world with excitement",
            "Ecstatic about joining the new AI research team",
            "I can't stop smiling because I'm so pumped",

            # ── love ─────────────────────────────────────────────────────────────
            "I love you more than words can say",
            "My heart overflows with love for this person",
            "Being with them makes everything perfect",
            "I feel warm and deeply loved",
            "I cherish every moment we spend together",
            "Love is what makes life worth living",
            "I adore everything about them — this is real",
            "This relationship fills my heart completely",
            "I feel such a deep bond and connection with them",
            "I fall more in love every single day",

            # ── neutral ──────────────────────────────────────────────────────────
            "I went to the store and bought groceries",
            "The meeting is scheduled for 3pm",
            "I read a book today and it was okay",
            "The weather is mild with some clouds",
            "I completed my tasks as planned",
            "Nothing unusual happened today",
            "I took a walk and it was fine",
            "The code compiles and works as expected",
            "Dinner was alright, nothing special",
            "I finished the report and submitted it",
        ],
        "emotion": [
            # joy (14)
            *["joy"] * 14,
            # sadness (14)
            *["sadness"] * 14,
            # anger (14)
            *["anger"] * 14,
            # anxiety (14)
            *["anxiety"] * 14,
            # fear (12)
            *["fear"] * 12,
            # surprise (11)
            *["surprise"] * 11,
            # disgust (10)
            *["disgust"] * 10,
            # excitement (12)
            *["excitement"] * 12,
            # love (10)
            *["love"] * 10,
            # neutral (10)
            *["neutral"] * 10,
        ]
    }
    return pd.DataFrame(data)


if __name__ == "__main__":
    print("=" * 55)
    print("  mini Emotion Cipher — Model Training")
    print("=" * 55)
    
    df = get_data()
    print(f"\nDataset loaded: {len(df)} samples across {df['emotion'].nunique()} emotion classes")
    print(f"Classes: {sorted(df['emotion'].unique())}\n")
    
    # TF-IDF Features (bigrams for better context capture)
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),     # unigrams + bigrams
        sublinear_tf=True       # Apply log normalization to tf
    )
    X = vectorizer.fit_transform(df['text'])
    y = df['emotion']
    
    # Train/test split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Logistic Regression with multi-class probability output
    model = LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
        C=1.0,
        solver='lbfgs',
        multi_class='multinomial'
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save models
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vec_path   = os.path.join(base_dir, 'vectorizer.pkl')
    model_path = os.path.join(base_dir, 'saved_model.pkl')
    
    with open(vec_path, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"\n✅ Training complete! Files saved:")
    print(f"   Model     → {model_path}")
    print(f"   Vectorizer→ {vec_path}")
    print(f"\nEmotion classes: {list(model.classes_)}")
