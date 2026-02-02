"""
ML Model Trainer for Medical Query Intent Classification
Trains a TF-IDF + Logistic Regression model on labeled query data
"""

import json
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def load_training_data(filepath='data/training_data.json'):
    """Load training data from JSON file"""
    print("=" * 70)
    print("LOADING TRAINING DATA")
    print("=" * 70)
    print(f"\nLoading from: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\n❌ Training data not found at: {filepath}\n"
            f"Please ensure training_data.json is in the data/ folder"
        )
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(f"✓ Successfully loaded {len(data)} samples")
    return data


def prepare_features_and_labels(data):
    """Extract queries and intent labels from data"""
    queries = [item['query'] for item in data]
    intents = [item['intent'] for item in data]
    
    # Count unique intents
    unique_intents = set(intents)
    print(f"✓ Found {len(unique_intents)} unique intent categories:")
    for intent in sorted(unique_intents):
        count = intents.count(intent)
        print(f"  - {intent}: {count} samples")
    
    return queries, intents


def create_model():
    """Create ML pipeline with TF-IDF and Logistic Regression"""
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)
    
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=100,      # Limit vocabulary size
            ngram_range=(1, 2),    # Use unigrams and bigrams (1-word and 2-word phrases)
            lowercase=True,        # Convert to lowercase
            stop_words='english'   # Remove common English words (the, a, is, etc.)
        )),
        ('classifier', LogisticRegression(
            max_iter=1000,         # Maximum training iterations
            random_state=42,       # For reproducibility
            #multi_class='ovr'      # One-vs-rest for multi-class classification
        ))
    ])
    
    print("✓ Model pipeline created:")
    print("  - TF-IDF Vectorizer (max_features=100, ngram_range=(1,2))")
    print("  - Logistic Regression (max_iter=1000)")
    
    return model


def train_model(model, X_train, y_train):
    """Train the model"""
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    print("\nTraining in progress...")
    
    model.fit(X_train, y_train)
    
    print("✓ Training complete!")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✓ Test Accuracy: {accuracy:.2%}")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print("-" * 70)
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    print("Confusion Matrix:")
    print("-" * 70)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    print(f"Classes: {list(model.classes_)}")
    print(cm)
    
    return accuracy


def save_model(model, filepath='models/intent_classifier.pkl'):
    """Save trained model to disk"""
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✓ Model saved to: {filepath}")


def test_model_predictions(model):
    """Test model with example queries"""
    print("\n" + "=" * 70)
    print("TESTING MODEL WITH EXAMPLE QUERIES")
    print("=" * 70)
    
    test_queries = [
        "How many CT scans?",
        "List all MRI studies",
        "What modalities are available?",
        "Show me brain imaging",
        "Count chest X-rays",
        "What body parts can be scanned?",
        "Give me ultrasound abdomen studies",
        "How many spine MRI scans?"
    ]
    
    for query in test_queries:
        intent = model.predict([query])[0]
        probabilities = model.predict_proba([query])[0]
        confidence = max(probabilities)
        
        print(f"\nQuery: '{query}'")
        print(f"  → Predicted Intent: {intent}")
        print(f"  → Confidence: {confidence:.2%}")


def main():
    """Main training pipeline"""
    print("\n" + "=" * 70)
    print("MEDICAL QUERY INTENT CLASSIFIER - TRAINING PIPELINE")
    print("=" * 70)
    print()
    
    # Step 1: Load data
    data = load_training_data()
    
    # Step 2: Prepare features and labels
    print("\n" + "=" * 70)
    print("PREPARING DATA")
    print("=" * 70)
    queries, intents = prepare_features_and_labels(data)
    
    # Step 3: Split data (80% train, 20% test)
    print(f"\n✓ Splitting data into train/test sets (80/20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        queries, intents, 
        test_size=0.2,        # 20% for testing
        random_state=42,      # For reproducibility
        stratify=intents      # Maintain class distribution in both sets
    )
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Testing samples: {len(X_test)}")
    
    # Step 4: Create model
    model = create_model()
    
    # Step 5: Train model
    model = train_model(model, X_train, y_train)
    
    # Step 6: Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Step 7: Save model
    save_model(model)
    
    # Step 8: Test with examples
    test_model_predictions(model)
    
    # Final summary
    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Final Model Performance:")
    print(f"  - Test Accuracy: {accuracy:.2%}")
    print(f"  - Training Samples: {len(X_train)}")
    print(f"  - Test Samples: {len(X_test)}")
    print(f"\n Model Location:")
    print(f"  - models/intent_classifier.pkl")
    print(f"\n Next Steps:")
    print(f"  1. Test entity extraction: python ml_model/entity_extractor.py")
    print(f"  2. Create MCP server: (request server.py code)")
    print(f"  3. Start FastAPI: (request fastapi_app.py code)")
    print()


if __name__ == "__main__":
    main()
