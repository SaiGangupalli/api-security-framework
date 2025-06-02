# joblib_explorer.py - Script to explore .joblib files

import joblib
import pandas as pd
from pprint import pprint

def explore_vectorizer(file_path):
    """Explore a vectorizer.joblib file"""
    print("="*60)
    print("🔍 EXPLORING VECTORIZER.JOBLIB")
    print("="*60)
    
    try:
        # Load the vectorizer
        vectorizer = joblib.load(file_path)
        
        # Basic info
        print(f"📁 File: {file_path}")
        print(f"📊 Type: {type(vectorizer).__name__}")
        print()
        
        # Vocabulary info
        if hasattr(vectorizer, 'vocabulary_'):
            vocab = vectorizer.vocabulary_
            print(f"📚 Vocabulary Size: {len(vocab)} words")
            print(f"🔤 Max Features: {getattr(vectorizer, 'max_features', 'Unknown')}")
            print(f"🔠 N-gram Range: {getattr(vectorizer, 'ngram_range', 'Unknown')}")
            print()
            
            # Show top 20 words by position
            print("🏆 TOP 20 WORDS IN VOCABULARY:")
            sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])[:20]
            for word, position in sorted_vocab:
                print(f"  Position {position:3d}: '{word}'")
            print()
            
            # Show security-related words
            print("🔒 SECURITY-RELATED WORDS FOUND:")
            security_words = [word for word in vocab.keys() if any(keyword in word.lower() 
                            for keyword in ['auth', 'password', 'security', 'encrypt', 'bypass', 'hack'])]
            for word in sorted(security_words)[:15]:
                print(f"  Position {vocab[word]:3d}: '{word}'")
            print()
            
            # Show fraud-related words  
            print("💰 FRAUD-RELATED WORDS FOUND:")
            fraud_words = [word for word in vocab.keys() if any(keyword in word.lower() 
                         for keyword in ['payment', 'fraud', 'money', 'transaction', 'verify', 'credit'])]
            for word in sorted(fraud_words)[:15]:
                print(f"  Position {vocab[word]:3d}: '{word}'")
            print()
        
        # IDF scores (word importance)
        if hasattr(vectorizer, 'idf_'):
            idf_scores = vectorizer.idf_
            print(f"📈 IDF Scores Available: {len(idf_scores)} scores")
            
            # Create word-to-IDF mapping
            if hasattr(vectorizer, 'vocabulary_'):
                word_idf = [(word, idf_scores[pos]) for word, pos in vocab.items()]
                word_idf.sort(key=lambda x: x[1], reverse=True)  # Sort by IDF score
                
                print("🎯 TOP 15 MOST IMPORTANT WORDS (Highest IDF scores):")
                for word, score in word_idf[:15]:
                    print(f"  {score:.4f}: '{word}'")
                print()
                
                print("📉 TOP 15 LEAST IMPORTANT WORDS (Lowest IDF scores):")
                for word, score in word_idf[-15:]:
                    print(f"  {score:.4f}: '{word}'")
                print()
        
        # Other attributes
        print("🔧 OTHER VECTORIZER SETTINGS:")
        important_attrs = ['min_df', 'max_df', 'stop_words', 'lowercase', 'token_pattern']
        for attr in important_attrs:
            if hasattr(vectorizer, attr):
                value = getattr(vectorizer, attr)
                if attr == 'stop_words' and value is not None:
                    print(f"  {attr}: {len(value)} stop words" if hasattr(value, '__len__') else f"  {attr}: {value}")
                else:
                    print(f"  {attr}: {value}")
        
    except Exception as e:
        print(f"❌ Error loading vectorizer: {e}")

def explore_model(file_path):
    """Explore a model.joblib file"""
    print("="*60)
    print("🤖 EXPLORING MODEL.JOBLIB")
    print("="*60)
    
    try:
        # Load the model
        models_data = joblib.load(file_path)
        
        print(f"📁 File: {file_path}")
        print(f"📊 Type: {type(models_data)}")
        print()
        
        if isinstance(models_data, dict):
            print("📦 CONTENTS:")
            for key in models_data.keys():
                print(f"  🔑 {key}: {type(models_data[key])}")
            print()
            
            # Explore security model
            if 'security_model' in models_data:
                sec_model = models_data['security_model']
                print("🛡️ SECURITY MODEL:")
                print(f"  Type: {type(sec_model).__name__}")
                print(f"  Classes: {getattr(sec_model, 'classes_', 'Unknown')}")
                print(f"  Features: {getattr(sec_model, 'n_features_in_', 'Unknown')}")
                if hasattr(sec_model, 'feature_importances_'):
                    print(f"  Top 5 important features: {sec_model.feature_importances_[:5]}")
                print()
            
            # Explore fraud model
            if 'fraud_model' in models_data:
                fraud_model = models_data['fraud_model']
                print("💳 FRAUD MODEL:")
                print(f"  Type: {type(fraud_model).__name__}")
                print(f"  Classes: {getattr(fraud_model, 'classes_', 'Unknown')}")
                print(f"  Features: {getattr(fraud_model, 'n_features_in_', 'Unknown')}")
                if hasattr(fraud_model, 'feature_importances_'):
                    print(f"  Top 5 important features: {fraud_model.feature_importances_[:5]}")
                print()
            
            # Metadata
            if 'metadata' in models_data:
                print("📋 METADATA:")
                pprint(models_data['metadata'], width=80)
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")

def demo_vectorizer_usage(vectorizer_path):
    """Show how to use the vectorizer"""
    print("="*60)
    print("🎯 VECTORIZER USAGE DEMO")
    print("="*60)
    
    try:
        vectorizer = joblib.load(vectorizer_path)
        
        # Example texts
        examples = [
            "Users can bypass two-factor authentication",
            "Implement secure password storage with encryption",
            "Allow refunds without receipt verification",
            "Add new button to homepage"
        ]
        
        print("📝 EXAMPLE TRANSFORMATIONS:")
        for text in examples:
            # Transform text to numbers
            vector = vectorizer.transform([text])
            
            # Get non-zero elements (words that were found)
            if hasattr(vector, 'toarray'):
                dense_vector = vector.toarray()[0]
                non_zero_indices = [i for i, val in enumerate(dense_vector) if val > 0]
                
                print(f"\n📄 Text: '{text}'")
                print(f"   📊 Non-zero features: {len(non_zero_indices)} out of {len(dense_vector)}")
                
                # Show which words were recognized
                if hasattr(vectorizer, 'vocabulary_'):
                    reverse_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
                    recognized_words = [(reverse_vocab[i], dense_vector[i]) for i in non_zero_indices[:5]]
                    print(f"   🔤 Top words found: {recognized_words}")
    
    except Exception as e:
        print(f"❌ Error in demo: {e}")

def main():
    """Main function to explore joblib files"""
    import os
    
    # Look for joblib files in common locations
    possible_paths = [
        'models/trained_models/vectorizer.joblib',
        'models/trained_models/security_fraud_model.joblib',
        'vectorizer.joblib',
        'security_fraud_model.joblib'
    ]
    
    print("🔍 JOBLIB FILE EXPLORER")
    print("="*60)
    
    # Find existing files
    found_files = [path for path in possible_paths if os.path.exists(path)]
    
    if not found_files:
        print("❌ No .joblib files found in common locations.")
        print("💡 Make sure you've trained the model first:")
        print("   python main.py --train --input your_data.xlsx")
        return
    
    print(f"✅ Found {len(found_files)} joblib files:")
    for path in found_files:
        print(f"   📁 {path}")
    print()
    
    # Explore each file
    for file_path in found_files:
        if 'vectorizer' in file_path:
            explore_vectorizer(file_path)
            demo_vectorizer_usage(file_path)
        elif 'model' in file_path:
            explore_model(file_path)
        print()

if __name__ == "__main__":
    main()
