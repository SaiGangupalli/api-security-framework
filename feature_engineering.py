# src/model_training/feature_engineering.py
import logging
import pandas as pd
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Try to import NLTK components with fallbacks
try:
    from nltk.corpus import stopwords
except ImportError:
    stopwords = None

try:
    from nltk.stem import WordNetLemmatizer
except ImportError:
    WordNetLemmatizer = None

try:
    from nltk.tokenize import word_tokenize
except ImportError:
    word_tokenize = None

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Class to handle feature engineering for the ML model."""
    
    def __init__(self, nltk_data_path=None):
        """
        Initialize the feature engineer.
        
        Args:
            nltk_data_path (str): Path to NLTK data directory
        """
        # Set NLTK data path if provided
        if nltk_data_path:
            nltk.data.path.append(nltk_data_path)
            logger.info(f"Added NLTK data path: {nltk_data_path}")
            
            # Print available NLTK data
            try:
                from nltk.data import path as nltk_path
                logger.info(f"NLTK data paths: {nltk_path}")
            except:
                pass
        
        # Fallback stopwords list (will be used if NLTK stopwords can't be loaded)
        default_stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
            'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than',
            'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during',
            'to', 'from', 'in', 'on', 'by', 'with', 'at', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'but', 'can', 'could', 'should',
            'would', 'might', 'will', 'shall', 'may', 'must', 'ought'
        }
        
        # Try to load stopwords
        try:
            if stopwords is not None:
                self.stop_words = set(stopwords.words('english'))
                logger.info(f"Successfully loaded {len(self.stop_words)} NLTK stopwords.")
            else:
                raise ImportError("NLTK stopwords module not available")
        except Exception as e:
            logger.warning(f"Could not load NLTK stopwords: {e}")
            self.stop_words = default_stopwords
            logger.info(f"Using {len(self.stop_words)} default stopwords instead.")
            
        # Try to initialize lemmatizer
        try:
            if WordNetLemmatizer is not None:
                self.lemmatizer = WordNetLemmatizer()
                logger.info("Successfully loaded NLTK WordNetLemmatizer.")
            else:
                raise ImportError("NLTK WordNetLemmatizer not available")
        except Exception as e:
            logger.warning(f"Could not load NLTK lemmatizer: {e}")
            # Simple lemmatizer function (just returns the word)
            self.lemmatizer = lambda word: word
            logger.info("Using identity function as fallback lemmatizer.")
    
    def preprocess_text(self, text):
        """
        Clean and preprocess text data.
        
        Args:
            text (str): Raw text input
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Custom tokenization that doesn't rely on punkt_tab
        def simple_tokenize(text):
            # First split by whitespace
            tokens = text.split()
            # Then handle punctuation - separate punctuation from words
            result = []
            for token in tokens:
                # Split out punctuation at beginning and end of tokens
                cleaned = token.strip('.,!?;:()[]{}""\'')
                if cleaned:
                    result.append(cleaned)
            return result
            
        try:
            # Try NLTK's word_tokenize if available
            if word_tokenize is not None:
                tokens = word_tokenize(text)
                logger.debug("Used NLTK word_tokenize successfully")
            else:
                # If word_tokenize isn't available, try the basic tokenizer
                try:
                    from nltk.tokenize import TreebankWordTokenizer
                    tokenizer = TreebankWordTokenizer()
                    tokens = tokenizer.tokenize(text)
                    logger.debug("Used NLTK TreebankWordTokenizer successfully")
                except (ImportError, LookupError):
                    # If that fails too, use our simple tokenizer
                    tokens = simple_tokenize(text)
                    logger.debug("Used custom simple tokenizer")
        except Exception as e:
            logger.warning(f"All NLTK tokenization methods failed: {e}. Using simple tokenizer.")
            tokens = simple_tokenize(text)
        
        # Remove stopwords and lemmatize
        try:
            if callable(getattr(self.lemmatizer, 'lemmatize', None)):
                # If we have the real NLTK lemmatizer
                cleaned_tokens = [
                    self.lemmatizer.lemmatize(token) for token in tokens 
                    if token not in self.stop_words and len(token) > 2
                ]
            else:
                # If we have the fallback simple lemmatizer
                cleaned_tokens = [
                    self.lemmatizer(token) for token in tokens 
                    if token not in self.stop_words and len(token) > 2
                ]
        except Exception as e:
            logger.warning(f"Error during lemmatization/stopword removal: {e}. Using simple filtering.")
            cleaned_tokens = [token for token in tokens if len(token) > 2]
        
        return ' '.join(cleaned_tokens)
    
    def extract_features(self, df, vectorizer=None, fit=True):
        """
        Extract features from the data.
        
        Args:
            df (pandas.DataFrame): Input DataFrame
            vectorizer (TfidfVectorizer, optional): Pre-trained vectorizer for text features
            fit (bool): Whether to fit the vectorizer on the data
            
        Returns:
            tuple: (X_features, vectorizer)
        """
        # Preprocess text
        logger.info("Preprocessing text data")
        df['processed_text'] = df['combined_text'].apply(self.preprocess_text)
        
        # Create text features using TF-IDF
        if vectorizer is None and fit:
            logger.info("Creating new TF-IDF vectorizer")
            vectorizer = TfidfVectorizer(
                max_features=2000,
                min_df=5,
                max_df=0.85,
                ngram_range=(1, 2)
            )
        
        # Check if vectorizer is None (this could happen if it wasn't provided and fit is False)
        if vectorizer is None:
            logger.error("Vectorizer is None. Cannot transform text features.")
            # Create an empty DataFrame with expected structure
            text_features_df = pd.DataFrame()
        else:
            if fit:
                logger.info("Fitting TF-IDF vectorizer")
                text_features = vectorizer.fit_transform(df['processed_text'])
            else:
                logger.info("Transforming text using existing vectorizer")
                text_features = vectorizer.transform(df['processed_text'])
            
            # Convert sparse matrix to DataFrame
            text_feature_names = vectorizer.get_feature_names_out()
            text_features_df = pd.DataFrame(
                text_features.toarray(),
                columns=[f'tfidf_{name}' for name in text_feature_names]
            )
        
        # Extract other features (non-text features)
        logger.info("Extracting additional features")
        
        # Make sure required columns exist
        required_columns = ['security_keyword_count', 'fraud_keyword_count']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Required column '{col}' not found in dataframe. Adding zeros.")
                df[col] = 0
        
        feature_df = df[['security_keyword_count', 'fraud_keyword_count']].copy()
        
        # Add length features
        feature_df['summary_length'] = df['summary'].str.len()
        feature_df['description_length'] = df['description'].str.len()
        
        # Add text complexity features (average word length) with error handling
        def safe_avg_word_length(text):
            if not text or not text.split():
                return 0
            try:
                return np.mean([len(w) for w in text.split()])
            except:
                return 0
        
        feature_df['avg_word_length'] = df['processed_text'].apply(safe_avg_word_length)
        
        # Number of components with error handling
        try:
            feature_df['component_count'] = df['components'].str.count(',') + 1
            feature_df.loc[df['components'] == "", 'component_count'] = 0
        except:
            logger.warning("Error calculating component count. Using zeros.")
            feature_df['component_count'] = 0
        
        # Number of labels with error handling
        try:
            feature_df['label_count'] = df['labels'].str.count(',') + 1
            feature_df.loc[df['labels'] == "", 'label_count'] = 0
        except:
            logger.warning("Error calculating label count. Using zeros.")
            feature_df['label_count'] = 0
        
        # One-hot encode status with error handling
        try:
            status_dummies = pd.get_dummies(df['status'], prefix='status')
        except:
            logger.warning("Error creating status dummies. Skipping.")
            status_dummies = pd.DataFrame(index=df.index)
        
        # Combine all features
        logger.info("Combining all features")
        all_features = pd.concat([feature_df, text_features_df, status_dummies], axis=1)
        
        # Make sure we have at least some features
        if all_features.empty:
            logger.warning("No features extracted. Creating minimal feature set.")
            all_features = pd.DataFrame(index=df.index)
            all_features['fallback_feature'] = 1.0
        
        return all_features, vectorizer
