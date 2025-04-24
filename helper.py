# Enhanced keyword matching function for src/data_processing/excel_processor.py

def _find_matching_keywords(self, text, keyword_list):
    """
    Find which keywords from the list match in the text,
    including similar words and variations.
    
    Args:
        text (str): Text to search in
        keyword_list (list): List of keywords to search for
        
    Returns:
        tuple: (matching_keywords, count)
    """
    if not isinstance(text, str) or not text:
        return [], 0
    
    # Convert to lowercase for case-insensitive matching
    text = text.lower()
    
    # Tokenize the text
    try:
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(text)
    except:
        # Simple tokenization as fallback
        tokens = text.split()
    
    # Try to lemmatize tokens (if possible)
    try:
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    except:
        # Use tokens as is if lemmatization fails
        lemmatized_tokens = tokens
    
    # Create stems for fuzzy matching
    try:
        from nltk.stem.porter import PorterStemmer
        stemmer = PorterStemmer()
        stems = [stemmer.stem(token) for token in tokens]
    except:
        # No stemming if it fails
        stems = []
    
    # Find all matching keywords (direct matches)
    matching_keywords = []
    matched_indices = set()  # To avoid duplicate matches
    
    # 1. Direct substring matching (original method)
    for i, keyword in enumerate(keyword_list):
        if i in matched_indices:
            continue
            
        keyword_lower = keyword.lower()
        if keyword_lower in text:
            matching_keywords.append(keyword)
            matched_indices.add(i)
    
    # 2. Token-level exact matching
    for i, keyword in enumerate(keyword_list):
        if i in matched_indices:
            continue
            
        # Handle multi-word keywords
        keyword_tokens = keyword.lower().split()
        
        # Check if all tokens in the keyword appear in the text tokens
        if all(kt in tokens for kt in keyword_tokens):
            matching_keywords.append(keyword)
            matched_indices.add(i)
    
    # 3. Lemma matching (finds "encrypt" when searching for "encryption")
    try:
        lemmatizer = WordNetLemmatizer()
        for i, keyword in enumerate(keyword_list):
            if i in matched_indices:
                continue
                
            # Get lemmas for the keyword
            keyword_tokens = keyword.lower().split()
            keyword_lemmas = [lemmatizer.lemmatize(kt) for kt in keyword_tokens]
            
            # Check if any lemmatized token matches keyword lemmas
            if any(kl in lemmatized_tokens for kl in keyword_lemmas):
                matching_keywords.append(keyword)
                matched_indices.add(i)
    except:
        # Skip lemma matching if NLTK is not available
        pass
    
    # 4. Stem matching (captures more variations)
    try:
        stemmer = PorterStemmer()
        for i, keyword in enumerate(keyword_list):
            if i in matched_indices:
                continue
                
            # Get stems for the keyword
            keyword_tokens = keyword.lower().split()
            keyword_stems = [stemmer.stem(kt) for kt in keyword_tokens]
            
            # Check if any stem matches keyword stems
            if any(ks in stems for ks in keyword_stems):
                matching_keywords.append(keyword)
                matched_indices.add(i)
    except:
        # Skip stem matching if NLTK is not available
        pass
    
    # 5. Simple fuzzy matching (allows for minor typos/variations)
    try:
        from difflib import SequenceMatcher
        
        for i, keyword in enumerate(keyword_list):
            if i in matched_indices:
                continue
                
            keyword_lower = keyword.lower()
            
            # Check similarity with tokens
            for token in tokens:
                if len(token) > 3 and len(keyword_lower) > 3:  # Only meaningful words
                    similarity = SequenceMatcher(None, token, keyword_lower).ratio()
                    if similarity > 0.85:  # High similarity threshold
                        matching_keywords.append(f"{keyword} (similar to '{token}')")
                        matched_indices.add(i)
                        break
    except:
        # Skip fuzzy matching if it fails
        pass
    
    return matching_keywords, len(matching_keywords)





# Add this to config/settings.py

# Keyword synonyms for expanded matching
SECURITY_KEYWORD_SYNONYMS = {
    "authentication": ["auth", "login", "signin", "sign-in", "credentials", "authenticate"],
    "authorization": ["authz", "permissions", "access control", "privilege"],
    "encrypt": ["encryption", "encrypted", "cipher", "crypto"],
    "decrypt": ["decryption", "decrypted", "decipher"],
    "password": ["passwd", "pwd", "passphrase", "credentials"],
    "vulnerability": ["vuln", "weakness", "exploit", "flaw", "bug"],
    "injection": ["sql injection", "xss", "code injection", "script injection"],
    "malware": ["virus", "trojan", "spyware", "ransomware", "malicious code"],
    "phishing": ["spoofing", "social engineering", "fake login"],
    "firewall": ["fw", "waf", "filtering", "network security"],
    "certificate": ["cert", "ssl", "tls", "x509"]
}

FRAUD_KEYWORD_SYNONYMS = {
    "fraud": ["fraudulent", "scam", "deceptive", "illicit"],
    "suspicious": ["suspect", "questionable", "doubtful", "unusual"],
    "anomaly": ["abnormal", "outlier", "irregular", "deviation"],
    "detection": ["detect", "identify", "discover", "flag"],
    "prevention": ["prevent", "avoid", "mitigate", "block"],
    "verification": ["verify", "validate", "confirm", "check"],
    "monitoring": ["monitor", "surveillance", "tracking", "observing"],
    "transaction": ["payment", "transfer", "purchase", "activity"],
    "alert": ["alarm", "notification", "warning", "flag"],
    "unauthorized": ["illegal", "illicit", "prohibited", "not permitted"]
}

# Function to expand keyword lists with synonyms
def expand_keyword_list(keyword_list, synonym_map):
    """
    Expand a keyword list by adding synonyms.
    
    Args:
        keyword_list (list): Original keyword list
        synonym_map (dict): Dictionary mapping keywords to their synonyms
        
    Returns:
        list: Expanded keyword list with synonyms
    """
    expanded_list = keyword_list.copy()
    
    # Add synonyms for keywords that have them
    for keyword in keyword_list:
        if keyword in synonym_map:
            expanded_list.extend(synonym_map[keyword])
    
    # Remove duplicates and return
    return list(set(expanded_list))

# Expand the keyword lists with synonyms
EXPANDED_SECURITY_KEYWORDS = expand_keyword_list(SECURITY_KEYWORDS, SECURITY_KEYWORD_SYNONYMS)
EXPANDED_FRAUD_KEYWORDS = expand_keyword_list(FRAUD_KEYWORDS, FRAUD_KEYWORD_SYNONYMS)






# Update this in src/data_processing/excel_processor.py - _add_keyword_features method

def _add_keyword_features(self):
    """Add keyword-based features using the predefined security and fraud keywords with similar word matching."""
    try:
        # Try to import expanded keyword lists first
        from config.settings import (
            EXPANDED_SECURITY_KEYWORDS, 
            EXPANDED_FRAUD_KEYWORDS,
            SECURITY_KEYWORDS,
            FRAUD_KEYWORDS
        )
        
        # Use expanded lists if available
        security_keywords = EXPANDED_SECURITY_KEYWORDS
        fraud_keywords = EXPANDED_FRAUD_KEYWORDS
        logger.info(f"Using expanded keyword lists: {len(security_keywords)} security keywords, {len(fraud_keywords)} fraud keywords")
    except ImportError:
        # Fall back to regular keyword lists
        from config.settings import SECURITY_KEYWORDS, FRAUD_KEYWORDS
        security_keywords = SECURITY_KEYWORDS
        fraud_keywords = FRAUD_KEYWORDS
        logger.info(f"Using standard keyword lists: {len(security_keywords)} security keywords, {len(fraud_keywords)} fraud keywords")
    
    # Find matching security keywords and their count
    security_matches = self.df['combined_text'].apply(
        lambda x: self._find_matching_keywords(x, security_keywords)
    )
    
    # Separate matching keywords and counts
    self.df['security_matching_keywords'] = security_matches.apply(lambda x: ', '.join(x[0]))
    self.df['security_keyword_count'] = security_matches.apply(lambda x: x[1])
    
    # Find matching fraud keywords and their count
    fraud_matches = self.df['combined_text'].apply(
        lambda x: self._find_matching_keywords(x, fraud_keywords)
    )
    
    # Separate matching keywords and counts
    self.df['fraud_matching_keywords'] = fraud_matches.apply(lambda x: ', '.join(x[0]))
    self.df['fraud_keyword_count'] = fraud_matches.apply(lambda x: x[1])
    
    # Add original keyword matches (for reference)
    original_security_matches = self.df['combined_text'].apply(
        lambda x: ', '.join([k for k in SECURITY_KEYWORDS if k.lower() in str(x).lower()])
    )
    original_fraud_matches = self.df['combined_text'].apply(
        lambda x: ', '.join([k for k in FRAUD_KEYWORDS if k.lower() in str(x).lower()])
    )
    
    self.df['original_security_matches'] = original_security_matches
    self.df['original_fraud_matches'] = original_fraud_matches
    
    # Flag if any keywords are present
    self.df['has_security_keywords'] = self.df['security_keyword_count'] > 0
    self.df['has_fraud_keywords'] = self.df['fraud_keyword_count'] > 0
    
    # Initial labels based purely on keywords - to be refined by ML model
    self.df['initial_security_flag'] = self.df['has_security_keywords']
    self.df['initial_fraud_flag'] = self.df['has_fraud_keywords']
    
    logger.info("Added keyword-based features with matching keywords and similar word detection")
