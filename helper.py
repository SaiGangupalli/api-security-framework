# Add this function to src/data_processing/excel_processor.py

def _find_matching_keywords(self, text, keyword_list):
    """
    Find which keywords from the list match in the text.
    
    Args:
        text (str): Text to search in
        keyword_list (list): List of keywords to search for
        
    Returns:
        tuple: (matching_keywords, count)
    """
    if not isinstance(text, str):
        return [], 0
    
    # Convert to lowercase for case-insensitive matching
    text = text.lower()
    
    # Find all matching keywords
    matching_keywords = []
    for keyword in keyword_list:
        if keyword.lower() in text:
            matching_keywords.append(keyword)
    
    return matching_keywords, len(matching_keywords)

# Update the _add_keyword_features method in ExcelProcessor class
def _add_keyword_features(self):
    """Add keyword-based features using the predefined security and fraud keywords."""
    from config.settings import SECURITY_KEYWORDS, FRAUD_KEYWORDS
    
    # Find matching security keywords and their count
    self.df['security_matching_keywords'] = self.df['combined_text'].apply(
        lambda x: ', '.join(self._find_matching_keywords(x, SECURITY_KEYWORDS)[0])
    )
    
    # Count security keywords
    self.df['security_keyword_count'] = self.df['combined_text'].apply(
        lambda x: self._find_matching_keywords(x, SECURITY_KEYWORDS)[1]
    )
    
    # Find matching fraud keywords and their count
    self.df['fraud_matching_keywords'] = self.df['combined_text'].apply(
        lambda x: ', '.join(self._find_matching_keywords(x, FRAUD_KEYWORDS)[0])
    )
    
    # Count fraud keywords
    self.df['fraud_keyword_count'] = self.df['combined_text'].apply(
        lambda x: self._find_matching_keywords(x, FRAUD_KEYWORDS)[1]
    )
    
    # Flag if any keywords are present
    self.df['has_security_keywords'] = self.df['security_keyword_count'] > 0
    self.df['has_fraud_keywords'] = self.df['fraud_keyword_count'] > 0
    
    # Initial labels based purely on keywords - to be refined by ML model
    self.df['initial_security_flag'] = self.df['has_security_keywords']
    self.df['initial_fraud_flag'] = self.df['has_fraud_keywords']
    
    logger.info("Added keyword-based features with matching keywords listed")





# Update this part in src/model_training/training.py within the predict_on_new_data method

def predict_on_new_data(self, new_data):
    """
    Make predictions on new data.
    
    Args:
        new_data (pandas.DataFrame): New data to predict on
        
    Returns:
        pandas.DataFrame: Data with prediction results
    """
    # Load the model and vectorizer if not already loaded
    if self.model.model_security is None or self.model.vectorizer is None:
        logger.info("Loading saved model and vectorizer")
        self.model.load_model()
        self.model.load_vectorizer()
    
    # Extract features using the saved vectorizer
    X, _ = self.feature_engineer.extract_features(new_data, vectorizer=self.model.vectorizer, fit=False)
    
    # Make predictions
    predictions = self.model.predict(X)
    
    # Add prediction results to the data
    result_df = new_data.copy()
    result_df['security_prediction'] = predictions['security_prediction']
    result_df['security_probability'] = predictions['security_probability']
    result_df['fraud_prediction'] = predictions['fraud_prediction']
    result_df['fraud_probability'] = predictions['fraud_probability']
    
    # Add prediction explanation columns if matching keywords are available
    if 'security_matching_keywords' in result_df.columns:
        # Add a column that shows matched keywords only for positive predictions
        result_df['security_matches'] = result_df.apply(
            lambda row: row['security_matching_keywords'] if row['security_prediction'] == 1 else "",
            axis=1
        )
    
    if 'fraud_matching_keywords' in result_df.columns:
        # Add a column that shows matched keywords only for positive predictions
        result_df['fraud_matches'] = result_df.apply(
            lambda row: row['fraud_matching_keywords'] if row['fraud_prediction'] == 1 else "",
            axis=1
        )
    
    # Add explanation column
    result_df['prediction_explanation'] = result_df.apply(
        lambda row: self._get_prediction_explanation(row), 
        axis=1
    )
    
    return result_df

# Add this helper method to the ModelTrainer class
def _get_prediction_explanation(self, row):
    """Generate a human-readable explanation for the prediction."""
    explanation = []
    
    # Add security explanation if predicted as security issue
    if row['security_prediction'] == 1:
        if 'security_matching_keywords' in row and row['security_matching_keywords']:
            explanation.append(f"Security risk detected (probability: {row['security_probability']:.2f}) "
                               f"with keywords: {row['security_matching_keywords']}")
        else:
            explanation.append(f"Security risk detected (probability: {row['security_probability']:.2f})")
    
    # Add fraud explanation if predicted as fraud issue
    if row['fraud_prediction'] == 1:
        if 'fraud_matching_keywords' in row and row['fraud_matching_keywords']:
            explanation.append(f"Fraud risk detected (probability: {row['fraud_probability']:.2f}) "
                              f"with keywords: {row['fraud_matching_keywords']}")
        else:
            explanation.append(f"Fraud risk detected (probability: {row['fraud_probability']:.2f})")
    
    # If no risks detected
    if not explanation:
        explanation.append("No security or fraud risks detected")
    
    return "\n".join(explanation)





# Update this part in the export_predictions_to_excel function in src/utils/helpers.py

# Add these columns to the column_widths dictionary:
column_widths = {
    'project_key': 10,
    'issue_key': 15,
    'summary': 40,
    'description': 60,
    'status': 12,
    'created_date': 18,
    'assignee': 20,
    'reporter': 20,
    'priority': 10,
    'labels': 25,
    'components': 25,
    'impact_label': 20,
    'security_prediction': 15,
    'security_probability': 15,
    'security_matches': 30,
    'fraud_prediction': 15,
    'fraud_probability': 15,
    'fraud_matches': 30,
    'prediction_explanation': 60
}

# Also, add this code to ensure text wrapping is applied to the explanation and matches columns:

# Find indices for explanation and matches columns
explanation_idx = df.columns.get_loc('prediction_explanation') + 1 if 'prediction_explanation' in df.columns else None
security_matches_idx = df.columns.get_loc('security_matches') + 1 if 'security_matches' in df.columns else None
fraud_matches_idx = df.columns.get_loc('fraud_matches') + 1 if 'fraud_matches' in df.columns else None

# In the cell formatting loop, add:
if explanation_idx:
    explanation_cell = worksheet.cell(row=row_idx, column=explanation_idx)
    explanation_cell.alignment = Alignment(wrap_text=True, vertical='top')

if security_matches_idx:
    security_matches_cell = worksheet.cell(row=row_idx, column=security_matches_idx)
    security_matches_cell.alignment = Alignment(wrap_text=True, vertical='top')

if fraud_matches_idx:
    fraud_matches_cell = worksheet.cell(row=row_idx, column=fraud_matches_idx)
    fraud_matches_cell.alignment = Alignment(wrap_text=True, vertical='top')


