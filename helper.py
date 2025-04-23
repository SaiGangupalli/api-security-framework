# Updated evaluate method for the SecurityFraudModel class

def evaluate(self, X_test, y_security_test, y_fraud_test):
    """
    Evaluate the models on test data.
    
    Args:
        X_test (pandas.DataFrame): Feature matrix for testing
        y_security_test (pandas.Series): Security impact labels
        y_fraud_test (pandas.Series): Fraud impact labels
        
    Returns:
        dict: Evaluation metrics
    """
    predictions = self.predict(X_test)
    
    # Get classification reports
    security_report = classification_report(
        y_security_test, 
        predictions['security_prediction'],
        output_dict=True
    )
    
    fraud_report = classification_report(
        y_fraud_test, 
        predictions['fraud_prediction'],
        output_dict=True
    )
    
    # Get confusion matrices
    security_cm = confusion_matrix(y_security_test, predictions['security_prediction'])
    fraud_cm = confusion_matrix(y_fraud_test, predictions['fraud_prediction'])
    
    evaluation = {
        'security_report': security_report,
        'security_confusion_matrix': security_cm,
        'fraud_report': fraud_report,
        'fraud_confusion_matrix': fraud_cm
    }
    
    # Log the evaluation results
    logger.info("Security Model Performance:")
    logger.info(f"Accuracy: {security_report['accuracy']:.4f}")
    
    # Safely log class-specific metrics with error handling
    try:
        if '1' in security_report:
            logger.info(f"Precision (Class 1): {security_report['1']['precision']:.4f}")
            logger.info(f"Recall (Class 1): {security_report['1']['recall']:.4f}")
            logger.info(f"F1-Score (Class 1): {security_report['1']['f1-score']:.4f}")
        else:
            logger.warning("No positive examples (Class 1) in security test set")
    except KeyError as e:
        logger.warning(f"Could not log security metrics for class 1: {e}")
    
    logger.info("Fraud Model Performance:")
    logger.info(f"Accuracy: {fraud_report['accuracy']:.4f}")
    
    # Safely log class-specific metrics with error handling
    try:
        if '1' in fraud_report:
            logger.info(f"Precision (Class 1): {fraud_report['1']['precision']:.4f}")
            logger.info(f"Recall (Class 1): {fraud_report['1']['recall']:.4f}")
            logger.info(f"F1-Score (Class 1): {fraud_report['1']['f1-score']:.4f}")
        else:
            logger.warning("No positive examples (Class 1) in fraud test set")
    except KeyError as e:
        logger.warning(f"Could not log fraud metrics for class 1: {e}")
    
    return evaluation




# Updated visualize_results function for src/utils/helpers.py

def visualize_results(evaluation, feature_importances, output_dir="reports/figures"):
    """
    Generate visualizations of model results.
    
    Args:
        evaluation (dict): Evaluation metrics
        feature_importances (dict): Feature importances
        output_dir (str): Directory to save visualizations
        
    Returns:
        list: Paths to the generated visualization files
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    visualization_files = []
    
    # 1. Plot confusion matrices
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(
        evaluation['security_confusion_matrix'], 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['No Security Impact', 'Security Impact'],
        yticklabels=['No Security Impact', 'Security Impact']
    )
    plt.title('Security Impact Confusion Matrix')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(
        evaluation['fraud_confusion_matrix'], 
        annot=True, 
        fmt='d', 
        cmap='Oranges',
        xticklabels=['No Fraud Impact', 'Fraud Impact'],
        yticklabels=['No Fraud Impact', 'Fraud Impact']
    )
    plt.title('Fraud Impact Confusion Matrix')
    
    plt.tight_layout()
    confusion_matrix_path = os.path.join(output_dir, f"confusion_matrices_{timestamp}.png")
    plt.savefig(confusion_matrix_path)
    plt.close()
    visualization_files.append(confusion_matrix_path)
    
    # 2. Plot top feature importances
    plt.figure(figsize=(14, 12))
    
    # Handle potential KeyError or empty importances
    try:
        # Security model feature importances
        plt.subplot(2, 1, 1)
        if feature_importances and 'security' in feature_importances and feature_importances['security']:
            sec_features = list(feature_importances['security'].keys())[:15]
            sec_values = list(feature_importances['security'].values())[:15]
            
            if sec_features and sec_values:
                sns.barplot(x=sec_values, y=sec_features)
                plt.title('Top 15 Features for Security Impact Detection')
                plt.xlabel('Importance')
            else:
                plt.text(0.5, 0.5, "No feature importance data available for security model", 
                        horizontalalignment='center', verticalalignment='center')
                plt.title('Security Feature Importances (Not Available)')
        else:
            plt.text(0.5, 0.5, "No feature importance data available for security model", 
                    horizontalalignment='center', verticalalignment='center')
            plt.title('Security Feature Importances (Not Available)')
        
        # Fraud model feature importances
        plt.subplot(2, 1, 2)
        if feature_importances and 'fraud' in feature_importances and feature_importances['fraud']:
            fraud_features = list(feature_importances['fraud'].keys())[:15]
            fraud_values = list(feature_importances['fraud'].values())[:15]
            
            if fraud_features and fraud_values:
                sns.barplot(x=fraud_values, y=fraud_features)
                plt.title('Top 15 Features for Fraud Impact Detection')
                plt.xlabel('Importance')
            else:
                plt.text(0.5, 0.5, "No feature importance data available for fraud model", 
                        horizontalalignment='center', verticalalignment='center')
                plt.title('Fraud Feature Importances (Not Available)')
        else:
            plt.text(0.5, 0.5, "No feature importance data available for fraud model", 
                    horizontalalignment='center', verticalalignment='center')
            plt.title('Fraud Feature Importances (Not Available)')
        
    except Exception as e:
        logger.warning(f"Error creating feature importance plots: {e}")
        plt.text(0.5, 0.5, f"Error creating feature importance plots: {e}", 
                horizontalalignment='center', verticalalignment='center')
        plt.title('Feature Importances (Error)')
    
    plt.tight_layout()
    feature_imp_path = os.path.join(output_dir, f"feature_importances_{timestamp}.png")
    plt.savefig(feature_imp_path)
    plt.close()
    visualization_files.append(feature_imp_path)
    
    logger.info(f"Visualizations saved to: {', '.join(visualization_files)}")
    return visualization_files


# Updated update_keyword_lists function for src/utils/helpers.py

def update_keyword_lists(feature_importances, threshold=0.01):
    """
    Update security and fraud keyword lists based on model feature importances.
    
    Args:
        feature_importances (dict): Feature importances
        threshold (float): Minimum importance threshold
        
    Returns:
        tuple: (updated_security_keywords, updated_fraud_keywords)
    """
    security_keywords = set(SECURITY_KEYWORDS)
    fraud_keywords = set(FRAUD_KEYWORDS)
    
    # Check if feature_importances is valid
    if not feature_importances:
        logger.warning("No feature importances provided, using original keyword lists")
        return list(security_keywords), list(fraud_keywords)
    
    # Extract new keywords from TF-IDF features
    try:
        if 'security' in feature_importances and feature_importances['security']:
            for feature, importance in feature_importances['security'].items():
                if importance >= threshold and feature.startswith('tfidf_'):
                    keyword = feature.replace('tfidf_', '')
                    if len(keyword) > 3:  # Avoid short words
                        security_keywords.add(keyword)
    except Exception as e:
        logger.warning(f"Error updating security keywords: {e}")
    
    try:
        if 'fraud' in feature_importances and feature_importances['fraud']:
            for feature, importance in feature_importances['fraud'].items():
                if importance >= threshold and feature.startswith('tfidf_'):
                    keyword = feature.replace('tfidf_', '')
                    if len(keyword) > 3:  # Avoid short words
                        fraud_keywords.add(keyword)
    except Exception as e:
        logger.warning(f"Error updating fraud keywords: {e}")
    
    updated_security_keywords = list(security_keywords)
    updated_fraud_keywords = list(fraud_keywords)
    
    logger.info(f"Updated security keywords: {len(updated_security_keywords)} keywords")
    logger.info(f"Updated fraud keywords: {len(updated_fraud_keywords)} keywords")
    
    return updated_security_keywords, updated_fraud_keywords


def generate_feedback_template(predictions_df, output_path):
    """
    Generate a feedback template for manual review.
    
    Args:
        predictions_df (pandas.DataFrame): DataFrame with predictions
        output_path (str): Path to save the template
        
    Returns:
        str: Path to the feedback template
    """
    # Create a copy of the DataFrame with only the necessary columns
    template_df = predictions_df[['project_key', 'issue_key', 'summary', 
                                  'security_prediction', 'security_probability',
                                  'fraud_prediction', 'fraud_probability']].copy()
    
    # Add columns for manual verification
    template_df['security_verified'] = ""
    template_df['fraud_verified'] = ""
    template_df['notes'] = ""
    
    # Focus on issues with higher probability or conflicting predictions
    # Fix the ambiguous comparison
    template_df['needs_review'] = (
        (template_df['security_probability'] > 0.3) | 
        (template_df['fraud_probability'] > 0.3) |
        # Fix the comparison between prediction and probability
        ((template_df['security_prediction'] == 1) & (template_df['security_probability'] < 0.3)) |
        ((template_df['security_prediction'] == 0) & (template_df['security_probability'] > 0.3)) |
        ((template_df['fraud_prediction'] == 1) & (template_df['fraud_probability'] < 0.3)) |
        ((template_df['fraud_prediction'] == 0) & (template_df['fraud_probability'] > 0.3))
    )
    
    try:
        template_df.to_excel(output_path, index=False)
        logger.info(f"Feedback template saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to create feedback template: {e}")
        return None
