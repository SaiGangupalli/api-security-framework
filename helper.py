# src/utils/helpers.py - Updated export_predictions_to_excel function that works with raw data too

def export_predictions_to_excel(df, output_path):
    """
    Export data to Excel with improved formatting.
    Works with both raw Jira data and prediction results.
    
    Args:
        df (pandas.DataFrame): DataFrame with data
        output_path (str): Path to save the Excel file
        
    Returns:
        str: Path to the exported Excel file
    """
    try:
        # Clean up text fields by removing extra spaces
        text_columns = ['summary', 'description', 'combined_text']
        for col in text_columns:
            if col in df.columns:
                # Replace multiple spaces with single space
                df[col] = df[col].astype(str).str.replace(r'\s+', ' ', regex=True)
                # Strip leading/trailing whitespace
                df[col] = df[col].str.strip()
        
        # Check if this is prediction data or raw data
        is_prediction_data = all(col in df.columns for col in ['security_prediction', 'fraud_prediction'])
        
        # If it's prediction data, add an impact label
        if is_prediction_data:
            df['impact_label'] = 'No Impact'
            # If either security or fraud impact exists, mark as 'Security/Fraud Impact'
            df.loc[(df['security_prediction'] == 1) | (df['fraud_prediction'] == 1), 'impact_label'] = 'Security/Fraud Impact'
            # More specific labels
            df.loc[(df['security_prediction'] == 1) & (df['fraud_prediction'] == 0), 'impact_label'] = 'Security Impact'
            df.loc[(df['security_prediction'] == 0) & (df['fraud_prediction'] == 1), 'impact_label'] = 'Fraud Impact'
            df.loc[(df['security_prediction'] == 1) & (df['fraud_prediction'] == 1), 'impact_label'] = 'Security & Fraud Impact'
        
        # Create a Pandas Excel writer using openpyxl
        writer = pd.ExcelWriter(output_path, engine='openpyxl')
        
        # Convert DataFrame to Excel
        sheet_name = 'Predictions' if is_prediction_data else 'Jira Stories'
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Get the workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        
        # Define column widths
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
            'fraud_prediction': 15,
            'fraud_probability': 15
        }
        
        # Apply column widths
        for col_name, width in column_widths.items():
            if col_name in df.columns:
                col_idx = df.columns.get_loc(col_name) + 1
                col_letter = get_column_letter(col_idx)
                worksheet.column_dimensions[col_letter].width = width
        
        # Get column indices for text columns
        summary_idx = df.columns.get_loc('summary') + 1 if 'summary' in df.columns else None
        desc_idx = df.columns.get_loc('description') + 1 if 'description' in df.columns else None
        
        # Apply formatting to all cells
        for row_idx in range(2, len(df) + 2):  # +2 for header and 1-based indexing
            # Apply text wrapping to summary and description
            if summary_idx:
                cell = worksheet.cell(row=row_idx, column=summary_idx)
                cell.alignment = Alignment(wrap_text=True, vertical='top')
            
            if desc_idx:
                cell = worksheet.cell(row=row_idx, column=desc_idx)
                cell.alignment = Alignment(wrap_text=True, vertical='top')
        
            # Format impact label if present
            if is_prediction_data:
                impact_col_idx = df.columns.get_loc('impact_label') + 1
                impact_cell = worksheet.cell(row=row_idx, column=impact_col_idx)
                impact_text = impact_cell.value
                
                # Apply conditional formatting to impact label
                if impact_text == 'Security & Fraud Impact':
                    impact_cell.fill = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")  # Red
                    impact_cell.font = Font(color="FFFFFF", bold=True)  # White bold text
                elif impact_text == 'Security Impact':
                    impact_cell.fill = PatternFill(start_color="FFC000", end_color="FFC000", fill_type="solid")  # Orange
                    impact_cell.font = Font(bold=True)
                elif impact_text == 'Fraud Impact':
                    impact_cell.fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")  # Yellow
                    impact_cell.font = Font(bold=True)
                
                # Format probability cells with gradient coloring based on value if present
                if 'security_probability' in df.columns:
                    security_prob_idx = df.columns.get_loc('security_probability') + 1
                    sec_prob_cell = worksheet.cell(row=row_idx, column=security_prob_idx)
                    if isinstance(sec_prob_cell.value, (int, float)) and 0 <= sec_prob_cell.value <= 1:
                        intensity = int(255 - (sec_prob_cell.value * 255))
                        color = f"{intensity:02X}{255:02X}{intensity:02X}"  # Green gradient
                        sec_prob_cell.fill = PatternFill(start_color=color, end_color=color, fill_type="solid")
                
                if 'fraud_probability' in df.columns:
                    fraud_prob_idx = df.columns.get_loc('fraud_probability') + 1
                    fraud_prob_cell = worksheet.cell(row=row_idx, column=fraud_prob_idx)
                    if isinstance(fraud_prob_cell.value, (int, float)) and 0 <= fraud_prob_cell.value <= 1:
                        intensity = int(255 - (fraud_prob_cell.value * 255))
                        color = f"{intensity:02X}{intensity:02X}{255:02X}"  # Blue gradient
                        fraud_prob_cell.fill = PatternFill(start_color=color, end_color=color, fill_type="solid")
        
        # Format header row
        for col_idx in range(1, len(df.columns) + 1):
            cell = worksheet.cell(row=1, column=col_idx)
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="DDDDDD", end_color="DDDDDD", fill_type="solid")
            cell.alignment = Alignment(horizontal='center')
        
        # Add borders to all cells
        thin_border = Border(
            left=Side(style='thin'), 
            right=Side(style='thin'), 
            top=Side(style='thin'), 
            bottom=Side(style='thin')
        )
        
        for row in worksheet.iter_rows(min_row=1, max_row=len(df) + 1, min_col=1, max_col=len(df.columns)):
            for cell in row:
                cell.border = thin_border
        
        # Apply auto-filter to header row
        worksheet.auto_filter.ref = f"A1:{get_column_letter(len(df.columns))}{len(df) + 1}"
        
        # Freeze the header row
        worksheet.freeze_panes = 'A2'
        
        # Create a summary sheet if this is prediction data
        if is_prediction_data:
            summary_sheet = workbook.create_sheet("Summary")
            
            # Add summary statistics
            summary_data = [
                ["Jira User Stories Security & Fraud Analysis Summary"],
                [""],
                ["Total user stories analyzed:", len(df)],
                ["Stories with Security Impact:", df['security_prediction'].sum()],
                ["Stories with Fraud Impact:", df['fraud_prediction'].sum()],
                ["Stories with both Security & Fraud Impact:", ((df['security_prediction'] == 1) & (df['fraud_prediction'] == 1)).sum()],
                ["Stories with Security or Fraud Impact:", ((df['security_prediction'] == 1) | (df['fraud_prediction'] == 1)).sum()],
                [""],
                ["Analysis Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            ]
            
            for i, row_data in enumerate(summary_data, 1):
                for j, value in enumerate(row_data, 1):
                    cell = summary_sheet.cell(row=i, column=j, value=value)
                    if i == 1:  # Title
                        cell.font = Font(size=14, bold=True)
                        summary_sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=2)
                        cell.alignment = Alignment(horizontal='center')
                    
                    if j == 1 and i > 2:  # Left column headers
                        cell.font = Font(bold=True)
            
            # Adjust column widths in summary sheet
            summary_sheet.column_dimensions['A'].width = 40
            summary_sheet.column_dimensions['B'].width = 15
        
        # Save the workbook
        writer.close()
        logger.info(f"Data exported to {output_path} with formatting")
        return output_path
    except Exception as e:
        logger.error(f"Failed to export data: {e}")
        return None



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





