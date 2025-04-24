# src/utils/helpers.py - Updated export_predictions_to_excel function with column filtering

def export_predictions_to_excel(df, output_path, exclude_columns=None, hidden_columns=None):
    """
    Export data to Excel with improved formatting.
    Works with both raw Jira data and prediction results.
    Allows excluding or hiding specific columns.
    
    Args:
        df (pandas.DataFrame): DataFrame with data
        output_path (str): Path to save the Excel file
        exclude_columns (list): Columns to exclude completely from the output
        hidden_columns (list): Columns to include but hide in Excel
        
    Returns:
        str: Path to the exported Excel file
    """
    try:
        # Set default values for column lists if None
        exclude_columns = exclude_columns or []
        hidden_columns = hidden_columns or []
        
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
        
        # Create a copy of the dataframe excluding specified columns
        filtered_df = df.drop(columns=[col for col in exclude_columns if col in df.columns])
        
        # Create a Pandas Excel writer using openpyxl
        writer = pd.ExcelWriter(output_path, engine='openpyxl')
        
        # Convert DataFrame to Excel
        sheet_name = 'Predictions' if is_prediction_data else 'Jira Stories'
        filtered_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
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
            'security_matches': 30,
            'fraud_prediction': 15,
            'fraud_probability': 15,
            'fraud_matches': 30,
            'prediction_explanation': 60
        }
        
        # Apply column widths and hide columns as needed
        for col_idx, col_name in enumerate(filtered_df.columns, 1):
            col_letter = get_column_letter(col_idx)
            
            # Apply width if defined
            if col_name in column_widths:
                worksheet.column_dimensions[col_letter].width = column_widths[col_name]
            
            # Hide column if in hidden_columns list
            if col_name in hidden_columns:
                worksheet.column_dimensions[col_letter].hidden = True
        
        # Get indices for specific columns
        summary_idx = filtered_df.columns.get_loc('summary') + 1 if 'summary' in filtered_df.columns else None
        desc_idx = filtered_df.columns.get_loc('description') + 1 if 'description' in filtered_df.columns else None
        explanation_idx = filtered_df.columns.get_loc('prediction_explanation') + 1 if 'prediction_explanation' in filtered_df.columns else None
        security_matches_idx = filtered_df.columns.get_loc('security_matches') + 1 if 'security_matches' in filtered_df.columns else None
        fraud_matches_idx = filtered_df.columns.get_loc('fraud_matches') + 1 if 'fraud_matches' in filtered_df.columns else None
        
        # Apply formatting to all cells
        for row_idx in range(2, len(filtered_df) + 2):  # +2 for header and 1-based indexing
            # Apply vertical alignment to ALL cells
            for col_idx in range(1, len(filtered_df.columns) + 1):
                cell = worksheet.cell(row=row_idx, column=col_idx)
                cell.alignment = Alignment(vertical='top')
            
            # Add text wrapping for text columns
            for idx in [summary_idx, desc_idx, explanation_idx, security_matches_idx, fraud_matches_idx]:
                if idx:
                    cell = worksheet.cell(row=row_idx, column=idx)
                    cell.alignment = Alignment(wrap_text=True, vertical='top')
            
            # Format impact label if present
            if is_prediction_data and 'impact_label' in filtered_df.columns:
                impact_col_idx = filtered_df.columns.get_loc('impact_label') + 1
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
            if 'security_probability' in filtered_df.columns:
                security_prob_idx = filtered_df.columns.get_loc('security_probability') + 1
                sec_prob_cell = worksheet.cell(row=row_idx, column=security_prob_idx)
                if isinstance(sec_prob_cell.value, (int, float)) and 0 <= sec_prob_cell.value <= 1:
                    intensity = int(255 - (sec_prob_cell.value * 255))
                    color = f"{intensity:02X}{255:02X}{intensity:02X}"  # Green gradient
                    sec_prob_cell.fill = PatternFill(start_color=color, end_color=color, fill_type="solid")
            
            if 'fraud_probability' in filtered_df.columns:
                fraud_prob_idx = filtered_df.columns.get_loc('fraud_probability') + 1
                fraud_prob_cell = worksheet.cell(row=row_idx, column=fraud_prob_idx)
                if isinstance(fraud_prob_cell.value, (int, float)) and 0 <= fraud_prob_cell.value <= 1:
                    intensity = int(255 - (fraud_prob_cell.value * 255))
                    color = f"{intensity:02X}{intensity:02X}{255:02X}"  # Blue gradient
                    fraud_prob_cell.fill = PatternFill(start_color=color, end_color=color, fill_type="solid")
        
        # Format header row
        for col_idx in range(1, len(filtered_df.columns) + 1):
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
        
        for row in worksheet.iter_rows(min_row=1, max_row=len(filtered_df) + 1, min_col=1, max_col=len(filtered_df.columns)):
            for cell in row:
                cell.border = thin_border
        
        # Apply auto-filter to header row
        worksheet.auto_filter.ref = f"A1:{get_column_letter(len(filtered_df.columns))}{len(filtered_df) + 1}"
        
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





# Update the predict_and_analyze function in main.py

def predict_and_analyze(model_trainer, input_data, nltk_data_path=None, exclude_columns=None, hidden_columns=None):
    """
    Make predictions and analyze data.
    
    Args:
        model_trainer (ModelTrainer): Trained model instance
        input_data: Input data (DataFrame or path to file)
        nltk_data_path (str): Path to NLTK data directory
        exclude_columns (list): Columns to exclude from Excel output
        hidden_columns (list): Columns to hide in Excel output
        
    Returns:
        tuple: (predictions_df, predictions_path, template_path)
    """
    logger.info("Running predictions and analysis")
    
    # Set default values if None
    exclude_columns = exclude_columns or []
    hidden_columns = hidden_columns or []
    
    # Common columns to consider excluding/hiding
    common_exclude_candidates = [
        'combined_text',               # Usually large and redundant with summary+description
        'original_security_matches',   # Internal use, redundant with security_matches
        'original_fraud_matches',      # Internal use, redundant with fraud_matches
        'has_security_keywords',       # Internal use, redundant with security_prediction
        'has_fraud_keywords',          # Internal use, redundant with fraud_prediction
        'initial_security_flag',       # Internal use, superseded by security_prediction
        'initial_fraud_flag'           # Internal use, superseded by fraud_prediction
    ]
    
    common_hide_candidates = [
        'security_keyword_count',      # Technical detail, but might be useful
        'fraud_keyword_count',         # Technical detail, but might be useful
        'security_matching_keywords',  # Raw matches before filtering
        'fraud_matching_keywords'      # Raw matches before filtering
    ]
    
    # Update feature engineer if NLTK data path is provided
    if nltk_data_path:
        model_trainer.feature_engineer = FeatureEngineer(nltk_data_path)
    
    # Convert input_data to DataFrame if it's a file path
    if isinstance(input_data, str):
        input_df = pd.read_excel(input_data)
    else:
        input_df = input_data
    
    # Make predictions
    predictions_df = model_trainer.predict_on_new_data(input_df)
    
    # Export predictions to Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions_path = os.path.join(
        PROCESSED_DATA_PATH, 
        f"security_fraud_predictions_{timestamp}.xlsx"
    )
    export_predictions_to_excel(
        predictions_df, 
        predictions_path,
        exclude_columns=exclude_columns,
        hidden_columns=hidden_columns
    )
    
    # Generate feedback template
    template_path = os.path.join(
        PROCESSED_DATA_PATH, 
        f"feedback_template_{timestamp}.xlsx"
    )
    generate_feedback_template(predictions_df, template_path)
    
    logger.info(f"Predictions saved to {predictions_path}")
    logger.info(f"Feedback template saved to {template_path}")
    
    # Summary statistics
    security_count = predictions_df['security_prediction'].sum()
    fraud_count = predictions_df['fraud_prediction'].sum()
    both_count = ((predictions_df['security_prediction'] == 1) & 
                  (predictions_df['fraud_prediction'] == 1)).sum()
    
    logger.info(f"Analysis Summary:")
    logger.info(f"Total user stories analyzed: {len(predictions_df)}")
    logger.info(f"User stories with security impacts: {security_count} ({security_count/len(predictions_df)*100:.1f}%)")
    logger.info(f"User stories with fraud impacts: {fraud_count} ({fraud_count/len(predictions_df)*100:.1f}%)")
    logger.info(f"User stories with both security and fraud impacts: {both_count} ({both_count/len(predictions_df)*100:.1f}%)")
    
    return predictions_df, predictions_path, template_path




# Update the parse_arguments function in main.py

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Jira User Story Security and Fraud Analysis')
    
    parser.add_argument('--extract', action='store_true', 
                        help='Extract user stories from Jira')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze existing data without extraction')
    parser.add_argument('--train', action='store_true',
                        help='Train or update the ML model')
    parser.add_argument('--predict', action='store_true',
                        help='Make predictions on existing data')
    parser.add_argument('--projects', nargs='+', type=str,
                        help='List of Jira project keys to extract (comma-separated or space-separated)')
    parser.add_argument('--project-file', type=str,
                        help='Path to file containing list of Jira project keys (one per line)')
    parser.add_argument('--input', type=str,
                        help='Input file path (for analyze, train, or predict modes)')
    parser.add_argument('--output', type=str,
                        help='Output file path (optional)')
    parser.add_argument('--feedback', type=str,
                        help='Path to feedback file for model updating')
    parser.add_argument('--nltk-data', type=str, default='C:/nltk_data',
                        help='Path to NLTK data directory (default: C:/nltk_data)')
    parser.add_argument('--exclude-columns', nargs='+', type=str,
                        help='Columns to exclude from Excel output (space-separated)')
    parser.add_argument('--hide-columns', nargs='+', type=str,
                        help='Columns to hide in Excel output (space-separated)')
    parser.add_argument('--excel-config', type=str,
                        help='Path to Excel configuration file (JSON format)')
    
    return parser.parse_args()




# excel_config.json - Example configuration for Excel output

{
  "exclude_columns": [
    "combined_text", 
    "original_security_matches", 
    "original_fraud_matches", 
    "has_security_keywords", 
    "has_fraud_keywords", 
    "initial_security_flag", 
    "initial_fraud_flag"
  ],
  "hidden_columns": [
    "security_keyword_count", 
    "fraud_keyword_count"
  ],
  "column_widths": {
    "summary": 50,
    "description": 70,
    "security_matches": 35,
    "fraud_matches": 35,
    "prediction_explanation": 65
  }
}


# Update the main function in main.py to handle the new column filtering options

def main():
    """Main function to orchestrate the workflow."""
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    setup_directories()
    
    # Parse command line arguments
    args = parse_arguments()
    
    if not any([args.extract, args.analyze, args.train, args.predict]):
        logger.info("No action specified. Running complete workflow.")
        args.extract = args.analyze = args.train = args.predict = True
    
    # Initialize column filtering options
    exclude_columns = args.exclude_columns or []
    hidden_columns = args.hide_columns or []
    
    # Load Excel configuration if provided
    if args.excel_config and os.path.exists(args.excel_config):
        try:
            import json
            with open(args.excel_config, 'r') as f:
                excel_config = json.load(f)
                
            # Update exclude and hidden columns from config
            if 'exclude_columns' in excel_config:
                exclude_columns.extend([col for col in excel_config['exclude_columns'] 
                                       if col not in exclude_columns])
            
            if 'hidden_columns' in excel_config:
                hidden_columns.extend([col for col in excel_config['hidden_columns'] 
                                     if col not in hidden_columns])
                
            logger.info(f"Loaded Excel configuration from {args.excel_config}")
        except Exception as e:
            logger.error(f"Failed to load Excel configuration: {e}")
    
    # Extract data from Jira
    if args.extract:
        data_path = extract_jira_data(args.projects, args.project_file, args.output)
        if not data_path:
            logger.error("Data extraction failed. Exiting.")
            return
    else:
        data_path = args.input or EXCEL_EXPORT_PATH
    
    # Preprocess data
    if args.analyze or (not args.input and args.predict):
        processed_path, processed_data = preprocess_data(data_path)
        if not processed_path:
            logger.error("Data preprocessing failed. Exiting.")
            return
    else:
        processed_path = args.input or PROCESSED_EXCEL_PATH
        processed_data = None
    
    # Train or update the model
    if args.train:
        model, trainer = train_model(processed_path, args.nltk_data, args.feedback)
        if not model:
            logger.error("Model training failed. Exiting.")
            return
    else:
        # Initialize trainer with NLTK data path
        trainer = ModelTrainer()
        if args.nltk_data:
            trainer.feature_engineer = FeatureEngineer(args.nltk_data)
    
    # Make predictions and analyze
    if args.predict or args.analyze:
        input_data = processed_data if processed_data is not None else processed_path
        predictions_df, predictions_path, template_path = predict_and_analyze(
            trainer, 
            input_data, 
            args.nltk_data,
            exclude_columns=exclude_columns,
            hidden_columns=hidden_columns
        )
        
        logger.info("Analysis complete!")
        logger.info(f"Results available at {predictions_path}")
        logger.info(f"Feedback template available at {template_path}")
    
    logger.info("All operations completed successfully")



{
  "exclude_columns": [
    "combined_text", 
    "original_security_matches", 
    "has_security_keywords"
  ],
  "hidden_columns": [
    "security_keyword_count", 
    "fraud_keyword_count"
  ],
  "column_widths": {
    "summary": 50,
    "description": 70
  }
}
