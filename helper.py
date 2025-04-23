# src/utils/helpers.py - Add this text cleaning function

def clean_text(text):
    """
    Clean and normalize text data from Jira.
    
    Args:
        text (str): Raw text input from Jira
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Replace HTML tags and entities
    import re
    
    # Handle HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Handle common HTML entities
    html_entities = {
        '&nbsp;': ' ', '&amp;': '&', '&lt;': '<', '&gt;': '>', 
        '&quot;': '"', '&#39;': "'", '&apos;': "'", '&#x2F;': '/',
        '&#x27;': "'", '&#x2f;': '/'
    }
    for entity, replacement in html_entities.items():
        text = text.replace(entity, replacement)
    
    # Handle Jira markup (basic)
    jira_markup = {
        '{code}': '', '{code:.*?}': '', '{noformat}': '', '{quote}': '"', 
        '{color:.*?}': '', '{color}': '', '----': '-', '----': '-', 
        '----': '-', '\\\\': '\n', '{panel}': '', '{panel:.*?}': '', 
        '{panel}': '', '{table}': '', '{table:.*?}': '', '{table}': '',
        '{column}': '', '{column:.*?}': '', '{column}': '',
        '{section}': '', '{section:.*?}': '', '{section}': ''
    }
    for markup, replacement in jira_markup.items():
        text = re.sub(markup, replacement, text)
    
    # Remove markdown formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
    text = re.sub(r'__(.*?)__', r'\1', text)      # Underline
    text = re.sub(r'~~(.*?)~~', r'\1', text)      # Strikethrough
    text = re.sub(r'`(.*?)`', r'\1', text)        # Code
    
    # Remove special characters and control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Replace multiple spaces, tabs, and newlines with single space
    text = re.sub(r'[\s\t\n\r]+', ' ', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '[URL]', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text



# Add to src/data_processing/jira_connector.py

from src.utils.helpers import clean_text

# Add a text cleaning method to the JiraConnector class
def clean_extracted_data(self, df):
    """
    Clean text data in the DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame with extracted Jira data
        
    Returns:
        pandas.DataFrame: DataFrame with cleaned text
    """
    if df.empty:
        return df
    
    logger.info("Cleaning extracted text data")
    
    # Clean text fields
    text_columns = ['summary', 'description', 'labels', 'components']
    for column in text_columns:
        if column in df.columns:
            logger.info(f"Cleaning text in column: {column}")
            df[column] = df[column].apply(clean_text)
    
    logger.info("Text cleaning completed")
    return df

# Update the export_to_excel method to include text cleaning and formatting
def export_to_excel(self, data, export_path=None):
    """
    Export user stories data to Excel with text cleaning and formatting.
    
    Args:
        data (pandas.DataFrame): DataFrame containing user stories
        export_path (str): Path to export the Excel file
    
    Returns:
        str: Path to the exported Excel file
    """
    if data.empty:
        logger.warning("No data to export.")
        return None
        
    if export_path is None:
        os.makedirs(RAW_DATA_PATH, exist_ok=True)
        export_path = os.path.join(RAW_DATA_PATH, "jira_user_stories.xlsx")
    
    try:
        # Clean the text data
        data = self.clean_extracted_data(data)
        
        from src.utils.helpers import export_predictions_to_excel
        
        # Use the enhanced export function with formatting
        export_path = export_predictions_to_excel(data, export_path)
        
        logger.info(f"Successfully exported {len(data)} records to {export_path}")
        return export_path
    except Exception as e:
        logger.error(f"Failed to export data: {e}")
        return None



# Update src/data_processing/excel_processor.py

from src.utils.helpers import clean_text

# Update the preprocess_data method in the ExcelProcessor class
def preprocess_data(self):
    """Preprocess the data for ML analysis."""
    if self.df is None:
        if not self.load_data():
            return False
    
    try:
        # Clean text data first
        logger.info("Cleaning text data")
        text_columns = ['summary', 'description', 'labels', 'components']
        for column in text_columns:
            if column in self.df.columns:
                self.df[column] = self.df[column].apply(clean_text)
        
        # Fill NaN values
        self.df.fillna("", inplace=True)
        
        # Combine text fields for analysis
        self.df['combined_text'] = self.df['summary'] + " " + self.df['description']
        
        # Clean text data
        self.df['combined_text'] = self.df['combined_text'].str.lower()
        
        # Add initial keyword-based features
        self._add_keyword_features()
        
        # Add date features
        self.df['created_date'] = pd.to_datetime(self.df['created_date'])
        self.df['created_year'] = self.df['created_date'].dt.year
        self.df['created_month'] = self.df['created_date'].dt.month
        self.df['created_day'] = self.df['created_date'].dt.day
        
        logger.info("Data preprocessing completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return False

# Update the save_processed_data method to use the enhanced Excel export
def save_processed_data(self, output_path=None):
    """Save the preprocessed data to an Excel file with formatting."""
    if self.df is None:
        logger.error("No data to save.")
        return None
    
    if output_path is None:
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        output_path = os.path.join(PROCESSED_DATA_PATH, "processed_user_stories.xlsx")
    
    try:
        from src.utils.helpers import export_predictions_to_excel
        
        # Use the enhanced export function with formatting
        output_path = export_predictions_to_excel(self.df, output_path)
        
        logger.info(f"Preprocessed data saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to save preprocessed data: {e}")
        return None



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
