# src/utils/helpers.py - Updated export_predictions_to_excel function

def export_predictions_to_excel(df, output_path):
    """
    Export predictions to Excel with improved formatting.
    
    Args:
        df (pandas.DataFrame): DataFrame with predictions
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
        
        # Add an overall impact column
        df['impact_label'] = 'No Impact'
        # If either security or fraud impact exists, mark as 'Security/Fraud Impact'
        df.loc[(df['security_prediction'] == 1) | (df['fraud_prediction'] == 1), 'impact_label'] = 'Security/Fraud Impact'
        # More specific labels
        df.loc[(df['security_prediction'] == 1) & (df['fraud_prediction'] == 0), 'impact_label'] = 'Security Impact'
        df.loc[(df['security_prediction'] == 0) & (df['fraud_prediction'] == 1), 'impact_label'] = 'Fraud Impact'
        df.loc[(df['security_prediction'] == 1) & (df['fraud_prediction'] == 1), 'impact_label'] = 'Security & Fraud Impact'
        
        # Create a Pandas Excel writer using XlsxWriter
        writer = pd.ExcelWriter(output_path, engine='openpyxl')
        
        # Convert DataFrame to Excel
        df.to_excel(writer, sheet_name='Predictions', index=False)
        
        # Get the workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Predictions']
        
        # Get column indices for formatting
        impact_col_idx = df.columns.get_loc('impact_label') + 1  # +1 for Excel 1-based indexing
        security_pred_idx = df.columns.get_loc('security_prediction') + 1
        security_prob_idx = df.columns.get_loc('security_probability') + 1
        fraud_pred_idx = df.columns.get_loc('fraud_prediction') + 1
        fraud_prob_idx = df.columns.get_loc('fraud_probability') + 1
        
        # Find text column indices
        summary_idx = df.columns.get_loc('summary') + 1 if 'summary' in df.columns else None
        desc_idx = df.columns.get_loc('description') + 1 if 'description' in df.columns else None
        
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
        
        # Apply formatting to all cells
        for row_idx in range(2, len(df) + 2):  # +2 for header and 1-based indexing
            # Apply text wrapping to summary and description
            if summary_idx:
                cell = worksheet.cell(row=row_idx, column=summary_idx)
                cell.alignment = Alignment(wrap_text=True, vertical='top')
            
            if desc_idx:
                cell = worksheet.cell(row=row_idx, column=desc_idx)
                cell.alignment = Alignment(wrap_text=True, vertical='top')
            
            # Format the impact label cell
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
            
            # Format probability cells with gradient coloring based on value
            sec_prob_cell = worksheet.cell(row=row_idx, column=security_prob_idx)
            if isinstance(sec_prob_cell.value, (int, float)) and 0 <= sec_prob_cell.value <= 1:
                intensity = int(255 - (sec_prob_cell.value * 255))
                color = f"{intensity:02X}{255:02X}{intensity:02X}"  # Green gradient
                sec_prob_cell.fill = PatternFill(start_color=color, end_color=color, fill_type="solid")
            
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
        
        # Create a second sheet for a summary
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
        logger.info(f"Predictions exported to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to export predictions: {e}")
        return None




from datetime import datetime
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation





# src/utils/helpers.py - Updated generate_feedback_template function

def generate_feedback_template(predictions_df, output_path):
    """
    Generate a feedback template for manual review with improved formatting.
    
    Args:
        predictions_df (pandas.DataFrame): DataFrame with predictions
        output_path (str): Path to save the template
        
    Returns:
        str: Path to the feedback template
    """
    try:
        # Create a copy of the DataFrame with only the necessary columns
        template_df = predictions_df[['project_key', 'issue_key', 'summary', 
                                     'security_prediction', 'security_probability',
                                     'fraud_prediction', 'fraud_probability']].copy()
        
        # Add overall impact label
        template_df['impact_label'] = 'No Impact'
        # If either security or fraud impact exists, mark as 'Security/Fraud Impact'
        template_df.loc[(template_df['security_prediction'] == 1) | (template_df['fraud_prediction'] == 1), 'impact_label'] = 'Security/Fraud Impact'
        # More specific labels
        template_df.loc[(template_df['security_prediction'] == 1) & (template_df['fraud_prediction'] == 0), 'impact_label'] = 'Security Impact'
        template_df.loc[(template_df['security_prediction'] == 0) & (template_df['fraud_prediction'] == 1), 'impact_label'] = 'Fraud Impact'
        template_df.loc[(template_df['security_prediction'] == 1) & (template_df['fraud_prediction'] == 1), 'impact_label'] = 'Security & Fraud Impact'
        
        # Add columns for manual verification
        template_df['security_verified'] = ""
        template_df['fraud_verified'] = ""
        template_df['notes'] = ""
        
        # Focus on issues with higher probability or conflicting predictions
        template_df['needs_review'] = (
            (template_df['security_probability'] > 0.3) | 
            (template_df['fraud_probability'] > 0.3) |
            # Check for cases where prediction and probability disagree
            ((template_df['security_prediction'] == 1) & (template_df['security_probability'] < 0.3)) |
            ((template_df['security_prediction'] == 0) & (template_df['security_probability'] > 0.3)) |
            ((template_df['fraud_prediction'] == 1) & (template_df['fraud_probability'] < 0.3)) |
            ((template_df['fraud_prediction'] == 0) & (template_df['fraud_probability'] > 0.3))
        )
        
        # Create a Pandas Excel writer using XlsxWriter
        writer = pd.ExcelWriter(output_path, engine='openpyxl')
        
        # Convert DataFrame to Excel
        template_df.to_excel(writer, sheet_name='Feedback', index=False)
        
        # Get the workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Feedback']
        
        # Get column indices for formatting
        impact_col_idx = template_df.columns.get_loc('impact_label') + 1  # +1 for Excel 1-based indexing
        needs_review_idx = template_df.columns.get_loc('needs_review') + 1
        security_verified_idx = template_df.columns.get_loc('security_verified') + 1
        fraud_verified_idx = template_df.columns.get_loc('fraud_verified') + 1
        notes_idx = template_df.columns.get_loc('notes') + 1
        
        # Define column widths
        column_widths = {
            'project_key': 10,
            'issue_key': 15,
            'summary': 40,
            'security_prediction': 15,
            'security_probability': 15,
            'fraud_prediction': 15,
            'fraud_probability': 15,
            'impact_label': 20,
            'security_verified': 15,
            'fraud_verified': 15,
            'notes': 40
        }
        
        # Apply column widths
        for col_name, width in column_widths.items():
            if col_name in template_df.columns:
                col_idx = template_df.columns.get_loc(col_name) + 1
                col_letter = get_column_letter(col_idx)
                worksheet.column_dimensions[col_letter].width = width
        
        # Add data validation for verified columns (dropdown list)
        dv = DataValidation(type="list", formula1='"Yes,No,Maybe"', allow_blank=True)
        worksheet.add_data_validation(dv)
        
        # Apply data validation to the security_verified and fraud_verified columns
        sec_verified_col = get_column_letter(security_verified_idx)
        fraud_verified_col = get_column_letter(fraud_verified_idx)
        
        dv.add(f"{sec_verified_col}2:{sec_verified_col}{len(template_df) + 1}")
        dv.add(f"{fraud_verified_col}2:{fraud_verified_col}{len(template_df) + 1}")
        
        # Apply formatting to all cells
        for row_idx in range(2, len(template_df) + 2):  # +2 for header and 1-based indexing
            # Format the needs_review column
            needs_review_cell = worksheet.cell(row=row_idx, column=needs_review_idx)
            if needs_review_cell.value is True:
                needs_review_cell.fill = PatternFill(start_color="FFCCCC", end_color="FFCCCC", fill_type="solid")  # Light red
                
            # Format the impact label cell
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
            
            # Format the notes column
            notes_cell = worksheet.cell(row=row_idx, column=notes_idx)
            notes_cell.alignment = Alignment(wrap_text=True, vertical='top')
            
            # Apply wrap text to summary
            summary_idx = template_df.columns.get_loc('summary') + 1
            summary_cell = worksheet.cell(row=row_idx, column=summary_idx)
            summary_cell.alignment = Alignment(wrap_text=True, vertical='top')
        
        # Format header row
        for col_idx in range(1, len(template_df.columns) + 1):
            cell = worksheet.cell(row=1, column=col_idx)
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="DDDDDD", end_color="DDDDDD", fill_type="solid")
            cell.alignment = Alignment(horizontal='center')
        
        # Hide the needs_review column (used for filtering only)
        needs_review_col = get_column_letter(needs_review_idx)
        worksheet.column_dimensions[needs_review_col].hidden = True
        
        # Add borders to all cells
        thin_border = Border(
            left=Side(style='thin'), 
            right=Side(style='thin'), 
            top=Side(style='thin'), 
            bottom=Side(style='thin')
        )
        
        for row in worksheet.iter_rows(min_row=1, max_row=len(template_df) + 1, min_col=1, max_col=len(template_df.columns)):
            for cell in row:
                cell.border = thin_border
        
        # Apply auto-filter to header row
        worksheet.auto_filter.ref = f"A1:{get_column_letter(len(template_df.columns))}{len(template_df) + 1}"
        
        # Freeze the header row
        worksheet.freeze_panes = 'A2'
        
        # Add instructions sheet
        instructions_sheet = workbook.create_sheet("Instructions", 0)  # Make it the first sheet
        
        # Add instructions
        instructions = [
            ["Jira Security & Fraud Analysis - Feedback Form"],
            [""],
            ["Instructions:"],
            ["1. Review the items marked for review (highlighted in light red)."],
            ["2. In the 'security_verified' and 'fraud_verified' columns, select Yes, No, or Maybe from the dropdown."],
            ["3. Add any notes or observations in the 'notes' column."],
            ["4. Save this file and provide it back for model retraining."],
            [""],
            ["Color Coding:"],
            ["- Red: Stories with both Security & Fraud Impact"],
            ["- Orange: Stories with Security Impact only"],
            ["- Yellow: Stories with Fraud Impact only"],
            ["- Light Red background: Flagged for review (high probability or conflicting prediction)"],
            [""],
            ["Thank you for your feedback!"]
        ]
        
        for i, row_data in enumerate(instructions, 1):
            for j, value in enumerate(row_data, 1):
                cell = instructions_sheet.cell(row=i, column=j, value=value)
                if i == 1:  # Title
                    cell.font = Font(size=14, bold=True)
                    
                if i == 3:  # Instructions header
                    cell.font = Font(bold=True)
        
        # Adjust column widths in instructions sheet
        instructions_sheet.column_dimensions['A'].width = 80
        
        # Save the workbook
        writer.close()
        
        logger.info(f"Feedback template saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to create feedback template: {e}")
        return None
