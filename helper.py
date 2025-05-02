# src/utils/pivot_analyzer.py - Enhanced with multiple pivot tables

import pandas as pd
import numpy as np
import logging
import io
import os
import re
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.chart import BarChart, Reference
from openpyxl.utils import get_column_letter
from config.settings import SECURITY_KEYWORDS, FRAUD_KEYWORDS

logger = logging.getLogger(__name__)

class PivotAnalyzer:
    """Class to generate pivot tables and charts for security and fraud analysis."""
    
    def __init__(self, key_programs=None, key_security_keywords=None, key_fraud_keywords=None):
        """
        Initialize the pivot analyzer with optional lists of key items to track.
        
        Args:
            key_programs (list): List of program keys to specifically track
            key_security_keywords (list): List of security keywords to specifically track
            key_fraud_keywords (list): List of fraud keywords to specifically track
        """
        self.key_programs = key_programs or []
        self.key_security_keywords = key_security_keywords or []
        self.key_fraud_keywords = key_fraud_keywords or []
    
    def generate_program_impact_pivot(self, predictions_df):
        """
        Generate a pivot table showing test program impacts on security and fraud.
        
        Args:
            predictions_df (pandas.DataFrame): DataFrame with predictions
            
        Returns:
            pandas.DataFrame: Pivot table DataFrame
        """
        try:
            # Check if predictions_df is valid
            if predictions_df is None or len(predictions_df) == 0:
                logger.warning("No data available for pivot table analysis")
                return None
            
            # Extract program name from issue_key or project_key if available
            if 'issue_key' in predictions_df.columns:
                # Try to extract program name from issue key (assuming format PROJECT-123)
                predictions_df['program'] = predictions_df['issue_key'].str.split('-').str[0]
            elif 'project_key' in predictions_df.columns:
                # Use project_key as program
                predictions_df['program'] = predictions_df['project_key']
            else:
                logger.warning("No program/project identifier found in data")
                return None
            
            # Filter to key programs if specified
            if self.key_programs:
                # Make sure all specified programs are included even if they have no data
                all_programs = set(predictions_df['program'].unique())
                missing_programs = set(self.key_programs) - all_programs
                
                # Keep only specified programs in the data
                pivot_df = predictions_df[predictions_df['program'].isin(self.key_programs)].copy()
                
                # If some specified programs are missing, log a warning
                if missing_programs:
                    logger.warning(f"Some specified key programs were not found in the data: {missing_programs}")
            else:
                # Use all programs
                pivot_df = predictions_df.copy()
            
            # Create a copy of the dataframe with only the columns we need
            if not pivot_df.empty:
                pivot_df = pivot_df[['program', 'security_prediction', 'fraud_prediction', 
                                     'security_probability', 'fraud_probability']].copy()
                
                # Calculate both impacts count separately
                pivot_df['both_impacts'] = ((pivot_df['security_prediction'] == 1) & 
                                           (pivot_df['fraud_prediction'] == 1)).astype(int)
                
                # Create pure security and fraud counts (excluding overlaps)
                pivot_df['security_only'] = ((pivot_df['security_prediction'] == 1) & 
                                            (pivot_df['fraud_prediction'] == 0)).astype(int)
                pivot_df['fraud_only'] = ((pivot_df['security_prediction'] == 0) & 
                                         (pivot_df['fraud_prediction'] == 1)).astype(int)
                
                # Group by program and count security and fraud impacts
                pivot_table = pd.pivot_table(
                    pivot_df,
                    index='program',
                    values=['security_only', 'fraud_only', 'both_impacts', 
                            'security_prediction', 'fraud_prediction'],
                    aggfunc='sum'
                )
                
                # Calculate total impacts correctly (avoiding double-counting)
                pivot_table['total_impacts'] = pivot_table['security_only'] + pivot_table['fraud_only'] + pivot_table['both_impacts']
                
                # Add average probability columns
                security_prob_pivot = pd.pivot_table(
                    pivot_df[pivot_df['security_prediction'] == 1],
                    index='program',
                    values='security_probability',
                    aggfunc='mean'
                )
                
                fraud_prob_pivot = pd.pivot_table(
                    pivot_df[pivot_df['fraud_prediction'] == 1],
                    index='program',
                    values='fraud_probability',
                    aggfunc='mean'
                )
                
                # Add these to the main pivot table
                pivot_table = pivot_table.join(security_prob_pivot, how='left')
                pivot_table = pivot_table.join(fraud_prob_pivot, how='left')
                
                # Rename columns for clarity and rearrange
                final_columns = [
                    'security_prediction', 
                    'fraud_prediction', 
                    'security_only',
                    'fraud_only',
                    'both_impacts',
                    'total_impacts',
                    'security_probability',
                    'fraud_probability'
                ]
                
                # Make sure all columns exist
                for col in final_columns:
                    if col not in pivot_table.columns:
                        pivot_table[col] = 0
                
                # Select and rename columns
                pivot_table = pivot_table[final_columns]
                
                pivot_table.columns = [
                    'Total Security Impacts', 
                    'Total Fraud Impacts',
                    'Security Only',
                    'Fraud Only',
                    'Both Impacts',
                    'Total Impacts',
                    'Avg Security Probability',
                    'Avg Fraud Probability'
                ]
                
                # Sort by total impacts descending
                pivot_table = pivot_table.sort_values('Total Impacts', ascending=False)
                
                # Calculate percentages
                total_security = pivot_table['Total Security Impacts'].sum()
                total_fraud = pivot_table['Total Fraud Impacts'].sum()
                
                if total_security > 0:
                    pivot_table['Security %'] = (pivot_table['Total Security Impacts'] / total_security * 100).round(1)
                else:
                    pivot_table['Security %'] = 0
                    
                if total_fraud > 0:
                    pivot_table['Fraud %'] = (pivot_table['Total Fraud Impacts'] / total_fraud * 100).round(1)
                else:
                    pivot_table['Fraud %'] = 0
                
                # Fill NaN values
                pivot_table = pivot_table.fillna(0)
                
                # Round probability columns
                for col in ['Avg Security Probability', 'Avg Fraud Probability']:
                    if col in pivot_table.columns:
                        pivot_table[col] = pivot_table[col].round(2)
                
                return pivot_table
            else:
                logger.warning("No data available for the specified programs")
                return None
            
        except Exception as e:
            logger.error(f"Error generating program impact pivot table: {e}")
            return None
    
    def generate_security_keyword_pivot(self, predictions_df):
        """
        Generate a pivot table showing security keyword impacts across programs.
        
        Args:
            predictions_df (pandas.DataFrame): DataFrame with predictions
            
        Returns:
            pandas.DataFrame: Pivot table DataFrame with security keyword counts
        """
        try:
            # Check if predictions_df is valid
            if predictions_df is None or len(predictions_df) == 0:
                logger.warning("No data available for security keyword analysis")
                return None
            
            # Make sure we have the required columns
            if 'security_matching_keywords' not in predictions_df.columns:
                logger.warning("No security_matching_keywords column found in the data")
                return None
            
            # Extract program name if not already present
            if 'program' not in predictions_df.columns:
                if 'issue_key' in predictions_df.columns:
                    predictions_df['program'] = predictions_df['issue_key'].str.split('-').str[0]
                elif 'project_key' in predictions_df.columns:
                    predictions_df['program'] = predictions_df['project_key']
                else:
                    logger.warning("No program/project identifier found in data")
                    return None
            
            # Prepare data for keyword analysis
            keyword_data = []
            
            # Use only security-flagged stories
            security_stories = predictions_df[predictions_df['security_prediction'] == 1].copy()
            
            # Get all unique security keywords from the data
            all_keywords = set()
            for keywords_str in security_stories['security_matching_keywords'].dropna():
                keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
                all_keywords.update(keywords)
            
            # Filter to key security keywords if specified, otherwise use the top ones from the data
            if self.key_security_keywords:
                target_keywords = [k for k in self.key_security_keywords if k in all_keywords]
                if len(target_keywords) < len(self.key_security_keywords):
                    missing = set(self.key_security_keywords) - all_keywords
                    logger.warning(f"Some specified security keywords were not found: {missing}")
            else:
                # Get the top 20 most common keywords from the data
                keyword_counts = {}
                for keywords_str in security_stories['security_matching_keywords'].dropna():
                    keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
                    for keyword in keywords:
                        if keyword in keyword_counts:
                            keyword_counts[keyword] += 1
                        else:
                            keyword_counts[keyword] = 1
                
                # Sort by count and get top 20
                target_keywords = [k for k, v in sorted(keyword_counts.items(), 
                                                      key=lambda item: item[1], 
                                                      reverse=True)[:20]]
            
            # Check if we have any target keywords
            if not target_keywords:
                logger.warning("No security keywords to analyze")
                return None
            
            # Prepare data for pivot table
            for _, story in security_stories.iterrows():
                program = story['program']
                keywords_str = story['security_matching_keywords']
                
                if pd.isna(keywords_str) or not keywords_str.strip():
                    continue
                
                keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
                
                # Count each target keyword for this story
                for keyword in target_keywords:
                    if keyword in keywords:
                        keyword_data.append({
                            'program': program,
                            'keyword': keyword,
                            'count': 1
                        })
            
            # Create DataFrame from the keyword data
            if not keyword_data:
                logger.warning("No security keyword data to analyze")
                return None
                
            keyword_df = pd.DataFrame(keyword_data)
            
            # Create pivot table
            keyword_pivot = pd.pivot_table(
                keyword_df,
                index='keyword',
                columns='program',
                values='count',
                aggfunc='sum',
                fill_value=0
            )
            
            # Filter to key programs if specified
            if self.key_programs:
                # Keep only specified programs
                program_cols = [col for col in keyword_pivot.columns if col in self.key_programs]
                if not program_cols:
                    logger.warning("None of the specified key programs have security keywords")
                    # Add the programs as columns with zeros
                    for program in self.key_programs:
                        keyword_pivot[program] = 0
                    program_cols = self.key_programs
                
                keyword_pivot = keyword_pivot[program_cols]
            
            # Add a total column
            keyword_pivot['Total'] = keyword_pivot.sum(axis=1)
            
            # Sort by total count descending
            keyword_pivot = keyword_pivot.sort_values('Total', ascending=False)
            
            return keyword_pivot
            
        except Exception as e:
            logger.error(f"Error generating security keyword pivot table: {e}")
            return None
    
    def generate_fraud_keyword_pivot(self, predictions_df):
        """
        Generate a pivot table showing fraud keyword impacts across programs.
        
        Args:
            predictions_df (pandas.DataFrame): DataFrame with predictions
            
        Returns:
            pandas.DataFrame: Pivot table DataFrame with fraud keyword counts
        """
        try:
            # Check if predictions_df is valid
            if predictions_df is None or len(predictions_df) == 0:
                logger.warning("No data available for fraud keyword analysis")
                return None
            
            # Make sure we have the required columns
            if 'fraud_matching_keywords' not in predictions_df.columns:
                logger.warning("No fraud_matching_keywords column found in the data")
                return None
            
            # Extract program name if not already present
            if 'program' not in predictions_df.columns:
                if 'issue_key' in predictions_df.columns:
                    predictions_df['program'] = predictions_df['issue_key'].str.split('-').str[0]
                elif 'project_key' in predictions_df.columns:
                    predictions_df['program'] = predictions_df['project_key']
                else:
                    logger.warning("No program/project identifier found in data")
                    return None
            
            # Prepare data for keyword analysis
            keyword_data = []
            
            # Use only fraud-flagged stories
            fraud_stories = predictions_df[predictions_df['fraud_prediction'] == 1].copy()
            
            # Get all unique fraud keywords from the data
            all_keywords = set()
            for keywords_str in fraud_stories['fraud_matching_keywords'].dropna():
                keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
                all_keywords.update(keywords)
            
            # Filter to key fraud keywords if specified, otherwise use the top ones from the data
            if self.key_fraud_keywords:
                target_keywords = [k for k in self.key_fraud_keywords if k in all_keywords]
                if len(target_keywords) < len(self.key_fraud_keywords):
                    missing = set(self.key_fraud_keywords) - all_keywords
                    logger.warning(f"Some specified fraud keywords were not found: {missing}")
            else:
                # Get the top 20 most common keywords from the data
                keyword_counts = {}
                for keywords_str in fraud_stories['fraud_matching_keywords'].dropna():
                    keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
                    for keyword in keywords:
                        if keyword in keyword_counts:
                            keyword_counts[keyword] += 1
                        else:
                            keyword_counts[keyword] = 1
                
                # Sort by count and get top 20
                target_keywords = [k for k, v in sorted(keyword_counts.items(), 
                                                      key=lambda item: item[1], 
                                                      reverse=True)[:20]]
            
            # Check if we have any target keywords
            if not target_keywords:
                logger.warning("No fraud keywords to analyze")
                return None
            
            # Prepare data for pivot table
            for _, story in fraud_stories.iterrows():
                program = story['program']
                keywords_str = story['fraud_matching_keywords']
                
                if pd.isna(keywords_str) or not keywords_str.strip():
                    continue
                
                keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
                
                # Count each target keyword for this story
                for keyword in target_keywords:
                    if keyword in keywords:
                        keyword_data.append({
                            'program': program,
                            'keyword': keyword,
                            'count': 1
                        })
            
            # Create DataFrame from the keyword data
            if not keyword_data:
                logger.warning("No fraud keyword data to analyze")
                return None
                
            keyword_df = pd.DataFrame(keyword_data)
            
            # Create pivot table
            keyword_pivot = pd.pivot_table(
                keyword_df,
                index='keyword',
                columns='program',
                values='count',
                aggfunc='sum',
                fill_value=0
            )
            
            # Filter to key programs if specified
            if self.key_programs:
                # Keep only specified programs
                program_cols = [col for col in keyword_pivot.columns if col in self.key_programs]
                if not program_cols:
                    logger.warning("None of the specified key programs have fraud keywords")
                    # Add the programs as columns with zeros
                    for program in self.key_programs:
                        keyword_pivot[program] = 0
                    program_cols = self.key_programs
                
                keyword_pivot = keyword_pivot[program_cols]
            
            # Add a total column
            keyword_pivot['Total'] = keyword_pivot.sum(axis=1)
            
            # Sort by total count descending
            keyword_pivot = keyword_pivot.sort_values('Total', ascending=False)
            
            return keyword_pivot
            
        except Exception as e:
            logger.error(f"Error generating fraud keyword pivot table: {e}")
            return None
    
    def create_pivot_excel(self, program_pivot=None, security_keyword_pivot=None, 
                          fraud_keyword_pivot=None, output_path=None):
        """
        Create a formatted Excel file with pivot tables and charts.
        
        Args:
            program_pivot (pandas.DataFrame): Program impact pivot table
            security_keyword_pivot (pandas.DataFrame): Security keyword pivot table
            fraud_keyword_pivot (pandas.DataFrame): Fraud keyword pivot table
            output_path (str): Path to save the Excel file
            
        Returns:
            str: Path to the created Excel file or None if failed
        """
        try:
            # Check if we have any pivot tables
            if program_pivot is None and security_keyword_pivot is None and fraud_keyword_pivot is None:
                logger.warning("No pivot table data available to export")
                return None
            
            # Create Excel writer
            if output_path is None:
                output_path = 'data/processed/program_impact_pivot.xlsx'
                
            writer = pd.ExcelWriter(output_path, engine='openpyxl')
            
            # Create the workbook
            workbook = writer.book
            
            # Write program impact pivot table if available
            if program_pivot is not None and not program_pivot.empty:
                # Write to Excel
                program_pivot.to_excel(writer, sheet_name='Program Impacts')
                
                # Get the worksheet
                worksheet = writer.sheets['Program Impacts']
                
                # Format the worksheet
                self._format_pivot_worksheet(worksheet, program_pivot)
                
                # Add a bar chart for security and fraud impacts
                chart_sheet = workbook.create_sheet("Impact Charts")
                
                # Create the chart (Security and Fraud Impacts)
                impact_chart = BarChart()
                impact_chart.title = "Security and Fraud Impacts by Program"
                impact_chart.style = 10  # Use a nice style
                impact_chart.y_axis.title = "Number of Impacts"
                impact_chart.x_axis.title = "Program"
                
                # Define data ranges for the chart (first 10 programs only if more exist)
                num_programs = min(10, len(program_pivot))
                data = Reference(worksheet, min_col=2, max_col=3, min_row=1, max_row=num_programs+1)
                cats = Reference(worksheet, min_col=1, min_row=2, max_row=num_programs+1)
                
                # Add data and categories
                impact_chart.add_data(data, titles_from_data=True)
                impact_chart.set_categories(cats)
                
                # Add the chart to the chart sheet
                chart_sheet.add_chart(impact_chart, "A1")
                
                # Adjust chart sheet layout
                for col in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
                    chart_sheet.column_dimensions[col].width = 12
                    
                # Make the chart large enough
                impact_chart.height = 15
                impact_chart.width = 20
            
            # Write security keyword pivot table if available
            if security_keyword_pivot is not None and not security_keyword_pivot.empty:
                # Write to Excel
                security_keyword_pivot.to_excel(writer, sheet_name='Security Keywords')
                
                # Get the worksheet
                worksheet = writer.sheets['Security Keywords']
                
                # Format the worksheet
                self._format_pivot_worksheet(worksheet, security_keyword_pivot)
                
                # Adjust column formatting for better readability
                for col_idx in range(1, len(security_keyword_pivot.columns) + 2):  # +2 for index and total
                    # Set column width
                    col_letter = get_column_letter(col_idx)
                    worksheet.column_dimensions[col_letter].width = 15
                
                # Add totals row at the bottom
                totals_row = len(security_keyword_pivot) + 2
                worksheet.cell(row=totals_row, column=1).value = "TOTAL"
                worksheet.cell(row=totals_row, column=1).font = Font(bold=True)
                
                # Add sum formulas for each program column
                for col_idx in range(2, len(security_keyword_pivot.columns) + 2):
                    col_letter = get_column_letter(col_idx)
                    worksheet.cell(row=totals_row, column=col_idx).value = f"=SUM({col_letter}2:{col_letter}{len(security_keyword_pivot)+1})"
                    worksheet.cell(row=totals_row, column=col_idx).font = Font(bold=True)
            
            # Write fraud keyword pivot table if available
            if fraud_keyword_pivot is not None and not fraud_keyword_pivot.empty:
                # Write to Excel
                fraud_keyword_pivot.to_excel(writer, sheet_name='Fraud Keywords')
                
                # Get the worksheet
                worksheet = writer.sheets['Fraud Keywords']
                
                # Format the worksheet
                self._format_pivot_worksheet(worksheet, fraud_keyword_pivot)
                
                # Adjust column formatting for better readability
                for col_idx in range(1, len(fraud_keyword_pivot.columns) + 2):  # +2 for index and total
                    # Set column width
                    col_letter = get_column_letter(col_idx)
                    worksheet.column_dimensions[col_letter].width = 15
                
                # Add totals row at the bottom
                totals_row = len(fraud_keyword_pivot) + 2
                worksheet.cell(row=totals_row, column=1).value = "TOTAL"
                worksheet.cell(row=totals_row, column=1).font = Font(bold=True)
                
                # Add sum formulas for each program column
                for col_idx in range(2, len(fraud_keyword_pivot.columns) + 2):
                    col_letter = get_column_letter(col_idx)
                    worksheet.cell(row=totals_row, column=col_idx).value = f"=SUM({col_letter}2:{col_letter}{len(fraud_keyword_pivot)+1})"
                    worksheet.cell(row=totals_row, column=col_idx).font = Font(bold=True)
            
            # Add a summary sheet
            summary_sheet = workbook.create_sheet("Summary", 0)  # Make it the first sheet
            
            # Add summary data
            summary_data = [
                ["Program and Keyword Impact Analysis"],
                [""],
                ["This workbook contains pivot tables analyzing security and fraud impacts."],
                [""],
                ["Sheets:"],
                ["- Program Impacts: Count of security and fraud impacts by program"],
                ["- Security Keywords: Frequency of security keywords by program"],
                ["- Fraud Keywords: Frequency of fraud keywords by program"],
                ["- Impact Charts: Visualizations of the impact data"],
                [""],
                ["Analysis Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            ]
            
            from datetime import datetime
            
            for i, row_data in enumerate(summary_data, 1):
                for j, value in enumerate(row_data, 1):
                    cell = summary_sheet.cell(row=i, column=j, value=value)
                    if i == 1:  # Title
                        cell.font = Font(size=14, bold=True)
                        summary_sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=3)
                        cell.alignment = Alignment(horizontal='center')
                    
                    if j == 1 and i > 2 and i < len(summary_data):  # Left column headers
                        cell.font = Font(bold=True)
            
            # Adjust column widths in summary sheet
            summary_sheet.column_dimensions['A'].width = 40
            summary_sheet.column_dimensions['B'].width = 40
            
            # Save the workbook
            writer.close()
            
            logger.info(f"Pivot tables and charts saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating pivot Excel file: {e}")
            return None
    
    def _format_pivot_worksheet(self, worksheet, pivot_table):
        """Helper method to format a pivot table worksheet."""
        # Format header row
        for col_idx in range(1, len(pivot_table.columns) + 2):  # +2 for index column
            cell = worksheet.cell(row=1, column=col_idx)
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="DDDDDD", end_color="DDDDDD", fill_type="solid")
            cell.alignment = Alignment(horizontal='center')
        
        # Format index column (row headers)
        for row_idx in range(2, len(pivot_table) + 2):
            cell = worksheet.cell(row=row_idx, column=1)
            cell.font = Font(bold=True)
        
        # Format numeric columns
        for col_idx in range(2, len(pivot_table.columns) + 2):
            # Set column width
            worksheet.column_dimensions[get_column_letter(col_idx)].width = 18
            
            # Apply number formatting
            for row_idx in range(2, len(pivot_table) + 2):
                cell = worksheet.cell(row=row_idx, column=col_idx)
                cell.number_format = '0'
        
        # Add borders to all cells
        thin_border = Border(
            left=Side(style='thin'), 
            right=Side(style='thin'), 
            top=Side(style='thin'), 
            bottom=Side(style='thin')
        )
        
        for row in worksheet.iter_rows(min_row=1, max_row=len(pivot_table) + 1, min_col=1, max_col=len(pivot_table.columns) + 1):
            for cell in row:
                cell.border = thin_border
        
        # Apply auto-filter
        worksheet.auto_filter.ref = f"A1:{get_column_letter(len(pivot_table.columns) + 1)}{len(pivot_table) + 1}"
