# src/utils/pivot_analyzer.py
import pandas as pd
import numpy as np
import logging
import io
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.chart import BarChart, Reference
from openpyxl.utils import get_column_letter
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class PivotAnalyzer:
    """Class to generate pivot tables and charts for security and fraud analysis."""
    
    def __init__(self):
        """Initialize the pivot analyzer."""
        pass
    
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
            
            # Create a copy of the dataframe with only the columns we need
            pivot_df = predictions_df[['program', 'security_prediction', 'fraud_prediction', 'security_probability', 'fraud_probability']].copy()
            
            # Group by program and count security and fraud impacts
            pivot_table = pd.pivot_table(
                pivot_df,
                index='program',
                values=['security_prediction', 'fraud_prediction'],
                aggfunc={
                    'security_prediction': 'sum',
                    'fraud_prediction': 'sum'
                }
            )
            
            # Add a combined impact column
            pivot_table['total_impacts'] = pivot_table['security_prediction'] + pivot_table['fraud_prediction']
            
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
            
            # Rename columns for clarity
            pivot_table.columns = [
                'Security Impacts', 
                'Fraud Impacts', 
                'Total Impacts',
                'Avg Security Probability',
                'Avg Fraud Probability'
            ]
            
            # Sort by total impacts descending
            pivot_table = pivot_table.sort_values('Total Impacts', ascending=False)
            
            # Calculate percentages
            total_security = pivot_table['Security Impacts'].sum()
            total_fraud = pivot_table['Fraud Impacts'].sum()
            
            if total_security > 0:
                pivot_table['Security %'] = (pivot_table['Security Impacts'] / total_security * 100).round(1)
            else:
                pivot_table['Security %'] = 0
                
            if total_fraud > 0:
                pivot_table['Fraud %'] = (pivot_table['Fraud Impacts'] / total_fraud * 100).round(1)
            else:
                pivot_table['Fraud %'] = 0
            
            # Fill NaN values
            pivot_table = pivot_table.fillna(0)
            
            # Round probability columns
            for col in ['Avg Security Probability', 'Avg Fraud Probability']:
                if col in pivot_table.columns:
                    pivot_table[col] = pivot_table[col].round(2)
            
            return pivot_table
        
        except Exception as e:
            logger.error(f"Error generating program impact pivot table: {e}")
            return None
    
    def create_pivot_excel(self, pivot_table, output_path=None):
        """
        Create a formatted Excel file with the pivot table and charts.
        
        Args:
            pivot_table (pandas.DataFrame): Pivot table DataFrame
            output_path (str): Path to save the Excel file
            
        Returns:
            str: Path to the created Excel file or None if failed
        """
        try:
            if pivot_table is None or len(pivot_table) == 0:
                logger.warning("No pivot table data available to export")
                return None
            
            # Create Excel writer
            if output_path is None:
                output_path = 'data/processed/program_impact_pivot.xlsx'
                
            writer = pd.ExcelWriter(output_path, engine='openpyxl')
            
            # Write pivot table to Excel
            pivot_table.to_excel(writer, sheet_name='Program Impacts', index=True)
            
            # Get workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets['Program Impacts']
            
            # Format header row
            for col_idx in range(1, len(pivot_table.columns) + 2):  # +2 for index column
                cell = worksheet.cell(row=1, column=col_idx)
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color="DDDDDD", end_color="DDDDDD", fill_type="solid")
                cell.alignment = Alignment(horizontal='center')
            
            # Format index column (program names)
            for row_idx in range(2, len(pivot_table) + 2):
                cell = worksheet.cell(row=row_idx, column=1)
                cell.font = Font(bold=True)
            
            # Format numeric columns
            for col_idx, col_name in enumerate(pivot_table.columns, start=2):  # Start at 2 because column 1 is the index
                # Set column width
                worksheet.column_dimensions[get_column_letter(col_idx)].width = 18
                
                # Apply number formatting
                for row_idx in range(2, len(pivot_table) + 2):
                    cell = worksheet.cell(row=row_idx, column=col_idx)
                    
                    # Format based on column type
                    if 'Impacts' in col_name:
                        # Integer format
                        cell.number_format = '0'
                    elif 'Probability' in col_name:
                        # Percentage format
                        cell.number_format = '0.00'
                        # Color gradient based on value
                        value = cell.value or 0
                        if value > 0:
                            intensity = int(255 - (value * 155))  # Less intense to keep text readable
                            color = f"{intensity:02X}{intensity:02X}{255:02X}" if 'Fraud' in col_name else f"{intensity:02X}{255:02X}{intensity:02X}"
                            cell.fill = PatternFill(start_color=color, end_color=color, fill_type="solid")
                    elif '%' in col_name:
                        # Percentage format
                        cell.number_format = '0.0"%"'
            
            # Add a bar chart for security and fraud impacts
            chart_sheet = workbook.create_sheet("Impact Charts")
            
            # Create the first chart (Security and Fraud Impacts)
            impact_chart = BarChart()
            impact_chart.title = "Security and Fraud Impacts by Program"
            impact_chart.style = 10  # Use a nice style
            impact_chart.y_axis.title = "Number of Impacts"
            impact_chart.x_axis.title = "Program"
            
            # Define data ranges for the chart (first 10 programs only if more exist)
            num_programs = min(10, len(pivot_table))
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
            
            # Add the total row at the bottom
            totals_row = len(pivot_table) + 2
            worksheet.cell(row=totals_row, column=1).value = "TOTAL"
            worksheet.cell(row=totals_row, column=1).font = Font(bold=True)
            
            # Add sum formulas for the numeric columns
            for col_idx, col_name in enumerate(pivot_table.columns, start=2):
                if 'Impacts' in col_name or '%' in col_name:
                    col_letter = get_column_letter(col_idx)
                    worksheet.cell(row=totals_row, column=col_idx).value = f"=SUM({col_letter}2:{col_letter}{len(pivot_table)+1})"
                    worksheet.cell(row=totals_row, column=col_idx).font = Font(bold=True)
                    worksheet.cell(row=totals_row, column=col_idx).fill = PatternFill(
                        start_color="EEEEEE", 
                        end_color="EEEEEE", 
                        fill_type="solid"
                    )
            
            # Save the workbook
            writer.close()
            
            logger.info(f"Pivot table and charts saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating pivot Excel file: {e}")
            return None
    
    def generate_pivot_chart_image(self, pivot_table):
        """
        Generate a chart image from the pivot table for email reports.
        
        Args:
            pivot_table (pandas.DataFrame): Pivot table DataFrame
            
        Returns:
            bytes: Image data as bytes
        """
        try:
            if pivot_table is None or len(pivot_table) == 0:
                logger.warning("No pivot table data available for chart image")
                return None
            
            # Limit to top 10 programs for readability
            display_pivot = pivot_table.head(10).copy()
            
            # Create a figure
            plt.figure(figsize=(10, 6))
            
            # Create a grouped bar chart
            ax = display_pivot[['Security Impacts', 'Fraud Impacts']].plot(
                kind='bar',
                color=['#4285F4', '#EA4335'],  # Blue and Red
                alpha=0.8,
                rot=45
            )
            
            # Add labels and title
            plt.title('Security and Fraud Impacts by Program', fontsize=14)
            plt.ylabel('Number of Impacts', fontsize=12)
            plt.xlabel('Program', fontsize=12)
            
            # Add data labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%d', padding=3)
            
            # Add a legend
            plt.legend(loc='best')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the figure to a bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            
            # Close the figure to free memory
            plt.close()
            
            return buf.getvalue()
            
        except Exception as e:
            logger.error(f"Error generating pivot chart image: {e}")
            return None








# Update the EmailReporter class in src/utils/email_reporter.py to include pivot table

# Add this import at the top
from src.utils.pivot_analyzer import PivotAnalyzer

# Update the generate_email_report method
def generate_email_report(self, predictions_df, evaluation=None, predictions_path=None):
    """
    Generate and send email report with analysis results.
    
    Args:
        predictions_df (pandas.DataFrame): DataFrame with predictions
        evaluation (dict): Evaluation metrics (optional)
        predictions_path (str): Path to the Excel file with predictions (optional)
        
    Returns:
        str: HTML content of the email
    """
    # Calculate summary statistics
    total_stories = len(predictions_df)
    security_count = predictions_df['security_prediction'].sum()
    fraud_count = predictions_df['fraud_prediction'].sum()
    both_count = ((predictions_df['security_prediction'] == 1) & 
                  (predictions_df['fraud_prediction'] == 1)).sum()
    any_count = ((predictions_df['security_prediction'] == 1) | 
                 (predictions_df['fraud_prediction'] == 1)).sum()
    
    security_percent = round(security_count / total_stories * 100, 1) if total_stories > 0 else 0
    fraud_percent = round(fraud_count / total_stories * 100, 1) if total_stories > 0 else 0
    both_percent = round(both_count / total_stories * 100, 1) if total_stories > 0 else 0
    any_percent = round(any_count / total_stories * 100, 1) if total_stories > 0 else 0
    
    # Get top security and fraud stories
    top_security_stories = predictions_df[predictions_df['security_prediction'] == 1].sort_values(
        by='security_probability', ascending=False).head(5)
    
    top_fraud_stories = predictions_df[predictions_df['fraud_prediction'] == 1].sort_values(
        by='fraud_probability', ascending=False).head(5)
    
    # Generate pivot table for program impacts
    pivot_analyzer = PivotAnalyzer()
    program_impact_pivot = pivot_analyzer.generate_program_impact_pivot(predictions_df)
    
    # Convert pivot table to HTML for email
    pivot_html = ""
    has_pivot_data = False
    
    if program_impact_pivot is not None and not program_impact_pivot.empty:
        has_pivot_data = True
        # Get top 10 programs for email display
        display_pivot = program_impact_pivot.head(10)
        
        # Create HTML with custom styling
        pivot_html = display_pivot.to_html(
            classes='pivot-table',
            float_format=lambda x: f'{x:.1f}' if isinstance(x, float) else str(x)
        )
        
        # Apply custom formatting to the HTML table
        pivot_html = pivot_html.replace('<table', '<table style="width:100%; border-collapse:collapse; margin:15px 0;"')
        pivot_html = pivot_html.replace('<th>', '<th style="background-color:#004080; color:white; padding:8px; text-align:left;">')
        pivot_html = pivot_html.replace('<td>', '<td style="border:1px solid #ddd; padding:8px;">')
        pivot_html = pivot_html.replace('<tr>', '<tr style="border-bottom:1px solid #ddd;">')
    
    # Prepare template data
    template_data = {
        'report_date': datetime.now().strftime("%B %d, %Y"),
        'analysis_date': datetime.now().strftime("%B %d, %Y at %H:%M"),
        'total_stories': total_stories,
        'security_count': int(security_count),
        'fraud_count': int(fraud_count),
        'both_count': int(both_count),
        'any_count': int(any_count),
        'security_percent': security_percent,
        'fraud_percent': fraud_percent,
        'both_percent': both_percent,
        'any_percent': any_percent,
        'top_security_stories': top_security_stories.to_dict('records'),
        'top_fraud_stories': top_fraud_stories.to_dict('records'),
        'has_pivot_data': has_pivot_data,
        'pivot_html': pivot_html
    }
    
    # Add model performance metrics if available
    has_model_metrics = False
    if evaluation:
        logger.info("Adding model performance metrics to email report")
        try:
            security_report = evaluation.get('security_report', {})
            fraud_report = evaluation.get('fraud_report', {})
            
            # Check if we have valid classification reports
            if isinstance(security_report, dict) and isinstance(fraud_report, dict):
                has_model_metrics = True
                
                # Handle security metrics
                security_metrics = {}
                if 'accuracy' in security_report:
                    security_metrics['accuracy'] = round(security_report['accuracy'] * 100, 1)
                
                # Check for class '1' metrics (positive class)
                if '1' in security_report:
                    security_metrics['precision'] = round(security_report['1']['precision'] * 100, 1)
                    security_metrics['recall'] = round(security_report['1']['recall'] * 100, 1)
                    security_metrics['f1'] = round(security_report['1']['f1-score'] * 100, 1)
                else:
                    # If no positive class, check for macro avg
                    if 'macro avg' in security_report:
                        security_metrics['precision'] = round(security_report['macro avg']['precision'] * 100, 1)
                        security_metrics['recall'] = round(security_report['macro avg']['recall'] * 100, 1)
                        security_metrics['f1'] = round(security_report['macro avg']['f1-score'] * 100, 1)
                    else:
                        security_metrics['precision'] = security_metrics['recall'] = security_metrics['f1'] = 'N/A'
                
                # Handle fraud metrics
                fraud_metrics = {}
                if 'accuracy' in fraud_report:
                    fraud_metrics['accuracy'] = round(fraud_report['accuracy'] * 100, 1)
                
                # Check for class '1' metrics (positive class)
                if '1' in fraud_report:
                    fraud_metrics['precision'] = round(fraud_report['1']['precision'] * 100, 1)
                    fraud_metrics['recall'] = round(fraud_report['1']['recall'] * 100, 1)
                    fraud_metrics['f1'] = round(fraud_report['1']['f1-score'] * 100, 1)
                else:
                    # If no positive class, check for macro avg
                    if 'macro avg' in fraud_report:
                        fraud_metrics['precision'] = round(fraud_report['macro avg']['precision'] * 100, 1)
                        fraud_metrics['recall'] = round(fraud_report['macro avg']['recall'] * 100, 1)
                        fraud_metrics['f1'] = round(fraud_report['macro avg']['f1-score'] * 100, 1)
                    else:
                        fraud_metrics['precision'] = fraud_metrics['recall'] = fraud_metrics['f1'] = 'N/A'
                
                template_data.update({
                    'security_accuracy': security_metrics.get('accuracy', 'N/A'),
                    'security_precision': security_metrics.get('precision', 'N/A'),
                    'security_recall': security_metrics.get('recall', 'N/A'),
                    'security_f1': security_metrics.get('f1', 'N/A'),
                    'fraud_accuracy': fraud_metrics.get('accuracy', 'N/A'),
                    'fraud_precision': fraud_metrics.get('precision', 'N/A'),
                    'fraud_recall': fraud_metrics.get('recall', 'N/A'),
                    'fraud_f1': fraud_metrics.get('f1', 'N/A')
                })
                
                logger.info("Successfully added model metrics to email report")
            else:
                logger.warning("Evaluation data does not contain valid classification reports")
        except Exception as e:
            logger.error(f"Error processing evaluation metrics for email: {e}")
            has_model_metrics = False
    
    # If we don't have metrics or there was an error, use default N/A values
    if not has_model_metrics:
        logger.info("Using default N/A values for model metrics in email report")
        template_data.update({
            'security_accuracy': 'N/A',
            'security_precision': 'N/A',
            'security_recall': 'N/A',
            'security_f1': 'N/A',
            'fraud_accuracy': 'N/A',
            'fraud_precision': 'N/A',
            'fraud_recall': 'N/A',
            'fraud_f1': 'N/A'
        })
    
    # Check if we have confusion matrices
    template_data['has_confusion_matrices'] = (
        evaluation and 
        'security_confusion_matrix' in evaluation and
        'fraud_confusion_matrix' in evaluation
    )
    
    # Generate HTML using Jinja2 template
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('email_template.html')
    html_content = template.render(**template_data)
    
    return html_content








# Update the _create_email_template method in EmailReporter class to include pivot table section

def _create_email_template(self):
    """Create the HTML email template if it doesn't exist."""
    template_path = 'templates/email_template.html'
    
    if not os.path.exists(template_path):
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                }
                .header {
                    background-color: #004080;
                    color: white;
                    padding: 20px;
                    text-align: center;
                }
                .content {
                    padding: 20px;
                }
                .summary-box {
                    background-color: #f5f5f5;
                    border-left: 4px solid #004080;
                    padding: 15px;
                    margin-bottom: 20px;
                }
                .summary-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }
                .summary-table th {
                    background-color: #004080;
                    color: white;
                    text-align: left;
                    padding: 10px;
                }
                .summary-table td {
                    border: 1px solid #ddd;
                    padding: 10px;
                }
                .pivot-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                }
                .pivot-table th {
                    background-color: #004080;
                    color: white;
                    text-align: left;
                    padding: 8px;
                }
                .pivot-table td {
                    border: 1px solid #ddd;
                    padding: 8px;
                }
                .charts {
                    text-align: center;
                    margin: 30px 0;
                }
                .chart {
                    margin-bottom: 20px;
                }
                .footer {
                    background-color: #f5f5f5;
                    padding: 15px;
                    text-align: center;
                    font-size: 0.8em;
                    color: #666;
                }
                .highlight {
                    background-color: #ffffcc;
                    padding: 2px 5px;
                    font-weight: bold;
                }
                .security {
                    color: #e67e00;
                }
                .fraud {
                    color: #9c27b0;
                }
                .both {
                    color: #d32f2f;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Jira Security & Fraud Analysis Report</h1>
                <p>{{ report_date }}</p>
            </div>
            <div class="content">
                <h2>Analysis Summary</h2>
                <div class="summary-box">
                    <p>This report analyzes <strong>{{ total_stories }}</strong> Jira user stories for security and fraud impacts.</p>
                    <p>Analysis completed on <strong>{{ analysis_date }}</strong></p>
                </div>
                
                <h3>Key Findings</h3>
                <table class="summary-table">
                    <tr>
                        <th>Metric</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                    <tr>
                        <td>Stories with <span class="security">Security Impact</span></td>
                        <td>{{ security_count }}</td>
                        <td>{{ security_percent }}%</td>
                    </tr>
                    <tr>
                        <td>Stories with <span class="fraud">Fraud Impact</span></td>
                        <td>{{ fraud_count }}</td>
                        <td>{{ fraud_percent }}%</td>
                    </tr>
                    <tr>
                        <td>Stories with <span class="both">Both Impacts</span></td>
                        <td>{{ both_count }}</td>
                        <td>{{ both_percent }}%</td>
                    </tr>
                    <tr>
                        <td>Stories with Any Impact</td>
                        <td>{{ any_count }}</td>
                        <td>{{ any_percent }}%</td>
                    </tr>
                </table>
                
                {% if has_pivot_data %}
                <h3>Program Impact Analysis</h3>
                <p>The table below shows the distribution of security and fraud impacts across test programs:</p>
                
                {{ pivot_html|safe }}
                
                <p><em>Note: The full pivot table analysis is available in the attached Excel file.</em></p>
                
                <div class="charts">
                    <h4>Program Impact Visualization</h4>
                    <div class="chart">
                        <img src="cid:program_impact_chart" alt="Program Impact Chart" style="max-width: 100%;">
                    </div>
                </div>
                {% endif %}
                
                <h3>Model Performance</h3>
                <p>The model achieved the following performance metrics:</p>
                <table class="summary-table">
                    <tr>
                        <th>Model</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                    </tr>
                    <tr>
                        <td>Security Impact</td>
                        <td>{{ security_accuracy }}</td>
                        <td>{{ security_precision }}</td>
                        <td>{{ security_recall }}</td>
                        <td>{{ security_f1 }}</td>
                    </tr>
                    <tr>
                        <td>Fraud Impact</td>
                        <td>{{ fraud_accuracy }}</td>
                        <td>{{ fraud_precision }}</td>
                        <td>{{ fraud_recall }}</td>
                        <td>{{ fraud_f1 }}</td>
                    </tr>
                </table>
                
                {% if has_confusion_matrices %}
                <div class="charts">
                    <h3>Confusion Matrices</h3>
                    <div class="chart">
                        <img src="cid:confusion_matrices" alt="Confusion Matrices" style="max-width: 100%;">
                    </div>
                </div>
                {% endif %}
                
                <h3>Highlighted Stories</h3>
                <p>Top security-impacted stories:</p>
                <ul>
                    {% for story in top_security_stories %}
                    <li><strong>{{ story.issue_key }}</strong>: {{ story.summary }} 
                        <span class="highlight security">({{ story.security_probability|round(2) }})</span></li>
                    {% endfor %}
                </ul>
                
                <p>Top fraud-impacted stories:</p>
                <ul>
                    {% for story in top_fraud_stories %}
                    <li><strong>{{ story.issue_key }}</strong>: {{ story.summary }} 
                        <span class="highlight fraud">({{ story.fraud_probability|round(2) }})</span></li>
                    {% endfor %}
                </ul>
            </div>
            <div class="footer">
                <p>This report was automatically generated by the Jira Security & Fraud Analysis System</p>
                <p>Full results are available in the attached Excel file.</p>
            </div>
        </body>
        </html>
        """
        
        with open(template_path, 'w') as f:
            f.write(html_template)
        
        logger.info(f"Created email template at {template_path}")







# Update the send_email_report method in EmailReporter class to include pivot table chart

def send_email_report(self, recipients, subject, html_content, evaluation=None, predictions_path=None, predictions_df=None):
    """
    Send email report with analysis results.
    
    Args:
        recipients (list): List of recipient email addresses
        subject (str): Email subject
        html_content (str): HTML content of the email
        evaluation (dict): Evaluation metrics (optional, for confusion matrix)
        predictions_path (str): Path to the Excel file with predictions (optional)
        predictions_df (pandas.DataFrame): DataFrame with predictions (optional, for pivot table)
        
    Returns:
        bool: Success status
    """
    try:
        # Validate required parameters
        if not recipients or not subject or not html_content:
            logger.error("Missing required email parameters")
            return False
        
        # Validate email settings
        if not self.smtp_server or not self.smtp_port or not self.username or not self.password:
            logger.error("Email settings not properly configured")
            return False
            
        # Create email message
        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = subject
        
        # Attach HTML content
        msg.attach(MIMEText(html_content, 'html'))
        
        # Attach confusion matrix image if evaluation is provided
        if evaluation and 'security_confusion_matrix' in evaluation and 'fraud_confusion_matrix' in evaluation:
            try:
                image_data = self.generate_confusion_matrix_image(evaluation)
                if image_data:
                    image = MIMEImage(image_data)
                    image.add_header('Content-ID', '<confusion_matrices>')
                    image.add_header('Content-Disposition', 'inline')
                    msg.attach(image)
                    logger.info("Confusion matrix image attached to email")
                else:
                    logger.warning("No confusion matrix image generated")
            except Exception as e:
                logger.error(f"Error attaching confusion matrix: {e}")
        else:
            logger.info("No confusion matrices available to include in email")
        
        # Generate and attach pivot table chart if predictions_df is provided
        if predictions_df is not None and not predictions_df.empty:
            try:
                # Create pivot analyzer
                from src.utils.pivot_analyzer import PivotAnalyzer
                pivot_analyzer = PivotAnalyzer()
                
                # Generate pivot table
                pivot_table = pivot_analyzer.generate_program_impact_pivot(predictions_df)
                
                if pivot_table is not None and not pivot_table.empty:
                    # Generate chart image
                    chart_image = pivot_analyzer.generate_pivot_chart_image(pivot_table)
                    
                    if chart_image:
                        # Attach the chart image
                        image = MIMEImage(chart_image)
                        image.add_header('Content-ID', '<program_impact_chart>')
                        image.add_header('Content-Disposition', 'inline')
                        msg.attach(image)
                        logger.info("Program impact chart attached to email")
                        
                        # Create and attach the pivot Excel file
                        pivot_path = pivot_analyzer.create_pivot_excel(pivot_table)
                        if pivot_path and os.path.exists(pivot_path):
                            with open(pivot_path, 'rb') as f:
                                pivot_attachment = MIMEApplication(f.read(), _subtype='xlsx')
                                pivot_attachment.add_header('Content-Disposition', 'attachment', 
                                                         filename='program_impact_analysis.xlsx')
                                msg.attach(pivot_attachment)
                                logger.info("Pivot table Excel file attached to email")
                    else:
                        logger.warning("Failed to generate program impact chart")
                else:
                    logger.warning("No valid pivot table data to include in email")
            except Exception as e:
                logger.error(f"Error generating pivot table chart: {e}")
        
        # Attach Excel file if provided
        if predictions_path and os.path.exists(predictions_path):
            try:
                with open(predictions_path, 'rb') as f:
                    attachment = MIMEApplication(f.read(), _subtype='xlsx')
                    attachment.add_header('Content-Disposition', 'attachment', 
                                         filename=os.path.basename(predictions_path))
                    msg.attach(attachment)
                logger.info(f"Excel file attached: {os.path.basename(predictions_path)}")
            except Exception as e:
                logger.error(f"Error attaching Excel file: {e}")
        
        # Send email
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            logger.info(f"Connecting to SMTP server {self.smtp_server}:{self.smtp_port}")
            server.starttls()
            server.login(self.username, self.password)
            logger.info(f"SMTP login successful")
            server.send_message(msg)
            logger.info(f"Email sent to {', '.join(recipients)}")
        
        return True
    except smtplib.SMTPAuthenticationError:
        logger.error("SMTP authentication failed. Check your username and password.")
        return False
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to send email report: {e}")
        return False








# Update the predict_and_analyze function in main.py to pass predictions_df to email reporter

def predict_and_analyze(model_trainer, input_data, nltk_data_path=None, exclude_columns=None, 
                        hidden_columns=None, send_email=False, email_recipients=None, 
                        email_subject=None, evaluation=None):
    """
    Make predictions and analyze data.
    
    Args:
        model_trainer (ModelTrainer): Trained model instance
        input_data: Input data (DataFrame or path to file)
        nltk_data_path (str): Path to NLTK data directory
        exclude_columns (list): Columns to exclude from Excel output
        hidden_columns (list): Columns to hide in Excel output
        send_email (bool): Whether to send email report
        email_recipients (list): List of email recipients
        email_subject (str): Email subject
        evaluation (dict): Evaluation metrics from model training (optional)
        
    Returns:
        tuple: (predictions_df, predictions_path, template_path)
    """
    logger.info("Running predictions and analysis")
    
    # Set default values if None
    exclude_columns = exclude_columns or []
    hidden_columns = hidden_columns or []
    
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
    
    # Generate pivot table analysis
    try:
        from src.utils.pivot_analyzer import PivotAnalyzer
        pivot_analyzer = PivotAnalyzer()
        program_impact_pivot = pivot_analyzer.generate_program_impact_pivot(predictions_df)
        
        if program_impact_pivot is not None and not program_impact_pivot.empty:
            pivot_path = pivot_analyzer.create_pivot_excel(program_impact_pivot)
            if pivot_path:
                logger.info(f"Program impact pivot table saved to {pivot_path}")
    except Exception as e:
        logger.error(f"Error generating pivot table: {e}")
    
    # Generate and send email report if requested
    if send_email:
        try:
            # Use default recipients if none provided
            recipients = email_recipients or EMAIL_SETTINGS.get('default_recipients')
            if not recipients:
                logger.warning("No email recipients specified. Email report will not be sent.")
            else:
                # Create email reporter
                email_reporter = EmailReporter(
                    smtp_server=EMAIL_SETTINGS.get('smtp_server'),
                    smtp_port=EMAIL_SETTINGS.get('smtp_port'),
                    username=EMAIL_SETTINGS.get('username'),
                    password=EMAIL_SETTINGS.get('password'),
                    sender_email=EMAIL_SETTINGS.get('sender_email')
                )
                
                # Generate and send email report
                html_content = email_reporter.generate_email_report(
                    predictions_df, 
                    evaluation=evaluation,
                    predictions_path=predictions_path
                )
                
                subject = email_subject or "Jira Security & Fraud Analysis Report"
                
                success = email_reporter.send_email_report(
                    recipients=recipients,
                    subject=subject,
                    html_content=html_content,
                    evaluation=evaluation,
                    predictions_path=predictions_path,
                    predictions_df=predictions_df  # Pass the predictions DataFrame for pivot table
                )
                
                if success:
                    logger.info(f"Email report sent to: {', '.join(recipients)}")
                else:
                    logger.error("Failed to send email report.")
        except Exception as e:
            logger.error(f"Error sending email report: {e}")
    
    # Summary statistics
    security_count = predictions_df['security_prediction'].sum()
    fraud_count = predictions_df['fraud_prediction'].sum()
    both_count = ((predictions_df['security_prediction'] == 1) & 
                  (predictions_df['fraud_prediction'] == 1)).sum()
    any_count = ((predictions_df['security_prediction'] == 1) | 
                 (predictions_df['fraud_prediction'] == 1)).sum()
    
    logger.info(f"Analysis Summary:")
    logger.info(f"Total user stories analyzed: {len(predictions_df)}")
    logger.info(f"User stories with security impacts: {security_count} ({security_count/len(predictions_df)*100:.1f}%)")
    logger.info(f"User stories with fraud impacts: {fraud_count} ({fraud_count/len(predictions_df)*100:.1f}%)")
    logger.info(f"User stories with both security and fraud impacts: {both_count} ({both_count/len(predictions_df)*100:.1f}%)")
    logger.info(f"User stories with any security or fraud impact: {any_count} ({any_count/len(predictions_df)*100:.1f}%)")
    
    return predictions_df, predictions_path, template_path






