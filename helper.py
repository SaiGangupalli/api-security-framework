# src/utils/email_reporter.py
import os
import logging
import smtplib
import io
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Environment, FileSystemLoader

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmailReporter:
    """Class to generate and send email reports with analysis results."""
    
    def __init__(self, smtp_server, smtp_port, username, password, sender_email):
        """
        Initialize the email reporter.
        
        Args:
            smtp_server (str): SMTP server address
            smtp_port (int): SMTP server port
            username (str): SMTP account username
            password (str): SMTP account password
            sender_email (str): Sender email address
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.sender_email = sender_email
        
        # Create templates directory if it doesn't exist
        os.makedirs('templates', exist_ok=True)
        
        # Create the email template if it doesn't exist
        self._create_email_template()
    
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
                            <td>{{ security_accuracy }}%</td>
                            <td>{{ security_precision }}%</td>
                            <td>{{ security_recall }}%</td>
                            <td>{{ security_f1 }}%</td>
                        </tr>
                        <tr>
                            <td>Fraud Impact</td>
                            <td>{{ fraud_accuracy }}%</td>
                            <td>{{ fraud_precision }}%</td>
                            <td>{{ fraud_recall }}%</td>
                            <td>{{ fraud_f1 }}%</td>
                        </tr>
                    </table>
                    
                    <div class="charts">
                        <h3>Confusion Matrices</h3>
                        <div class="chart">
                            <img src="cid:confusion_matrices" alt="Confusion Matrices" style="max-width: 100%;">
                        </div>
                    </div>
                    
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
    
    def generate_confusion_matrix_image(self, evaluation):
        """
        Generate confusion matrix visualization.
        
        Args:
            evaluation (dict): Evaluation metrics including confusion matrices
            
        Returns:
            bytes: Image data as bytes
        """
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
        
        # Save the figure to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        
        # Close the figure to free memory
        plt.close()
        
        return buf.getvalue()
    
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
            'top_fraud_stories': top_fraud_stories.to_dict('records')
        }
        
        # Add model performance metrics if available
        if evaluation:
            security_report = evaluation.get('security_report', {})
            fraud_report = evaluation.get('fraud_report', {})
            
            # Handle missing class data
            try:
                if '1' in security_report:
                    security_precision = round(security_report['1']['precision'] * 100, 1)
                    security_recall = round(security_report['1']['recall'] * 100, 1)
                    security_f1 = round(security_report['1']['f1-score'] * 100, 1)
                else:
                    security_precision = security_recall = security_f1 = "N/A"
            except (KeyError, TypeError):
                security_precision = security_recall = security_f1 = "N/A"
                
            try:
                if '1' in fraud_report:
                    fraud_precision = round(fraud_report['1']['precision'] * 100, 1)
                    fraud_recall = round(fraud_report['1']['recall'] * 100, 1)
                    fraud_f1 = round(fraud_report['1']['f1-score'] * 100, 1)
                else:
                    fraud_precision = fraud_recall = fraud_f1 = "N/A"
            except (KeyError, TypeError):
                fraud_precision = fraud_recall = fraud_f1 = "N/A"
            
            template_data.update({
                'security_accuracy': round(security_report.get('accuracy', 0) * 100, 1),
                'security_precision': security_precision,
                'security_recall': security_recall,
                'security_f1': security_f1,
                'fraud_accuracy': round(fraud_report.get('accuracy', 0) * 100, 1),
                'fraud_precision': fraud_precision,
                'fraud_recall': fraud_recall,
                'fraud_f1': fraud_f1
            })
        else:
            # Default values if evaluation is not provided
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
        
        # Generate HTML using Jinja2 template
        env = Environment(loader=FileSystemLoader('templates'))
        template = env.get_template('email_template.html')
        html_content = template.render(**template_data)
        
        return html_content
    
    def send_email_report(self, recipients, subject, html_content, evaluation=None, predictions_path=None):
        """
        Send email report with analysis results.
        
        Args:
            recipients (list): List of recipient email addresses
            subject (str): Email subject
            html_content (str): HTML content of the email
            evaluation (dict): Evaluation metrics (optional, for confusion matrix)
            predictions_path (str): Path to the Excel file with predictions (optional)
            
        Returns:
            bool: Success status
        """
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            # Attach HTML content
            msg.attach(MIMEText(html_content, 'html'))
            
            # Attach confusion matrix image if evaluation is provided
            if evaluation:
                image_data = self.generate_confusion_matrix_image(evaluation)
                image = MIMEImage(image_data)
                image.add_header('Content-ID', '<confusion_matrices>')
                image.add_header('Content-Disposition', 'inline')
                msg.attach(image)
            
            # Attach Excel file if provided
            if predictions_path and os.path.exists(predictions_path):
                with open(predictions_path, 'rb') as f:
                    attachment = MIMEApplication(f.read(), _subtype='xlsx')
                    attachment.add_header('Content-Disposition', 'attachment', 
                                          filename=os.path.basename(predictions_path))
                    msg.attach(attachment)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"Email report sent to {', '.join(recipients)}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email report: {e}")
            return False




# Add this to config/settings.py

# Email settings
EMAIL_SETTINGS = {
    'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
    'smtp_port': int(os.getenv('SMTP_PORT', 587)),
    'username': os.getenv('EMAIL_USERNAME', ''),
    'password': os.getenv('EMAIL_PASSWORD', ''),
    'sender_email': os.getenv('SENDER_EMAIL', ''),
    'default_recipients': os.getenv('DEFAULT_RECIPIENTS', '').split(',')
}




# Add these imports to main.py
from src.utils.email_reporter import EmailReporter
from config.settings import EMAIL_SETTINGS

# Add these arguments to parse_arguments() function
def parse_arguments():
    """Parse command line arguments."""
    # ... existing code ...
    
    # Add email reporting arguments
    parser.add_argument('--email-report', action='store_true',
                        help='Generate and send email report')
    parser.add_argument('--email-recipients', nargs='+', type=str,
                        help='Email recipients (space-separated)')
    parser.add_argument('--email-subject', type=str, default='Jira Security & Fraud Analysis Report',
                        help='Email subject')
    
    return parser.parse_args()

# Update predict_and_analyze function to include email reporting
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
    # ... existing code ...
    
    # Export predictions to Excel
    # ... existing code ...
    
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
                    predictions_path=predictions_path
                )
                
                if success:
                    logger.info(f"Email report sent to: {', '.join(recipients)}")
                else:
                    logger.error("Failed to send email report.")
        except Exception as e:
            logger.error(f"Error sending email report: {e}")
    
    # ... rest of the function ...
    
    return predictions_df, predictions_path, template_path

# Update main function to handle email reporting
def main():
    # ... existing code ...
    
    # Make predictions and analyze
    if args.predict or args.analyze:
        input_data = processed_data if processed_data is not None else processed_path
        
        # Get evaluation metrics if we've just trained a model
        evaluation = None
        if args.train and 'evaluation' in locals():
            evaluation = locals()['evaluation']
        
        predictions_df, predictions_path, template_path = predict_and_analyze(
            trainer, 
            input_data, 
            args.nltk_data,
            exclude_columns=exclude_columns,
            hidden_columns=hidden_columns,
            send_email=args.email_report,
            email_recipients=args.email_recipients,
            email_subject=args.email_subject,
            evaluation=evaluation
        )
        
        # ... rest of the function ...




# .env
# Jira API Configuration
JIRA_URL=https://your-jira-instance.atlassian.net
# API Token for Bearer Authentication
JIRA_API_TOKEN=your_bearer_token

# Email Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
SENDER_EMAIL=your_email@gmail.com
DEFAULT_RECIPIENTS=recipient1@example.com,recipient2@example.com




