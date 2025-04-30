# Update the _create_email_template method in EmailReporter class to remove chart section

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
                    font-size: 0.9em;
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
                .pivot-table tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
                .pivot-table tr:hover {
                    background-color: #f1f1f1;
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







# Update the generate_email_report method in EmailReporter class

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
        
        # Get top 20 programs for email display (increased from 10)
        display_pivot = program_impact_pivot.head(20)
        
        # Select specific columns for display
        display_columns = [
            'Total Security Impacts', 
            'Total Fraud Impacts',
            'Security Only',
            'Fraud Only', 
            'Both Impacts',
            'Total Impacts'
        ]
        
        # Make sure all columns exist
        for col in display_columns:
            if col not in display_pivot.columns:
                display_pivot[col] = 0
        
        # Use only the columns we want to display
        email_pivot = display_pivot[display_columns]
        
        # Create HTML with custom styling
        pivot_html = email_pivot.to_html(
            classes='pivot-table',
            float_format=lambda x: f'{x:.0f}' if isinstance(x, float) else str(x)
        )
        
        # Apply custom formatting to the HTML table
        pivot_html = pivot_html.replace('<table', '<table style="width:100%; border-collapse:collapse; margin:15px 0; font-size:0.9em;"')
        pivot_html = pivot_html.replace('<th>', '<th style="background-color:#004080; color:white; padding:8px; text-align:left;">')
        pivot_html = pivot_html.replace('<td>', '<td style="border:1px solid #ddd; padding:8px;">')
        pivot_html = pivot_html.replace('<tr>', '<tr style="border-bottom:1px solid #ddd;">')
        
        # Add alternating row colors
        pivot_html = pivot_html.replace('<tr style="border-bottom:1px solid #ddd;">', 
                                       '<tr style="border-bottom:1px solid #ddd; background-color:#f9f9f9;">', 
                                       len(display_pivot) // 2)
    
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






# Update the send_email_report method in EmailReporter class to remove chart

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
        
        # Generate and attach pivot table Excel file if predictions_df is provided
        if predictions_df is not None and not predictions_df.empty:
            try:
                # Import within the try block to handle missing dependencies
                from src.utils.pivot_analyzer import PivotAnalyzer
                
                logger.info("Generating pivot table for email report")
                
                # Create pivot analyzer
                pivot_analyzer = PivotAnalyzer()
                
                # Generate pivot table
                pivot_table = pivot_analyzer.generate_program_impact_pivot(predictions_df)
                
                if pivot_table is not None and not pivot_table.empty:
                    # Create and attach the pivot Excel file only (no chart)
                    pivot_path = pivot_analyzer.create_pivot_excel(pivot_table)
                    if pivot_path and os.path.exists(pivot_path):
                        with open(pivot_path, 'rb') as f:
                            pivot_attachment = MIMEApplication(f.read(), _subtype='xlsx')
                            pivot_attachment.add_header('Content-Disposition', 'attachment', 
                                                     filename='program_impact_analysis.xlsx')
                            msg.attach(pivot_attachment)
                            logger.info("Pivot table Excel file attached to email")
                else:
                    logger.warning("No valid pivot table data to include in email")
            except Exception as e:
                logger.error(f"Error generating pivot table: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
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
        import traceback
        logger.error(traceback.format_exc())
        return False
