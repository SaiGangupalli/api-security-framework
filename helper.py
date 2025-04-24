# Update in src/utils/email_reporter.py - fix generate_confusion_matrix_image method

def generate_confusion_matrix_image(self, evaluation):
    """
    Generate confusion matrix visualization.
    
    Args:
        evaluation (dict): Evaluation metrics including confusion matrices
        
    Returns:
        bytes: Image data as bytes
    """
    try:
        # Check if confusion matrices are available in the evaluation
        if ('security_confusion_matrix' not in evaluation or 
            'fraud_confusion_matrix' not in evaluation):
            logger.warning("Confusion matrices not found in evaluation data")
            return None
            
        # Create the figure
        plt.figure(figsize=(12, 5))
        
        # Security confusion matrix
        plt.subplot(1, 2, 1)
        security_cm = evaluation['security_confusion_matrix']
        sns.heatmap(
            security_cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['No Security Impact', 'Security Impact'],
            yticklabels=['No Security Impact', 'Security Impact']
        )
        plt.title('Security Impact Confusion Matrix')
        
        # Fraud confusion matrix
        plt.subplot(1, 2, 2)
        fraud_cm = evaluation['fraud_confusion_matrix']
        sns.heatmap(
            fraud_cm, 
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
    except Exception as e:
        logger.error(f"Error generating confusion matrix image: {e}")
        # Return None instead of failing
        return None






# Update the train_model function in main.py to return evaluation metrics

def train_model(input_data, nltk_data_path=None, feedback_path=None):
    """
    Train or update the ML model.
    
    Args:
        input_data: Input data (DataFrame or path to file)
        nltk_data_path (str): Path to NLTK data directory
        feedback_path (str): Path to feedback file for model updating
        
    Returns:
        tuple: (model, trainer, evaluation, importances)
    """
    logger.info("Starting model training")
    
    # Initialize feature engineer with NLTK data path
    feature_engineer = FeatureEngineer(nltk_data_path)
    
    # Convert input_data to DataFrame if it's a file path
    if isinstance(input_data, str):
        trainer = ModelTrainer(input_data)
        # Set the feature engineer
        trainer.feature_engineer = feature_engineer
    else:
        trainer = ModelTrainer()
        trainer.df = input_data
        # Set the feature engineer
        trainer.feature_engineer = feature_engineer
    
    # If feedback is provided, update the model
    if feedback_path and os.path.exists(feedback_path):
        logger.info(f"Incorporating feedback from {feedback_path}")
        feedback_data = pd.read_excel(feedback_path)
        model, evaluation, importances = trainer.update_model_with_feedback(feedback_data)
    else:
        # Train a new model
        model, evaluation, importances = trainer.train_model()
    
    if model is None:
        logger.error("Model training failed")
        return None, None, None, None
    
    # Visualize results
    visualize_results(evaluation, importances)
    
    # Update keyword lists based on feature importances
    update_keyword_lists(importances)
    
    logger.info("Model training complete")
    return model, trainer, evaluation, importances




# Update the main function in main.py to capture and pass evaluation metrics

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
    
    # Initialize evaluation variable
    evaluation = None
    
    # Train or update the model
    if args.train:
        model, trainer, evaluation, importances = train_model(processed_path, args.nltk_data, args.feedback)
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
            hidden_columns=hidden_columns,
            send_email=args.email_report,
            email_recipients=args.email_recipients,
            email_subject=args.email_subject,
            evaluation=evaluation  # Pass the evaluation from training
        )
        
        logger.info("Analysis complete!")
        logger.info(f"Results available at {predictions_path}")
        logger.info(f"Feedback template available at {template_path}")
    
    logger.info("All operations completed successfully")




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




# Update in EmailReporter._create_email_template method - modify the template

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




# Update the send_email_report method in EmailReporter class

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
