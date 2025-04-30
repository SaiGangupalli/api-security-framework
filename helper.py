# Fix for the generate_program_impact_pivot method in PivotAnalyzer class

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
            values=['security_only', 'fraud_only', 'both_impacts', 'security_prediction', 'fraud_prediction'],
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
    
    except Exception as e:
        logger.error(f"Error generating program impact pivot table: {e}")
        return None



# Updated generate_pivot_chart_image method in PivotAnalyzer class

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
        
        # Verify matplotlib is available
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Limit to top 10 programs for readability
        display_pivot = pivot_table.head(10).copy()
        
        # Select columns to plot (ensure they exist)
        if 'Both Impacts' in display_pivot.columns and 'Security Only' in display_pivot.columns and 'Fraud Only' in display_pivot.columns:
            plot_data = display_pivot[['Security Only', 'Fraud Only', 'Both Impacts']]
        else:
            # Fallback to whatever columns are available
            plot_cols = [col for col in ['Total Security Impacts', 'Total Fraud Impacts'] if col in display_pivot.columns]
            plot_data = display_pivot[plot_cols]
        
        # Create a figure
        plt.figure(figsize=(10, 6))
        
        # Create a grouped bar chart
        ax = plot_data.plot(
            kind='bar',
            color=['#4285F4', '#EA4335', '#FBBC05'],  # Blue, Red, Yellow
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
        
        logger.info("Successfully generated pivot chart image")
        return buf.getvalue()
            
    except Exception as e:
        logger.error(f"Error generating pivot chart image: {e}")
        # Add more detailed error logging
        import traceback
        logger.error(traceback.format_exc())
        return None






# Updated send_email_report method in EmailReporter class with improved pivot chart handling

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
                # Import within the try block to handle missing dependencies
                from src.utils.pivot_analyzer import PivotAnalyzer
                import io
                
                logger.info("Generating pivot table and chart for email report")
                
                # Create pivot analyzer
                pivot_analyzer = PivotAnalyzer()
                
                # Generate pivot table
                pivot_table = pivot_analyzer.generate_program_impact_pivot(predictions_df)
                
                if pivot_table is not None and not pivot_table.empty:
                    # Generate chart image
                    logger.info("Generating pivot chart image")
                    chart_image = pivot_analyzer.generate_pivot_chart_image(pivot_table)
                    
                    if chart_image:
                        # Attach the chart image
                        logger.info("Attaching pivot chart image to email")
                        pivot_image = MIMEImage(chart_image)
                        pivot_image.add_header('Content-ID', '<program_impact_chart>')
                        pivot_image.add_header('Content-Disposition', 'inline')
                        msg.attach(pivot_image)
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

