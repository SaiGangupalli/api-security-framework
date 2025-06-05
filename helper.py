requests==2.31.0
python-docx==0.8.11
openai==1.3.7
smtplib-ssl==1.0.0
email-validator==2.1.0
python-dotenv==1.0.0
jira==3.5.0
beautifulsoup4==4.12.2
lxml==4.9.4



"""
Configuration file for Jira Analysis System
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Jira Configuration
JIRA_URL = os.getenv('JIRA_URL')  # e.g., 'https://yourcompany.atlassian.net'
JIRA_USERNAME = os.getenv('JIRA_USERNAME')
JIRA_API_TOKEN = os.getenv('JIRA_API_TOKEN')

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Email Configuration
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SENDER_EMAIL = os.getenv('SENDER_EMAIL')
SENDER_PASSWORD = os.getenv('SENDER_PASSWORD')  # App password for Gmail

# LLM Prompts
SUMMARY_PROMPT = """
Analyze the following Jira story details and provide a comprehensive summary:

Title: {title}
Description: {description}
Acceptance Criteria: {acceptance_criteria}
Attachments Summary: {attachments_summary}

Please provide:
1. A concise summary of the requirement
2. Key functional points
3. Technical considerations
4. Business value
5. Implementation complexity (Low/Medium/High)

Format your response in a structured manner.
"""

FRAUD_SECURITY_PROMPT = """
Analyze the following Jira story for potential fraud and security considerations:

Title: {title}
Description: {description}
Acceptance Criteria: {acceptance_criteria}

Please identify:
1. Potential fraud scenarios that could arise
2. Security vulnerabilities or risks
3. Recommended security controls
4. Data protection considerations
5. Compliance requirements (if any)
6. Risk level assessment (Low/Medium/High)

If no significant fraud or security concerns are identified, please state that clearly.
Format your response in a structured manner.
"""





"""
Jira Client for extracting issue details
"""
import requests
from jira import JIRA
import base64
from bs4 import BeautifulSoup
import logging
from typing import Dict, List, Optional

class JiraClient:
    def __init__(self, url: str, username: str, api_token: str):
        self.url = url
        self.username = username
        self.api_token = api_token
        self.session = requests.Session()
        self.session.auth = (username, api_token)
        
        # Initialize JIRA client
        try:
            self.jira = JIRA(server=url, basic_auth=(username, api_token))
        except Exception as e:
            logging.error(f"Failed to initialize JIRA client: {e}")
            raise
    
    def extract_issue_details(self, issue_key: str) -> Dict:
        """
        Extract comprehensive details from a Jira issue
        """
        try:
            issue = self.jira.issue(issue_key, expand='attachment')
            
            # Extract basic details
            summary = issue.fields.summary
            description = self._clean_html(issue.fields.description or "")
            
            # Extract acceptance criteria (often in description or custom field)
            acceptance_criteria = self._extract_acceptance_criteria(issue)
            
            # Extract attachments
            attachments_info = self._extract_attachments(issue)
            
            return {
                'key': issue_key,
                'summary': summary,
                'description': description,
                'acceptance_criteria': acceptance_criteria,
                'attachments': attachments_info,
                'status': str(issue.fields.status),
                'assignee': str(issue.fields.assignee) if issue.fields.assignee else "Unassigned",
                'reporter': str(issue.fields.reporter) if issue.fields.reporter else "Unknown",
                'created': str(issue.fields.created),
                'updated': str(issue.fields.updated),
                'priority': str(issue.fields.priority) if issue.fields.priority else "Not Set",
                'issue_type': str(issue.fields.issuetype)
            }
            
        except Exception as e:
            logging.error(f"Error extracting details for {issue_key}: {e}")
            return {
                'key': issue_key,
                'error': str(e),
                'summary': f"Error fetching {issue_key}",
                'description': f"Failed to fetch issue details: {e}",
                'acceptance_criteria': "",
                'attachments': []
            }
    
    def _clean_html(self, text: str) -> str:
        """Clean HTML content from Jira fields"""
        if not text:
            return ""
        
        try:
            soup = BeautifulSoup(text, 'html.parser')
            return soup.get_text(strip=True)
        except:
            return text
    
    def _extract_acceptance_criteria(self, issue) -> str:
        """
        Extract acceptance criteria from various possible fields
        """
        acceptance_criteria = ""
        
        # Check common custom fields for acceptance criteria
        try:
            # Try common custom field names
            custom_fields = ['customfield_10100', 'customfield_10200', 'customfield_10300']
            
            for field in custom_fields:
                if hasattr(issue.fields, field):
                    field_value = getattr(issue.fields, field)
                    if field_value and 'acceptance' in str(field_value).lower():
                        acceptance_criteria = self._clean_html(str(field_value))
                        break
            
            # If not found in custom fields, look in description
            if not acceptance_criteria and issue.fields.description:
                desc_text = self._clean_html(issue.fields.description)
                lines = desc_text.split('\n')
                
                capture = False
                ac_lines = []
                
                for line in lines:
                    line_lower = line.lower().strip()
                    if any(keyword in line_lower for keyword in ['acceptance criteria', 'acceptance', 'ac:']):
                        capture = True
                        if ':' in line:
                            ac_lines.append(line.split(':', 1)[1].strip())
                        continue
                    
                    if capture:
                        if line.strip() and not line_lower.startswith(('description', 'summary', 'notes')):
                            ac_lines.append(line.strip())
                        elif not line.strip() and ac_lines:
                            break
                
                acceptance_criteria = '\n'.join(ac_lines)
        
        except Exception as e:
            logging.warning(f"Error extracting acceptance criteria: {e}")
        
        return acceptance_criteria or "No acceptance criteria found"
    
    def _extract_attachments(self, issue) -> List[Dict]:
        """
        Extract attachment information and content
        """
        attachments_info = []
        
        try:
            for attachment in issue.fields.attachment:
                att_info = {
                    'filename': attachment.filename,
                    'size': attachment.size,
                    'created': str(attachment.created),
                    'author': str(attachment.author),
                    'content_summary': ""
                }
                
                # Try to read attachment content if it's a text file
                if attachment.filename.lower().endswith(('.txt', '.md', '.doc', '.docx')):
                    try:
                        att_content = attachment.get()
                        if len(att_content) < 10000:  # Only for small files
                            att_info['content_summary'] = att_content[:500] + "..." if len(att_content) > 500 else att_content
                    except:
                        att_info['content_summary'] = "Could not read file content"
                
                attachments_info.append(att_info)
        
        except Exception as e:
            logging.warning(f"Error extracting attachments: {e}")
        
        return attachments_info
    
    def validate_connection(self) -> bool:
        """Validate Jira connection"""
        try:
            self.jira.myself()
            return True
        except Exception as e:
            logging.error(f"Jira connection validation failed: {e}")
            return False






"""
LLM Analyzer for processing Jira issues
"""
import openai
import logging
from typing import Dict, List
from config import OPENAI_API_KEY, SUMMARY_PROMPT, FRAUD_SECURITY_PROMPT

class LLMAnalyzer:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or OPENAI_API_KEY
        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def analyze_issue_summary(self, issue_data: Dict) -> str:
        """
        Generate comprehensive summary using LLM
        """
        try:
            # Prepare attachments summary
            attachments_summary = ""
            if issue_data.get('attachments'):
                attachments_summary = f"Attachments ({len(issue_data['attachments'])}): "
                for att in issue_data['attachments']:
                    attachments_summary += f"{att['filename']} ({att.get('content_summary', 'No content available')}); "
            else:
                attachments_summary = "No attachments"
            
            prompt = SUMMARY_PROMPT.format(
                title=issue_data.get('summary', ''),
                description=issue_data.get('description', ''),
                acceptance_criteria=issue_data.get('acceptance_criteria', ''),
                attachments_summary=attachments_summary
            )
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a business analyst expert in analyzing software requirements."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logging.error(f"Error in LLM analysis for summary: {e}")
            return f"Error analyzing issue: {str(e)}"
    
    def analyze_fraud_security(self, issue_data: Dict) -> str:
        """
        Analyze fraud and security implications using LLM
        """
        try:
            prompt = FRAUD_SECURITY_PROMPT.format(
                title=issue_data.get('summary', ''),
                description=issue_data.get('description', ''),
                acceptance_criteria=issue_data.get('acceptance_criteria', '')
            )
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a cybersecurity expert specializing in fraud prevention and security analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.2
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logging.error(f"Error in fraud/security analysis: {e}")
            return f"Error analyzing fraud/security aspects: {str(e)}"
    
    def generate_executive_summary(self, all_issues: List[Dict]) -> str:
        """
        Generate an executive summary of all analyzed issues
        """
        try:
            issues_summary = ""
            for issue in all_issues:
                issues_summary += f"- {issue['key']}: {issue['summary']}\n"
            
            prompt = f"""
            Based on the following Jira issues, provide an executive summary:
            
            Issues Analyzed:
            {issues_summary}
            
            Please provide:
            1. Overall project scope and objectives
            2. Key deliverables and features
            3. High-level risk assessment
            4. Resource and timeline considerations
            5. Strategic recommendations
            
            Keep it concise but comprehensive for executive leadership.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a senior project manager providing executive briefings."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logging.error(f"Error generating executive summary: {e}")
            return f"Error generating executive summary: {str(e)}"






"""
Document Generator for creating Word documents
"""
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.shared import OxmlElement, qn
import logging
from typing import List, Dict
from datetime import datetime

class DocumentGenerator:
    def __init__(self):
        self.doc = Document()
        self._setup_document_styles()
    
    def _setup_document_styles(self):
        """Setup document styles and formatting"""
        # Add title style
        styles = self.doc.styles
        
        # Create or modify heading styles
        try:
            heading1 = styles['Heading 1']
            heading1.font.size = Pt(16)
            heading1.font.bold = True
        except:
            pass
        
        try:
            heading2 = styles['Heading 2']
            heading2.font.size = Pt(14)
            heading2.font.bold = True
        except:
            pass
    
    def create_comprehensive_report(self, issues_data: List[Dict], 
                                  executive_summary: str, 
                                  output_filename: str = "jira_analysis_report.docx") -> str:
        """
        Create a comprehensive analysis report
        """
        try:
            # Document Header
            self._add_document_header()
            
            # Executive Summary
            self._add_executive_summary(executive_summary)
            
            # Individual Issue Analysis
            self._add_issues_analysis(issues_data)
            
            # Consolidated Fraud & Security Analysis
            self._add_consolidated_security_analysis(issues_data)
            
            # Appendix
            self._add_appendix(issues_data)
            
            # Save document
            self.doc.save(output_filename)
            logging.info(f"Document saved as {output_filename}")
            return output_filename
            
        except Exception as e:
            logging.error(f"Error creating document: {e}")
            raise
    
    def _add_document_header(self):
        """Add document header and title"""
        # Title
        title = self.doc.add_heading('Jira Issues Analysis Report', 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Subtitle
        subtitle = self.doc.add_paragraph()
        subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = subtitle.add_run('Comprehensive Analysis with Fraud & Security Assessment')
        run.italic = True
        run.font.size = Pt(12)
        
        # Date and metadata
        meta_para = self.doc.add_paragraph()
        meta_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        meta_para.add_run(f'Generated on: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}')
        
        # Add page break
        self.doc.add_page_break()
    
    def _add_executive_summary(self, executive_summary: str):
        """Add executive summary section"""
        self.doc.add_heading('Executive Summary', level=1)
        
        para = self.doc.add_paragraph(executive_summary)
        para.space_after = Pt(12)
        
        self.doc.add_page_break()
    
    def _add_issues_analysis(self, issues_data: List[Dict]):
        """Add detailed analysis for each issue"""
        self.doc.add_heading('Detailed Issue Analysis', level=1)
        
        for i, issue in enumerate(issues_data, 1):
            # Issue header
            issue_heading = self.doc.add_heading(f"{i}. {issue['key']}: {issue['summary']}", level=2)
            
            # Basic Information Table
            self._add_issue_info_table(issue)
            
            # LLM Analysis
            if 'llm_summary' in issue:
                self.doc.add_heading('Analysis Summary', level=3)
                self.doc.add_paragraph(issue['llm_summary'])
            
            # Fraud & Security Analysis
            if 'fraud_security_analysis' in issue:
                self.doc.add_heading('Fraud & Security Assessment', level=3)
                self.doc.add_paragraph(issue['fraud_security_analysis'])
            
            # Attachments
            if issue.get('attachments'):
                self.doc.add_heading('Attachments', level=3)
                for att in issue['attachments']:
                    att_para = self.doc.add_paragraph()
                    att_para.add_run(f"• {att['filename']} ").bold = True
                    att_para.add_run(f"(Size: {att['size']} bytes, Created: {att['created']})")
                    if att.get('content_summary'):
                        self.doc.add_paragraph(f"   Content: {att['content_summary']}")
            
            # Add separator
            if i < len(issues_data):
                self.doc.add_paragraph("_" * 80)
                self.doc.add_paragraph()
    
    def _add_issue_info_table(self, issue: Dict):
        """Add issue information table"""
        table = self.doc.add_table(rows=0, cols=2)
        table.style = 'Table Grid'
        
        info_items = [
            ('Issue Key', issue['key']),
            ('Status', issue.get('status', 'Unknown')),
            ('Priority', issue.get('priority', 'Not Set')),
            ('Type', issue.get('issue_type', 'Unknown')),
            ('Assignee', issue.get('assignee', 'Unassigned')),
            ('Reporter', issue.get('reporter', 'Unknown')),
            ('Created', issue.get('created', 'Unknown')),
            ('Updated', issue.get('updated', 'Unknown'))
        ]
        
        for label, value in info_items:
            row = table.add_row()
            row.cells[0].text = label
            row.cells[0].paragraphs[0].runs[0].bold = True
            row.cells[1].text = str(value)
        
        self.doc.add_paragraph()
        
        # Description
        if issue.get('description'):
            self.doc.add_heading('Description', level=3)
            self.doc.add_paragraph(issue['description'])
        
        # Acceptance Criteria
        if issue.get('acceptance_criteria') and issue['acceptance_criteria'] != "No acceptance criteria found":
            self.doc.add_heading('Acceptance Criteria', level=3)
            self.doc.add_paragraph(issue['acceptance_criteria'])
    
    def _add_consolidated_security_analysis(self, issues_data: List[Dict]):
        """Add consolidated fraud and security analysis"""
        self.doc.add_page_break()
        self.doc.add_heading('Consolidated Fraud & Security Analysis', level=1)
        
        # Summary of all security concerns
        high_risk_issues = []
        medium_risk_issues = []
        low_risk_issues = []
        
        for issue in issues_data:
            fraud_analysis = issue.get('fraud_security_analysis', '')
            if 'high' in fraud_analysis.lower() and 'risk' in fraud_analysis.lower():
                high_risk_issues.append(issue['key'])
            elif 'medium' in fraud_analysis.lower() and 'risk' in fraud_analysis.lower():
                medium_risk_issues.append(issue['key'])
            else:
                low_risk_issues.append(issue['key'])
        
        # Risk Summary
        self.doc.add_heading('Risk Level Summary', level=2)
        
        if high_risk_issues:
            self.doc.add_paragraph().add_run('High Risk Issues: ').bold = True
            self.doc.add_paragraph(', '.join(high_risk_issues))
        
        if medium_risk_issues:
            self.doc.add_paragraph().add_run('Medium Risk Issues: ').bold = True
            self.doc.add_paragraph(', '.join(medium_risk_issues))
        
        if low_risk_issues:
            self.doc.add_paragraph().add_run('Low Risk Issues: ').bold = True
            self.doc.add_paragraph(', '.join(low_risk_issues))
        
        # Detailed Security Recommendations
        self.doc.add_heading('Security Recommendations', level=2)
        
        recommendations = [
            "Implement comprehensive input validation and sanitization",
            "Ensure proper authentication and authorization controls",
            "Regular security testing and vulnerability assessments",
            "Implement fraud detection and monitoring systems",
            "Ensure compliance with relevant data protection regulations",
            "Establish incident response procedures",
            "Regular security awareness training for development team"
        ]
        
        for rec in recommendations:
            para = self.doc.add_paragraph()
            para.add_run('• ').bold = True
            para.add_run(rec)
    
    def _add_appendix(self, issues_data: List[Dict]):
        """Add appendix with additional information"""
        self.doc.add_page_break()
        self.doc.add_heading('Appendix', level=1)
        
        # Issue Summary Table
        self.doc.add_heading('Issues Summary Table', level=2)
        
        table = self.doc.add_table(rows=1, cols=4)
        table.style = 'Table Grid'
        
        # Header row
        header_cells = table.rows[0].cells
        header_cells[0].text = 'Issue Key'
        header_cells[1].text = 'Summary'
        header_cells[2].text = 'Status'
        header_cells[3].text = 'Priority'
        
        for cell in header_cells:
            cell.paragraphs[0].runs[0].bold = True
        
        # Data rows
        for issue in issues_data:
            row = table.add_row()
            row.cells[0].text = issue['key']
            row.cells[1].text = issue['summary'][:50] + '...' if len(issue['summary']) > 50 else issue['summary']
            row.cells[2].text = issue.get('status', 'Unknown')
            row.cells[3].text = issue.get('priority', 'Not Set')
        
        # Generation info
        self.doc.add_paragraph()
        self.doc.add_heading('Document Information', level=2)
        info_para = self.doc.add_paragraph()
        info_para.add_run('Total Issues Analyzed: ').bold = True
        info_para.add_run(str(len(issues_data)))
        
        info_para = self.doc.add_paragraph()
        info_para.add_run('Generated by: ').bold = True
        info_para.add_run('Jira Analysis System')
        
        info_para = self.doc.add_paragraph()
        info_para.add_run('Analysis Date: ').bold = True
        info_para.add_run(datetime.now().strftime("%B %d, %Y"))
    
    def generate_html_summary(self, issues_data: List[Dict], executive_summary: str) -> str:
        """Generate HTML summary for email"""
        html_content = f"""
        <html>
        <head>
            <title>Jira Analysis Report Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; }}
                .summary {{ background-color: #e8f4f8; padding: 15px; margin: 15px 0; border-radius: 5px; }}
                .issue {{ border-left: 4px solid #007cba; padding-left: 15px; margin: 10px 0; }}
                .high-risk {{ border-left-color: #d32f2f; }}
                .medium-risk {{ border-left-color: #f57c00; }}
                .low-risk {{ border-left-color: #388e3c; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Jira Issues Analysis Report</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime("%B %d, %Y at %I:%M %p")}</p>
                <p><strong>Total Issues Analyzed:</strong> {len(issues_data)}</p>
            </div>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <p>{executive_summary.replace(chr(10), '<br>')}</p>
            </div>
            
            <h2>Issues Overview</h2>
            <table>
                <tr>
                    <th>Issue Key</th>
                    <th>Summary</th>
                    <th>Status</th>
                    <th>Priority</th>
                    <th>Risk Level</th>
                </tr>
        """
        
        for issue in issues_data:
            risk_level = "Low"
            risk_class = "low-risk"
            
            fraud_analysis = issue.get('fraud_security_analysis', '').lower()
            if 'high' in fraud_analysis and 'risk' in fraud_analysis:
                risk_level = "High"
                risk_class = "high-risk"
            elif 'medium' in fraud_analysis and 'risk' in fraud_analysis:
                risk_level = "Medium"
                risk_class = "medium-risk"
            
            html_content += f"""
                <tr class="{risk_class}">
                    <td>{issue['key']}</td>
                    <td>{issue['summary'][:60] + '...' if len(issue['summary']) > 60 else issue['summary']}</td>
                    <td>{issue.get('status', 'Unknown')}</td>
                    <td>{issue.get('priority', 'Not Set')}</td>
                    <td>{risk_level}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <div class="summary">
                <h3>Key Recommendations</h3>
                <ul>
                    <li>Review high-risk issues immediately</li>
                    <li>Implement recommended security controls</li>
                    <li>Conduct regular security assessments</li>
                    <li>Ensure compliance with data protection requirements</li>
                </ul>
            </div>
            
            <p><em>Please find the detailed analysis report attached to this email.</em></p>
        </body>
        </html>
        """
        
        return html_content







"""
Email Sender for sending reports
"""
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
from typing import List
from config import SMTP_SERVER, SMTP_PORT, SENDER_EMAIL, SENDER_PASSWORD

class EmailSender:
    def __init__(self, smtp_server: str = None, smtp_port: int = None, 
                 sender_email: str = None, sender_password: str = None):
        self.smtp_server = smtp_server or SMTP_SERVER
        self.smtp_port = smtp_port or SMTP_PORT
        self.sender_email = sender_email or SENDER_EMAIL
        self.sender_password = sender_password or SENDER_PASSWORD
    
    def send_report_email(self, recipient_email: str, html_content: str, 
                         attachment_path: str, jira_keys: List[str]) -> bool:
        """
        Send the analysis report via email
        """
        try:
            # Create message
            message = MIMEMultipart('mixed')
            message['From'] = self.sender_email
            message['To'] = recipient_email
            message['Subject'] = f'Jira Analysis Report - {", ".join(jira_keys)}'
            
            # Create HTML part
            html_part = MIMEText(html_content, 'html')
            message.attach(html_part)
            
            # Add attachment
            if os.path.exists(attachment_path):
                with open(attachment_path, "rb") as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {os.path.basename(attachment_path)}'
                )
                message.attach(part)
            else:
                logging.warning(f"Attachment file not found: {attachment_path}")
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            
            text = message.as_string()
            server.sendmail(self.sender_email, recipient_email, text)
            server.quit()
            
            logging.info(f"Email sent successfully to {recipient_email}")
            return True
            
        except Exception as e:
            logging.error(f"Error sending email: {e}")
            return False
    
    def validate_email_config(self) -> bool:
        """Validate email configuration"""
        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.quit()
            return True
        except Exception as e:
            logging.error(f"Email configuration validation failed: {e}")
            return False



"""
Main script for Jira Analysis System
"""
import logging
import os
import sys
from typing import List
import re
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('jira_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Import custom modules
from jira_client import JiraClient
from llm_analyzer import LLMAnalyzer
from document_generator import DocumentGenerator
from email_sender import EmailSender
from config import JIRA_URL, JIRA_USERNAME, JIRA_API_TOKEN, OPENAI_API_KEY

class JiraAnalysisSystem:
    def __init__(self):
        self.jira_client = None
        self.llm_analyzer = None
        self.doc_generator = None
        self.email_sender = None
        
    def initialize_components(self):
        """Initialize all system components"""
        try:
            # Initialize Jira client
            if not all([JIRA_URL, JIRA_USERNAME, JIRA_API_TOKEN]):
                raise ValueError("Jira configuration is incomplete. Please check your .env file.")
            
            self.jira_client = JiraClient(JIRA_URL, JIRA_USERNAME, JIRA_API_TOKEN)
            if not self.jira_client.validate_connection():
                raise ConnectionError("Failed to connect to Jira")
            
            # Initialize LLM analyzer
            if not OPENAI_API_KEY:
                raise ValueError("OpenAI API key is missing. Please check your .env file.")
            
            self.llm_analyzer = LLMAnalyzer(OPENAI_API_KEY)
            
            # Initialize document generator
            self.doc_generator = DocumentGenerator()
            
            # Initialize email sender
            self.email_sender = EmailSender()
            
            logging.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize components: {e}")
            return False
    
    def get_jira_keys_from_user(self) -> List[str]:
        """Get Jira keys from user input"""
        while True:
            try:
                print("\n" + "="*60)
                print("JIRA ANALYSIS SYSTEM")
                print("="*60)
                
                jira_input = input("\nEnter Jira keys (comma-separated): ").strip()
                
                if not jira_input:
                    print("❌ Please enter at least one Jira key.")
                    continue
                
                # Split by comma and clean up
                jira_keys = [key.strip().upper() for key in jira_input.split(',')]
                
                # Validate format (basic validation)
                invalid_keys = []
                valid_keys = []
                
                for key in jira_keys:
                    if re.match(r'^[A-Z]+-\d+$', key):
                        valid_keys.append(key)
                    else:
                        invalid_keys.append(key)
                
                if invalid_keys:
                    print(f"❌ Invalid Jira key format: {', '.join(invalid_keys)}")
                    print("   Jira keys should be in format: PROJECT-123")
                    continue
                
                if valid_keys:
                    print(f"✅ Valid Jira keys found: {', '.join(valid_keys)}")
                    return valid_keys
                
            except KeyboardInterrupt:
                print("\n❌ Operation cancelled by user.")
                sys.exit(0)
            except Exception as e:
                print(f"❌ Error processing input: {e}")
                continue
    
    def get_email_from_user(self) -> str:
        """Get recipient email from user input"""
        while True:
            try:
                email = input("\nEnter recipient email address: ").strip()
                
                if not email:
                    print("❌ Please enter an email address.")
                    continue
                
                # Basic email validation
                if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
                    print(f"✅ Email address validated: {email}")
                    return email
                else:
                    print("❌ Invalid email format. Please enter a valid email address.")
                    continue
                    
            except KeyboardInterrupt:
                print("\n❌ Operation cancelled by user.")
                sys.exit(0)
            except Exception as e:
                print(f"❌ Error processing email: {e}")
                continue
    
    def process_jira_issues(self, jira_keys: List[str]) -> List[dict]:
        """Process all Jira issues and generate analysis"""
        processed_issues = []
        
        print(f"\n📋 Processing {len(jira_keys)} Jira issue(s)...")
        
        for i, key in enumerate(jira_keys, 1):
            try:
                print(f"   [{i}/{len(jira_keys)}] Processing {key}...")
                
                # Extract issue details
                issue_data = self.jira_client.extract_issue_details(key)
                
                if 'error' in issue_data:
                    print(f"   ⚠️  Warning: Could not fully process {key}: {issue_data['error']}")
                    processed_issues.append(issue_data)
                    continue
                
                print(f"   ✅ Extracted details for {key}")
                
                # Generate LLM summary
                print(f"   🤖 Generating AI analysis for {key}...")
                llm_summary = self.llm_analyzer.analyze_issue_summary(issue_data)
                issue_data['llm_summary'] = llm_summary
                
                # Generate fraud & security analysis
                print(f"   🔒 Analyzing fraud & security implications for {key}...")
                fraud_security = self.llm_analyzer.analyze_fraud_security(issue_data)
                issue_data['fraud_security_analysis'] = fraud_security
                
                processed_issues.append(issue_data)
                print(f"   ✅ Completed analysis for {key}")
                
            except Exception as e:
                logging.error(f"Error processing {key}: {e}")
                print(f"   ❌ Error processing {key}: {e}")
                
                # Add error issue to maintain the list
                error_issue = {
                    'key': key,
                    'summary': f"Error processing {key}",
                    'description': f"Failed to process issue: {str(e)}",
                    'error': str(e),
                    'llm_summary': f"Could not analyze {key} due to processing error.",
                    'fraud_security_analysis': "Could not perform security analysis due to processing error."
                }
                processed_issues.append(error_issue)
        
        return processed_issues
    
    def generate_report(self, issues_data: List[dict], jira_keys: List[str]) -> tuple:
        """Generate Word document and HTML summary"""
        try:
            print("\n📄 Generating comprehensive report...")
            
            # Generate executive summary
            print("   🎯 Creating executive summary...")
            executive_summary = self.llm_analyzer.generate_executive_summary(issues_data)
            
            # Generate Word document
            print("   📋 Creating Word document...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            doc_filename = f"jira_analysis_report_{timestamp}.docx"
            
            document_path = self.doc_generator.create_comprehensive_report(
                issues_data, executive_summary, doc_filename
            )
            
            # Generate HTML summary for email
            print("   📧 Creating HTML email summary...")
            html_summary = self.doc_generator.generate_html_summary(issues_data, executive_summary)
            
            print(f"   ✅ Report generated successfully: {doc_filename}")
            return document_path, html_summary
            
        except Exception as e:
            logging.error(f"Error generating report: {e}")
            raise
    
    def send_email_report(self, recipient_email: str, html_content: str, 
                         document_path: str, jira_keys: List[str]) -> bool:
        """Send the report via email"""
        try:
            print(f"\n📧 Sending report to {recipient_email}...")
            
            # Validate email configuration
            if not self.email_sender.validate_email_config():
                print("   ❌ Email configuration validation failed.")
                return False
            
            # Send email
            success = self.email_sender.send_report_email(
                recipient_email, html_content, document_path, jira_keys
            )
            
            if success:
                print("   ✅ Email sent successfully!")
                return True
            else:
                print("   ❌ Failed to send email.")
                return False
                
        except Exception as e:
            logging.error(f"Error sending email: {e}")
            print(f"   ❌ Error sending email: {e}")
            return False
    
    def run(self):
        """Main execution method"""
        try:
            print("🚀 Starting Jira Analysis System...")
            
            # Initialize components
            if not self.initialize_components():
                print("❌ Failed to initialize system components. Please check your configuration.")
                return False
            
            # Get user inputs
            jira_keys = self.get_jira_keys_from_user()
            recipient_email = self.get_email_from_user()
            
            # Process issues
            issues_data = self.process_jira_issues(jira_keys)
            
            if not issues_data:
                print("❌ No issues were successfully processed.")
                return False
            
            # Generate report
            document_path, html_summary = self.generate_report(issues_data, jira_keys)
            
            # Send email
            email_success = self.send_email_report(recipient_email, html_summary, document_path, jira_keys)
            
            # Final summary
            print("\n" + "="*60)
            print("ANALYSIS COMPLETE")
            print("="*60)
            print(f"📊 Issues Processed: {len(issues_data)}")
            print(f"📄 Report Generated: {document_path}")
            print(f"📧 Email Status: {'✅ Sent' if email_success else '❌ Failed'}")
            
            if not email_success:
                print(f"📁 You can find the report at: {os.path.abspath(document_path)}")
            
            print("\n🎉 Jira Analysis System completed successfully!")
            return True
            
        except KeyboardInterrupt:
            print("\n❌ Operation cancelled by user.")
            return False
        except Exception as e:
            logging.error(f"System error: {e}")
            print(f"❌ System error: {e}")
            return False

def main():
    """Main entry point"""
    system = JiraAnalysisSystem()
    success = system.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()





# Jira Configuration
JIRA_URL=https://yourcompany.atlassian.net
JIRA_USERNAME=your-email@company.com
JIRA_API_TOKEN=your-jira-api-token

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key

# Email Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your-email@gmail.com
SENDER_PASSWORD=your-app-password
