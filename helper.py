"""
Jira Client for extracting issue details using Bearer Token authentication
"""
import requests
import json
from bs4 import BeautifulSoup
import logging
from typing import Dict, List, Optional

class JiraClient:
    def __init__(self, url: str, bearer_token: str):
        self.url = url.rstrip('/')
        self.bearer_token = bearer_token
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {bearer_token}',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
        # Test connection
        self._validate_connection()
    
    def _validate_connection(self):
        """Validate the connection to Jira"""
        try:
            response = self.session.get(f'{self.url}/rest/api/3/myself')
            response.raise_for_status()
            user_info = response.json()
            logging.info(f"Successfully connected to Jira as: {user_info.get('displayName', 'Unknown')}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to connect to Jira: {e}")
            raise ConnectionError(f"Cannot connect to Jira: {e}")
    
    def extract_issue_details(self, issue_key: str) -> Dict:
        """
        Extract comprehensive details from a Jira issue using REST API
        """
        try:
            # Get issue details with expanded fields
            url = f'{self.url}/rest/api/3/issue/{issue_key}'
            params = {
                'expand': 'attachment,changelog,renderedFields'
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            issue_data = response.json()
            fields = issue_data.get('fields', {})
            
            # Extract basic details
            summary = fields.get('summary', '')
            description = self._clean_html(fields.get('description', {}).get('content', '') if isinstance(fields.get('description'), dict) else str(fields.get('description', '')))
            
            # Extract acceptance criteria
            acceptance_criteria = self._extract_acceptance_criteria(fields, description)
            
            # Extract attachments
            attachments_info = self._extract_attachments(fields.get('attachment', []))
            
            return {
                'key': issue_key,
                'summary': summary,
                'description': description,
                'acceptance_criteria': acceptance_criteria,
                'attachments': attachments_info,
                'status': fields.get('status', {}).get('name', 'Unknown'),
                'assignee': fields.get('assignee', {}).get('displayName', 'Unassigned') if fields.get('assignee') else 'Unassigned',
                'reporter': fields.get('reporter', {}).get('displayName', 'Unknown') if fields.get('reporter') else 'Unknown',
                'created': fields.get('created', 'Unknown'),
                'updated': fields.get('updated', 'Unknown'),
                'priority': fields.get('priority', {}).get('name', 'Not Set') if fields.get('priority') else 'Not Set',
                'issue_type': fields.get('issuetype', {}).get('name', 'Unknown'),
                'labels': fields.get('labels', []),
                'components': [comp.get('name', '') for comp in fields.get('components', [])],
                'fix_versions': [ver.get('name', '') for ver in fields.get('fixVersions', [])]
            }
            
        except requests.exceptions.RequestException as e:
            logging.error(f"HTTP error extracting details for {issue_key}: {e}")
            return self._create_error_response(issue_key, f"HTTP error: {e}")
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error for {issue_key}: {e}")
            return self._create_error_response(issue_key, f"JSON decode error: {e}")
        except Exception as e:
            logging.error(f"Unexpected error extracting details for {issue_key}: {e}")
            return self._create_error_response(issue_key, f"Unexpected error: {e}")
    
    def _create_error_response(self, issue_key: str, error_msg: str) -> Dict:
        """Create a standardized error response"""
        return {
            'key': issue_key,
            'error': error_msg,
            'summary': f"Error fetching {issue_key}",
            'description': f"Failed to fetch issue details: {error_msg}",
            'acceptance_criteria': "Could not extract acceptance criteria",
            'attachments': [],
            'status': 'Unknown',
            'assignee': 'Unknown',
            'reporter': 'Unknown',
            'created': 'Unknown',
            'updated': 'Unknown',
            'priority': 'Unknown',
            'issue_type': 'Unknown',
            'labels': [],
            'components': [],
            'fix_versions': []
        }
    
    def _clean_html(self, text) -> str:
        """Clean HTML content from Jira fields"""
        if not text:
            return ""
        
        try:
            # Handle Atlassian Document Format (ADF)
            if isinstance(text, dict):
                return self._extract_text_from_adf(text)
            elif isinstance(text, list):
                return '\n'.join([self._extract_text_from_adf(item) if isinstance(item, dict) else str(item) for item in text])
            else:
                # Handle HTML content
                soup = BeautifulSoup(str(text), 'html.parser')
                return soup.get_text(strip=True)
        except Exception as e:
            logging.warning(f"Error cleaning HTML content: {e}")
            return str(text) if text else ""
    
    def _extract_text_from_adf(self, adf_content: dict) -> str:
        """Extract text from Atlassian Document Format"""
        try:
            if not isinstance(adf_content, dict):
                return str(adf_content)
            
            text_parts = []
            
            def extract_text_recursive(node):
                if isinstance(node, dict):
                    if node.get('type') == 'text':
                        text_parts.append(node.get('text', ''))
                    elif 'content' in node:
                        for child in node['content']:
                            extract_text_recursive(child)
                elif isinstance(node, list):
                    for item in node:
                        extract_text_recursive(item)
            
            extract_text_recursive(adf_content)
            return ' '.join(text_parts).strip()
            
        except Exception as e:
            logging.warning(f"Error extracting text from ADF: {e}")
            return str(adf_content)
    
    def _extract_acceptance_criteria(self, fields: dict, description: str) -> str:
        """
        Extract acceptance criteria from various possible fields
        """
        acceptance_criteria = ""
        
        try:
            # Check common custom fields for acceptance criteria
            custom_field_patterns = [
                'customfield_10100', 'customfield_10200', 'customfield_10300',
                'customfield_10400', 'customfield_10500', 'customfield_11000',
                'customfield_12000', 'customfield_13000'
            ]
            
            for field_name in custom_field_patterns:
                if field_name in fields and fields[field_name]:
                    field_value = fields[field_name]
                    
                    # Handle different field types
                    if isinstance(field_value, dict):
                        field_text = self._extract_text_from_adf(field_value)
                    elif isinstance(field_value, str):
                        field_text = self._clean_html(field_value)
                    else:
                        field_text = str(field_value)
                    
                    # Check if this field contains acceptance criteria
                    if field_text and any(keyword in field_text.lower() for keyword in ['acceptance', 'criteria', 'ac:', 'given', 'when', 'then']):
                        acceptance_criteria = field_text
                        break
            
            # If not found in custom fields, look in description
            if not acceptance_criteria and description:
                lines = description.split('\n')
                
                capture = False
                ac_lines = []
                
                for line in lines:
                    line_clean = line.strip()
                    line_lower = line_clean.lower()
                    
                    # Look for acceptance criteria headers
                    if any(keyword in line_lower for keyword in ['acceptance criteria', 'acceptance', 'ac:', 'given when then']):
                        capture = True
                        if ':' in line_clean:
                            ac_part = line_clean.split(':', 1)[1].strip()
                            if ac_part:
                                ac_lines.append(ac_part)
                        continue
                    
                    if capture:
                        if line_clean and not any(stop_word in line_lower for stop_word in ['description', 'summary', 'notes', 'background']):
                            ac_lines.append(line_clean)
                        elif not line_clean and ac_lines:
                            # Empty line might indicate end of AC section
                            break
                
                if ac_lines:
                    acceptance_criteria = '\n'.join(ac_lines)
        
        except Exception as e:
            logging.warning(f"Error extracting acceptance criteria: {e}")
        
        return acceptance_criteria or "No acceptance criteria found"
    
    def _extract_attachments(self, attachments: List[dict]) -> List[Dict]:
        """
        Extract attachment information and content
        """
        attachments_info = []
        
        try:
            for attachment in attachments:
                att_info = {
                    'filename': attachment.get('filename', 'Unknown'),
                    'size': attachment.get('size', 0),
                    'created': attachment.get('created', 'Unknown'),
                    'author': attachment.get('author', {}).get('displayName', 'Unknown'),
                    'content_summary': "",
                    'mime_type': attachment.get('mimeType', 'Unknown')
                }
                
                # Try to read attachment content if it's a text file and small enough
                content_url = attachment.get('content')
                if content_url and att_info['size'] < 50000:  # Only for files smaller than 50KB
                    try:
                        if any(file_type in att_info['mime_type'].lower() for file_type in ['text', 'json', 'xml']):
                            response = self.session.get(content_url)
                            if response.status_code == 200:
                                content = response.text
                                att_info['content_summary'] = content[:500] + "..." if len(content) > 500 else content
                    except Exception as e:
                        att_info['content_summary'] = f"Could not read file content: {e}"
                
                attachments_info.append(att_info)
        
        except Exception as e:
            logging.warning(f"Error extracting attachments: {e}")
        
        return attachments_info
    
    def validate_connection(self) -> bool:
        """Validate Jira connection"""
        try:
            response = self.session.get(f'{self.url}/rest/api/3/myself')
            response.raise_for_status()
            return True
        except Exception as e:
            logging.error(f"Jira connection validation failed: {e}")
            return False
    
    def get_issue_exists(self, issue_key: str) -> bool:
        """Check if an issue exists"""
        try:
            response = self.session.get(f'{self.url}/rest/api/3/issue/{issue_key}')
            return response.status_code == 200
        except:
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
from config import JIRA_URL, JIRA_BEARER_TOKEN, OPENAI_API_KEY

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
            if not all([JIRA_URL, JIRA_BEARER_TOKEN]):
                raise ValueError("Jira configuration is incomplete. Please check your .env file.")
            
            self.jira_client = JiraClient(JIRA_URL, JIRA_BEARER_TOKEN)
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



requests==2.31.0
python-docx==0.8.11
openai==1.3.7
smtplib-ssl==1.0.0
email-validator==2.1.0
python-dotenv==1.0.0
beautifulsoup4==4.12.2
lxml==4.9.4


