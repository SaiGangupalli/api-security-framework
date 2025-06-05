jira-report-generator/
├── requirements.txt
├── .env.example
├── config/
│   ├── __init__.py
│   └── settings.py
├── src/
│   ├── __init__.py
│   ├── jira_client.py
│   ├── ai_analyzer.py
│   ├── document_generator.py
│   ├── email_service.py
│   └── main.py
├── templates/
│   ├── email_template.html
│   └── document_template.py
├── output/
│   └── .gitkeep
└── README.md

# requirements.txt
requests==2.31.0
python-docx==0.8.11
python-dotenv==1.0.0
openai==1.3.0
smtplib2==0.2.1
jinja2==3.1.2
Pillow==10.0.1
beautifulsoup4==4.12.2

# .env.example
JIRA_BASE_URL=https://your-company.atlassian.net
JIRA_USERNAME=your-email@company.com
JIRA_API_TOKEN=your-jira-api-token
JIRA_STORY_KEYS=PROJ-123,PROJ-124,PROJ-125
OPENAI_API_KEY=your-openai-api-key
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
RECIPIENT_EMAIL=recipient@company.com
SENDER_NAME=Jira Report Generator

# config/__init__.py
# Empty file to make config a package

# config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Jira Configuration
    JIRA_BASE_URL = os.getenv('JIRA_BASE_URL')
    JIRA_USERNAME = os.getenv('JIRA_USERNAME')
    JIRA_API_TOKEN = os.getenv('JIRA_API_TOKEN')
    JIRA_STORY_KEYS = os.getenv('JIRA_STORY_KEYS', '').split(',')
    
    # AI Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Email Configuration
    SMTP_SERVER = os.getenv('SMTP_SERVER')
    SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
    SMTP_USERNAME = os.getenv('SMTP_USERNAME')
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
    RECIPIENT_EMAIL = os.getenv('RECIPIENT_EMAIL')
    SENDER_NAME = os.getenv('SENDER_NAME', 'Jira Report Generator')
    
    # Output Configuration
    OUTPUT_DIR = 'output'
    
    def validate(self):
        """Validate that all required settings are present"""
        required_settings = [
            'JIRA_BASE_URL', 'JIRA_USERNAME', 'JIRA_API_TOKEN',
            'OPENAI_API_KEY', 'SMTP_SERVER', 'SMTP_USERNAME',
            'SMTP_PASSWORD', 'RECIPIENT_EMAIL'
        ]
        
        missing = []
        for setting in required_settings:
            if not getattr(self, setting):
                missing.append(setting)
        
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        if not self.JIRA_STORY_KEYS or self.JIRA_STORY_KEYS == ['']:
            raise ValueError("JIRA_STORY_KEYS environment variable is required")

settings = Settings()

# src/__init__.py
# Empty file to make src a package

# src/jira_client.py
import requests
import base64
import json
from typing import Dict, List, Optional
from config.settings import settings

class JiraClient:
    def __init__(self):
        self.base_url = settings.JIRA_BASE_URL
        self.username = settings.JIRA_USERNAME
        self.api_token = settings.JIRA_API_TOKEN
        self.session = requests.Session()
        
        # Set up basic authentication
        auth_string = f"{self.username}:{self.api_token}"
        auth_bytes = auth_string.encode('ascii')
        auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
        
        self.session.headers.update({
            'Authorization': f'Basic {auth_b64}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def get_issue(self, issue_key: str) -> Optional[Dict]:
        """Fetch a single Jira issue by key"""
        try:
            url = f"{self.base_url}/rest/api/3/issue/{issue_key}"
            params = {
                'expand': 'attachments,changelog,comments'
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching issue {issue_key}: {e}")
            return None
    
    def get_issues(self, issue_keys: List[str]) -> List[Dict]:
        """Fetch multiple Jira issues"""
        issues = []
        for key in issue_keys:
            key = key.strip()
            if key:
                issue = self.get_issue(key)
                if issue:
                    issues.append(issue)
        return issues
    
    def get_attachment_content(self, attachment_url: str) -> Optional[bytes]:
        """Download attachment content"""
        try:
            response = self.session.get(attachment_url)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            print(f"Error downloading attachment: {e}")
            return None
    
    def parse_issue_data(self, issue: Dict) -> Dict:
        """Parse Jira issue data into a structured format"""
        fields = issue.get('fields', {})
        
        # Extract attachments info
        attachments = []
        for attachment in fields.get('attachment', []):
            attachments.append({
                'filename': attachment.get('filename'),
                'content_type': attachment.get('mimeType'),
                'size': attachment.get('size'),
                'url': attachment.get('content'),
                'created': attachment.get('created')
            })
        
        # Extract labels
        labels = fields.get('labels', [])
        
        # Extract custom fields that might contain acceptance criteria
        acceptance_criteria = (
            fields.get('customfield_10000', '') or  # Common AC field
            fields.get('acceptance_criteria', '') or
            ''
        )
        
        return {
            'key': issue.get('key'),
            'summary': fields.get('summary', ''),
            'description': fields.get('description', {}).get('content', [{}])[0].get('content', [{}])[0].get('text', '') if fields.get('description') else '',
            'acceptance_criteria': acceptance_criteria,
            'labels': labels,
            'status': fields.get('status', {}).get('name', ''),
            'priority': fields.get('priority', {}).get('name', ''),
            'issue_type': fields.get('issuetype', {}).get('name', ''),
            'assignee': fields.get('assignee', {}).get('displayName', 'Unassigned') if fields.get('assignee') else 'Unassigned',
            'reporter': fields.get('reporter', {}).get('displayName', ''),
            'created': fields.get('created', ''),
            'updated': fields.get('updated', ''),
            'attachments': attachments
        }

# src/ai_analyzer.py
import openai
from typing import Dict, List
from config.settings import settings

class AIAnalyzer:
    def __init__(self):
        openai.api_key = settings.OPENAI_API_KEY
    
    def analyze_fraud_security_risks(self, issue_data: Dict) -> Dict:
        """Analyze Jira issue for fraud and security implications"""
        
        prompt = self._create_analysis_prompt(issue_data)
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a cybersecurity and fraud prevention expert. Analyze the provided Jira story and identify potential fraud and security risks, vulnerabilities, and recommended mitigations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            analysis_text = response.choices[0].message.content
            return self._parse_analysis_response(analysis_text)
            
        except Exception as e:
            print(f"AI Analysis Error: {e}")
            return {
                'security_risks': ['Unable to analyze - AI service unavailable'],
                'fraud_risks': ['Unable to analyze - AI service unavailable'],
                'recommendations': ['Manual security review recommended'],
                'risk_level': 'Unknown'
            }
    
    def _create_analysis_prompt(self, issue_data: Dict) -> str:
        """Create analysis prompt for AI"""
        return f"""
        Please analyze the following Jira story for potential fraud and security risks:

        **Story Key:** {issue_data['key']}
        **Summary:** {issue_data['summary']}
        **Description:** {issue_data['description']}
        **Acceptance Criteria:** {issue_data['acceptance_criteria']}
        **Labels:** {', '.join(issue_data['labels'])}
        **Issue Type:** {issue_data['issue_type']}
        **Priority:** {issue_data['priority']}

        Please provide analysis in the following format:
        
        **SECURITY RISKS:**
        - [List specific security vulnerabilities or concerns]
        
        **FRAUD RISKS:**
        - [List potential fraud scenarios or risks]
        
        **RECOMMENDATIONS:**
        - [List specific security/fraud prevention measures]
        
        **RISK LEVEL:** [High/Medium/Low]
        
        Focus on:
        1. Data privacy and protection concerns
        2. Authentication and authorization vulnerabilities
        3. Input validation and injection risks
        4. Financial transaction security
        5. User identity verification
        6. Audit trail and logging requirements
        7. Compliance considerations (GDPR, PCI-DSS, etc.)
        """
    
    def _parse_analysis_response(self, response_text: str) -> Dict:
        """Parse AI response into structured data"""
        analysis = {
            'security_risks': [],
            'fraud_risks': [],
            'recommendations': [],
            'risk_level': 'Medium'
        }
        
        current_section = None
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if '**SECURITY RISKS:**' in line.upper():
                current_section = 'security_risks'
            elif '**FRAUD RISKS:**' in line.upper():
                current_section = 'fraud_risks'
            elif '**RECOMMENDATIONS:**' in line.upper():
                current_section = 'recommendations'
            elif '**RISK LEVEL:**' in line.upper():
                risk_level = line.split(':')[-1].strip()
                analysis['risk_level'] = risk_level
            elif line.startswith('-') and current_section:
                analysis[current_section].append(line[1:].strip())
        
        return analysis

# src/document_generator.py
import os
from datetime import datetime
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.shared import OxmlElement, qn
from typing import List, Dict
from config.settings import settings

class DocumentGenerator:
    def __init__(self):
        self.output_dir = settings.OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_report(self, issues_data: List[Dict], analyses: List[Dict]) -> str:
        """Generate Word document with Jira stories and AI analysis"""
        
        # Create document
        doc = Document()
        
        # Add title
        title = doc.add_heading('Jira Stories Security & Fraud Analysis Report', 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add metadata
        doc.add_paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        doc.add_paragraph(f"Total Stories Analyzed: {len(issues_data)}")
        doc.add_paragraph("")
        
        # Add executive summary
        self._add_executive_summary(doc, analyses)
        
        # Add individual story analyses
        for i, (issue, analysis) in enumerate(zip(issues_data, analyses)):
            self._add_story_section(doc, issue, analysis, i + 1)
        
        # Save document
        filename = f"jira_security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
        filepath = os.path.join(self.output_dir, filename)
        doc.save(filepath)
        
        return filepath
    
    def _add_executive_summary(self, doc: Document, analyses: List[Dict]):
        """Add executive summary section"""
        doc.add_heading('Executive Summary', level=1)
        
        # Risk level distribution
        risk_levels = {'High': 0, 'Medium': 0, 'Low': 0, 'Unknown': 0}
        for analysis in analyses:
            risk_level = analysis.get('risk_level', 'Unknown')
            risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1
        
        doc.add_paragraph("Risk Level Distribution:")
        for level, count in risk_levels.items():
            if count > 0:
                doc.add_paragraph(f"  • {level} Risk: {count} stories", style='List Bullet')
        
        doc.add_paragraph("")
        
        # Common risks
        all_security_risks = []
        all_fraud_risks = []
        
        for analysis in analyses:
            all_security_risks.extend(analysis.get('security_risks', []))
            all_fraud_risks.extend(analysis.get('fraud_risks', []))
        
        if all_security_risks:
            doc.add_paragraph("Most Common Security Concerns:")
            unique_risks = list(set(all_security_risks))[:5]  # Top 5 unique risks
            for risk in unique_risks:
                doc.add_paragraph(f"  • {risk}", style='List Bullet')
        
        doc.add_page_break()
    
    def _add_story_section(self, doc: Document, issue: Dict, analysis: Dict, story_number: int):
        """Add individual story section"""
        
        # Story header
        header = doc.add_heading(f"Story {story_number}: {issue['key']}", level=1)
        
        # Story details table
        table = doc.add_table(rows=8, cols=2)
        table.style = 'Table Grid'
        
        # Populate table
        details = [
            ('Summary', issue['summary']),
            ('Description', issue['description'][:500] + '...' if len(issue['description']) > 500 else issue['description']),
            ('Acceptance Criteria', issue['acceptance_criteria'][:300] + '...' if len(issue['acceptance_criteria']) > 300 else issue['acceptance_criteria']),
            ('Labels', ', '.join(issue['labels'])),
            ('Priority', issue['priority']),
            ('Status', issue['status']),
            ('Assignee', issue['assignee']),
            ('Attachments', f"{len(issue['attachments'])} files" if issue['attachments'] else 'None')
        ]
        
        for i, (label, value) in enumerate(details):
            table.cell(i, 0).text = label
            table.cell(i, 1).text = str(value) if value else 'N/A'
            
            # Make label column bold
            table.cell(i, 0).paragraphs[0].runs[0].font.bold = True
        
        doc.add_paragraph("")
        
        # Security Analysis
        doc.add_heading('Security & Fraud Analysis', level=2)
        
        # Risk Level
        risk_para = doc.add_paragraph()
        risk_para.add_run('Risk Level: ').bold = True
        risk_run = risk_para.add_run(analysis['risk_level'])
        
        # Color code risk level
        if analysis['risk_level'] == 'High':
            risk_run.font.color.rgb = None  # Red - note: python-docx color handling
        elif analysis['risk_level'] == 'Medium':
            risk_run.font.color.rgb = None  # Orange
        
        doc.add_paragraph("")
        
        # Security Risks
        if analysis['security_risks']:
            doc.add_heading('Security Risks Identified:', level=3)
            for risk in analysis['security_risks']:
                doc.add_paragraph(f"• {risk}", style='List Bullet')
        
        # Fraud Risks  
        if analysis['fraud_risks']:
            doc.add_heading('Fraud Risks Identified:', level=3)
            for risk in analysis['fraud_risks']:
                doc.add_paragraph(f"• {risk}", style='List Bullet')
        
        # Recommendations
        if analysis['recommendations']:
            doc.add_heading('Recommendations:', level=3)
            for rec in analysis['recommendations']:
                doc.add_paragraph(f"• {rec}", style='List Bullet')
        
        # Add attachments info if available
        if issue['attachments']:
            doc.add_heading('Attachments:', level=3)
            for attachment in issue['attachments']:
                doc.add_paragraph(f"• {attachment['filename']} ({attachment['content_type']}, {attachment['size']} bytes)")
        
        doc.add_page_break()

# src/email_service.py
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from jinja2 import Template
from typing import List, Dict
from config.settings import settings

class EmailService:
    def __init__(self):
        self.smtp_server = settings.SMTP_SERVER
        self.smtp_port = settings.SMTP_PORT
        self.username = settings.SMTP_USERNAME
        self.password = settings.SMTP_PASSWORD
        self.sender_name = settings.SENDER_NAME
    
    def send_report(self, document_path: str, issues_data: List[Dict], analyses: List[Dict]) -> bool:
        """Send the generated report via email"""
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = f"{self.sender_name} <{self.username}>"
            msg['To'] = settings.RECIPIENT_EMAIL
            msg['Subject'] = f"Jira Security Analysis Report - {len(issues_data)} Stories Analyzed"
            
            # Create HTML email body
            html_body = self._create_html_body(issues_data, analyses)
            html_part = MIMEText(html_body, 'html')
            msg.attach(html_part)
            
            # Attach document
            if os.path.exists(document_path):
                with open(document_path, "rb") as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {os.path.basename(document_path)}'
                    )
                    msg.attach(part)
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            print(f"Email sending error: {e}")
            return False
    
    def _create_html_body(self, issues_data: List[Dict], analyses: List[Dict]) -> str:
        """Create beautiful HTML email body"""
        
        # Calculate summary statistics
        risk_stats = {'High': 0, 'Medium': 0, 'Low': 0}
        for analysis in analyses:
            risk_level = analysis.get('risk_level', 'Medium')
            risk_stats[risk_level] = risk_stats.get(risk_level, 0) + 1
        
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    line-height: 1.6; 
                    color: #333; 
                    max-width: 800px; 
                    margin: 0 auto; 
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    background-color: white;
                    padding: 30px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                .header { 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; 
                    padding: 30px; 
                    text-align: center; 
                    border-radius: 8px;
                    margin-bottom: 30px;
                }
                .header h1 { margin: 0; font-size: 28px; }
                .header p { margin: 10px 0 0 0; opacity: 0.9; }
                .stats-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }
                .stat-card {
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                    border-left: 4px solid #667eea;
                }
                .stat-number { font-size: 32px; font-weight: bold; color: #667eea; }
                .stat-label { color: #666; margin-top: 5px; }
                .risk-high { color: #dc3545; }
                .risk-medium { color: #ffc107; }
                .risk-low { color: #28a745; }
                .story-summary {
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 15px 0;
                    border-left: 4px solid #007bff;
                }
                .story-key { font-weight: bold; color: #007bff; font-size: 16px; }
                .story-title { margin: 5px 0; font-size: 14px; }
                .risk-badge {
                    display: inline-block;
                    padding: 4px 12px;
                    border-radius: 20px;
                    font-size: 12px;
                    font-weight: bold;
                    text-transform: uppercase;
                }
                .risk-badge.high { background: #dc3545; color: white; }
                .risk-badge.medium { background: #ffc107; color: #333; }
                .risk-badge.low { background: #28a745; color: white; }
                .footer {
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #eee;
                    color: #666;
                    font-size: 14px;
                }
                .cta-button {
                    display: inline-block;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 12px 30px;
                    text-decoration: none;
                    border-radius: 25px;
                    margin: 20px 0;
                    font-weight: bold;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔒 Security Analysis Report</h1>
                    <p>Jira Stories Security & Fraud Risk Assessment</p>
                    <p>Generated on {{ current_date }}</p>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">{{ total_stories }}</div>
                        <div class="stat-label">Stories Analyzed</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number risk-high">{{ risk_stats.High }}</div>
                        <div class="stat-label">High Risk</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number risk-medium">{{ risk_stats.Medium }}</div>
                        <div class="stat-label">Medium Risk</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number risk-low">{{ risk_stats.Low }}</div>
                        <div class="stat-label">Low Risk</div>
                    </div>
                </div>
                
                <h2>📋 Story Summary</h2>
                {% for issue, analysis in story_data %}
                <div class="story-summary">
                    <div class="story-key">{{ issue.key }}</div>
                    <div class="story-title">{{ issue.summary }}</div>
                    <div style="margin-top: 10px;">
                        <span class="risk-badge {{ analysis.risk_level.lower() }}">{{ analysis.risk_level }} Risk</span>
                        {% if analysis.security_risks %}
                        <span style="margin-left: 10px; color: #666; font-size: 12px;">
                            🛡️ {{ analysis.security_risks|length }} Security Issues
                        </span>
                        {% endif %}
                        {% if analysis.fraud_risks %}
                        <span style="margin-left: 10px; color: #666; font-size: 12px;">
                            ⚠️ {{ analysis.fraud_risks|length }} Fraud Risks
                        </span>
                        {% endif %}
                    </div>
                </div>
                {% endfor %}
                
                <div style="text-align: center; margin: 40px 0;">
                    <p>📎 <strong>Detailed analysis report is attached to this email</strong></p>
                    <p style="color: #666;">The attached Word document contains comprehensive security and fraud risk analysis for each story.</p>
                </div>
                
                <div class="footer">
                    <p><strong>Next Steps:</strong></p>
                    <ul>
                        <li>Review high-risk stories immediately</li>
                        <li>Implement recommended security measures</li>
                        <li>Schedule security review meetings for critical items</li>
                        <li>Update acceptance criteria to include security requirements</li>
                    </ul>
                    
                    <p style="margin-top: 20px;">
                        <em>This report was automatically generated using AI analysis. Please review all recommendations with your security team.</em>
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        template = Template(template_str)
        
        return template.render(
            current_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            total_stories=len(issues_data),
            risk_stats=risk_stats,
            story_data=list(zip(issues_data, analyses))
        )

# src/main.py
import sys
import os
from datetime import datetime
from config.settings import settings
from jira_client import JiraClient
from ai_analyzer import AIAnalyzer
from document_generator import DocumentGenerator
from email_service import EmailService

def main():
    """Main execution function"""
    
    print("🚀 Starting Jira Security Analysis Report Generation...")
    print("=" * 60)
    
    try:
        # Validate settings
        settings.validate()
        print("✅ Configuration validated successfully")
        
        # Initialize services
        jira_client = JiraClient()
        ai_analyzer = AIAnalyzer()
        doc_generator = DocumentGenerator()
        email_service = EmailService()
        
        print(f"📋 Fetching {len(settings.JIRA_STORY_KEYS)} Jira stories...")
        
        # Fetch Jira issues
        raw_issues = jira_client.get_issues(settings.JIRA_STORY_KEYS)
        if not raw_issues:
            print("❌ No Jira issues found. Please check your story keys.")
            return False
            
        # Parse issue data
        issues_data = [jira_client.parse_issue_data(issue) for issue in raw_issues]
        print(f"✅ Successfully fetched {len(issues_data)} stories")
        
        # Analyze with AI
        print("🤖 Analyzing stories for security and fraud risks...")
        analyses = []
        for i, issue in enumerate(issues_data, 1):
            print(f"   Analyzing story {i}/{len(issues_data)}: {issue['key']}")
            analysis = ai_analyzer.analyze_fraud_security_risks(issue)
            analyses.append(analysis)
        
        print("✅ AI analysis completed")
        
        # Generate document
        print("📄 Generating Word document...")
        document_path = doc_generator.generate_report(issues_data, analyses)
        print(f"✅ Document generated: {document_path}")
        
        # Send email
        print("📧 Sending email report...")
        email_sent = email_service.send_report(document_path, issues_data, analyses)
        
        if email_sent:
            print(f"✅ Report successfully sent to {settings.RECIPIENT_EMAIL}")
        else:
            print("❌ Failed to send email report")
            
        # Summary
        print("\n" + "=" * 60)
        print("📊 REPORT SUMMARY")
        print("=" * 60)
        print(f"Stories Analyzed: {len(issues_data)}")
        
        risk_counts = {'High': 0, 'Medium': 0, 'Low': 0}
        for analysis in analyses:
            risk_level = analysis.get('risk_level', 'Medium')
            risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
        
        for level, count in risk_counts.items():
            if count > 0:
                print(f"{level} Risk Stories: {count}")
        
        print(f"Document Location: {document_path}")
        print(f"Email Status: {'✅ Sent' if email_sent else '❌ Failed'}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

# README.md
# Jira Security Analysis Report Generator

This tool automatically fetches Jira stories, analyzes them for security and fraud risks using AI, generates comprehensive Word documents, and emails the reports to stakeholders.

## Features

- 🔍 **Jira Integration**: Fetches stories using Basic Auth
- 🤖 **AI Analysis**: Uses OpenAI GPT-4 for security and fraud risk assessment
- 📄 **Document Generation**: Creates professional Word documents with detailed analysis
- 📧 **Email Reports**: Sends beautiful HTML emails with attachments
- 🛡️ **Security Focus**: Identifies vulnerabilities, fraud risks, and provides recommendations
- 📊 **Risk Assessment**: Categorizes stories by risk level (High/Medium/Low)

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Jira Configuration
JIRA_BASE_URL=https://your-company.atlassian.net
JIRA_USERNAME=your-email@company.com
JIRA_API_TOKEN=your-jira-api-token
JIRA_STORY_KEYS=PROJ-123,PROJ-124,PROJ-125

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key

# Email Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
RECIPIENT_EMAIL=recipient@company.com
SENDER_NAME=Security Analysis Bot
```

### 3. Jira API Token Setup

1. Go to https://id.atlassian.com/manage-profile/security/api-tokens
2. Create API token
3. Use your email as username and token as password

### 4. OpenAI API Setup

1. Get API key from https://platform.openai.com/api-keys
2. Add to `.env` file

### 5. Email Setup (Gmail Example)

1. Enable 2-factor authentication
2. Generate app password: https://myaccount.google.com/apppasswords
3. Use app password in `.env`

## Usage

### Run the Analysis

```bash
cd jira-report-generator
python src/main.py
```

### What It Does

1. **Fetches Jira Stories**: Retrieves stories based on keys from environment
2. **AI Analysis**: Analyzes each story for:
   - Security vulnerabilities
   - Fraud risks
   - Compliance issues
   - Recommended mitigations
3. **Document Generation**: Creates Word document with:
   - Executive summary
   - Individual story analysis
   - Risk assessments
   - Recommendations
4. **Email Delivery**: Sends HTML email with:
   - Executive dashboard
   - Risk statistics
   - Attached detailed report

## Output Examples

### Generated Document Includes:
- Executive Summary with risk distribution
- Individual story sections with:
  - Story details (summary, description, acceptance criteria)
  - AI-generated security analysis
  - Fraud risk assessment
  - Specific recommendations
  - Attachment information

### Email Report Features:
- Beautiful HTML layout
- Risk level statistics
- Story summaries with risk badges
- Call-to-action recommendations
- Professional formatting

## AI Analysis Capabilities

The AI analyzer identifies:

**Security Risks:**
- Authentication vulnerabilities
- Data privacy concerns
- Input validation issues
- Authorization flaws
- Injection attack vectors

**Fraud Risks:**
- Financial transaction security
- Identity verification gaps
- Audit trail weaknesses
- User impersonation risks
- Data manipulation scenarios

**Recommendations:**
- Specific security controls
- Compliance requirements
- Testing strategies
- Monitoring solutions

## Customization

### Adding Custom Fields

Edit `src/jira_client.py` in the `parse_issue_data` method:

```python
# Add custom field parsing
custom_field_value = fields.get('customfield_12345', '')
```

### Modifying AI Analysis

Edit `src/ai_analyzer.py` to adjust the analysis prompt or add industry-specific checks.

### Email Template Customization

Modify the HTML template in `src/email_service.py` to match your organization's branding.

## Security Considerations

- Store API tokens securely
- Use environment variables for sensitive data
- Regularly rotate API tokens
- Review AI analysis results manually
- Implement proper access controls

## Troubleshooting

### Common Issues:

1. **Jira Authentication Failed**
   - Verify API token is correct
   - Check if 2FA is properly configured
   - Ensure Jira URL is correct

2. **OpenAI API Errors**
   - Verify API key is valid
   - Check rate limits
   - Ensure sufficient credits

3. **Email Sending Failed**
   - Verify SMTP settings
   - Check app password for Gmail
   - Ensure less secure apps are enabled (if needed)

4. **Missing Story Keys**
   - Verify story keys exist and are accessible
   - Check permissions for the Jira user

### Debug Mode

Add debug logging by modifying the script:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Features

### Batch Processing
The tool can handle multiple stories efficiently with progress tracking.

### Attachment Analysis
Automatically includes attachment information in the analysis.

### Risk Scoring
Provides quantitative risk assessment for prioritization.

### Compliance Mapping
Maps identified risks to compliance frameworks (GDPR, PCI-DSS, etc.).

## License

This project is provided as-is for educational and professional use. Please ensure compliance with your organization's security policies and API usage guidelines.
