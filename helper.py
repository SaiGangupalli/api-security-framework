"""
Document Generator for creating Word documents with clean text formatting
"""
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.shared import OxmlElement, qn
import logging
import re
from typing import List, Dict
from datetime import datetime

class DocumentGenerator:
    def __init__(self):
        self.doc = Document()
        self._setup_document_styles()
    
    def _clean_text_content(self, text: str) -> str:
        """
        Clean text content by removing excessive whitespace and formatting issues
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove HTML entities
        text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Replace multiple newlines with double newline
        text = re.sub(r'^\s+|\s+
    
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
        
        self._add_clean_paragraph(executive_summary)
        
        self.doc.add_page_break()
    
    def _add_issues_analysis(self, issues_data: List[Dict]):
        """Add detailed analysis for each issue"""
        self.doc.add_heading('Detailed Issue Analysis', level=1)
        
        for i, issue in enumerate(issues_data, 1):
            # Issue header
            clean_summary = self._clean_text_content(issue.get('summary', ''))
            issue_heading = self.doc.add_heading(f"{i}. {issue['key']}: {clean_summary}", level=2)
            
            # Basic Information Table
            self._add_issue_info_table(issue)
            
            # LLM Analysis
            if 'llm_summary' in issue and issue['llm_summary']:
                self.doc.add_heading('Analysis Summary', level=3)
                self._add_clean_paragraph(issue['llm_summary'])
            
            # Fraud & Security Analysis
            if 'fraud_security_analysis' in issue and issue['fraud_security_analysis']:
                self.doc.add_heading('Fraud & Security Assessment', level=3)
                self._add_clean_paragraph(issue['fraud_security_analysis'])
            
            # Attachments
            if issue.get('attachments'):
                self.doc.add_heading('Attachments', level=3)
                for att in issue['attachments']:
                    att_para = self.doc.add_paragraph()
                    att_para.add_run(f"• {att['filename']} ").bold = True
                    att_para.add_run(f"(Size: {att['size']} bytes, Created: {att['created']})")
                    if att.get('content_summary'):
                        clean_content = self._clean_text_content(att['content_summary'])
                        if clean_content:
                            self._add_clean_paragraph(f"   Content: {clean_content}")
            
            # Add separator
            if i < len(issues_data):
                self.doc.add_paragraph("_" * 80)
                self.doc.add_paragraph()
    
    def _add_issue_info_table(self, issue: Dict):
        """Add issue information table with cleaned content"""
        table = self.doc.add_table(rows=0, cols=2)
        table.style = 'Table Grid'
        
        info_items = [
            ('Issue Key', issue['key']),
            ('Status', self._clean_text_content(str(issue.get('status', 'Unknown')))),
            ('Priority', self._clean_text_content(str(issue.get('priority', 'Not Set')))),
            ('Type', self._clean_text_content(str(issue.get('issue_type', 'Unknown')))),
            ('Assignee', self._clean_text_content(str(issue.get('assignee', 'Unassigned')))),
            ('Reporter', self._clean_text_content(str(issue.get('reporter', 'Unknown')))),
            ('Created', self._clean_text_content(str(issue.get('created', 'Unknown')))),
            ('Updated', self._clean_text_content(str(issue.get('updated', 'Unknown'))))
        ]
        
        for label, value in info_items:
            row = table.add_row()
            row.cells[0].text = label
            row.cells[0].paragraphs[0].runs[0].bold = True
            row.cells[1].text = value
        
        self.doc.add_paragraph()
        
        # Description
        if issue.get('description'):
            self.doc.add_heading('Description', level=3)
            self._add_clean_paragraph(issue['description'])
        
        # Acceptance Criteria
        if issue.get('acceptance_criteria') and issue['acceptance_criteria'] != "No acceptance criteria found":
            self.doc.add_heading('Acceptance Criteria', level=3)
            self._add_clean_paragraph(issue['acceptance_criteria'])
    
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
            
            # Clean and truncate summary
            clean_summary = self._clean_text_content(issue['summary'])
            row.cells[1].text = clean_summary[:50] + '...' if len(clean_summary) > 50 else clean_summary
            
            row.cells[2].text = self._clean_text_content(str(issue.get('status', 'Unknown')))
            row.cells[3].text = self._clean_text_content(str(issue.get('priority', 'Not Set')))
        
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
                    <td>{self._clean_text_content(issue['summary'])[:60] + '...' if len(self._clean_text_content(issue['summary'])) > 60 else self._clean_text_content(issue['summary'])}</td>
                    <td>{self._clean_text_content(str(issue.get('status', 'Unknown')))}</td>
                    <td>{self._clean_text_content(str(issue.get('priority', 'Not Set')))}</td>
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
        
        return html_content, '', text, flags=re.MULTILINE)  # Remove leading/trailing spaces from each line
        
        # Remove empty lines at the beginning and end
        text = text.strip()
        
        # Normalize line breaks
        text = re.sub(r'\r\n', '\n', text)  # Convert Windows line breaks
        text = re.sub(r'\r', '\n', text)  # Convert Mac line breaks
        
        # Remove more than 2 consecutive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Clean up bullet points and list formatting
        text = re.sub(r'^\s*[-*•]\s*', '• ', text, flags=re.MULTILINE)
        
        # Remove trailing periods from headers/titles (common in Jira)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _format_multiline_text(self, text: str) -> List[str]:
        """
        Format multiline text into properly structured paragraphs
        """
        if not text:
            return [""]
        
        cleaned_text = self._clean_text_content(text)
        
        # Split into paragraphs (separated by double newlines or more)
        paragraphs = re.split(r'\n\s*\n', cleaned_text)
        
        formatted_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if para:
                # Handle bullet points
                if para.startswith('•') or para.startswith('-') or para.startswith('*'):
                    # Keep bullet points as separate items
                    bullet_items = re.split(r'\n(?=[•\-*])', para)
                    for item in bullet_items:
                        item = item.strip()
                        if item:
                            formatted_paragraphs.append(item)
                else:
                    # Regular paragraph
                    formatted_paragraphs.append(para)
        
        return formatted_paragraphs if formatted_paragraphs else [""]
    
    def _add_clean_paragraph(self, text: str, style=None):
        """
        Add a paragraph with cleaned text content
        """
        if not text:
            return
        
        cleaned_text = self._clean_text_content(text)
        if not cleaned_text:
            return
        
        # Handle multiline content
        paragraphs = self._format_multiline_text(cleaned_text)
        
        for para_text in paragraphs:
            if para_text.strip():
                para = self.doc.add_paragraph(para_text, style=style)
                para.space_after = Pt(6)
    
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
Jira Client for extracting issue details using Bearer Token authentication
"""
import requests
import json
import re
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
        """Clean HTML content from Jira fields and remove excessive whitespace"""
        if not text:
            return ""
        
        try:
            # Handle Atlassian Document Format (ADF)
            if isinstance(text, dict):
                cleaned_text = self._extract_text_from_adf(text)
            elif isinstance(text, list):
                cleaned_text = '\n'.join([self._extract_text_from_adf(item) if isinstance(item, dict) else str(item) for item in text])
            else:
                # Handle HTML content
                soup = BeautifulSoup(str(text), 'html.parser')
                cleaned_text = soup.get_text(strip=True)
            
            # Additional cleaning for better formatting
            if cleaned_text:
                # Remove excessive whitespace
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Multiple spaces to single space
                cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)  # Multiple newlines to double newline
                cleaned_text = re.sub(r'^\s+|\s+
    
    def _extract_text_from_adf(self, adf_content: dict) -> str:
        """Extract text from Atlassian Document Format with better formatting"""
        try:
            if not isinstance(adf_content, dict):
                return str(adf_content)
            
            text_parts = []
            
            def extract_text_recursive(node, depth=0):
                if isinstance(node, dict):
                    node_type = node.get('type', '')
                    
                    if node_type == 'text':
                        text_parts.append(node.get('text', ''))
                    elif node_type == 'hardBreak':
                        text_parts.append('\n')
                    elif node_type == 'paragraph':
                        if text_parts and not text_parts[-1].endswith('\n'):
                            text_parts.append('\n')
                        if 'content' in node:
                            for child in node['content']:
                                extract_text_recursive(child, depth + 1)
                        text_parts.append('\n')
                    elif node_type in ['bulletList', 'orderedList']:
                        text_parts.append('\n')
                        if 'content' in node:
                            for child in node['content']:
                                extract_text_recursive(child, depth + 1)
                        text_parts.append('\n')
                    elif node_type == 'listItem':
                        text_parts.append('• ')
                        if 'content' in node:
                            for child in node['content']:
                                extract_text_recursive(child, depth + 1)
                        text_parts.append('\n')
                    elif 'content' in node:
                        for child in node['content']:
                            extract_text_recursive(child, depth + 1)
                elif isinstance(node, list):
                    for item in node:
                        extract_text_recursive(item, depth)
            
            extract_text_recursive(adf_content)
            
            # Join and clean up the text
            result = ''.join(text_parts)
            
            # Clean up excessive newlines and spaces
            result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)
            result = re.sub(r'^\s+|\s+
    
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
            return False, '', cleaned_text, flags=re.MULTILINE)  # Leading/trailing spaces per line
                cleaned_text = cleaned_text.strip()
                
                # Normalize line breaks
                cleaned_text = re.sub(r'\r\n', '\n', cleaned_text)
                cleaned_text = re.sub(r'\r', '\n', cleaned_text)
                
                # Remove HTML entities
                cleaned_text = cleaned_text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            
            return cleaned_text
            
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
            return False, '', result, flags=re.MULTILINE)
            result = result.strip()
            
            return result
            
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
            return False, '', cleaned_text, flags=re.MULTILINE)  # Leading/trailing spaces per line
                cleaned_text = cleaned_text.strip()
                
                # Normalize line breaks
                cleaned_text = re.sub(r'\r\n', '\n', cleaned_text)
                cleaned_text = re.sub(r'\r', '\n', cleaned_text)
                
                # Remove HTML entities
                cleaned_text = cleaned_text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            
            return cleaned_text
            
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
