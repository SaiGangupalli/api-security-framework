#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import requests
import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import openai
from datetime import datetime, timedelta
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom JSON Encoder to handle enums and other objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

app = Flask(__name__)
app.json_encoder = CustomJSONEncoder
app.secret_key = os.environ.get('SECRET_KEY', 'jira-ai-secret-key-change-in-production')
CORS(app)

# Get configuration from environment variables
JIRA_URL = os.environ.get('JIRA_URL')
JIRA_USERNAME = os.environ.get('JIRA_USERNAME')
JIRA_TOKEN = os.environ.get('JIRA_TOKEN')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Validate required environment variables
required_env_vars = {
    'JIRA_URL': JIRA_URL,
    'JIRA_USERNAME': JIRA_USERNAME,
    'JIRA_TOKEN': JIRA_TOKEN,
    'OPENAI_API_KEY': OPENAI_API_KEY
}

missing_vars = [var for var, value in required_env_vars.items() if not value]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
    print("📝 Please create a .env file with the required variables:")
    print("   JIRA_URL=https://yourcompany.atlassian.net")
    print("   JIRA_USERNAME=your-email@company.com")
    print("   JIRA_TOKEN=your-jira-api-token")
    print("   OPENAI_API_KEY=your-openai-api-key")
    exit(1)

logger.info("✅ All required environment variables are configured")

class QueryType(Enum):
    STORY_SEARCH = "story_search"
    EPIC_SEARCH = "epic_search"
    STATUS_FILTER = "status_filter"
    ASSIGNEE_FILTER = "assignee_filter"
    DATE_RANGE = "date_range"
    PROJECT_FILTER = "project_filter"

@dataclass
class JiraQuery:
    query_type: QueryType
    project_key: Optional[str] = None
    assignee: Optional[str] = None
    status: Optional[str] = None
    epic_key: Optional[str] = None
    story_key: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    keywords: Optional[List[str]] = None

class JiraAgent:
    def __init__(self, jira_url: str, username: str, api_token: str, openai_api_key: str):
        self.jira_url = jira_url.rstrip('/')
        self.auth = (username, api_token)
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        openai.api_key = openai_api_key
    
    def parse_natural_language_query(self, user_query: str) -> JiraQuery:
        """Use AI to parse natural language queries into structured JiraQuery objects"""
        prompt = f"""
        Parse the following user query about Jira tickets and extract relevant information:
        
        User Query: "{user_query}"
        
        Extract and return a JSON object with the following fields (set to null if not mentioned):
        - query_type: one of ["story_search", "epic_search", "status_filter", "assignee_filter", "date_range", "project_filter"]
        - project_key: project abbreviation (e.g., "PROJ", "DEV")
        - assignee: person's name or username
        - status: ticket status (e.g., "To Do", "In Progress", "Done")
        - epic_key: specific epic identifier
        - story_key: specific story identifier
        - date_from: start date in YYYY-MM-DD format
        - date_to: end date in YYYY-MM-DD format
        - keywords: array of relevant search terms
        
        Examples:
        - "Show me all stories assigned to John" -> {{"query_type": "assignee_filter", "assignee": "John"}}
        - "Find epic PROJ-123" -> {{"query_type": "epic_search", "epic_key": "PROJ-123"}}
        - "Stories in progress for project DEV" -> {{"query_type": "status_filter", "project_key": "DEV", "status": "In Progress"}}
        
        Return only the JSON object:
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            parsed_data = json.loads(response.choices[0].message.content)
            
            return JiraQuery(
                query_type=QueryType(parsed_data.get('query_type', 'story_search')),
                project_key=parsed_data.get('project_key'),
                assignee=parsed_data.get('assignee'),
                status=parsed_data.get('status'),
                epic_key=parsed_data.get('epic_key'),
                story_key=parsed_data.get('story_key'),
                date_from=parsed_data.get('date_from'),
                date_to=parsed_data.get('date_to'),
                keywords=parsed_data.get('keywords', [])
            )
        except Exception as e:
            logger.error(f"Error parsing query: {e}")
            return JiraQuery(
                query_type=QueryType.STORY_SEARCH,
                keywords=user_query.split()
            )
    
    def build_jql_query(self, parsed_query: JiraQuery) -> str:
        """Convert parsed query into JQL (Jira Query Language)"""
        jql_parts = []
        
        if parsed_query.project_key:
            jql_parts.append(f'project = "{parsed_query.project_key}"')
        
        if parsed_query.query_type == QueryType.EPIC_SEARCH:
            jql_parts.append('issuetype = Epic')
        elif parsed_query.query_type == QueryType.STORY_SEARCH:
            jql_parts.append('issuetype = Story')
        
        if parsed_query.epic_key:
            jql_parts.append(f'key = "{parsed_query.epic_key}"')
        if parsed_query.story_key:
            jql_parts.append(f'key = "{parsed_query.story_key}"')
        
        if parsed_query.assignee:
            jql_parts.append(f'assignee ~ "{parsed_query.assignee}"')
        
        if parsed_query.status:
            jql_parts.append(f'status = "{parsed_query.status}"')
        
        if parsed_query.date_from:
            jql_parts.append(f'created >= "{parsed_query.date_from}"')
        if parsed_query.date_to:
            jql_parts.append(f'created <= "{parsed_query.date_to}"')
        
        if parsed_query.keywords:
            keyword_search = ' OR '.join([f'summary ~ "{kw}" OR description ~ "{kw}"' for kw in parsed_query.keywords])
            jql_parts.append(f'({keyword_search})')
        
        return ' AND '.join(jql_parts) if jql_parts else 'project is not EMPTY'
    
    def search_jira_issues(self, jql_query: str, max_results: int = 50) -> Dict[str, Any]:
        """Execute JQL query against Jira API"""
        search_url = f"{self.jira_url}/rest/api/3/search"
        
        payload = {
            'jql': jql_query,
            'maxResults': max_results,
            'fields': [
                'summary', 'description', 'status', 'assignee', 'reporter',
                'created', 'updated', 'priority', 'issuetype', 'project',
                'parent', 'subtasks', 'labels', 'components'
            ]
        }
        
        try:
            response = requests.post(
                search_url,
                headers=self.headers,
                auth=self.auth,
                data=json.dumps(payload),
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying Jira: {e}")
            raise Exception(f"Failed to query Jira: {str(e)}")
    
    def process_user_query(self, user_query: str) -> Dict[str, Any]:
        """Main method to process user queries end-to-end"""
        logger.info(f"Processing query: '{user_query}'")
        
        try:
            # Parse natural language query
            parsed_query = self.parse_natural_language_query(user_query)
            logger.info(f"Parsed query: {parsed_query}")
            
            # Build JQL query
            jql_query = self.build_jql_query(parsed_query)
            logger.info(f"JQL Query: {jql_query}")
            
            # Execute search
            issues_data = self.search_jira_issues(jql_query)
            
            # Create a simplified, JSON-serializable version of parsed_query
            parsed_query_simple = {
                'query_type': parsed_query.query_type.value,
                'project_key': parsed_query.project_key,
                'assignee': parsed_query.assignee,
                'status': parsed_query.status,
                'epic_key': parsed_query.epic_key,
                'story_key': parsed_query.story_key,
                'date_from': parsed_query.date_from,
                'date_to': parsed_query.date_to,
                'keywords': parsed_query.keywords
            }
            
            return {
                'success': True,
                'data': issues_data,
                'jql_query': jql_query,
                'parsed_query': parsed_query_simple
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'success': False,
                'error': str(e)
            }

class JiraSecurityAnalyzer:
    def __init__(self, jira_url: str, username: str, api_token: str, openai_api_key: str):
        self.jira_url = jira_url.rstrip('/')
        self.auth = (username, api_token)
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        openai.api_key = openai_api_key
    
    def get_issue_details(self, issue_key: str) -> Dict[str, Any]:
        """Get detailed information about a specific Jira issue"""
        issue_url = f"{self.jira_url}/rest/api/3/issue/{issue_key}"
        
        params = {
            'fields': 'summary,description,status,assignee,priority,issuetype,labels,components'
        }
        
        try:
            response = requests.get(
                issue_url,
                headers=self.headers,
                auth=self.auth,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Jira issue {issue_key}: {e}")
            raise Exception(f"Failed to fetch issue {issue_key}: {str(e)}")
    
    def analyze_issue_security_impact(self, issue_key: str) -> Dict[str, Any]:
        """Analyze a specific issue for fraud and security impact"""
        try:
            # Get issue details
            issue_data = self.get_issue_details(issue_key)
            
            if not issue_data:
                return {
                    'success': False,
                    'error': f'Issue {issue_key} not found or inaccessible'
                }
            
            # Prepare issue context for LLM
            issue_context = self._prepare_issue_context(issue_data)
            
            # Get LLM security analysis
            security_analysis = self._get_llm_security_analysis(issue_context, issue_key)
            
            return {
                'success': True,
                'issue_key': issue_key,
                'analysis': security_analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing issue {issue_key}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _prepare_issue_context(self, issue_data: Dict) -> str:
        """Prepare issue context for LLM analysis"""
        fields = issue_data['fields']
        
        context = f"""
        Issue Key: {issue_data['key']}
        Summary: {fields.get('summary', '')}
        Description: {fields.get('description', 'No description provided')}
        Type: {fields.get('issuetype', {}).get('name', '')}
        Priority: {fields.get('priority', {}).get('name', '')}
        Status: {fields.get('status', {}).get('name', '')}
        Labels: {', '.join(fields.get('labels', []))}
        Components: {', '.join([comp['name'] for comp in fields.get('components', [])])}
        """
        
        return context.strip()
    
    def _get_llm_security_analysis(self, issue_context: str, issue_key: str) -> str:
        """Get LLM security analysis for the issue"""
        prompt = f"""
        You are a cybersecurity expert analyzing a Jira issue for fraud and security risks.
        
        Issue Details:
        {issue_context}
        
        Provide a fraud & security impact analysis in simple, non-technical language that a business manager can understand.
        
        Focus on:
        1. What security risks this issue might create
        2. How it could potentially be exploited for fraud
        3. What business impact this could have
        4. Simple recommendations to mitigate risks
        
        Keep your response short, crisp, and actionable. Use bullet points where appropriate.
        Avoid technical jargon and focus on business implications.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error getting LLM analysis: {e}")
            return f"Unable to generate security analysis. Error: {str(e)}"

# HTML Template as string to avoid file creation issues
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jira AI Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 1200px;
            height: 90vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #0052cc, #2684ff);
            color: white;
            padding: 25px 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2rem;
            font-weight: 600;
            margin-bottom: 8px;
        }

        .header p {
            opacity: 0.9;
            font-size: 1.1rem;
        }

        .status-bar {
            background: #e8f5e8;
            border-left: 4px solid #36b37e;
            padding: 15px 30px;
            color: #006644;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .status-indicator {
            width: 12px;
            height: 12px;
            background: #36b37e;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .chat-container {
            flex: 1;
            padding: 30px;
            overflow-y: auto;
            scroll-behavior: smooth;
        }

        .welcome-message {
            text-align: center;
            color: #5e6c84;
            margin-bottom: 30px;
        }

        .welcome-message h2 {
            font-size: 1.5rem;
            margin-bottom: 15px;
            color: #172b4d;
        }

        .example-queries {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }

        .example-query {
            background: #f4f5f7;
            padding: 15px;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.2s;
            border-left: 4px solid #0052cc;
        }

        .example-query:hover {
            background: #e4edff;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 82, 204, 0.15);
        }

        .example-query h4 {
            color: #0052cc;
            margin-bottom: 5px;
            font-size: 0.9rem;
        }

        .message {
            margin-bottom: 20px;
            animation: fadeIn 0.3s ease-in;
        }

        .message.user {
            display: flex;
            justify-content: flex-end;
        }

        .message.assistant {
            display: flex;
            justify-content: flex-start;
        }

        .message-content {
            max-width: 70%;
            padding: 15px 20px;
            border-radius: 18px;
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .message.user .message-content {
            background: #0052cc;
            color: white;
            border-bottom-right-radius: 6px;
        }

        .message.assistant .message-content {
            background: #f4f5f7;
            color: #172b4d;
            border-bottom-left-radius: 6px;
        }

        .input-container {
            padding: 25px 30px;
            background: white;
            border-top: 1px solid #e0e6ff;
            display: flex;
            gap: 15px;
            align-items: flex-end;
        }

        .input-wrapper {
            flex: 1;
            position: relative;
        }

        #queryInput {
            width: 100%;
            padding: 15px 20px;
            border: 2px solid #dfe1e6;
            border-radius: 25px;
            font-size: 1rem;
            resize: none;
            min-height: 50px;
            max-height: 120px;
            transition: border-color 0.2s;
        }

        #queryInput:focus {
            outline: none;
            border-color: #0052cc;
        }

        .send-button {
            background: #0052cc;
            color: white;
            border: none;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
            font-size: 1.2rem;
        }

        .send-button:hover:not(:disabled) {
            background: #003d99;
            transform: scale(1.05);
        }

        .send-button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }

        .loading {
            display: flex;
            align-items: center;
            gap: 10px;
            color: #5e6c84;
            font-style: italic;
        }

        .loading-spinner {
            width: 20px;
            height: 20px;
            border: 2px solid #e0e6ff;
            border-top: 2px solid #0052cc;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        .jira-issue {
            background: white;
            border: 1px solid #dfe1e6;
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            transition: transform 0.2s;
        }

        .jira-issue:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
        }

        .issue-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 15px;
        }

        .issue-key {
            background: #0052cc;
            color: white;
            padding: 6px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9rem;
        }

        .issue-status {
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .status-todo { background: #ddd; color: #666; }
        .status-progress { background: #fff4e6; color: #ff8b00; }
        .status-done { background: #e3fcef; color: #00875a; }

        .issue-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #172b4d;
            margin-bottom: 10px;
        }

        .issue-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            font-size: 0.9rem;
            color: #5e6c84;
        }

        .security-form {
            background: #fff4e6;
            border: 2px solid #ff8b00;
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
            box-shadow: 0 4px 12px rgba(255, 139, 0, 0.15);
        }

        .security-form h3 {
            color: #ff8b00;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .form-group {
            margin-bottom: 15px;
        }

        .form-group label {
            display: block;
            font-weight: 600;
            color: #172b4d;
            margin-bottom: 5px;
        }

        .form-group input {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #dfe1e6;
            border-radius: 8px;
            font-size: 1rem;
            transition: border-color 0.2s;
        }

        .form-group input:focus {
            outline: none;
            border-color: #ff8b00;
        }

        .analyze-button {
            background: #ff8b00;
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            width: 100%;
        }

        .analyze-button:hover:not(:disabled) {
            background: #cc6f00;
            transform: translateY(-1px);
        }

        .analyze-button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }

        .security-result {
            background: white;
            border-left: 4px solid #0052cc;
            border-radius: 12px;
            padding: 25px;
            margin: 15px 0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        }

        .security-result h3 {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 1.3rem;
        }

        .error-message {
            background: #ffebe6;
            border: 1px solid #ff8f73;
            color: #de350b;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @media (max-width: 768px) {
            .container {
                height: 100vh;
                border-radius: 0;
            }
            
            .example-queries {
                grid-template-columns: 1fr;
            }
            
            .message-content {
                max-width: 85%;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Jira AI Assistant</h1>
            <p>Ask me anything about your Jira issues and epics in natural language</p>
        </div>

        <div class="status-bar">
            <div class="status-indicator"></div>
            <span>✅ Connected to Jira - Ready to process your queries</span>
        </div>

        <div class="main-content">
            <div class="chat-container" id="chatContainer">
                <div class="welcome-message">
                    <h2>Welcome to your Jira AI Assistant! 👋</h2>
                    <p>Your Jira connection is configured and ready. Start asking questions about your issues and epics!</p>
                    
                    <div class="example-queries">
                        <div class="example-query" onclick="useExampleQuery(this)">
                            <h4>📋 Find Stories by Assignee</h4>
                            <p>Show me all stories assigned to John Smith</p>
                        </div>
                        <div class="example-query" onclick="useExampleQuery(this)">
                            <h4>🐛 Search by Issue Type</h4>
                            <p>Find all bugs in the DEV project</p>
                        </div>
                        <div class="example-query" onclick="useExampleQuery(this)">
                            <h4>📈 Status Filtering</h4>
                            <p>What are the open epics for this sprint?</p>
                        </div>
                        <div class="example-query" onclick="useExampleQuery(this)">
                            <h4>📅 Date Range Queries</h4>
                            <p>Show me stories created in the last week</p>
                        </div>
                        <div class="example-query" onclick="useExampleQuery(this)">
                            <h4>🔒 Security Analysis</h4>
                            <p>Analyze issue PROJ-123 for security risks</p>
                        </div>
                        <div class="example-query" onclick="showSecurityForm()">
                            <h4>🛡️ Fraud & Security Impact</h4>
                            <p>Get security analysis for an issue</p>
                        </div>
                    </div>
                </div>
            </div>

            <div class="input-container">
                <div class="input-wrapper">
                    <textarea id="queryInput" placeholder="Ask me about your Jira issues... (e.g., 'Show me all stories assigned to John')" rows="1"></textarea>
                </div>
                <button class="send-button" onclick="sendQuery()" id="sendButton">
                    ➤
                </button>
            </div>
        </div>
    </div>

    <script>
        // Auto-resize textarea
        document.getElementById('queryInput').addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });

        // Send query on Enter (but allow Shift+Enter for new lines)
        document.getElementById('queryInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendQuery();
            }
        });

        function useExampleQuery(element) {
            const queryText = element.querySelector('p').textContent;
            document.getElementById('queryInput').value = queryText;
            sendQuery();
        }

        function showSecurityForm() {
            const chatContainer = document.getElementById('chatContainer');
            
            // Remove welcome message if it exists
            const welcomeMessage = chatContainer.querySelector('.welcome-message');
            if (welcomeMessage) {
                welcomeMessage.remove();
            }
            
            const securityForm = `
                <div class="security-form">
                    <h3>🛡️ Fraud & Security Impact Analysis</h3>
                    <p style="margin-bottom: 20px; color: #5e6c84;">
                        Enter a Jira issue key to get a comprehensive security risk assessment in simple terms.
                    </p>
                    <div class="form-group">
                        <label for="issueKey">Issue Key:</label>
                        <input type="text" id="issueKey" placeholder="e.g., PROJ-123, DEV-456, SEC-789" maxlength="20">
                    </div>
                    <button class="analyze-button" onclick="analyzeIssueSecurity()" id="analyzeBtn">
                        🔍 Analyze Security Impact
                    </button>
                </div>
            `;
            
            chatContainer.innerHTML = securityForm;
            document.getElementById('issueKey').focus();
            
            // Allow Enter key to submit
            document.getElementById('issueKey').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    analyzeIssueSecurity();
                }
            });
        }

        async function analyzeIssueSecurity() {
            const issueKey = document.getElementById('issueKey').value.trim().toUpperCase();
            const analyzeBtn = document.getElementById('analyzeBtn');
            
            if (!issueKey) {
                alert('Please enter an issue key');
                return;
            }
            
            // Show loading state
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = '🔄 Analyzing...';
            
            try {
                const response = await fetch('/api/security-analysis', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        issue_key: issueKey
                    })
                });

                const result = await response.json();

                if (result.success) {
                    displaySecurityResults(result);
                } else {
                    addMessage(`<div class="error-message">Error: ${result.error}</div>`, false);
                }
                
            } catch (error) {
                addMessage(`<div class="error-message">Network Error: ${error.message}</div>`, false);
            } finally {
                analyzeBtn.disabled = false;
                analyzeBtn.innerHTML = '🔍 Analyze Security Impact';
            }
        }

        function displaySecurityResults(result) {
            const securityHtml = `
                <div class="security-result">
                    <h3 style="color: #0052cc; margin-bottom: 20px;">
                        🛡️ Security Analysis: ${result.issue_key}
                    </h3>
                    
                    <div style="background: #f8f9ff; padding: 20px; border-radius: 8px; white-space: pre-line; line-height: 1.6;">
                        ${result.analysis}
                    </div>
                    
                    <div style="margin-top: 20px; text-align: center;">
                        <button onclick="showSecurityForm()" style="background: #0052cc; color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer;">
                            🔍 Analyze Another Issue
                        </button>
                    </div>
                </div>
            `;
            
            addMessage(securityHtml, false);
        }

        function addMessage(content, isUser, isLoading = false) {
            const chatContainer = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
            
            if (isLoading) {
                messageDiv.innerHTML = `
                    <div class="message-content loading">
                        <div class="loading-spinner"></div>
                        Processing your query...
                    </div>
                `;
                messageDiv.id = 'loadingMessage';
            } else {
                messageDiv.innerHTML = `<div class="message-content">${content}</div>`;
            }
            
            // Remove welcome message if it exists
            const welcomeMessage = chatContainer.querySelector('.welcome-message');
            if (welcomeMessage) {
                welcomeMessage.remove();
            }
            
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            
            return messageDiv;
        }

        function removeLoadingMessage() {
            const loadingMessage = document.getElementById('loadingMessage');
            if (loadingMessage) {
                loadingMessage.remove();
            }
        }

        function formatJiraResponse(data) {
            if (!data.issues || data.issues.length === 0) {
                return '<div class="error-message">No issues found for your query.</div>';
            }

            let html = `<div style="margin-bottom: 15px;"><strong>Found ${data.total} issue(s):</strong></div>`;
            
            data.issues.slice(0, 10).forEach(issue => {
                const fields = issue.fields;
                
                html += `
                    <div class="jira-issue">
                        <div class="issue-header">
                            <span class="issue-key">${issue.key}</span>
                            <span class="issue-status status-${getStatusClass(fields.status.name)}">${fields.status.name}</span>
                        </div>
                        <div class="issue-title">${fields.summary}</div>
                        <div class="issue-meta">
                            <span>📋 ${fields.issuetype.name}</span>
                            ${fields.assignee ? `<span>👤 ${fields.assignee.displayName}</span>` : '<span>👤 Unassigned</span>'}
                            ${fields.priority ? `<span>⚡ ${fields.priority.name}</span>` : ''}
                            <span>📅 ${new Date(fields.created).toLocaleDateString()}</span>
                        </div>
                        ${fields.description ? `<div style="margin-top: 10px; color: #5e6c84; font-size: 0.9rem;">${fields.description.substring(0, 200)}${fields.description.length > 200 ? '...' : ''}</div>` : ''}
                    </div>
                `;
            });

            if (data.total > 10) {
                html += `<div style="text-align: center; color: #5e6c84; margin-top: 15px;">... and ${data.total - 10} more issues</div>`;
            }

            return html;
        }

        function getStatusClass(status) {
            const statusLower = status.toLowerCase();
            if (statusLower.includes('progress') || statusLower.includes('development')) return 'progress';
            if (statusLower.includes('done') || statusLower.includes('closed') || statusLower.includes('resolved')) return 'done';
            return 'todo';
        }

        async function sendQuery() {
            const queryInput = document.getElementById('queryInput');
            const sendButton = document.getElementById('sendButton');
            const query = queryInput.value.trim();
            
            if (!query) return;

            // Add user message
            addMessage(query, true);
            queryInput.value = '';
            queryInput.style.height = 'auto';
            
            // Show loading
            sendButton.disabled = true;
            const loadingMessage = addMessage('', false, true);

            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query: query
                    })
                });

                const result = await response.json();

                removeLoadingMessage();

                if (result.success) {
                    addMessage(formatJiraResponse(result.data), false);
                } else {
                    addMessage(`<div class="error-message">Error: ${result.error}</div>`, false);
                }
                
            } catch (error) {
                removeLoadingMessage();
                addMessage(`<div class="error-message">Network Error: ${error.message}</div>`, false);
            } finally {
                sendButton.disabled = false;
            }
        }
    </script>
</body>
</html>"""

@app.route('/')
def index():
    """Serve the main application page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/security-analysis', methods=['POST'])
def api_security_analysis():
    """API endpoint for fraud & security impact analysis of a specific issue"""
    try:
        data = request.get_json()
        
        if not data or not data.get('issue_key'):
            return jsonify({
                'success': False,
                'error': 'Missing required field: issue_key'
            }), 400
        
        issue_key = data['issue_key'].strip().upper()
        
        # Initialize security analyzer
        analyzer = JiraSecurityAnalyzer(
            jira_url=JIRA_URL,
            username=JIRA_USERNAME,
            api_token=JIRA_TOKEN,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Perform analysis
        result = analyzer.analyze_issue_security_impact(issue_key)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Security analysis API error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/query', methods=['POST'])
def api_query():
    """API endpoint to process Jira queries"""
    try:
        data = request.get_json()
        
        # Only require the query from the request
        if not data or not data.get('query'):
            return jsonify({
                'success': False,
                'error': 'Missing required field: query'
            }), 400
        
        # Initialize Jira agent with environment variables
        agent = JiraAgent(
            jira_url=JIRA_URL,
            username=JIRA_USERNAME,
            api_token=JIRA_TOKEN,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Process query
        result = agent.process_user_query(data['query'])
        
        # Ensure the result is JSON serializable
        try:
            json.dumps(result)  # Test serialization
            return jsonify(result)
        except (TypeError, ValueError) as json_error:
            logger.error(f"JSON serialization error: {json_error}")
            # Return a simplified response if serialization fails
            return jsonify({
                'success': result.get('success', False),
                'data': result.get('data', {}),
                'jql_query': result.get('jql_query', ''),
                'error': result.get('error', None)
            })
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'jira_url': JIRA_URL,
        'jira_username': JIRA_USERNAME,
        'openai_configured': bool(OPENAI_API_KEY)
    })

@app.route('/api/test', methods=['GET'])
def test_connection():
    """Test Jira connection"""
    try:
        agent = JiraAgent(
            jira_url=JIRA_URL,
            username=JIRA_USERNAME,
            api_token=JIRA_TOKEN,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Test with a simple query
        test_jql = "project is not EMPTY ORDER BY created DESC"
        search_url = f"{JIRA_URL}/rest/api/3/search"
        
        payload = {
            'jql': test_jql,
            'maxResults': 1,
            'fields': ['summary', 'status', 'issuetype', 'project']
        }
        
        response = requests.post(
            search_url,
            headers=agent.headers,
            auth=agent.auth,
            data=json.dumps(payload),
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        
        return jsonify({
            'success': True,
            'message': 'Jira connection successful',
            'total_issues': result.get('total', 0)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Jira connection failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Jira AI Assistant...")
    print("=" * 60)
    print(f"📊 Configuration:")
    print(f"   - Jira URL: {JIRA_URL}")
    print(f"   - Jira Username: {JIRA_USERNAME}")
    print(f"   - OpenAI API Key: {'✅ Configured' if OPENAI_API_KEY else '❌ Missing'}")
    print("=" * 60)
    print("🔧 Required Python packages:")
    print("   pip install flask flask-cors requests openai python-dotenv")
    print("=" * 60)
    print("🌐 Server will start at: http://localhost:5000")
    print("🔍 Health check: http://localhost:5000/api/health")
    print("🧪 Test connection: http://localhost:5000/api/test")
    print("=" * 60)
    print("📋 Example queries to try:")
    print("   - 'Show me all stories assigned to John Smith'")
    print("   - 'Find all bugs in the DEV project'")
    print("   - 'What are the high priority issues in progress?'")
    print("   - Security Analysis: Enter issue key like 'PROJ-123'")
    print("=" * 60)
    
    # Run the Flask app
    try:
        app.run(
            debug=os.environ.get('FLASK_ENV') == 'development',
            host='0.0.0.0',
            port=int(os.environ.get('PORT', 5000))
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down Jira AI Assistant...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("💡 Make sure port 5000 is available or set PORT environment variable")
