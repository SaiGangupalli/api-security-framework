import requests
import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import openai
from datetime import datetime, timedelta

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
        """
        Initialize the Jira Agent
        
        Args:
            jira_url: Your Jira instance URL (e.g., 'https://yourcompany.atlassian.net')
            username: Your Jira username/email
            api_token: Your Jira API token
            openai_api_key: Your OpenAI API key for natural language processing
        """
        self.jira_url = jira_url.rstrip('/')
        self.auth = (username, api_token)
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        openai.api_key = openai_api_key
    
    def parse_natural_language_query(self, user_query: str) -> JiraQuery:
        """
        Use AI to parse natural language queries into structured JiraQuery objects
        """
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
        - "Show me all stories assigned to John" → {{"query_type": "assignee_filter", "assignee": "John"}}
        - "Find epic PROJ-123" → {{"query_type": "epic_search", "epic_key": "PROJ-123"}}
        - "Stories in progress for project DEV" → {{"query_type": "status_filter", "project_key": "DEV", "status": "In Progress"}}
        
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
            print(f"Error parsing query: {e}")
            # Fallback to basic keyword search
            return JiraQuery(
                query_type=QueryType.STORY_SEARCH,
                keywords=user_query.split()
            )
    
    def build_jql_query(self, parsed_query: JiraQuery) -> str:
        """
        Convert parsed query into JQL (Jira Query Language)
        """
        jql_parts = []
        
        # Project filter
        if parsed_query.project_key:
            jql_parts.append(f'project = "{parsed_query.project_key}"')
        
        # Issue type filter
        if parsed_query.query_type == QueryType.EPIC_SEARCH:
            jql_parts.append('issuetype = Epic')
        elif parsed_query.query_type == QueryType.STORY_SEARCH:
            jql_parts.append('issuetype = Story')
        
        # Specific issue key
        if parsed_query.epic_key:
            jql_parts.append(f'key = "{parsed_query.epic_key}"')
        if parsed_query.story_key:
            jql_parts.append(f'key = "{parsed_query.story_key}"')
        
        # Assignee filter
        if parsed_query.assignee:
            jql_parts.append(f'assignee ~ "{parsed_query.assignee}"')
        
        # Status filter
        if parsed_query.status:
            jql_parts.append(f'status = "{parsed_query.status}"')
        
        # Date range filter
        if parsed_query.date_from:
            jql_parts.append(f'created >= "{parsed_query.date_from}"')
        if parsed_query.date_to:
            jql_parts.append(f'created <= "{parsed_query.date_to}"')
        
        # Keyword search in summary and description
        if parsed_query.keywords:
            keyword_search = ' OR '.join([f'summary ~ "{kw}" OR description ~ "{kw}"' for kw in parsed_query.keywords])
            jql_parts.append(f'({keyword_search})')
        
        return ' AND '.join(jql_parts) if jql_parts else 'project is not EMPTY'
    
    def search_jira_issues(self, jql_query: str, max_results: int = 50) -> Dict[str, Any]:
        """
        Execute JQL query against Jira API
        """
        search_url = f"{self.jira_url}/rest/api/3/search"
        
        payload = {
            'jql': jql_query,
            'maxResults': max_results,
            'fields': [
                'summary',
                'description',
                'status',
                'assignee',
                'reporter',
                'created',
                'updated',
                'priority',
                'issuetype',
                'project',
                'parent',
                'subtasks',
                'labels',
                'components'
            ]
        }
        
        try:
            response = requests.post(
                search_url,
                headers=self.headers,
                auth=self.auth,
                data=json.dumps(payload)
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error querying Jira: {e}")
            return {'issues': [], 'total': 0}
    
    def get_epic_stories(self, epic_key: str) -> Dict[str, Any]:
        """
        Get all stories under a specific epic
        """
        jql_query = f'"Epic Link" = "{epic_key}"'
        return self.search_jira_issues(jql_query)
    
    def format_response(self, issues_data: Dict[str, Any], original_query: str) -> str:
        """
        Format the Jira response into a human-readable format
        """
        issues = issues_data.get('issues', [])
        total = issues_data.get('total', 0)
        
        if total == 0:
            return f"No issues found for query: '{original_query}'"
        
        response = f"Found {total} issue(s) for query: '{original_query}'\n\n"
        
        for issue in issues[:10]:  # Limit to first 10 results
            fields = issue['fields']
            
            response += f"🎫 **{issue['key']}**: {fields['summary']}\n"
            response += f"   📊 Status: {fields['status']['name']}\n"
            response += f"   🏷️ Type: {fields['issuetype']['name']}\n"
            
            if fields.get('assignee'):
                response += f"   👤 Assignee: {fields['assignee']['displayName']}\n"
            
            if fields.get('priority'):
                response += f"   ⚡ Priority: {fields['priority']['name']}\n"
            
            if fields.get('parent'):
                response += f"   📋 Epic: {fields['parent']['key']}\n"
            
            response += f"   📅 Created: {fields['created'][:10]}\n"
            
            if fields.get('description'):
                desc = fields['description'][:200] + "..." if len(fields['description']) > 200 else fields['description']
                response += f"   📝 Description: {desc}\n"
            
            response += "\n"
        
        if total > 10:
            response += f"... and {total - 10} more issues\n"
        
        return response
    
    def process_user_query(self, user_query: str) -> str:
        """
        Main method to process user queries end-to-end
        """
        print(f"Processing query: '{user_query}'")
        
        # Step 1: Parse natural language query
        parsed_query = self.parse_natural_language_query(user_query)
        print(f"Parsed query: {parsed_query}")
        
        # Step 2: Build JQL query
        jql_query = self.build_jql_query(parsed_query)
        print(f"JQL Query: {jql_query}")
        
        # Step 3: Execute search
        if parsed_query.epic_key and parsed_query.query_type == QueryType.EPIC_SEARCH:
            # If looking for stories under an epic
            issues_data = self.get_epic_stories(parsed_query.epic_key)
        else:
            issues_data = self.search_jira_issues(jql_query)
        
        # Step 4: Format and return response
        return self.format_response(issues_data, user_query)

# Usage Example
def main():
    # Initialize the agent
    agent = JiraAgent(
        jira_url="https://yourcompany.atlassian.net",
        username="your-email@company.com",
        api_token="your-jira-api-token",
        openai_api_key="your-openai-api-key"
    )
    
    # Example queries
    example_queries = [
        "Show me all stories assigned to John Smith",
        "Find all bugs in the DEV project",
        "What are the open epics for this sprint?",
        "Show me stories created in the last week",
        "Find epic PROJ-123 and all its stories",
        "List all high priority issues in progress",
        "Show me stories with 'authentication' in the title"
    ]
    
    for query in example_queries:
        print("=" * 80)
        response = agent.process_user_query(query)
        print(response)
        print("\n")

if __name__ == "__main__":
    main()
