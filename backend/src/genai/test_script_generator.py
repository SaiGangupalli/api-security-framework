import json
import aiohttp
import asyncio
import logging
from typing import Dict, List
import openai
from openai import AsyncOpenAI
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APITestScriptGenerator:
    def __init__(self):
        self.openai_client = AsyncOpenAI()

    async def generate_test_scripts(self, test_cases: List[Dict], api_details: Dict) -> List[Dict]:
        """Generate automated test scripts for test cases"""
        try:
            scripts = []
            for test_case in test_cases:
                # Generate Python test script using OpenAI
                prompt = self._create_test_script_prompt(test_case, api_details)
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an API testing expert. Generate Python test scripts using aiohttp for API security testing."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )

                test_script = response.choices[0].message.content

                # Clean and format the script
                test_script = self._format_test_script(test_script)

                scripts.append({
                    "test_name": test_case.get("name", "Unnamed Test"),
                    "test_type": test_case.get("type", "Unknown"),
                    "description": test_case.get("description", ""),
                    "script": test_script,
                    "priority": test_case.get("priority", "Medium")
                })

            return scripts
        except Exception as e:
            logger.error(f"Error generating test scripts: {str(e)}")
            return []

    def _create_test_script_prompt(self, test_case: Dict, api_details: Dict) -> str:
        """Create prompt for test script generation"""
        # Extract data from the test case
        steps = test_case.get('steps', [])
        expected_results = test_case.get('expected_results', '')
        remediation = test_case.get('remediation', '')

        # Determine method
        method = None
        for step in steps:
            if 'method' in step.lower():
                # Extract method from step text
                method_match = re.search(r'method to:?\s+([A-Z]+)', step, re.IGNORECASE)
                if method_match:
                    method = method_match.group(1).upper()
                    break

        # If method not found in steps, use API details
        if not method:
            method = (api_details.get('method') or
                      api_details.get('request', {}).get('method', 'POST'))

        # Get endpoint from api_details
        endpoint = api_details.get('request', {}).get('uri') or api_details.get('endpoint', '')

        # Get headers from api_details or generate sample security headers
        headers = api_details.get('request', {}).get('headers', {})

        # If headers are empty, provide some sample security headers based on the test type
        if not headers:
            test_type = test_case.get('type', '').lower()
            if 'command' in test_type or 'injection' in test_type:
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-API-Key": "test_api_key",
                    "User-Agent": "API Security Tester/1.0"
                }
            elif 'auth' in test_type:
                headers = {
                    "Authorization": "Bearer YOUR_JWT_TOKEN",
                    "Content-Type": "application/json"
                }
            else:
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }

        # Map test type to vulnerability category for risk scoring
        vulnerability_category = ""
        test_type = test_case.get('type', '').lower()
        if 'sql' in test_type or 'injection' in test_type:
            vulnerability_category = 'sql_injection'
        elif 'xss' in test_type or 'cross' in test_type:
            vulnerability_category = 'xss'
        elif 'auth' in test_type or 'authentication' in test_type or 'bypass' in test_type:
            vulnerability_category = 'auth_bypass'
        elif 'rate' in test_type or 'limit' in test_type:
            vulnerability_category = 'rate_limiting'
        elif 'command' in test_type:
            vulnerability_category = 'command_injection'
        elif 'path' in test_type or 'traversal' in test_type or 'directory' in test_type:
            vulnerability_category = 'path_traversal'

        return f"""
        Generate a Python async test script using aiohttp for the following API security test case:
        
        Test Name: {test_case.get('name')}
        Type: {test_case.get('type')}
        Description: {test_case.get('description')}
        Vulnerability Category: {vulnerability_category}
        
        API Details:
        - Method: {method}
        - Endpoint: {endpoint}
        - Headers: {json.dumps(headers)}
        
        Test Steps:
        {self._format_steps(steps)}
        
        Expected Results:
        {expected_results}
        
        Remediation:
        {remediation}
        
        Requirements:
        1. Create an async function named 'run_test()' that returns a dictionary with test results
        2. Include proper error handling with try/except blocks
        3. Make sure all imports (aiohttp, asyncio, json, logging) are at the top
        4. Keep the code clean and avoid unnecessary comments
        5. Make sure the test can be executed with asyncio.run(run_test())
        6. Ensure headers and parameters are properly passed to requests
        7. Return results as a dictionary with clear pass/fail status
        """
    def _format_steps(self, steps: List[str]) -> str:
        """Format test steps for prompt"""
        return "\n".join(f"- {step}" for step in steps)

    def _format_test_script(self, script: str) -> str:
        """Clean and format the generated test script"""
        # Remove markdown code block markers if present
        script = script.replace("```python", "").replace("```", "")

        # Add imports if not present
        required_imports = [
            "import aiohttp",
            "import asyncio",
            "import json",
            "from typing import Dict, List",
            "import logging"
        ]

        for imp in required_imports:
            if imp not in script:
                script = imp + "\n" + script

        return script.strip()