import json
import aiohttp
import asyncio
import logging
from typing import Dict, List
import openai
from openai import AsyncOpenAI
import re
import time
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APITestScriptGenerator:
    def __init__(self):
        self.openai_client = AsyncOpenAI()

        # Rate limiting configuration
        self.max_retries = 2  # Limit max retries to 2 attempts
        self.base_delay = 2   # Base delay in seconds
        self.max_requests_per_minute = 5  # Limit requests per minute
        self.request_timestamps = []  # Tracks recent requests for rate limiting

    async def generate_test_scripts(self, test_cases: List[Dict], api_details: Dict) -> List[Dict]:
        """Generate automated test scripts for test cases with rate limiting"""
        try:
            scripts = []
            for test_case in test_cases:
                # Apply rate limiting before making the API request
                await self._enforce_rate_limit()

                try:
                    # Generate Python test script using OpenAI with retry mechanism
                    script_content = await self._generate_script_with_retry(test_case, api_details)

                    if script_content:
                        scripts.append({
                            "test_name": test_case.get("name", "Unnamed Test"),
                            "test_type": test_case.get("type", "Unknown"),
                            "description": test_case.get("description", ""),
                            "script": script_content,
                            "priority": test_case.get("priority", "Medium")
                        })
                except Exception as e:
                    logger.error(f"Error generating script for test case {test_case.get('name')}: {str(e)}")
                    # Continue with other test cases if one fails
                    continue

            return scripts
        except Exception as e:
            logger.error(f"Error generating test scripts: {str(e)}")
            return []

    async def _generate_script_with_retry(self, test_case: Dict, api_details: Dict) -> str:
        """Generate a script with limited retries for rate limit handling"""
        prompt = self._create_test_script_prompt(test_case, api_details)

        # Try up to max_retries times
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"Generating script for: {test_case.get('name')} (attempt {attempt + 1}/{self.max_retries + 1})")

                # Make the API request
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an API testing expert. Generate Python test scripts using aiohttp for API security testing."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )

                # Record this request for rate limiting
                self.request_timestamps.append(time.time())

                # Extract and format the script
                test_script = response.choices[0].message.content
                return self._format_test_script(test_script)

            except openai.RateLimitError:
                # Handle rate limit error with exponential backoff
                if attempt < self.max_retries:
                    # Calculate backoff delay with jitter
                    delay = min(30, (2 ** attempt) * self.base_delay + random.uniform(0, 1))
                    logger.warning(f"Rate limit hit for {test_case.get('name')}. Waiting {delay:.2f} seconds before retry.")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Max retries reached for {test_case.get('name')} due to rate limits.")
                    return ""  # Return empty string if all retries failed
            except Exception as e:
                logger.error(f"Error generating script: {str(e)}")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.base_delay)
                else:
                    return ""  # Return empty string if all retries failed

        return ""  # Return empty string if we reach here

    async def _enforce_rate_limit(self):
        """Enforce rate limiting to avoid too many requests per minute"""
        # Remove timestamps older than 60 seconds
        current_time = time.time()
        self.request_timestamps = [ts for ts in self.request_timestamps if current_time - ts < 60]

        # If we've hit the rate limit, sleep until we can make a new request
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            # Calculate the earliest time we can make another request
            oldest_timestamp = min(self.request_timestamps)
            sleep_time = 60 - (current_time - oldest_timestamp) + 1  # +1 for safety margin

            logger.info(f"Rate limit reached. Waiting {sleep_time:.2f} seconds before next request.")
            await asyncio.sleep(max(0, sleep_time))

    def _create_test_script_prompt(self, test_case: Dict, api_details: Dict) -> str:
        """Create prompt for test script generation"""
        # Extract relevance information from test case
        # relevance_score = test_case.get('relevance_score', 0)
        # risk_score = test_case.get('risk_score', 0)

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
        8. Use consistent indentation with 4 spaces throughout the code
        9. Make sure to handle any special imports needed (like base64 for JWT tests)
        10. DO NOT use tabs for indentation, only use 4 spaces
        11. DO NOT provide explanation for the code, or comments on execution
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
            "from typing import Dict, List, Any",
            "import logging"
        ]

        for imp in required_imports:
            if imp not in script:
                script = imp + "\n" + script

        return script.strip()