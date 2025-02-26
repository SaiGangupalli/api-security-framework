import json
import aiohttp
import asyncio
import logging
from typing import Dict, List
import openai
from openai import AsyncOpenAI

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
        return f"""
        Generate a Python async test script using aiohttp for the following API security test case:
        
        Test Name: {test_case.get('name')}
        Type: {test_case.get('type')}
        Description: {test_case.get('description')}
        
        API Details:
        - Method: {api_details.get('method')}
        - Endpoint: {api_details.get('request', {}).get('uri')}
        
        Test Steps:
        {self._format_steps(test_case.get('test_cases', [{}])[0].get('steps', []))}
        
        Expected Results:
        {test_case.get('test_cases', [{}])[0].get('expected_results', '')}
        
        Requirements:
        1. Use async/await with aiohttp for HTTP requests
        2. Include proper assertions and error handling
        3. Generate formatted output with test results
        4. Include setup and teardown if needed
        5. Add comments explaining the test strategy
        6. Handle authentication if required
        7. Include validation for security-specific responses
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