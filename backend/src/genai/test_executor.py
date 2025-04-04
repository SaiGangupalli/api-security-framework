import asyncio
import logging
from typing import Dict, List
import aiohttp
import json

logger = logging.getLogger(__name__)

class APITestExecutor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def execute_test_script(self, script: Dict) -> Dict:
        """Execute a generated test script and return results"""
        try:
            # Create namespace for script execution
            namespace = {
                'aiohttp': aiohttp,
                'asyncio': asyncio,
                'json': json,
                'logging': logging
            }

            # Execute the script in a try-except block
            try:
                exec(script['script'], namespace)
            except SyntaxError as se:
                return {
                    "test_name": script['test_name'],
                    "test_type": script['test_type'],
                    "status": "failed",
                    "error": f"Syntax error: {str(se)}"
                }
            except Exception as e:
                return {
                    "test_name": script['test_name'],
                    "test_type": script['test_type'],
                    "status": "failed",
                    "error": f"Script error: {str(e)}"
                }

            # Run the main test function
            if 'run_test' in namespace:
                try:
                    result = await namespace['run_test']()
                    return {
                        "test_name": script['test_name'],
                        "test_type": script['test_type'],
                        "status": "completed",
                        "result": result,
                        "error": None
                    }
                except Exception as e:
                    return {
                        "test_name": script['test_name'],
                        "test_type": script['test_type'],
                        "status": "failed",
                        "error": f"Runtime error: {str(e)}"
                    }
            else:
                return {
                    "test_name": script['test_name'],
                    "test_type": script['test_type'],
                    "status": "failed",
                    "error": "No run_test function found in script"
                }
        except Exception as e:
            return {
                "test_name": script['test_name'],
                "test_type": script['test_type'],
                "status": "failed",
                "error": f"Execution error: {str(e)}"
            }

    async def execute_test_suite(self, scripts: List[Dict]) -> List[Dict]:
        """Execute multiple test scripts and return results"""
        results = []
        for script in scripts:
            result = await self.execute_test_script(script)
            results.append(result)
        return results