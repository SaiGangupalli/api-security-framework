from typing import List, Dict, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from .test_script_generator import APITestScriptGenerator
from .test_executor import APITestExecutor
import json
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

class TestScriptRequest(BaseModel):
    endpoint: str
    test_cases: List[dict]
    api_details: Optional[Dict] = None  # Added to accept api_details

class TestExecutionResponse(BaseModel):
    results: List[dict]

@router.post("/api/generate-scripts")
async def generate_test_scripts(request: TestScriptRequest):
    try:
        script_generator = APITestScriptGenerator()

        # Creating complete default api_details if not provided
        if not request.api_details:
            api_details = {
                "method": "POST",  # Default to POST
                "request": {
                    "uri": request.endpoint,
                    "method": "POST",
                    "headers": {}
                }
            }
        else:
            api_details = request.api_details
            # Ensuring method exists at the top level
            if not api_details.get('method'):
                api_details["method"] = api_details.get('request', {}).get('method', 'POST')

        # Log the request details for debugging
        logger.info(f"Generating scripts for endpoint: {request.endpoint}")
        logger.info(f"API details received: {json.dumps(api_details, default=str)}")
        logger.info(f"Test cases count: {len(request.test_cases)}")

        scripts = await script_generator.generate_test_scripts(
            request.test_cases,
            api_details
        )
        return {"scripts": scripts}
    except Exception as e:
        logger.error(f"Error generating scripts: {str(e)}")
        logger.exception("Detailed traceback:")
        raise HTTPException(status_code=500, detail=str(e))

def clean_script_for_execution(script_content):
    """Remove non-Python content from the script."""
    # Remove any markdown or descriptive text before imports
    import_index = script_content.find('import ')
    if import_index > 0:
        script_content = script_content[import_index:]

    # Remove explanation sections
    explanation_index = script_content.find('### Explanation')
    if explanation_index > 0:
        script_content = script_content[:explanation_index].strip()

    return script_content

@router.post("/api/execute-tests")
async def execute_tests(request: TestScriptRequest):
    try:
        test_executor = APITestExecutor()

        # Clean scripts before execution
        cleaned_scripts = []
        for script in request.test_cases:
            if "script" in script:
                script["script"] = clean_script_for_execution(script["script"])
            cleaned_scripts.append(script)

        results = await test_executor.execute_test_suite(cleaned_scripts)
        return {"results": results}
    except Exception as e:
        print(f"Error executing tests: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))