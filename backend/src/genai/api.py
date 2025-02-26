from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from .test_script_generator import APITestScriptGenerator
from .test_executor import APITestExecutor

router = APIRouter()

class TestScriptRequest(BaseModel):
    endpoint: str
    test_cases: List[dict]

class TestExecutionResponse(BaseModel):
    results: List[dict]

@router.post("/api/generate-scripts")
async def generate_test_scripts(request: TestScriptRequest):
    try:
        script_generator = APITestScriptGenerator()
        scripts = await script_generator.generate_test_scripts(request.test_cases, {
            "endpoint": request.endpoint
        })
        return {"scripts": scripts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/execute-tests")
async def execute_tests(request: TestScriptRequest):
    try:
        test_executor = APITestExecutor()
        results = await test_executor.execute_test_suite(request.test_cases)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))