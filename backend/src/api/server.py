import asyncio
import json
import logging
import time
from datetime import datetime
from typing import List, Optional, Dict
import uvicorn
from elasticsearch import AsyncElasticsearch
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ml.model import SecurityModel
from genai.test_generator import TestGenerator
from scripts.setup_elasticsearch import ElasticsearchSetup
from rag.enhanced_rag import EnhancedRAG
from genai.api import router as genai_router

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
class APIRequest(BaseModel):
    endpoint: str

rag = EnhancedRAG()
es_setup = ElasticsearchSetup()
app = FastAPI()
es = AsyncElasticsearch(['http://elasticsearch:9200'])
openai = TestGenerator()
security_model = SecurityModel()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(genai_router)

logger = logging.getLogger(__name__)

class APIAnalyzer:
    def __init__(self):
        self.es = AsyncElasticsearch(['http://elasticsearch:9200'])
        self.index = 'api_security_logs'

@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Starting server initialization...")

        # Setup RAG index and mapping
        logger.info("Setting up RAG index and mapping...")
        await rag.setup_index_mapping()
        logger.info("RAG index and mapping setup complete")

        # Get security patterns
        logger.info("Loading security patterns...")
        security_patterns = es_setup.setup_knowledge_base()
        logger.info(f"Loaded {len(security_patterns)} security patterns")

        # Bulk index the patterns with delay
        logger.info("Starting to index patterns with embeddings...")
        try:
            await rag.bulk_index_patterns(security_patterns)
        except Exception as index_error:
            logger.error(f"Error during pattern indexing: {str(index_error)}")
            raise

        # Refresh the index
        logger.info("Refreshing index...")
        await rag.refresh_index()

        logger.info("RAG system initialization completed successfully")

        # Add retry mechanism for model loading
        max_retries = 3
        for attempt in range(max_retries):
            try:
                security_model.load_model()
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to load models after {max_retries} attempts")
                    raise
                logger.warning(f"Model loading attempt {attempt + 1} failed: {str(e)}")
                await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        logger.exception("Detailed stack trace:")
        raise

@app.get("/api/check")
async def check_elasticsearch():
    try:
        # Get index stats
        stats = await es.indices.stats(index="api_security_logs")
        doc_count = stats["indices"]["api_security_logs"]["total"]["docs"]["count"]

        # Get a sample document
        sample = await es.search(
            index="api_security_logs",
            body={
                "query": {"match_all": {}},
                "size": 1
            }
        )

        return {
            "status": "connected",
            "document_count": doc_count,
            "sample_document": sample["hits"]["hits"][0]["_source"] if sample["hits"]["hits"] else None
        }
    except Exception as e:
        logger.error(f"Elasticsearch error: {str(e)}")
        return {"status": "error", "message": str(e)}

class EndpointResponse(BaseModel):
    endpoints: List[str]
    count: int
    sample_doc: Optional[Dict]

@app.get("/api/endpoints")
async def list_endpoints():
    try:
        # Get all documents to see what's actually in there
        results = await es.search(
            index="api_security_logs",
            body={
                "query": {"match_all": {}},
                "size": 10,  # Get 10 documents to inspect
                "_source": ["endpoint"]  # Only get the endpoint field
            }
        )

        endpoints = []
        if results["hits"]["hits"]:
            endpoints = [hit["_source"].get("endpoint") for hit in results["hits"]["hits"]]
            endpoints = list(set(filter(None, endpoints)))  # Remove duplicates and None values

        return {
            "total_documents": results["hits"]["total"]["value"],
            "endpoints": endpoints,
            "count": len(endpoints)
        }
    except Exception as e:
        logger.error(f"Error listing endpoints: {str(e)}")
        return {"status": "error", "message": str(e)}

async def get_api_details(endpoint: str) -> Dict:
    """Get API details from Elasticsearch"""
    try:
        search_query = {
            "query": {
                "match": {
                    "endpoint": endpoint
                }
            },
            "size": 1,
            "sort": [
                {"timestamp": {"order": "desc"}}
            ]
        }
        logger.debug(f"Executing ES query: {search_query}")
        result = await es.search(index="api_security_logs", body=search_query)

        if result["hits"]["total"]["value"] == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No API details found for endpoint: {endpoint}"
            )

        # Get the document from Elasticsearch
        doc = result["hits"]["hits"][0]["_source"]

        # Extract request method and details
        api_details = {
            "method": doc.get("method", "GET"),
            "request": {
                "method": doc.get("method", "GET"),
                "headers": doc.get("request", {}).get("headers", {}),
                "body": doc.get("request", {}).get("body", {}),
                "params": doc.get("request", {}).get("params", {}),
                "uri": doc.get("request", {}).get("uri", "")
            },
            "response": doc.get("response", {}),
            "timestamp": doc.get("timestamp")
        }

        logger.debug(f"Extracted API details: {api_details}")
        return api_details

    except Exception as e:
        logger.error(f"Error fetching API details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze")
async def analyze_endpoint(request: APIRequest):
    """Analyze API endpoint security"""
    try:
        es_check_result = await check_elasticsearch()
        if es_check_result.get("status") != "connected":
            raise HTTPException(status_code=500, detail="Elasticsearch connection error")

        start_time = time.time()

        try:
            api_details = await get_api_details(request.endpoint)
            es_status_code = 200
        except HTTPException as he:
            es_status_code = he.status_code
            raise
        except Exception as e:
            logger.error(f"Error getting API details: {str(e)}")
            es_status_code = 500
            raise
        finally:
            es_query_time = round(time.time() - start_time, 3)

        es_details = {
            "status_code": es_status_code,
            "response_time": es_query_time,
            "query_timestamp": datetime.now().isoformat()
        }

        # ML risk assessment
        try:
            # risk_assessment = security_model.perform_ml_risk_assessment(api_details)
            risk_assessment = security_model.analyze_risks(api_details)
            logger.debug(f"ML Risk Assessment: {json.dumps(risk_assessment, indent=2)}")
        except Exception as e:
            logger.error(f"ML prediction error: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"ML prediction failed: {str(e)}"}
            )

        # Get security patterns
        rag_patterns = await rag.get_rag_patterns(api_details, risk_assessment)

        # Generate OpenAI test cases
        try:
            openai_test_cases = await openai.generate_openai_test_cases(api_details, risk_assessment)
        except Exception as e:
            logger.error(f"OpenAI generation error: {str(e)}")
            openai_test_cases = []

        # Combine test cases
        combined_test_cases = enhance_test_cases(rag_patterns, openai_test_cases)

        response_data = {
            "api_details": api_details,         # API information from Elasticsearch
            "es_details": es_details,           # Elasticsearch query details
            "risk_assessment": risk_assessment, # ML model predictions
            "test_cases": combined_test_cases   # Combined test cases from RAG and OpenAI
        }

        # Log the complete response for debugging
        logger.debug(f"Complete response data: {json.dumps(response_data, indent=2)}")

        return response_data

    except Exception as e:
        logger.error(f"Error analyzing endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model-performance")
async def get_model_performance():
    """Get performance metrics for all models and vulnerability types including SQL Injection, XSS, Auth Bypass, Rate Limiting, Command Injection, and Path Traversal"""
    try:
        performance_metrics = {}

        # Get metrics from the security model
        if security_model.model_performance:
            for vuln_type, model_metrics in security_model.model_performance.items():
                performance_metrics[vuln_type] = {}
                for model_name, metrics in model_metrics.items():
                    performance_metrics[vuln_type][model_name] = {
                        'precision': round(metrics.get('precision', 0), 3),
                        'recall': round(metrics.get('recall', 0), 3),
                        'f1': round(metrics.get('f1', 0), 3),
                        'training_time': round(metrics.get('training_time', 0), 2)
                    }

        return {
            "status": "success",
            "data": performance_metrics
        }
    except Exception as e:
        logger.error(f"Error getting model performance: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }
def enhance_test_cases(rag_patterns: List[Dict], openai_cases: List[Dict]) -> List[Dict]:
    """Combine and enhance test cases from both sources"""
    combined_cases = []
    seen_patterns = set()

    # Add unique OpenAI cases
    for case in openai_cases:
        key = f"{case['type']}_{case['description']}"
        if key not in seen_patterns:
            seen_patterns.add(key)
            combined_cases.append(case)

    # Process RAG patterns later
    for pattern in rag_patterns:
        key = f"{pattern['type']}_{pattern['description']}"
        if key not in seen_patterns:
            seen_patterns.add(key)
            combined_cases.append(pattern)

    return combined_cases

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)