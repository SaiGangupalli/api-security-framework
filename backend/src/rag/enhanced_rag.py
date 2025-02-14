import numpy as np
from elasticsearch import AsyncElasticsearch
from typing import Dict, List
import logging
from sentence_transformers import SentenceTransformer
import torch
import asyncio
import os

# Set tokenizers parallelism to avoid warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

class EnhancedRAG:
    def __init__(self):
        # Connecting to Elasticsearch database
        self.es = AsyncElasticsearch(['http://elasticsearch:9200'])

        # Loading a pre-trained AI model for understanding text
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Name of the database index where security patterns are stored
        self.index = "security_knowledge_base"

    async def setup_index_mapping(self):
        """Set up Elasticsearch index with dense vector mapping"""
        logger.info("Setting up index mapping...")

        try:
            # Checking Elasticsearch connection
            if not await self.es.ping():
                logger.error("Cannot connect to Elasticsearch")
                raise ConnectionError("Elasticsearch connection failed")

            mapping = {
                "mappings": {
                    "properties": {
                        "category": {"type": "keyword"},
                        "vulnerability_type": {"type": "keyword"},
                        "test_pattern": {"type": "text"},
                        "payload": {"type": "text"},
                        "expected_response": {"type": "text"},
                        "remediation": {"type": "text"},
                        "risk_level": {"type": "keyword"},
                        "pattern_embedding": {
                            "type": "dense_vector",
                            "dims": 384,
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                }
            }

            if await self.es.indices.exists(index=self.index):
                await self.es.indices.delete(index=self.index)
            await self.es.indices.create(index=self.index, body=mapping)
            logger.info(f"Created index {self.index} with vector mapping")

        except Exception as e:
            logger.error(f"Error setting up index: {str(e)}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings for input text"""
        try:
            with torch.no_grad():
                # Using the SentenceTransformer model to create these numerical representations
                embedding = self.model.encode(text, convert_to_tensor=True)
                return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None

    async def index_security_pattern(self, pattern: Dict):
        """Index a security pattern with its embedding"""
        try:
            text_for_embedding = f"{pattern.get('vulnerability_type', '')} {pattern.get('test_pattern', '')} {pattern.get('payload', '')}"
            embedding = self.generate_embedding(text_for_embedding)

            if embedding is None:
                logger.error(f"Could not generate embedding for pattern: {pattern.get('vulnerability_type', 'unknown')}")
                return

            pattern['pattern_embedding'] = embedding
            await self.es.index(index=self.index, document=pattern)

        except Exception as e:
            logger.error(f"Error indexing pattern: {str(e)}")

    async def get_rag_patterns(self, api_details: Dict, risk_assessment: Dict) -> List[Dict]:
        """Get relevant security patterns using hybrid retrieval with risk-aware scoring"""
        try:
            # Build query context
            method = api_details.get('method', '').upper()
            uri = api_details.get('request', {}).get('uri', '')
            headers = str(api_details.get('request', {}).get('headers', {}))
            body = str(api_details.get('request', {}).get('body', {}))

            query_text = f"{method} {uri} {headers} {body}"
            query_embedding = self.generate_embedding(query_text)

            if query_embedding is None:
                logger.error("Failed to generate query embedding")
                return []

            # Map risk scores to vulnerability types
            risk_mapping = {
                'sql_injection_risk': {
                    'types': ['SQL Injection', 'Database Injection', 'NoSQL Injection'],
                    'score': risk_assessment.get('sql_injection_risk', 0)
                },
                'xss_risk': {
                    'types': ['Cross-Site Scripting', 'XSS', 'HTML Injection'],
                    'score': risk_assessment.get('xss_risk', 0)
                },
                'auth_bypass_risk': {
                    'types': ['Authentication Bypass', 'JWT Attack', 'OAuth Token', 'Authentication'],
                    'score': risk_assessment.get('auth_bypass_risk', 0)
                },
                'rate_limiting_risk': {
                    'types': ['Rate Limiting', 'DoS Protection', 'API Rate Limit'],
                    'score': risk_assessment.get('rate_limiting_risk', 0)
                },
                'command_injection_risk': {
                    'types': ['Command Injection', 'OS Command Injection', 'Shell Injection'],
                    'score': risk_assessment.get('command_injection_risk', 0)
                },
                'path_traversal_risk': {
                    'types': ['Path Traversal', 'Directory Traversal', 'File Inclusion'],
                    'score': risk_assessment.get('path_traversal_risk', 0)
                }
            }

            # Building should clauses based on risk scores
            should_clauses = []
            type_to_risk_score = {}  # Mapping vulnerability types to their risk scores

            for risk_key, risk_info in risk_mapping.items():
                risk_score = risk_info['score']
                # Scaling boost based on risk score
                boost = 1.0 + (risk_score * 2)  # Higher risk = higher boost

                # Storing risk score mapping for later use
                for vuln_type in risk_info['types']:
                    type_to_risk_score[vuln_type] = risk_score

                # Adding query clauses for each vulnerability type
                for vuln_type in risk_info['types']:
                    should_clauses.append({
                        "match": {
                            "vulnerability_type": {
                                "query": vuln_type,
                                "boost": boost
                            }
                        }
                    })

            # Building main query
            risk_aware_query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "match_all": {}
                            }
                        ],
                        "should": should_clauses,
                        "minimum_should_match": 0
                    }
                },
                "size": 50,
                "_source": [
                    "vulnerability_type",
                    "test_pattern",
                    "payload",
                    "expected_response",
                    "remediation",
                    "risk_level",
                    "pattern_embedding"
                ]
            }

            # Add method-specific boosting
            if method in ["POST", "PUT", "DELETE"]:
                for high_risk_type in ['SQL Injection', 'Command Injection', 'Authentication Bypass']:
                    risk_aware_query["query"]["bool"]["should"].append({
                        "match": {
                            "vulnerability_type": {
                                "query": high_risk_type,
                                "boost": 1.5
                            }
                        }
                    })

            # Add context-specific boosting
            context_boosts = {
                "auth": {
                    "condition": "auth" in headers.lower() or "jwt" in headers.lower(),
                    "types": ["Authentication", "JWT", "OAuth"],
                    "boost": 2.0
                },
                "json": {
                    "condition": "application/json" in headers.lower(),
                    "types": ["JSON Injection", "API Security"],
                    "boost": 1.5
                },
                "file": {
                    "condition": "multipart/form-data" in headers.lower(),
                    "types": ["File Upload", "Path Traversal"],
                    "boost": 1.8
                },
                "cors": {
                    "condition": any(h.lower().startswith('origin') for h in headers),
                    "types": ["CORS", "Cross-Origin"],
                    "boost": 1.3
                }
            }

            for context, boost_info in context_boosts.items():
                if boost_info["condition"]:
                    for vuln_type in boost_info["types"]:
                        risk_aware_query["query"]["bool"]["should"].append({
                            "match": {
                                "vulnerability_type": {
                                    "query": vuln_type,
                                    "boost": boost_info["boost"]
                                }
                            }
                        })

            logger.info("Fetching patterns with risk-aware scoring...")
            result = await self.es.search(index=self.index, body=risk_aware_query)
            logger.info(f"Found {len(result['hits']['hits'])} patterns")

            # Process results with risk-aware scoring
            scored_patterns = []
            for hit in result["hits"]["hits"]:
                pattern = hit["_source"]
                es_score = hit["_score"]

                # Get pattern embedding
                pattern_text = f"{pattern.get('vulnerability_type', '')} {pattern.get('test_pattern', '')} {pattern.get('payload', '')}"
                pattern_embedding = pattern.get('pattern_embedding') or self.generate_embedding(pattern_text)

                if pattern_embedding:
                    # Calculate vector similarity
                    similarity = float(np.dot(query_embedding, pattern_embedding) /
                                       (np.linalg.norm(query_embedding) * np.linalg.norm(pattern_embedding)))

                    # Get risk score for this vulnerability type
                    vuln_type = pattern["vulnerability_type"]
                    risk_score = 0.1  # Default low risk
                    for risk_info in risk_mapping.values():
                        if vuln_type in risk_info['types']:
                            risk_score = risk_info['score']
                            break

                    # Calculating combined score with risk weighting
                    # Risk score heavily influences final ranking
                    combined_score = (
                            similarity * 0.3 +  # 30% vector similarity
                            es_score * 0.3 +    # 30% elasticsearch score
                            risk_score * 0.4    # 40% risk assessment score
                    )

                    # Generate risk-appropriate steps
                    steps = [
                        f"Test with payload: {pattern['payload']}",
                        f"Set method to: {method}",
                        "Monitor response and error messages"
                    ]

                    # Add risk-specific steps
                    if risk_score > 0.7:  # High risk
                        steps.extend([
                            "Implement additional logging and monitoring",
                            "Verify all security controls are active",
                            "Test with multiple payloads"
                        ])
                    elif risk_score > 0.4:  # Medium risk
                        steps.extend([
                            "Verify security controls",
                            "Test edge cases"
                        ])

                    # Add context-aware steps
                    for context, boost_info in context_boosts.items():
                        if boost_info["condition"]:
                            if context == "auth":
                                steps.append("Verify authentication headers and tokens")
                            elif context == "json":
                                steps.append("Validate JSON payload structure")
                            elif context == "file":
                                steps.append("Verify file upload security controls")
                            elif context == "cors":
                                steps.append("Test CORS configuration")

                    scored_patterns.append({
                        "type": pattern["vulnerability_type"],
                        "description": pattern["test_pattern"],
                        "relevance_score": combined_score,
                        "risk_score": risk_score,
                        "test_cases": [{
                            "name": f"Test {pattern['vulnerability_type']}",
                            "description": pattern["test_pattern"],
                            "priority": pattern["risk_level"],
                            "steps": steps,
                            "expected_results": pattern["expected_response"],
                            "remediation": pattern["remediation"]
                        }]
                    })

            # Sort by combined score
            scored_patterns.sort(key=lambda x: x["relevance_score"], reverse=True)

            # Ensure diversity while maintaining risk awareness
            diverse_patterns = []
            seen_types = set()
            high_risk_count = 0
            medium_risk_count = 0

            # First add high-risk patterns (risk_score > 0.7)
            for pattern in scored_patterns:
                if pattern["risk_score"] > 0.7 and high_risk_count < 5:
                    if pattern["type"] not in seen_types:
                        diverse_patterns.append(pattern)
                        seen_types.add(pattern["type"])
                        high_risk_count += 1

            # Then add medium-risk patterns (risk_score > 0.4)
            for pattern in scored_patterns:
                if 0.4 < pattern["risk_score"] <= 0.7 and medium_risk_count < 3:
                    if pattern["type"] not in seen_types:
                        diverse_patterns.append(pattern)
                        seen_types.add(pattern["type"])
                        medium_risk_count += 1

            # Finally add remaining patterns up to limit
            for pattern in scored_patterns:
                if pattern["type"] not in seen_types:
                    diverse_patterns.append(pattern)
                    seen_types.add(pattern["type"])
                    if len(diverse_patterns) >= 10:
                        break

            logger.info(f"Generated {len(diverse_patterns)} risk-aware test cases")
            logger.debug(f"Risk distribution: {high_risk_count} high risk, {medium_risk_count} medium risk")

            return diverse_patterns

        except Exception as e:
            logger.error(f"Error in risk-aware RAG retrieval: {str(e)}")
            logger.exception("Detailed stack trace:")
            return []

    async def bulk_index_patterns(self, patterns: List[Dict]):
        """Bulk index multiple security patterns with embeddings"""
        try:
            logger.info(f"Starting bulk indexing of {len(patterns)} patterns")
            for idx, pattern in enumerate(patterns, 1):
                logger.debug(f"Indexing pattern {idx}: {pattern.get('vulnerability_type', 'unknown')}")
                await self.index_security_pattern(pattern)
                await asyncio.sleep(0.1)
            logger.info("Bulk indexing completed")
            logger.debug("First pattern indexed (sample):")
            if patterns:
                logger.debug(str(patterns[0]))
        except Exception as e:
            logger.error(f"Error in bulk indexing: {str(e)}")
            raise

    async def refresh_index(self):
        """Refresh the index to make new documents available for search"""
        try:
            await self.es.indices.refresh(index=self.index)
            logger.info("Index refreshed successfully")
        except Exception as e:
            logger.error(f"Error refreshing index: {str(e)}")
            raise