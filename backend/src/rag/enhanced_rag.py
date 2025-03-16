import random

import numpy as np
from elasticsearch import AsyncElasticsearch
from typing import Dict, List, Any, Tuple
import logging
from sentence_transformers import SentenceTransformer
import torch
import asyncio
import os
import re
from urllib.parse import urlparse

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

        # Initializing HTTP method weights for security risk correlation
        self.method_risk_weights = {
            'GET': 0.6,
            'POST': 1.0,
            'PUT': 0.9,
            'DELETE': 1.0,
            'PATCH': 0.9,
            'HEAD': 0.3,
            'OPTIONS': 0.2
        }

        # Setting up common API patterns to improve matching
        self.api_patterns = {
            'user': ['authentication', 'user management', 'access control'],
            'auth': ['authentication', 'jwt', 'token'],
            'login': ['authentication', 'credentials'],
            'payment': ['financial', 'transactions', 'sensitive data'],
            'admin': ['authorization', 'privilege escalation', 'admin access'],
            'file': ['file upload', 'path traversal', 'file handling'],
            'data': ['data validation', 'injection', 'sanitization'],
            'search': ['sqli', 'input validation'],
            'report': ['information disclosure', 'data leakage'],
            'config': ['configuration', 'settings', 'security controls']
        }

    async def setup_index_mapping(self):
        """Setting up Elasticsearch index with dense vector mapping"""
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
                        "test_pattern": {"type": "text",
                                         "analyzer": "standard",
                                         "fields": {
                                             "keyword": {"type": "keyword"}
                                         }},
                        "payload": {"type": "text"},
                        "expected_response": {"type": "text"},
                        "remediation": {"type": "text"},
                        "risk_level": {"type": "keyword"},
                        "pattern_embedding": {
                            "type": "dense_vector",
                            "dims": 384,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "relevance_factors": {
                            "properties": {
                                "http_methods": {"type": "keyword"},
                                "url_patterns": {"type": "keyword"},
                                "data_patterns": {"type": "keyword"},
                                "header_patterns": {"type": "keyword"}
                            }
                        }
                    }
                },
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "path_analyzer": {
                                "type": "custom",
                                "tokenizer": "path_tokenizer"
                            }
                        },
                        "tokenizer": {
                            "path_tokenizer": {
                                "type": "pattern",
                                "pattern": "[/\\\\]"
                            }
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
        """Generating embeddings for input text"""
        try:
            if not text or text.strip() == "":
                return None

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
            # Create a rich text representation for embedding
            vulnerability_text = pattern.get('vulnerability_type', '')
            test_pattern_text = pattern.get('test_pattern', '')
            payload_text = pattern.get('payload', '')

            text_for_embedding = f"{vulnerability_text} {test_pattern_text} {payload_text}"
            embedding = self.generate_embedding(text_for_embedding)

            if embedding is None:
                logger.error(f"Could not generate embedding for pattern: {pattern.get('vulnerability_type', 'unknown')}")
                return

            # Enhance the pattern with relevance factors
            enhanced_pattern = {**pattern}
            enhanced_pattern['pattern_embedding'] = embedding

            # Extract and add relevance factors for better matching
            enhanced_pattern['relevance_factors'] = self._extract_relevance_factors(pattern)

            await self.es.index(index=self.index, document=enhanced_pattern)

        except Exception as e:
            logger.error(f"Error indexing pattern: {str(e)}")

    def _extract_relevance_factors(self, pattern: Dict) -> Dict:
        """Extract relevance factors from a security pattern for better matching"""
        relevance_factors = {
            "http_methods": [],
            "url_patterns": [],
            "data_patterns": [],
            "header_patterns": []
        }

        # Extract HTTP methods
        vulnerability_type = pattern.get('vulnerability_type', '').lower()
        test_pattern = pattern.get('test_pattern', '').lower()
        payload = pattern.get('payload', '').lower()
        combined_text = f"{vulnerability_type} {test_pattern} {payload}"

        # Determine relevant HTTP methods
        if any(word in combined_text for word in ['post', 'submit', 'create', 'add']):
            relevance_factors["http_methods"].append("POST")

        if any(word in combined_text for word in ['get', 'retrieve', 'fetch', 'view']):
            relevance_factors["http_methods"].append("GET")

        if any(word in combined_text for word in ['put', 'update', 'modify']):
            relevance_factors["http_methods"].append("PUT")

        if any(word in combined_text for word in ['delete', 'remove']):
            relevance_factors["http_methods"].append("DELETE")

        # If no specific methods identified, it applies to all
        if not relevance_factors["http_methods"]:
            relevance_factors["http_methods"] = ["GET", "POST", "PUT", "DELETE", "PATCH"]

        # Extract URL patterns
        url_patterns = []
        for key_pattern, related_terms in self.api_patterns.items():
            if key_pattern in combined_text or any(term in combined_text for term in related_terms):
                url_patterns.append(key_pattern)

        relevance_factors["url_patterns"] = url_patterns

        # Extract data patterns
        if 'json' in combined_text or 'application/json' in combined_text:
            relevance_factors["data_patterns"].append("json")
        if 'form' in combined_text or 'multipart' in combined_text:
            relevance_factors["data_patterns"].append("form")
        if 'xml' in combined_text:
            relevance_factors["data_patterns"].append("xml")
        if 'file' in combined_text or 'upload' in combined_text:
            relevance_factors["data_patterns"].append("file")

        # Extract header patterns
        if 'auth' in combined_text or 'token' in combined_text or 'jwt' in combined_text:
            relevance_factors["header_patterns"].append("authorization")
        if 'content-type' in combined_text:
            relevance_factors["header_patterns"].append("content-type")
        if 'cors' in combined_text or 'origin' in combined_text:
            relevance_factors["header_patterns"].append("cors")
        if 'x-api-key' in combined_text:
            relevance_factors["header_patterns"].append("api-key")

        return relevance_factors

    def _extract_path_components(self, uri: str) -> List[str]:
        """Extract meaningful components from the API path"""
        if not uri:
            return []

        try:
            # Parse the URI and extract path
            parsed_uri = urlparse(uri)
            path = parsed_uri.path

            # Split the path by slashes and filter out empty strings
            components = [comp for comp in path.split('/') if comp]

            # Extract query parameters if present
            if parsed_uri.query:
                params = parsed_uri.query.split('&')
                param_names = [param.split('=')[0] for param in params]
                components.extend(param_names)

            return components
        except Exception as e:
            logger.error(f"Error extracting path components: {str(e)}")
            return []

    def _calculate_url_pattern_relevance(self, uri_components: List[str], pattern_url_patterns: List[str]) -> float:
        """Calculate relevance score based on URL pattern matching"""
        if not uri_components or not pattern_url_patterns:
            return 0.1  # Default minimal relevance

        # Check how many components match the pattern
        matches = sum(1 for comp in uri_components if any(pattern in comp.lower() for pattern in pattern_url_patterns))

        # Calculate a relevance score based on matches relative to total components
        max_matches = min(len(uri_components), len(pattern_url_patterns))
        if max_matches == 0:
            return 0.1

        return 0.1 + 0.9 * (matches / max_matches)

    def _calculate_method_relevance(self, request_method: str, pattern_methods: List[str]) -> float:
        """Calculate relevance score based on HTTP method matching"""
        # Direct method match gets highest score
        if request_method in pattern_methods:
            return 1.0

        # Similar semantic methods (POST/PUT/PATCH) get medium score
        write_methods = ['POST', 'PUT', 'PATCH']
        if request_method in write_methods and any(m in pattern_methods for m in write_methods):
            return 0.7

        # Default low relevance
        return 0.3

    def _extract_request_content_type(self, headers: Dict) -> str:
        """Extract content type from request headers"""
        # Check different header name variations
        for header_name in ['Content-Type', 'content-type', 'contentType']:
            if header_name in headers:
                content_type = headers[header_name].lower()
                if 'json' in content_type:
                    return 'json'
                elif 'form' in content_type:
                    return 'form'
                elif 'xml' in content_type:
                    return 'xml'
                elif 'multipart' in content_type:
                    return 'file'
        return 'unknown'

    def _extract_auth_type(self, headers: Dict) -> str:
        """Extract authentication type from request headers"""
        auth_header = headers.get('Authorization', headers.get('authorization', ''))
        if isinstance(auth_header, str):
            auth_header = auth_header.lower()
            if 'bearer' in auth_header:
                return 'jwt'
            elif 'basic' in auth_header:
                return 'basic'
            elif auth_header:
                return 'custom'

        # Check for API key headers
        for header in headers:
            if 'api-key' in header.lower() or 'apikey' in header.lower():
                return 'api-key'

        return 'none'

    def _calculate_data_pattern_relevance(self, request_content_type: str, pattern_data_patterns: List[str]) -> float:
        """Calculate relevance score based on data pattern matching"""
        if request_content_type in pattern_data_patterns:
            return 1.0
        return 0.2

    def _calculate_header_pattern_relevance(self, request_auth_type: str, pattern_header_patterns: List[str]) -> float:
        """Calculate relevance score based on header pattern matching"""
        # Authentication header matching
        if request_auth_type != 'none' and any(pattern in ['authorization', 'auth', 'jwt', 'api-key']
                                               for pattern in pattern_header_patterns):
            return 1.0

        # For other header patterns, check generic match
        if pattern_header_patterns and request_auth_type in pattern_header_patterns:
            return 0.8

        return 0.3

    def _calculate_contextual_relevance(self, api_details: Dict, relevance_factors: Dict) -> Dict[str, float]:
        """Calculate contextual relevance between API and pattern"""
        # Get request details
        method = api_details.get('method', '').upper()
        uri = api_details.get('request', {}).get('uri', '')
        headers = api_details.get('request', {}).get('headers', {})

        uri_components = self._extract_path_components(uri)
        request_content_type = self._extract_request_content_type(headers)
        request_auth_type = self._extract_auth_type(headers)

        # Calculate individual relevance scores
        method_relevance = self._calculate_method_relevance(
            method,
            relevance_factors.get('http_methods', [])
        )

        url_relevance = self._calculate_url_pattern_relevance(
            uri_components,
            relevance_factors.get('url_patterns', [])
        )

        data_relevance = self._calculate_data_pattern_relevance(
            request_content_type,
            relevance_factors.get('data_patterns', [])
        )

        header_relevance = self._calculate_header_pattern_relevance(
            request_auth_type,
            relevance_factors.get('header_patterns', [])
        )

        return {
            'method_relevance': method_relevance,
            'url_relevance': url_relevance,
            'data_relevance': data_relevance,
            'header_relevance': header_relevance
        }

    def _extract_uri_keywords(self, uri: str) -> List[str]:
        """Extract meaningful keywords from the URI"""
        if not uri:
            return []

        # Extract path components
        path_components = self._extract_path_components(uri)

        # Filter out common API version indicators and numbers-only segments
        filtered_components = []
        for comp in path_components:
            # Skip API version segments
            if re.match(r'^v\d+$', comp) or re.match(r'^\d+$', comp):
                continue
            filtered_components.append(comp)

        return filtered_components

    def _get_correlation_boost(self, vulnerability_type: str, method: str, headers: Dict) -> float:
        """Calculate a security correlation boost based on method and vulnerability type"""
        # Base boost value
        boost = 1.0

        # Method-specific boosts
        method_boost = self.method_risk_weights.get(method, 0.5)

        # Vulnerability type specific boosts
        if 'SQL Injection' in vulnerability_type and method in ['POST', 'PUT']:
            boost += 0.5
        elif 'XSS' in vulnerability_type and method == 'GET':
            boost += 0.3
        elif 'Command Injection' in vulnerability_type and method in ['POST', 'PUT']:
            boost += 0.6
        elif 'Authentication' in vulnerability_type:
            auth_header = headers.get('Authorization', '')
            if auth_header:
                boost += 0.4
        elif 'CSRF' in vulnerability_type and method in ['POST', 'PUT', 'DELETE']:
            boost += 0.4
        elif 'Path Traversal' in vulnerability_type and 'file' in vulnerability_type.lower():
            boost += 0.5

        return boost * method_boost

    async def get_rag_patterns(self, api_details: Dict, risk_assessment: Dict) -> List[Dict]:
        """
        Get relevant security patterns using improved retrieval with context-awareness
        """
        try:
            # Extracting API request details from input parameters
            method = api_details.get('method', '').upper()
            uri = api_details.get('request', {}).get('uri', '')
            headers = api_details.get('request', {}).get('headers', {})
            body = api_details.get('request', {}).get('body', {})

            # Identifying meaningful keywords from the API endpoint URL
            uri_keywords = self._extract_uri_keywords(uri)
            uri_keyword_str = " ".join(uri_keywords) if uri_keywords else ""

            # Building a rich query text that incorporates API context
            query_text = f"{method} {uri} {uri_keyword_str}"

            # Enhancing query with authentication and content-type information
            auth_header = headers.get('Authorization', '')
            content_type = headers.get('Content-Type', '')
            if auth_header:
                query_text += f" Authorization {auth_header}"
            if content_type:
                query_text += f" Content-Type {content_type}"

            # Converting text query into vector embedding for semantic similarity matching
            query_embedding = self.generate_embedding(query_text)
            if query_embedding is None:
                logger.error("Failed to generate query embedding")
                return []

            # Mapping vulnerability types to corresponding risk assessment scores
            risk_mapping = {
                'SQL Injection': {
                    'score': risk_assessment.get('sql_injection_risk', 0),
                    'related_terms': ['database injection', 'sql', 'query']
                },
                'Cross-Site Scripting': {
                    'score': risk_assessment.get('xss_risk', 0),
                    'related_terms': ['xss', 'html injection', 'script injection']
                },
                'Authentication Bypass': {
                    'score': risk_assessment.get('auth_bypass_risk', 0),
                    'related_terms': ['jwt attack', 'oauth', 'authentication']
                },
                'Rate Limiting': {
                    'score': risk_assessment.get('rate_limiting_risk', 0),
                    'related_terms': ['dos protection', 'rate limit', 'throttling']
                },
                'Command Injection': {
                    'score': risk_assessment.get('command_injection_risk', 0),
                    'related_terms': ['os command', 'shell', 'exec', 'command execution']
                },
                'Path Traversal': {
                    'score': risk_assessment.get('path_traversal_risk', 0),
                    'related_terms': ['directory traversal', 'file inclusion', 'lfi', 'rfi']
                },
                'Information Disclosure': {
                    'score': max(
                        risk_assessment.get('sql_injection_risk', 0),
                        risk_assessment.get('path_traversal_risk', 0)
                    ) * 0.8,
                    'related_terms': ['information leakage', 'sensitive data exposure']
                },
                'Server-Side Request Forgery': {
                    'score': max(
                        risk_assessment.get('command_injection_risk', 0),
                        risk_assessment.get('path_traversal_risk', 0)
                    ) * 0.7,
                    'related_terms': ['ssrf', 'request forgery']
                }
            }

            # Preparing Elasticsearch query clauses for structured search
            should_clauses = []
            must_clauses = []
            must_not_clauses = []

            # 1. Ensuring test cases match the API's HTTP method
            must_clauses.append({
                "bool": {
                    "should": [
                        {"term": {"relevance_factors.http_methods": method}},
                        {"match": {"test_pattern": method}}
                    ],
                    "minimum_should_match": 1
                }
            })

            # 2. Boosting relevance for test cases mentioning API URI keywords
            if uri_keywords:
                for keyword in uri_keywords:
                    if len(keyword) > 3:  # Only using meaningful keywords
                        should_clauses.append({
                            "match": {
                                "test_pattern": {
                                    "query": keyword,
                                    "boost": 3.0
                                }
                            }
                        })

            # 3. Adding content-type specific matching to find relevant test cases
            content_type_lower = content_type.lower() if content_type else ""
            if "json" in content_type_lower:
                should_clauses.append({"term": {"relevance_factors.data_patterns": "json"}})
            elif "form" in content_type_lower:
                should_clauses.append({"term": {"relevance_factors.data_patterns": "form"}})
            elif "xml" in content_type_lower:
                should_clauses.append({"term": {"relevance_factors.data_patterns": "xml"}})

            # 4. Adjusting search for authentication-related test cases based on headers
            if auth_header:
                if "bearer" in auth_header.lower():
                    should_clauses.append({"term": {"relevance_factors.header_patterns": "authorization"}})
                    should_clauses.append({"match": {"vulnerability_type": {"query": "JWT", "boost": 2.0}}})
                else:
                    should_clauses.append({"term": {"relevance_factors.header_patterns": "authorization"}})

            # 5. Adding risk-based scoring to prioritize high-risk vulnerabilities
            for vuln_type, risk_info in risk_mapping.items():
                risk_score = risk_info['score']
                boost = 1.0 + (risk_score * 2)  # Applying higher boost for higher risk

                # Adding main vulnerability type with appropriate boost
                should_clauses.append({
                    "match": {
                        "vulnerability_type": {
                            "query": vuln_type,
                            "boost": boost
                        }
                    }
                })

                # Including related terms with slightly lower boost factors
                for related_term in risk_info['related_terms']:
                    should_clauses.append({
                        "match": {
                            "vulnerability_type": {
                                "query": related_term,
                                "boost": boost * 0.8
                            }
                        }
                    })

                    should_clauses.append({
                        "match": {
                            "test_pattern": {
                                "query": related_term,
                                "boost": boost * 0.7
                            }
                        }
                    })

            # Constructing the complete Elasticsearch query with all clauses
            combined_query = {
                "query": {
                    "bool": {
                        "must": must_clauses,
                        "should": should_clauses,
                        "must_not": must_not_clauses,
                        "minimum_should_match": 0  # Retrieving documents even if no should clauses match
                    }
                },
                "size": 100,  # Requesting multiple documents for post-processing
                "_source": [
                    "vulnerability_type",
                    "test_pattern",
                    "payload",
                    "expected_response",
                    "remediation",
                    "risk_level",
                    "pattern_embedding",
                    "relevance_factors"
                ]
            }

            # Executing the search against Elasticsearch
            logger.info(f"Executing search for endpoint: {uri}, method: {method}")
            result = await self.es.search(index=self.index, body=combined_query)
            hits = result["hits"]["hits"]
            logger.info(f"Found {len(hits)} initial matches")

            # Implementing fallback search if no results were found
            if len(hits) == 0:
                logger.warning("No matches found - falling back to general search")
                general_query = {
                    "query": {
                        "match_all": {}
                    },
                    "size": 50
                }
                result = await self.es.search(index=self.index, body=general_query)
                hits = result["hits"]["hits"]
                logger.info(f"Fallback search returned {len(hits)} matches")

            # Preparing for post-search ranking and detailed scoring
            scored_patterns = []

            # Defining risk level priority scoring constants
            PRIORITY_SCORES = {
                'Critical': 1.0,
                'High': 0.7,
                'Medium': 0.4,
                'Low': 0.2
            }

            # Adding controlled randomization based on API characteristics
            random_seed = int(sum(ord(c) for c in uri) + ord(method[0]) if method else 0)
            random_offset = random_seed % 10  # Creating variability based on API endpoint

            # Processing each search result with advanced relevance scoring
            for idx, hit in enumerate(hits):
                pattern = hit["_source"]
                es_score = hit["_score"]

                # Retrieving pattern embedding for vector similarity comparison
                pattern_embedding = pattern.get('pattern_embedding')

                if pattern_embedding:
                    # Calculating vector similarity between query and pattern
                    similarity = float(np.dot(query_embedding, pattern_embedding) /
                                       (np.linalg.norm(query_embedding) * np.linalg.norm(pattern_embedding)))

                    # Determining risk score based on vulnerability type
                    vuln_type = pattern["vulnerability_type"]
                    risk_score = 0.1  # Starting with default low risk

                    for risk_type, risk_info in risk_mapping.items():
                        if risk_type in vuln_type or any(term in vuln_type.lower() for term in risk_info['related_terms']):
                            risk_score = max(risk_score, risk_info['score'])

                    # Computing contextual relevance based on API characteristics
                    relevance_factors = pattern.get('relevance_factors', {})
                    contextual_relevance = self._calculate_contextual_relevance(
                        api_details,
                        relevance_factors
                    )

                    # Weighting different aspects of contextual relevance
                    context_weight = (
                            contextual_relevance['method_relevance'] * 0.3 +
                            contextual_relevance['url_relevance'] * 0.35 +
                            contextual_relevance['data_relevance'] * 0.15 +
                            contextual_relevance['header_relevance'] * 0.2
                    )

                    # Applying security-specific correlation boost factors
                    correlation_boost = self._get_correlation_boost(
                        vuln_type,
                        method,
                        headers
                    )

                    # Combining all score components with appropriate weights
                    combined_score = (
                                             similarity * 0.25 +              # Vector similarity (25%)
                                             es_score * 0.15 +                # ES score (15%)
                                             risk_score * 0.25 +              # Risk assessment (25%)
                                             context_weight * 0.35            # Context relevance (35%)
                                     ) * correlation_boost

                    # Adding controlled randomization to prevent identical results
                    random_factor = 1.0 + 0.05 * (((idx + random_offset) % 10) / 10)
                    final_score = combined_score * random_factor

                    # Generating API-specific test steps for this pattern
                    steps = self._generate_contextual_steps(
                        pattern,
                        api_details,
                        risk_score
                    )

                    # Evaluating test case completeness for scoring
                    has_steps = len(steps) > 0
                    has_expected_results = bool(pattern.get("expected_response"))
                    has_remediation = bool(pattern.get("remediation"))

                    # Computing completeness score based on available components
                    completeness_score = (
                            (1.2 if has_steps else 0) +
                            (0.9 if has_expected_results else 0) +
                            (0.9 if has_remediation else 0)
                    )

                    # Converting risk level to numeric priority score
                    risk_level = pattern.get("risk_level", "Medium")
                    priority_score = PRIORITY_SCORES.get(risk_level, PRIORITY_SCORES["Medium"])

                    # Creating detailed explanation of relevance factors for transparency
                    relevance_explanation = {
                        'similarity': round(similarity, 3),
                        'es_score': round(es_score * 0.1, 3),  # Normalizing ES score
                        'risk_score': round(risk_score, 3),
                        'context_weight': round(context_weight, 3),
                        'method_match': round(contextual_relevance['method_relevance'], 3),
                        'url_match': round(contextual_relevance['url_relevance'], 3),
                        'correlation_boost': round(correlation_boost, 3)
                    }

                    # Building comprehensive test case data structure
                    scored_patterns.append({
                        "type": pattern["vulnerability_type"],
                        "description": pattern["test_pattern"],
                        "relevance_score": final_score,
                        "risk_score": risk_score,
                        "contextual_relevance": context_weight,
                        "relevance_explanation": relevance_explanation,
                        "test_cases": [{
                            "name": f"Test {pattern['vulnerability_type']}",
                            "description": pattern["test_pattern"],
                            "priority": pattern["risk_level"],
                            "steps": steps,
                            "expected_results": pattern["expected_response"],
                            "remediation": pattern["remediation"],
                            "total": final_score,
                            "breakdown": {
                                "riskBased": risk_score * 6.0,  # Scaling to expected range
                                "priority": priority_score,
                                "completeness": completeness_score
                            },
                            "components": {
                                "hasSteps": has_steps,
                                "hasExpectedResults": has_expected_results,
                                "hasRemediation": has_remediation
                            }
                        }]
                    })

            # Selecting a diverse set of patterns to ensure good coverage
            diverse_patterns = self._select_diverse_patterns(scored_patterns, api_details)

            logger.info(f"Generated {len(diverse_patterns)} diversified test cases")
            return diverse_patterns

        except Exception as e:
            logger.error(f"Error in retrieving security patterns: {str(e)}")
            logger.exception("Detailed stack trace:")
            return []

    def _generate_contextual_steps(self, pattern: Dict, api_details: Dict, risk_score: float) -> List[str]:
        """Generate contextually relevant test steps based on the API and pattern"""
        # Extract API information
        method = api_details.get('method', '').upper()
        uri = api_details.get('request', {}).get('uri', '')
        headers = api_details.get('request', {}).get('headers', {})

        # Base steps that apply to most test cases
        steps = [
            f"Send a {method} request to {uri}",
            f"Include payload: {pattern.get('payload', 'appropriate test value')}"
        ]

        # Add authentication step if relevant
        auth_header = headers.get('Authorization', '')
        if auth_header and 'Authentication' in pattern.get('vulnerability_type', ''):
            if 'bearer' in auth_header.lower():
                steps.append("Manipulate the JWT token to test for authentication bypass")
            else:
                steps.append("Modify the authorization header to test authentication mechanisms")

        # Add content-type specific steps
        content_type = headers.get('Content-Type', '')
        if 'json' in content_type.lower() and ('Injection' in pattern.get('vulnerability_type', '') or 'XSS' in pattern.get('vulnerability_type', '')):
            steps.append("Insert malicious payloads into JSON fields")

        elif 'form' in content_type.lower():
            steps.append("Modify form fields with test payloads")

        # Add high-risk specific steps
        if risk_score > 0.7:
            steps.extend([
                "Monitor server responses closely for error messages",
                "Check for information disclosure in error responses",
                "Test with multiple variants of the payload"
            ])
        elif risk_score > 0.4:
            steps.append("Observe server response for potential vulnerabilities")

        # Add vulnerability-specific steps
        vuln_type = pattern.get('vulnerability_type', '').lower()
        if 'sql' in vuln_type:
            steps.append("Look for database error messages in the response")
        elif 'xss' in vuln_type:
            steps.append("Check if the payload is reflected in the response")
        elif 'path traversal' in vuln_type:
            steps.append("Attempt to access files outside the intended directory")
        elif 'command injection' in vuln_type:
            steps.append("Look for command execution indicators in the response")
        elif 'authentication' in vuln_type:
            steps.append("Verify if access is incorrectly granted despite invalid credentials")

        return steps

    def _select_diverse_patterns(self, scored_patterns: List[Dict], api_details: Dict) -> List[Dict]:
        """Select diverse patterns with improved algorithm to avoid returning the same patterns"""
        if not scored_patterns:
            return []

        # First, sort by relevance score
        scored_patterns.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Track selected patterns and their types
        selected_patterns = []
        selected_types = set()
        selected_subtypes = set()

        # Extract relevant context for diversity
        method = api_details.get('method', '').upper()
        uri = api_details.get('request', {}).get('uri', '')

        # Map vulnerabilities to broader categories for diversity
        vulnerability_categories = {
            'sql': 'injection',
            'database': 'injection',
            'injection': 'injection',
            'xss': 'client-side',
            'cross-site': 'client-side',
            'script': 'client-side',
            'auth': 'authentication',
            'jwt': 'authentication',
            'token': 'authentication',
            'authentication': 'authentication',
            'path': 'file-access',
            'directory': 'file-access',
            'traversal': 'file-access',
            'file': 'file-access',
            'command': 'code-execution',
            'exec': 'code-execution',
            'shell': 'code-execution',
            'rate': 'availability',
            'limit': 'availability',
            'dos': 'availability'
        }

        # Function to get diversity category
        def get_category(pattern_type):
            pattern_type_lower = pattern_type.lower()
            for key, category in vulnerability_categories.items():
                if key in pattern_type_lower:
                    return category
            return 'other'

        # High-risk patterns first (up to 40% of our selection)
        high_risk_patterns = [p for p in scored_patterns if p.get("risk_score", 0) > 0.7]
        random.shuffle(high_risk_patterns)  # Add randomness

        high_risk_limit = min(5, len(high_risk_patterns))
        for pattern in high_risk_patterns[:high_risk_limit]:
            pattern_type = pattern.get("type", "")
            category = get_category(pattern_type)

            # Only add if we don't have too many of this category already
            if category not in selected_types or len([p for p in selected_patterns if get_category(p.get("type", "")) == category]) < 2:
                selected_patterns.append(pattern)
                selected_types.add(category)
                selected_subtypes.add(pattern_type)

            # Stop if we've selected enough
            if len(selected_patterns) >= 10:
                break

        # Medium risk patterns next (up to 30% of our selection)
        if len(selected_patterns) < 10:
            medium_risk_patterns = [p for p in scored_patterns
                                    if 0.4 <= p.get("risk_score", 0) <= 0.7
                                    and p not in selected_patterns]
            random.shuffle(medium_risk_patterns)  # Add randomness

            medium_risk_limit = min(3, len(medium_risk_patterns))
            for pattern in medium_risk_patterns[:medium_risk_limit]:
                pattern_type = pattern.get("type", "")
                category = get_category(pattern_type)

                # Ensure diversity
                if pattern_type not in selected_subtypes:
                    selected_patterns.append(pattern)
                    selected_types.add(category)
                    selected_subtypes.add(pattern_type)

                # Stop if we've selected enough
                if len(selected_patterns) >= 10:
                    break

        # Fill remaining slots with diverse patterns
        if len(selected_patterns) < 10:
            # Get patterns not yet selected
            remaining_patterns = [p for p in scored_patterns if p not in selected_patterns]

            # Sort by a combination of relevance and diversity
            def diversity_score(pattern):
                pattern_type = pattern.get("type", "")
                category = get_category(pattern_type)
                category_count = sum(1 for p in selected_patterns if get_category(p.get("type", "")) == category)
                return pattern.get("relevance_score", 0) * (1.0 - (category_count * 0.2))

            # Sort by diversity score
            remaining_patterns.sort(key=diversity_score, reverse=True)

            # Add patterns until we reach our limit
            for pattern in remaining_patterns:
                pattern_type = pattern.get("type", "")

                # Skip if we already have this exact type
                if pattern_type in selected_subtypes:
                    continue

                selected_patterns.append(pattern)
                selected_subtypes.add(pattern_type)

                # Stop if we've selected enough
                if len(selected_patterns) >= 10:
                    break

        # Ensure we don't return more than 10 patterns
        return selected_patterns[:10]

    async def bulk_index_patterns(self, patterns: List[Dict]):
        """Bulk index multiple security patterns with embeddings"""
        try:
            logger.info(f"Starting bulk indexing of {len(patterns)} patterns")
            for idx, pattern in enumerate(patterns, 1):
                logger.debug(f"Indexing pattern {idx}: {pattern.get('vulnerability_type', 'unknown')}")
                await self.index_security_pattern(pattern)
                await asyncio.sleep(0.1)  # Small delay to prevent overloading
            logger.info("Bulk indexing completed")

            # Log sample of indexed patterns
            if patterns:
                logger.debug("Sample of indexed patterns:")
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