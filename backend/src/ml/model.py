import json
import logging
import os
import warnings
from pathlib import Path
from typing import Dict
import joblib
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SecurityModel:
    def __init__(self):
        # Configuring warning filters to suppress unnecessary warnings during model operations
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        # Defining comprehensive feature set for security analysis, matching training data schema
        self.feature_names = [
            # Method features
            'is_get', 'is_post', 'is_put', 'is_delete_patch',

            # URI features
            'has_admin', 'has_login', 'has_sql_chars',
            'has_path_traversal', 'has_system_paths',
            'has_sensitive_endpoints',

            # Header features
            'has_auth', 'has_api_key', 'has_content_type',
            'has_jwt', 'has_basic_auth', 'has_rate_limit',
            'has_suspicious_jwt_alg', 'has_weak_jwt',
            'has_suspicious_user_agent',

            # Body and Parameter features
            'has_script_tags', 'has_sql_select', 'has_sql_union',
            'has_large_body', 'has_json_payload', 'has_file_upload',
            'has_command_injection', 'has_system_commands',
            'has_rce_patterns', 'has_serialization_data',

            # Authentication and Authorization features
            'missing_auth', 'invalid_auth', 'expired_token',
            'has_privilege_escalation', 'auth_bypass_attempt',

            # Error indicators
            'has_error_500', 'has_error_401', 'has_error_403',
            'error_indicates_sql', 'error_indicates_file',
            'error_indicates_memory', 'error_indicates_timeout',

            # Rate limiting features
            'rate_limit_low', 'rate_limit_critical',
            'rate_limit_exceeded', 'rate_limit_bypass_attempt',

            # Security headers
            'has_cors_headers', 'has_security_headers', 'has_csrf_token',
            'missing_security_headers', 'weak_security_config',

            # Scanner and tool detection
            'uses_security_scanner', 'uses_automated_tool',
            'uses_fuzzing_tool', 'uses_exploit_tool'
        ]

        # Defining supported vulnerability types for detection
        self.vulnerability_types = [
            'sql_injection',
            'xss',
            'auth_bypass',
            'rate_limiting',
            'command_injection',
            'path_traversal'
        ]

        # Initializing model configuration and default settings
        self.model_types = ['random_forest', 'gradient_boosting', 'neural_network', 'svm']
        self.model_performance = {}
        self.default_risks = {
            'sql_injection_risk': 0.5,
            'xss_risk': 0.5,
            'auth_bypass_risk': 0.5,
            'rate_limiting_risk': 0.5,
            'command_injection_risk': 0.5,
            'path_traversal_risk': 0.5
        }

        # Loading pre-trained models during initialization
        self.load_model()
        self.models = {}

    def load_model(self):
        """Loading trained models and metadata from filesystem"""
        base_path = Path('/app/src/ml/models')
        model_path = base_path / 'security_models.joblib'
        metadata_path = base_path / 'model_metadata.joblib'
        model_performance_path = base_path / 'model_performance.joblib'

        try:
            # Checking NumPy availability and version
            import numpy as np
            logger.debug(f"NumPy version: {np.__version__}")

            # Scanning model directory contents
            logger.info(f"Contents of {base_path}:")
            if base_path.exists():
                for file in base_path.iterdir():
                    logger.info(f"Found file: {file} ({os.path.getsize(file)} bytes)")
            else:
                logger.error(f"Models directory does not exist: {base_path}")

            # Verifying model file existence
            if not model_path.exists():
                logger.warning(f"Model file not found at {model_path}")
                return

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=FutureWarning)

                try:
                    # Loading and initializing model components
                    logger.info("Loading models...")
                    self.models = joblib.load(model_path)
                    metadata = joblib.load(metadata_path)
                    self.model_performance = joblib.load(model_performance_path)
                    self.feature_names = metadata.get('feature_names', [])
                    self.vulnerability_types = metadata.get('vulnerability_types', [])

                    # Validating loaded model data
                    if not self.models or not self.feature_names:
                        raise ValueError("Invalid model data")

                    logger.info("Models and metadata loaded successfully")
                    self._log_model_performance()

                except Exception as e:
                    logger.error(f"Error loading model files: {str(e)}")
                    self.models = {}

        except ImportError as e:
            logger.error(f"NumPy import error: {str(e)}")
            self.models = {}
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            self.models = {}

    def _verify_models(self, test_features):
        """Verifying models can make predictions"""
        try:
            for vuln_type in self.vulnerability_types:
                for model_type, model in self.models.get(vuln_type, {}).items():
                    model.predict_proba(test_features)
        except Exception as e:
            logger.error(f"Model verification failed: {str(e)}")
            self.models = {}

    def _log_model_performance(self):
        """Logging performance metrics for each model"""
        for vuln_type in self.vulnerability_types:
            logger.info(f"\nPerformance metrics for {vuln_type}:")
            for model_type in self.model_types:
                if vuln_type in self.model_performance and model_type in self.model_performance[vuln_type]:
                    metrics = self.model_performance[vuln_type][model_type]
                    logger.info(f"{model_type}:")
                    logger.info(f"F1 Score: {metrics['f1']:.3f}")
                    logger.info(f"Precision: {metrics['precision']:.3f}")
                    logger.info(f"Recall: {metrics['recall']:.3f}")

    def extract_features(self, api_details: Dict) -> np.ndarray:
        """Extracting and processing features from API request details"""
        features = {}

        # Extracting request components
        method = api_details.get('method', '').upper()
        headers = api_details.get('request', {}).get('headers', {})
        uri = api_details.get('request', {}).get('uri', '').lower()
        body = str(api_details.get('request', {}).get('body', {})).lower()
        params = str(api_details.get('request', {}).get('params', {})).lower()
        response = api_details.get('response', {})

        # Processing HTTP method features
        features.update({
            'is_get': 1.0 if method == 'GET' else 0.0,
            'is_post': 1.0 if method == 'POST' else 0.0,
            'is_put': 1.0 if method == 'PUT' else 0.0,
            'is_delete_patch': 1.0 if method in ['DELETE', 'PATCH'] else 0.0,
        })

        # Analyzing URI patterns for sensitive endpoints
        features.update({
            'has_admin': 1.0 if 'admin' in uri else 0.0,
            'has_login': 1.0 if 'login' in uri else 0.0,
            'has_sensitive_endpoints': 1.0 if any(x in uri for x in ['user', 'account', 'auth', 'payment']) else 0.0,
        })

        # Combining text fields for pattern analysis
        all_content = f"{uri} {body} {params}".lower()

        # Detecting SQL injection patterns
        sql_patterns = ["'", '"', ';', '--', 'union', 'select', 'from', 'where', 'drop', 'delete', 'update', 'insert']
        features.update({
            'has_sql_chars': 1.0 if any(char in all_content for char in sql_patterns) else 0.0,
            'has_sql_select': 1.0 if any(cmd in all_content for cmd in ['select', 'union select']) else 0.0,
            'has_sql_union': 1.0 if 'union' in all_content else 0.0
        })

        # Analyzing path traversal attempts
        path_traversal_patterns = ['../', '..\\', '/etc/', 'passwd', 'shadow', 'win.ini']
        features.update({
            'has_path_traversal': 1.0 if any(pattern in all_content for pattern in path_traversal_patterns) else 0.0,
            'has_system_paths': 1.0 if any(x in all_content for x in ['/var/', '/etc/', '/root/', 'c:\\', 'system32']) else 0.0,
        })

        # Processing authentication headers and tokens
        auth_header = str(headers.get('Authorization', '')).lower()
        jwt_token = auth_header.replace('bearer ', '') if 'bearer' in auth_header else ''
        features.update({
                            'has_auth': 1.0 if 'Authorization' in headers else 0.0,
                            'has_api_key': 1.0 if any(k.lower().startswith('x-api-key') for k in headers) else 0.0,
                            'has_jwt': 1.0 if jwt_token else 0.0,
                            'has_basic_auth': 1.0 if 'basic ' in auth_header else 0.0,
                            'has_suspicious_jwt_alg': 1.0 if any(x in jwt_token for x in ['none', 'null', '--']) else 0.0,
                            'has_weak_jwt': 1.0 if jwt_token and len(jwt_token) < 100 else 0.0,
                            'missing_auth': 1.0 if not any(h.lower().startswith(('authorization', 'x-api-key')) for h in headers) else 0.0,
                            'invalid_auth': 1.0 if response.get('status_code') in [401, 403] else 0.0,
                            'expired_token': 1.0 if 'token expired' in str(response.get('body', '')).lower() else 0.0,
                            'auth_bypass_attempt': 1.0 if 'admin' in jwt_token or 'admin' in str(body).lower() else 0.0
        })

        # Analyzing request body characteristics
        content_type = next((v for k, v in headers.items() if 'content-type' in k.lower()), '').lower()
        features.update({
            'has_script_tags': 1.0 if '<script' in all_content else 0.0,
            'has_large_body': 1.0 if len(str(body)) > 5000 else 0.0,
            'has_json_payload': 1.0 if 'application/json' in content_type else 0.0,
            'has_file_upload': 1.0 if 'multipart/form-data' in content_type else 0.0,
            'has_serialization_data': 1.0 if any(x in all_content for x in ['serialize', 'deserialize', '__proto__']) else 0.0
        })

        # Detecting command injection patterns
        cmd_patterns = [';', '&&', '||', '|', '`', '$(', 'cat ', 'echo ', 'rm ', 'mv ']
        features.update({
            'has_command_injection': 1.0 if any(cmd in all_content for cmd in cmd_patterns) else 0.0,
            'has_system_commands': 1.0 if any(cmd in all_content for cmd in ['eval', 'exec', 'system']) else 0.0,
            'has_rce_patterns': 1.0 if any(x in all_content for x in ['wget', 'curl', 'bash', 'nc']) else 0.0
        })

        # Analyzing error responses and status codes
        status_code = response.get('status_code', 0)
        error_body = str(response.get('body', {})).lower()
        features.update({
            'has_error_500': 1.0 if status_code == 500 else 0.0,
            'has_error_401': 1.0 if status_code == 401 else 0.0,
            'has_error_403': 1.0 if status_code == 403 else 0.0,
            'error_indicates_sql': 1.0 if any(x in error_body for x in ['sql', 'database', 'query']) else 0.0,
            'error_indicates_file': 1.0 if any(x in error_body for x in ['file', 'path', 'directory']) else 0.0,
            'error_indicates_memory': 1.0 if 'memory' in error_body else 0.0,
            'error_indicates_timeout': 1.0 if 'timeout' in error_body else 0.0
        })

        # Processing rate limiting features
        rate_headers = {k.lower(): v for k, v in headers.items()}
        remaining = int(rate_headers.get('x-ratelimit-remaining', 1000))
        features.update({
            'has_rate_limit': 1.0 if 'x-ratelimit-limit' in rate_headers else 0.0,
            'rate_limit_low': 1.0 if remaining < 50 else 0.0,
            'rate_limit_critical': 1.0 if remaining < 10 else 0.0,
            'rate_limit_exceeded': 1.0 if status_code == 429 else 0.0,
            'rate_limit_bypass_attempt': 1.0 if any(h.lower().startswith('x-forwarded-') for h in headers) else 0.0
        })

        # Analyzing security headers
        features.update({
            'has_cors_headers': 1.0 if any(h.lower().startswith('access-control-') for h in headers) else 0.0,
            'has_security_headers': 1.0 if any(h in headers for h in ['X-Content-Type-Options', 'X-Frame-Options']) else 0.0,
            'has_csrf_token': 1.0 if any(h.lower().startswith('x-csrf-') for h in headers) else 0.0,
            'missing_security_headers': 1.0 if not any(h in headers for h in ['X-Content-Type-Options', 'X-Frame-Options']) else 0.0,
            'weak_security_config': 1.0 if 'access-control-allow-origin: *' in str(headers).lower() else 0.0
        })

        # Detecting security scanning tools
        user_agent = headers.get('User-Agent', '').lower()
        features.update({
            'uses_security_scanner': 1.0 if any(x in user_agent for x in ['burp', 'zap', 'nikto', 'acunetix']) else 0.0,
            'uses_automated_tool': 1.0 if any(x in user_agent for x in ['python', 'curl', 'wget', 'postman']) else 0.0,
            'uses_fuzzing_tool': 1.0 if any(x in user_agent for x in ['wfuzz', 'ffuf', 'gobuster']) else 0.0,
            'uses_exploit_tool': 1.0 if any(x in user_agent for x in ['metasploit', 'sqlmap', 'hydra']) else 0.0
        })

        # Creating feature vector matching training features
        return np.array([features.get(name, 0.0) for name in self.feature_names])

    def analyze_input_features(self, api_details: Dict) -> Dict[str, float]:
        """Analyzing input features for dynamic risk weighting"""
        # Initializing base risk multipliers for each vulnerability type
        risk_multipliers = {
            'sql_injection': 1.0,
            'xss': 1.0,
            'auth_bypass': 1.0,
            'rate_limiting': 1.0,
            'command_injection': 1.0,
            'path_traversal': 1.0
        }

        # Analyzing HTTP method for risk assessment
        method = api_details.get('method', '').upper()
        if method in ['DELETE', 'PUT']:
            # Increasing risk weights for dangerous HTTP methods
            risk_multipliers['auth_bypass'] *= 1.5
            risk_multipliers['rate_limiting'] *= 1.3
            risk_multipliers['command_injection'] *= 1.4  # Evaluating dangerous methods for command injection
        elif method == 'POST':
            # Adjusting risks for POST requests
            risk_multipliers['sql_injection'] *= 1.3  # Accounting for higher SQL injection risk in POST
            risk_multipliers['xss'] *= 1.2  # Considering stored XSS possibilities
            risk_multipliers['command_injection'] *= 1.3

        # Examining request headers for security vulnerabilities
        headers = api_details.get('request', {}).get('headers', {})
        # Checking authentication headers
        if not any(k.lower().startswith('authorization') for k in headers):
            # Increasing auth bypass risk for missing authorization
            risk_multipliers['auth_bypass'] *= 1.8
        if not any(k.lower().startswith('x-api-key') for k in headers):
            # Adjusting risk for missing API key
            risk_multipliers['auth_bypass'] *= 1.4

        # Assessing security headers
        if not any(h in headers for h in ['X-Content-Type-Options', 'X-Frame-Options']):
            # Increasing XSS risk for missing security headers
            risk_multipliers['xss'] *= 1.3
        if not headers.get('Content-Security-Policy'):
            # Adjusting risks for missing CSP
            risk_multipliers['xss'] *= 1.2
            risk_multipliers['command_injection'] *= 1.2

        # Analyzing content type for specific vulnerabilities
        content_type = next((v for k, v in headers.items() if 'content-type' in k.lower()), '').lower()
        if 'application/json' in content_type:
            # Increasing SQL injection risk for JSON payloads
            risk_multipliers['sql_injection'] *= 1.2  # JSON payloads might contain SQL
        elif 'multipart/form-data' in content_type:
            # Adjusting risks for file upload scenarios
            risk_multipliers['path_traversal'] *= 1.4  # File uploads risk
            risk_multipliers['command_injection'] *= 1.3  # File-based command injection

        # Detecting suspicious user agents
        user_agent = headers.get('User-Agent', '').lower()
        suspicious_agents = [
            'postman', 'curl', 'burp', 'python-requests', 'sqlmap',
            'nikto', 'nmap', 'wfuzz', 'gobuster', 'dirbuster'
        ]
        if any(agent in user_agent for agent in suspicious_agents):
            # Increasing risks for known security testing tools
            risk_multipliers['auth_bypass'] *= 1.3
            risk_multipliers['rate_limiting'] *= 1.2
            risk_multipliers['sql_injection'] *= 1.3
            risk_multipliers['command_injection'] *= 1.4
            risk_multipliers['path_traversal'] *= 1.3

        # Examining URI for sensitive endpoints
        uri = api_details.get('request', {}).get('uri', '').lower()
        sensitive_endpoints = ['webhook', 'admin', 'payment', 'user', 'auth', 'upload', 'file']
        if any(endpoint in uri for endpoint in sensitive_endpoints):
            # Adjusting risks for sensitive endpoint access
            risk_multipliers['auth_bypass'] *= 1.4
            risk_multipliers['rate_limiting'] *= 1.2
            if 'file' in uri or 'upload' in uri:
                # Increasing risks for file-related endpoints
                risk_multipliers['path_traversal'] *= 1.5
                risk_multipliers['command_injection'] *= 1.3

        # Analyzing request body for attack patterns
        body = str(api_details.get('request', {}).get('body', {})).lower()
        # Checking SQL injection patterns
        if any(pattern in body for pattern in ["'", '"', ';', '--', 'union', 'select']):
            risk_multipliers['sql_injection'] *= 1.6
        # Detecting XSS pattern
        if any(pattern in body for pattern in ['<script', 'javascript:', 'onerror=', '<img']):
            risk_multipliers['xss'] *= 1.5
        # Identifying command injection patterns
        if any(pattern in body for pattern in [';', '&&', '||', '|', '`', '$(']):
            risk_multipliers['command_injection'] *= 1.7
        # Looking for path traversal attempts
        if any(pattern in body for pattern in ['../', '..\\', '/etc/', 'c:\\']):
            risk_multipliers['path_traversal'] *= 1.6

        # Evaluating rate limiting configuration
        response = api_details.get('response', {})
        rate_limit = response.get('headers', {}).get('X-RateLimit-Limit', '1000')
        try:
            if int(rate_limit) < 100:  # Low rate limit threshold
                risk_multipliers['rate_limiting'] *= 1.5
        except:
            # Increasing risk when rate limit headers are missing
            risk_multipliers['rate_limiting'] *= 1.2  # No rate limit headers

        # Analyzing response status codes
        status_code = response.get('status_code', 200)
        if status_code in [401, 403]:
            # Adjusting auth bypass risk for authentication failures
            risk_multipliers['auth_bypass'] *= 1.3
        elif status_code == 429:
            # Increasing rate limiting risk for rate limit exceeded
            risk_multipliers['rate_limiting'] *= 2.0
        elif status_code >= 500:
            # Adjusting risks for server errors
            risk_multipliers['sql_injection'] *= 1.2
            risk_multipliers['command_injection'] *= 1.2
            risk_multipliers['path_traversal'] *= 1.2

        # Examining error messages for vulnerability indicators
        error_body = str(response.get('body', '')).lower()
        if any(err in error_body for err in ['sql', 'database', 'query']):
            # Increasing SQL injection risk for database-related errors
            risk_multipliers['sql_injection'] *= 1.4
        if any(err in error_body for err in ['file', 'directory', 'path']):
            # Adjusting path traversal risk for file-related errors
            risk_multipliers['path_traversal'] *= 1.4
        if 'command' in error_body or 'execution' in error_body:
            # Increasing command injection risk for execution-related errors
            risk_multipliers['command_injection'] *= 1.4

        # Capping risk multipliers to prevent excessive scaling
        max_multiplier = 3.0
        return {k: min(v, max_multiplier) for k, v in risk_multipliers.items()}

    def analyze_risks(self, api_details: Dict) -> Dict:
        """Analyzing API security risks with enhanced risk assessment"""
        try:
            # Extracting features from API details and reshaping for model input
            features = self.extract_features(api_details)
            features = features.reshape(1, -1)

            # Getting context-aware risk multipliers based on request characteristics
            risk_multipliers = self.analyze_input_features(api_details)

            # Checking model availability and falling back to defaults if needed
            if not self.models:
                logger.warning("Models not available, using default risk assessment")
                return self._get_default_response(api_details)

            # Initializing containers for tracking risk analysis results
            risk_scores = {}
            model_contributions = {}
            confidence_scores = {}

            # Getting dynamic model weights based on request context
            model_weights = self._calculate_model_weights(api_details)

            # Processing each vulnerability type through ensemble models
            for vuln_type in self.vulnerability_types:
                # Initializing prediction tracking for current vulnerability
                predictions = {}
                model_confidences = {}

                if vuln_type in self.models:
                    # Processing predictions from each model in the ensemble
                    for model_name, model in self.models[vuln_type].items():
                        try:
                            # Getting raw probability predictions from model
                            prob = model.predict_proba(features)[0][1]

                            # Calculating confidence score based on prediction certainty
                            confidence = abs(prob - 0.5) * 2

                            # Applying context-aware weights and adjustments
                            weight = model_weights[vuln_type][model_name]
                            adjusted_prob = self._adjust_probability(prob, vuln_type, model_name, api_details)
                            weighted_prob = adjusted_prob * weight

                            # Storing model predictions and confidence scores
                            predictions[model_name] = weighted_prob
                            model_confidences[model_name] = confidence

                            # Logging prediction details for debugging
                            logger.debug(f"Raw probability for {vuln_type} from {model_name}: {prob:.3f}")
                            logger.debug(f"Weighted probability: {weighted_prob:.3f}")

                        except Exception as e:
                            # Handling prediction failures gracefully
                            logger.warning(f"Prediction failed for {model_name}: {str(e)}")
                            predictions[model_name] = 0.0
                            model_confidences[model_name] = 0.0

                    if predictions:
                        # Calculating confidence-weighted aggregate score
                        total_confidence = sum(model_confidences.values())
                        if total_confidence > 0:
                            # Computing weighted average based on model confidence
                            base_score = sum(
                                pred * conf / total_confidence
                                for pred, conf in zip(predictions.values(), model_confidences.values())
                            )
                        else:
                            # Falling back to simple average when confidence data unavailable
                            base_score = sum(predictions.values()) / len(predictions)

                        # Applying risk multipliers based on context
                        multiplier = risk_multipliers.get(vuln_type, 1.0)
                        final_score = self._calculate_final_score(base_score, multiplier, vuln_type)

                        # Storing final risk scores and model contributions
                        risk_scores[f"{vuln_type}_risk"] = final_score
                        model_contributions[vuln_type] = predictions
                        confidence_scores[vuln_type] = model_confidences

                        # Logging final risk assessment details
                        logger.debug(f"Final risk score for {vuln_type}: {final_score:.3f}")
                        logger.debug(f"Applied multiplier: {multiplier:.2f}")
                    else:
                        # Using default risk scores when predictions unavailable
                        risk_scores[f"{vuln_type}_risk"] = self.default_risks.get(f"{vuln_type}_risk", 0.5)
                else:
                    # Falling back to default risks when model unavailable
                    risk_scores[f"{vuln_type}_risk"] = self.default_risks.get(f"{vuln_type}_risk", 0.5)

            # Building enhanced response with context and confidence data
            response = self._build_enhanced_response(
                api_details,
                risk_scores,
                model_contributions,
                confidence_scores
            )

            return response

        except Exception as e:
            # Handling overall analysis failures
            logger.error(f"Risk analysis failed: {str(e)}")
            logger.exception("Detailed stack trace:")
            return self._get_default_response(api_details)

    def _calculate_model_weights(self, api_details: Dict) -> Dict[str, Dict[str, float]]:
        """Calculate dynamic model weights based on context and historic performance"""
        weights = {}

        # Base model weights
        base_weights = {
            'random_forest': 0.4,
            'gradient_boosting': 0.3,
            'neural_network': 0.1,
            'svm': 0.2
        }

        # Adjusting Weights Based on Expertise Areas
        for vuln_type in self.vulnerability_types:
            weights[vuln_type] = base_weights.copy()

            # Adjusting SQL Injection weights
            if vuln_type == 'sql_injection':
                if any(pattern in str(api_details) for pattern in ["'", '"', '--', 'UNION']):
                    weights[vuln_type]['random_forest'] *= 1.2
                    weights[vuln_type]['gradient_boosting'] *= 1.1

            # Adjusting Command Injection weights
            elif vuln_type == 'command_injection':
                if any(pattern in str(api_details) for pattern in [';', '&&', '|', '`']):
                    weights[vuln_type]['random_forest'] *= 1.2
                    weights[vuln_type]['neural_network'] *= 1.2

            # Adjusting XSS weights
            elif vuln_type == 'xss':
                if any(pattern in str(api_details) for pattern in ['<script', 'javascript:', 'onerror']):
                    weights[vuln_type]['gradient_boosting'] *= 1.2
                    weights[vuln_type]['svm'] *= 1.1

            # Adjusting Auth Bypass weights
            elif vuln_type == 'auth_bypass':
                auth_header = str(api_details.get('request', {}).get('headers', {}).get('Authorization', ''))
                if 'none' in auth_header.lower() or '--' in auth_header:
                    weights[vuln_type]['random_forest'] *= 1.3
                    weights[vuln_type]['gradient_boosting'] *= 1.2

            # Normalizing weights to sum to 1. Ensuring that all model opinions together make up 100% of the decision.
            total_weight = sum(weights[vuln_type].values())
            weights[vuln_type] = {
                k: v / total_weight for k, v in weights[vuln_type].items()
            }

        return weights

    def _adjust_probability(self, prob: float, vuln_type: str, model_name: str, api_details: Dict) -> float:
        """Apply context-aware probability adjustments"""

        # JWT token analysis for auth bypass
        if vuln_type == 'auth_bypass':
            auth_header = str(api_details.get('request', {}).get('headers', {}).get('Authorization', '')).lower()
            if 'none' in auth_header:
                return max(prob, 0.7)  # Minimum 0.7 for 'none' algorithm
            if 'admin' in auth_header and '--' in auth_header:
                return max(prob, 0.8)  # Minimum 0.8 for potential SQL injection in JWT

        # SQL Injection adjustments
        elif vuln_type == 'sql_injection':
            content = str(api_details)
            if "'--" in content or "1=1" in content:
                return max(prob, 0.7)
            if "UNION" in content.upper() and "SELECT" in content.upper():
                return max(prob, 0.8)

        # Command Injection adjustments
        elif vuln_type == 'command_injection':
            content = str(api_details)
            if any(cmd in content for cmd in [';', '&&', '|', '`']):
                return max(prob, 0.7)
            if '/bin/' in content or 'eval(' in content:
                return max(prob, 0.8)

        # Rate Limiting adjustments
        elif vuln_type == 'rate_limiting':
            if api_details.get('response', {}).get('status_code') == 429:
                return max(prob, 0.8)

            headers = api_details.get('request', {}).get('headers', {})
            if any(h.lower().startswith('x-forwarded-') for h in headers):
                return max(prob, 0.6)

        return prob

    def _calculate_final_score(self, base_score: float, multiplier: float, vuln_type: str) -> float:
        """Calculate final risk score with dampening for high scores"""

        # Apply dampening for high scores to prevent excessive amplification
        if base_score > 0.8:
            # Dampen multiplier effect for high base scores
            damped_multiplier = 1.0 + (multiplier - 1.0) * 0.5
            final_score = base_score * damped_multiplier
        else:
            final_score = base_score * multiplier

        # Ensure score is between 0 and 1
        return min(max(float(final_score), 0.0), 1.0)

    def _build_enhanced_response(self, api_details: Dict, risk_scores: Dict,
                                 model_contributions: Dict, confidence_scores: Dict) -> Dict:
        """Build enhanced response with additional context and insights"""

        response = {
            **risk_scores,
            "confidence_scores": {
                "method_confidence": self._calculate_method_confidence(api_details),
                "headers_confidence": self._calculate_headers_confidence(api_details),
                "body_confidence": self._calculate_body_confidence(api_details),
                "uri_confidence": self._calculate_uri_confidence(api_details)
            },
            "model_contributions": model_contributions,
            "model_confidence": confidence_scores
        }

        # Add analyzed components
        response["analyzed_components"] = {
            "method": bool(api_details.get('method')),
            "headers": bool(api_details.get('request', {}).get('headers')),
            "body": bool(api_details.get('request', {}).get('body')),
            "uri": bool(api_details.get('request', {}).get('uri'))
        }

        # Calculate overall accuracy
        response["overall_accuracy"] = sum(response["confidence_scores"].values()) / len(response["confidence_scores"])

        # Calculate risk levels
        high_risk_threshold = 0.7
        medium_risk_threshold = 0.4

        high_risks = [vuln for vuln, score in risk_scores.items() if score > high_risk_threshold]
        medium_risks = [vuln for vuln, score in risk_scores.items() if medium_risk_threshold < score <= high_risk_threshold]

        response["high_risk_count"] = len(high_risks)
        response["high_risk_vulnerabilities"] = high_risks
        response["medium_risk_vulnerabilities"] = medium_risks

        # Determine overall risk level
        max_risk = max(risk_scores.values())
        response["overall_risk_level"] = (
            'Critical' if len(high_risks) >= 3 or max_risk > 0.9 else
            'High' if high_risks else
            'Medium' if medium_risks else
            'Low'
        )

        return response

    def _get_default_response(self, api_details: Dict) -> Dict:
        """Generate default response when analysis fails"""
        confidence_scores = {
            "method_confidence": self._calculate_method_confidence(api_details),
            "headers_confidence": self._calculate_headers_confidence(api_details),
            "body_confidence": self._calculate_body_confidence(api_details),
            "uri_confidence": self._calculate_uri_confidence(api_details)
        }

        return {
            **self.default_risks,
            "confidence_scores": confidence_scores,
            "overall_accuracy": sum(confidence_scores.values()) / len(confidence_scores),
            "analyzed_components": {
                "method": bool(api_details.get('method')),
                "headers": bool(api_details.get('request', {}).get('headers')),
                "body": bool(api_details.get('request', {}).get('body')),
                "uri": bool(api_details.get('request', {}).get('uri'))
            }
        }

    def _build_response(self, api_details: Dict, risk_scores: Dict, model_contributions: Dict) -> Dict:
        """Build final response with all metrics"""
        confidence_scores = {
            "method_confidence": self._calculate_method_confidence(api_details),
            "headers_confidence": self._calculate_headers_confidence(api_details),
            "body_confidence": self._calculate_body_confidence(api_details),
            "uri_confidence": self._calculate_uri_confidence(api_details)
        }

        return {
            **risk_scores,
            "confidence_scores": confidence_scores,
            "overall_accuracy": sum(confidence_scores.values()) / len(confidence_scores),
            "analyzed_components": {
                "method": bool(api_details.get('method')),
                "headers": bool(api_details.get('request', {}).get('headers')),
                "body": bool(api_details.get('request', {}).get('body')),
                "uri": bool(api_details.get('request', {}).get('uri'))
            },
            "model_contributions": model_contributions
        }

    def _calculate_method_confidence(self, api_details: Dict) -> float:
        method = api_details.get('method', '').upper()
        if not method:
            return 0.0
        return 1.0 if method in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'] else 0.5

    def _calculate_headers_confidence(self, api_details: Dict) -> float:
        headers = api_details.get('request', {}).get('headers', {})
        if not headers:
            return 0.0
        important_headers = ['authorization', 'content-type', 'x-api-key']
        return min(1.0, sum(1 for h in headers if any(ih in h.lower() for ih in important_headers)) / len(important_headers))

    def _calculate_body_confidence(self, api_details: Dict) -> float:
        body = api_details.get('request', {}).get('body')
        if not body:
            return 0.0
        return 1.0 if isinstance(body, (dict, list)) else 0.5

    def _calculate_uri_confidence(self, api_details: Dict) -> float:
        uri = api_details.get('request', {}).get('uri')
        if not uri:
            return 0.0
        return 1.0 if uri.startswith(('http://', 'https://')) else 0.5

    def fallback_security_risk_assessment(self, api_details: Dict) -> Dict:
        """Comprehensive risk assessment based on request headers and body"""
        # Initialize risk scores for all vulnerability types
        risk_assessment = {
            "sql_injection_risk": 0.0,
            "xss_risk": 0.0,
            "auth_bypass_risk": 0.0,
            "rate_limiting_risk": 0.0,
            "command_injection_risk": 0.0,
            "path_traversal_risk": 0.0
        }

        # Extract request components
        headers = api_details.get('request', {}).get('headers', {})
        body = api_details.get('request', {}).get('body', {})
        params = api_details.get('request', {}).get('params', {})
        method = api_details.get('method', '').upper()
        uri = api_details.get('request', {}).get('uri', '')
        all_content = f"{uri} {str(body).lower()} {str(params).lower()}"

        # Authentication Risk Assessment
        auth_risk = 0.0
        if not any(h.lower().startswith(('authorization', 'x-api-key')) for h in headers):
            auth_risk += 0.8  # Missing auth headers
        if any('bearer' in str(h).lower() for h in headers.values()):
            token = next((v for k, v in headers.items() if 'bearer' in k.lower()), '')
            if token:
                if 'null' in str(token).lower() or 'none' in str(token).lower():
                    auth_risk += 0.9  # JWT 'none' algorithm attempt
                if len(token) < 100:
                    auth_risk += 0.7  # Suspiciously short token
        if 'admin' in str(body).lower() or 'admin' in uri.lower():
            auth_risk += 0.6  # Potential privilege escalation attempt
        risk_assessment["auth_bypass_risk"] = min(auth_risk, 1.0)

        # SQL Injection Risk Assessment
        sql_patterns = ["'", "OR 1=1", "SELECT", "UNION", "--", ";", "DROP", "DELETE", "UPDATE", "INSERT"]
        sql_risk = sum(0.3 for pattern in sql_patterns if pattern.lower() in all_content)
        if method in ['POST', 'PUT', 'DELETE']:
            sql_risk += 0.2
        if any(x in str(api_details.get('response', {}).get('body', '')).lower()
               for x in ['sql', 'database', 'query']):
            sql_risk += 0.4  # SQL error messages detected
        risk_assessment["sql_injection_risk"] = min(sql_risk, 1.0)

        # XSS Risk Assessment
        xss_patterns = ["<script>", "javascript:", "onerror=", "onload=", "<img", "<svg", "alert(",
                        "onclick", "onmouseover", "onfocus", "<iframe"]
        xss_risk = sum(0.3 for pattern in xss_patterns if pattern.lower() in all_content)
        content_type = next((v for k, v in headers.items() if 'content-type' in k.lower()), '')
        if 'html' in str(content_type).lower() or 'text' in str(content_type).lower():
            xss_risk += 0.2
        if not any(h in headers for h in ['X-Content-Type-Options', 'X-XSS-Protection']):
            xss_risk += 0.3  # Missing XSS protection headers
        risk_assessment["xss_risk"] = min(xss_risk, 1.0)

        # Command Injection Risk Assessment
        cmd_patterns = [';', '&&', '||', '|', '`', '$(', 'cat ', 'echo ', 'rm ', 'mv ', '/etc/', '/bin/',
                        'wget', 'curl', 'bash', 'nc', 'powershell', 'cmd.exe']
        cmd_risk = sum(0.4 for pattern in cmd_patterns if pattern in all_content)
        if any(cmd in all_content for cmd in ['eval(', 'exec(', 'system(']):
            cmd_risk += 0.6  # Direct command execution attempts
        if not headers.get('Content-Security-Policy'):
            cmd_risk += 0.2  # Missing CSP header
        risk_assessment["command_injection_risk"] = min(cmd_risk, 1.0)

        # Path Traversal Risk Assessment
        path_patterns = ['../', '..\\', '/etc/passwd', 'shadow', 'win.ini', '/proc/', '/var/log/',
                         'boot.ini', '/windows/system32/', 'web.config', '.htaccess']
        path_risk = sum(0.4 for pattern in path_patterns if pattern in all_content)
        if any(x in str(api_details.get('response', {}).get('body', '')).lower()
               for x in ['file', 'directory', 'path']):
            path_risk += 0.3  # File-related error messages
        if 'file' in str(params).lower() or 'path' in str(params).lower():
            path_risk += 0.2  # File/path parameters detected
        risk_assessment["path_traversal_risk"] = min(path_risk, 1.0)

        # Rate Limiting Risk Assessment
        rate_risk = 0.0
        if not any(h.lower().startswith('x-ratelimit') for h in headers):
            rate_risk += 0.7  # No rate limiting headers
        if api_details.get('response', {}).get('status_code') == 429:
            rate_risk += 0.8  # Rate limit exceeded

        # Check for rate limit bypass attempts
        if any(h.lower().startswith(('x-forwarded-', 'forwarded', 'x-real-ip')) for h in headers):
            rate_risk += 0.4  # Potential rate limit bypass attempt

        # User agent analysis
        user_agent = headers.get('User-Agent', '').lower()
        if any(tool in user_agent for tool in ['burp', 'postman', 'curl', 'wget', 'python-requests']):
            rate_risk += 0.3  # Automated tool detection
        risk_assessment["rate_limiting_risk"] = min(rate_risk, 1.0)

        # Calculate confidence scores for each component
        confidence_scores = {
            "method_confidence": self._calculate_method_confidence(api_details),
            "headers_confidence": self._calculate_headers_confidence(api_details),
            "body_confidence": self._calculate_body_confidence(api_details),
            "uri_confidence": self._calculate_uri_confidence(api_details)
        }

        # Calculate overall accuracy
        overall_accuracy = sum(confidence_scores.values()) / len(confidence_scores)

        # Add analyzed components status
        analyzed_components = {
            "method": bool(method),
            "headers": bool(headers),
            "body": bool(body),
            "uri": bool(uri)
        }

        # Add confidence scores and analysis details to risk assessment
        risk_assessment.update({
            "confidence_scores": confidence_scores,
            "overall_accuracy": overall_accuracy,
            "analyzed_components": analyzed_components
        })

        return risk_assessment