import base64
import random
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
import joblib
import json
import os
from typing import List, Tuple, Dict, Any
from datetime import datetime
import time
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SecurityModelTrainer:
    def __init__(self):
        # Define comprehensive feature set
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

        self.vulnerability_types = [
            'sql_injection',
            'xss',
            'auth_bypass',
            'rate_limiting',
            'command_injection',
            'path_traversal'
        ]

        # Initialize storage paths
        self.data_dir = Path('/app/src/ml/data')
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model storage
        self.models = {}
        self.model_metrics = {}

        # Load attack pattern definitions
        self.attack_patterns = self.load_attack_patterns()

    def load_attack_patterns(self) -> Dict:
        """Load attack pattern definitions"""
        return {
            'command_injection': {
                'patterns': [';', '&&', '||', '|', '`', '$(', 'cat ', 'echo ',
                             'rm ', 'mv ', '/etc/', '/bin/', '/usr/', 'curl ',
                             'wget ', 'bash ', 'sh ', 'cmd ', 'powershell ',
                             'exec ', 'eval ', 'system '],
                'risk_weight': 2.0
            },
            'path_traversal': {
                'patterns': ['../', '..\\', '/etc/passwd', 'shadow', 'win.ini',
                             'web.config', '.htaccess', '/proc/', '/var/log/',
                             'boot.ini', '/windows/system32/'],
                'risk_weight': 1.8
            },
            'sql_injection': {
                'patterns': ["'", '"', ';', '--', 'union', 'select', 'from',
                             'where', 'drop', 'delete', 'update', 'insert',
                             '1=1', 'or 1', 'admin\'--'],
                'risk_weight': 1.9
            },
            'xss': {
                'patterns': ['<script', 'javascript:', 'onerror=', 'onload=',
                             'eval(', 'alert(', 'document.cookie', 'onclick',
                             'onmouseover', '<img', '<svg', '<iframe'],
                'risk_weight': 1.7
            },
            'security_scanners': {
                'patterns': ['nikto', 'burp', 'zap', 'zgrab', 'nmap',
                             'gobuster', 'wfuzz', 'sqlmap', 'metasploit',
                             'acunetix', 'nessus'],
                'risk_weight': 1.5
            },
            'suspicious_jwt': {
                'algorithms': ['none', 'null', 'hs256', 'rs256'],
                'patterns': ['admin', 'root', 'system', 'role=admin',
                             'isAdmin=true'],
                'risk_weight': 2.0
            }
        }

    def analyze_jwt_token(self, token: str) -> Dict[str, bool]:
        """Analyze JWT token for suspicious patterns"""
        results = {
            'has_suspicious_jwt_alg': False,
            'has_weak_jwt': False,
            'has_privilege_escalation': False
        }

        try:
            # Split and decode JWT parts
            parts = token.split('.')
            if len(parts) >= 2:
                # Analyze header
                header = base64.b64decode(parts[0] + '==').decode()
                if any(alg in header.lower()
                       for alg in self.attack_patterns['suspicious_jwt']['algorithms']):
                    results['has_suspicious_jwt_alg'] = True
                    results['has_weak_jwt'] = True

                # Analyze payload
                payload = base64.b64decode(parts[1] + '==').decode()
                if any(pattern in payload.lower()
                       for pattern in self.attack_patterns['suspicious_jwt']['patterns']):
                    results['has_privilege_escalation'] = True
        except:
            pass

        return results

    def extract_attack_patterns(self, content: str) -> Dict[str, float]:
        """Extract attack patterns from content with risk scoring"""
        results = {}

        for attack_type, config in self.attack_patterns.items():
            matches = sum(1 for pattern in config['patterns']
                          if pattern in content.lower())
            if matches > 0:
                # Calculate risk score based on number of matches and weight
                risk_score = min(matches * config['risk_weight'] / 10.0, 1.0)
                results[f'has_{attack_type}'] = risk_score
            else:
                results[f'has_{attack_type}'] = 0.0

        return results

    def analyze_request_patterns(self) -> Dict[str, List[float]]:
        """Generate realistic probability distributions for request patterns"""
        return {
            'method_patterns': {
                'is_get': [0.7, 0.3],      # 70% GET requests
                'is_post': [0.6, 0.4],     # 40% POST requests
                'is_put': [0.8, 0.2],      # 20% PUT requests
                'is_delete_patch': [0.9, 0.1]  # 10% DELETE/PATCH requests
            },
            'uri_patterns': {
                'has_admin': [0.95, 0.05],  # 5% admin paths
                'has_login': [0.9, 0.1],    # 10% login paths
                'has_sql_chars': [0.99, 0.01],  # 1% SQL injection attempts
                'has_path_traversal': [0.99, 0.01],  # 1% path traversal
                'has_system_paths': [0.99, 0.01],    # 1% system paths
                'has_sensitive_endpoints': [0.95, 0.05]  # 5% sensitive endpoints
            },
            'header_patterns': {
                'has_auth': [0.2, 0.8],     # 80% authenticated
                'has_api_key': [0.3, 0.7],  # 70% with API keys
                'has_content_type': [0.1, 0.9],  # 90% content type
                'has_jwt': [0.4, 0.6],      # 60% JWT auth
                'has_basic_auth': [0.8, 0.2],  # 20% basic auth
                'has_suspicious_jwt_alg': [0.99, 0.01],  # 1% suspicious JWT
                'has_weak_jwt': [0.98, 0.02]  # 2% weak JWT
            },
            'body_patterns': {
                'has_script_tags': [0.99, 0.01],  # 1% XSS attempts
                'has_sql_select': [0.99, 0.01],   # 1% SQL queries
                'has_sql_union': [0.995, 0.005],  # 0.5% UNION attacks
                'has_command_injection': [0.999, 0.001],  # 0.1% command injection
                'has_system_commands': [0.998, 0.002],   # 0.2% system commands
                'has_large_body': [0.8, 0.2]      # 20% large payloads
            },
            'scanner_patterns': {
                'uses_security_scanner': [0.99, 0.01],  # 1% security scanners
                'uses_automated_tool': [0.95, 0.05],    # 5% automated tools
                'uses_fuzzing_tool': [0.995, 0.005],    # 0.5% fuzzing tools
                'uses_exploit_tool': [0.999, 0.001]     # 0.1% exploit tools
            }
        }

    def generate_training_data(self, n_samples: int = 10000) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """Generate synthetic training data with more diverse and realistic patterns"""
        data = []

        # Define traffic scenario probabilities
        scenario_probs = {
            'normal_traffic': 0.7,
            'suspicious_traffic': 0.2,
            'malicious_traffic': 0.1
        }

        for _ in range(n_samples):
            sample = {}
            scenario = np.random.choice(
                list(scenario_probs.keys()),
                p=list(scenario_probs.values())
            )

            # Generate request method features
            method = np.random.choice(['GET', 'POST', 'PUT', 'DELETE'], p=[0.7, 0.2, 0.05, 0.05])
            sample.update({
                'is_get': float(method == 'GET'),
                'is_post': float(method == 'POST'),
                'is_put': float(method == 'PUT'),
                'is_delete_patch': float(method == 'DELETE')
            })

            # Generate URI features
            uri_components = []
            if scenario == 'malicious_traffic':
                uri_components.extend([
                    '/admin', '/login', '/user', '/auth', '/payment',
                    '../', '..\\', '/etc/', '/proc/', 'cmd', 'eval'
                ])
            elif scenario == 'suspicious_traffic':
                uri_components.extend(['/api', '/data', '/public', '/docs'])
            else:
                uri_components.extend(['/home', '/products', '/categories', '/items'])

            uri = np.random.choice(uri_components)
            sample.update({
                'has_admin': float('admin' in uri.lower()),
                'has_login': float('login' in uri.lower()),
                'has_sql_chars': float(any(c in uri for c in ["'", '"', ';', '--'])),
                'has_path_traversal': float(any(p in uri for p in ['../', '..\\', '/etc/'])),
                'has_system_paths': float(any(p in uri for p in ['/var/', '/etc/', '/root/', '/proc/'])),
                'has_sensitive_endpoints': float(any(e in uri for e in ['admin', 'login', 'auth', 'payment']))
            })

            # Generate header features
            headers = self._generate_headers(scenario)
            sample.update(headers)

            # Generate body and parameter features
            body_features = self._generate_body_features(scenario)
            sample.update(body_features)

            # Generate error features
            error_features = self._generate_error_features(scenario)
            sample.update(error_features)

            # Generate rate limiting features
            rate_features = self._generate_rate_limiting_features(scenario)
            sample.update(rate_features)

            # Add noise
            sample = self._add_noise_to_features(sample)
            data.append(sample)

        # Create DataFrame
        X = pd.DataFrame(data)

        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in X.columns:
                X[feature] = 0

        # Generate vulnerability labels
        y_dict = self.generate_vulnerability_labels(X)

        # Clean data
        X = self.clean_dataframe(X)

        return X, y_dict

    def _generate_headers(self, scenario: str) -> Dict[str, float]:
        """Generate header-related features based on traffic scenario"""
        headers = {}

        # Authentication headers
        if scenario == 'normal_traffic':
            headers.update({
                'has_auth': 1.0,
                'has_api_key': float(np.random.random() > 0.3),
                'has_content_type': 1.0,
                'has_jwt': float(np.random.random() > 0.4),
                'has_basic_auth': float(np.random.random() > 0.7),
                'has_rate_limit': 1.0,
                'has_suspicious_jwt_alg': 0.0,
                'has_weak_jwt': 0.0,
                'has_suspicious_user_agent': 0.0
            })
        elif scenario == 'suspicious_traffic':
            headers.update({
                'has_auth': float(np.random.random() > 0.3),
                'has_api_key': float(np.random.random() > 0.5),
                'has_content_type': float(np.random.random() > 0.2),
                'has_jwt': float(np.random.random() > 0.5),
                'has_basic_auth': float(np.random.random() > 0.5),
                'has_rate_limit': float(np.random.random() > 0.3),
                'has_suspicious_jwt_alg': float(np.random.random() > 0.8),
                'has_weak_jwt': float(np.random.random() > 0.8),
                'has_suspicious_user_agent': float(np.random.random() > 0.7)
            })
        else:  # malicious_traffic
            headers.update({
                'has_auth': float(np.random.random() > 0.7),
                'has_api_key': float(np.random.random() > 0.8),
                'has_content_type': float(np.random.random() > 0.5),
                'has_jwt': 1.0,
                'has_basic_auth': float(np.random.random() > 0.8),
                'has_rate_limit': float(np.random.random() > 0.8),
                'has_suspicious_jwt_alg': float(np.random.random() > 0.3),
                'has_weak_jwt': float(np.random.random() > 0.3),
                'has_suspicious_user_agent': float(np.random.random() > 0.2)
            })

        # Add security headers
        headers.update({
            'has_cors_headers': float(scenario == 'normal_traffic'),
            'has_security_headers': float(scenario == 'normal_traffic'),
            'has_csrf_token': float(scenario == 'normal_traffic'),
            'missing_security_headers': float(scenario != 'normal_traffic'),
            'weak_security_config': float(scenario == 'malicious_traffic')
        })

        return headers

    def _generate_body_features(self, scenario: str) -> Dict[str, float]:
        """Generate body and parameter related features"""
        features = {}

        if scenario == 'malicious_traffic':
            features.update({
                'has_script_tags': float(np.random.random() > 0.3),
                'has_sql_select': float(np.random.random() > 0.3),
                'has_sql_union': float(np.random.random() > 0.4),
                'has_large_body': float(np.random.random() > 0.5),
                'has_json_payload': float(np.random.random() > 0.5),
                'has_file_upload': float(np.random.random() > 0.6),
                'has_command_injection': float(np.random.random() > 0.3),
                'has_system_commands': float(np.random.random() > 0.4),
                'has_rce_patterns': float(np.random.random() > 0.3),
                'has_serialization_data': float(np.random.random() > 0.5)
            })
        elif scenario == 'suspicious_traffic':
            features.update({
                'has_script_tags': float(np.random.random() > 0.7),
                'has_sql_select': float(np.random.random() > 0.7),
                'has_sql_union': float(np.random.random() > 0.8),
                'has_large_body': float(np.random.random() > 0.6),
                'has_json_payload': float(np.random.random() > 0.3),
                'has_file_upload': float(np.random.random() > 0.7),
                'has_command_injection': float(np.random.random() > 0.8),
                'has_system_commands': float(np.random.random() > 0.8),
                'has_rce_patterns': float(np.random.random() > 0.8),
                'has_serialization_data': float(np.random.random() > 0.7)
            })
        else:  # normal_traffic
            features.update({
                'has_script_tags': 0.0,
                'has_sql_select': float(np.random.random() > 0.9),
                'has_sql_union': 0.0,
                'has_large_body': float(np.random.random() > 0.8),
                'has_json_payload': float(np.random.random() > 0.2),
                'has_file_upload': float(np.random.random() > 0.8),
                'has_command_injection': 0.0,
                'has_system_commands': 0.0,
                'has_rce_patterns': 0.0,
                'has_serialization_data': float(np.random.random() > 0.9)
            })

        return features

    def _generate_error_features(self, scenario: str) -> Dict[str, float]:
        """Generate error-related features"""
        features = {}

        if scenario == 'malicious_traffic':
            features.update({
                'has_error_500': float(np.random.random() > 0.5),
                'has_error_401': float(np.random.random() > 0.3),
                'has_error_403': float(np.random.random() > 0.3),
                'error_indicates_sql': float(np.random.random() > 0.5),
                'error_indicates_file': float(np.random.random() > 0.5),
                'error_indicates_memory': float(np.random.random() > 0.7),
                'error_indicates_timeout': float(np.random.random() > 0.7)
            })
        elif scenario == 'suspicious_traffic':
            features.update({
                'has_error_500': float(np.random.random() > 0.7),
                'has_error_401': float(np.random.random() > 0.5),
                'has_error_403': float(np.random.random() > 0.5),
                'error_indicates_sql': float(np.random.random() > 0.8),
                'error_indicates_file': float(np.random.random() > 0.8),
                'error_indicates_memory': float(np.random.random() > 0.8),
                'error_indicates_timeout': float(np.random.random() > 0.8)
            })
        else:  # normal_traffic
            features.update({
                'has_error_500': float(np.random.random() > 0.95),
                'has_error_401': float(np.random.random() > 0.9),
                'has_error_403': float(np.random.random() > 0.9),
                'error_indicates_sql': 0.0,
                'error_indicates_file': float(np.random.random() > 0.95),
                'error_indicates_memory': float(np.random.random() > 0.95),
                'error_indicates_timeout': float(np.random.random() > 0.95)
            })

        return features

    def _generate_rate_limiting_features(self, scenario: str) -> Dict[str, float]:
        """Generate rate limiting related features"""
        features = {}

        if scenario == 'malicious_traffic':
            features.update({
                'rate_limit_low': float(np.random.random() > 0.3),
                'rate_limit_critical': float(np.random.random() > 0.4),
                'rate_limit_exceeded': float(np.random.random() > 0.5),
                'rate_limit_bypass_attempt': float(np.random.random() > 0.3)
            })
        elif scenario == 'suspicious_traffic':
            features.update({
                'rate_limit_low': float(np.random.random() > 0.5),
                'rate_limit_critical': float(np.random.random() > 0.7),
                'rate_limit_exceeded': float(np.random.random() > 0.8),
                'rate_limit_bypass_attempt': float(np.random.random() > 0.8)
            })
        else:  # normal_traffic
            features.update({
                'rate_limit_low': float(np.random.random() > 0.9),
                'rate_limit_critical': float(np.random.random() > 0.95),
                'rate_limit_exceeded': 0.0,
                'rate_limit_bypass_attempt': 0.0
            })

        return features

    def _add_noise_to_features(self, sample: Dict[str, float]) -> Dict[str, float]:
        """Add random noise to feature values"""
        noisy_sample = {}
        for key, value in sample.items():
            if value == 1.0:
                # Add small negative noise to 1.0 values
                noise = np.random.uniform(0, 0.1)
                noisy_sample[key] = max(0.0, min(1.0, value - noise))
            elif value == 0.0:
                # Add small positive noise to 0.0 values
                noise = np.random.uniform(0, 0.1)
                noisy_sample[key] = max(0.0, min(1.0, value + noise))
            else:
                # Add bidirectional noise to intermediate values
                noise = np.random.uniform(-0.1, 0.1)
                noisy_sample[key] = max(0.0, min(1.0, value + noise))

        return noisy_sample

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the DataFrame to remove NaNs, infinities, and ensure data type consistency
        """
        # Replace NaNs with 0
        df = df.fillna(0)

        # Replace infinities with large finite numbers
        df = df.replace([np.inf, -np.inf], np.finfo(np.float32).max)

        # Ensure all columns are numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Clip values to prevent extremely large numbers
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].clip(
            lower=-1e10,
            upper=1e10
        )

        return df

    def generate_vulnerability_labels(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate vulnerability labels with weighted risk factors"""
        y_dict = {}
        X_bool = X > 0.5  # Convert to boolean, any value > 0.5 is considered True

        # SQL Injection risk factors
        sql_factors = np.zeros(len(X))
        sql_factors += (2.0 * (X_bool['has_sql_chars'] & (X_bool['is_post'] | X_bool['is_put'])))
        sql_factors += (1.8 * (X_bool['has_sql_select'] & X_bool['has_sql_union']))
        sql_factors += (1.5 * (X_bool['has_sql_chars'] & ~X_bool['has_auth']))
        sql_factors += (1.2 * (X_bool['error_indicates_sql'] & X_bool['has_error_500']))
        sql_factors += (1.4 * (X_bool['has_sql_chars'] & X_bool['uses_security_scanner']))
        y_dict['sql_injection'] = (sql_factors > 0).astype(np.int32)

        # XSS risk factors
        xss_factors = np.zeros(len(X))
        xss_factors += (2.0 * (X_bool['has_script_tags'] & ~X_bool['has_content_type']))
        xss_factors += (1.8 * (X_bool['has_script_tags'] & X_bool['has_large_body']))
        xss_factors += (1.5 * (X_bool['has_script_tags'] & X_bool['is_post']))
        xss_factors += (1.2 * (X_bool['has_script_tags'] & ~X_bool['has_security_headers']))
        xss_factors += (1.4 * (X_bool['has_script_tags'] & X_bool['uses_security_scanner']))
        y_dict['xss'] = (xss_factors > 0).astype(np.int32)

        # Auth Bypass risk factors
        auth_factors = np.zeros(len(X))
        auth_factors += (2.0 * X_bool['has_suspicious_jwt_alg'])
        auth_factors += (1.8 * (X_bool['has_admin'] & ~X_bool['has_auth']))
        auth_factors += (1.5 * (~X_bool['has_auth'] & ~X_bool['has_api_key']))
        auth_factors += (1.3 * (X_bool['has_jwt'] & ~X_bool['has_security_headers']))
        auth_factors += (1.4 * X_bool['uses_security_scanner'])
        auth_factors += (1.6 * X_bool['auth_bypass_attempt'])
        auth_factors += (1.2 * (X_bool['has_error_401'] | X_bool['has_error_403']))
        y_dict['auth_bypass'] = (auth_factors > 0).astype(np.int32)

        # Command Injection risk factors
        cmd_factors = np.zeros(len(X))
        cmd_factors += (2.0 * X_bool['has_command_injection'])
        cmd_factors += (1.8 * X_bool['has_system_commands'])
        cmd_factors += (1.5 * (X_bool['has_command_injection'] & ~X_bool['has_auth']))
        cmd_factors += (1.3 * (X_bool['has_error_500'] & X_bool['error_indicates_file']))
        cmd_factors += (1.4 * (X_bool['has_command_injection'] & X_bool['uses_security_scanner']))
        y_dict['command_injection'] = (cmd_factors > 0).astype(np.int32)

        # Rate Limiting risk factors
        rate_factors = np.zeros(len(X))
        rate_factors += (2.0 * X_bool['rate_limit_exceeded'])
        rate_factors += (1.8 * X_bool['rate_limit_bypass_attempt'])
        rate_factors += (1.5 * (X_bool['rate_limit_critical'] & X_bool['uses_automated_tool']))
        rate_factors += (1.3 * (~X_bool['has_rate_limit'] & X_bool['has_large_body']))
        rate_factors += (1.4 * (X_bool['uses_security_scanner'] & ~X_bool['has_rate_limit']))
        y_dict['rate_limiting'] = (rate_factors > 0).astype(np.int32)

        # Path Traversal risk factors
        path_factors = np.zeros(len(X))
        path_factors += (2.0 * X_bool['has_path_traversal'])
        path_factors += (1.8 * X_bool['has_system_paths'])
        path_factors += (1.5 * (X_bool['has_path_traversal'] & ~X_bool['has_auth']))
        path_factors += (1.3 * (X_bool['has_error_500'] & X_bool['error_indicates_file']))
        path_factors += (1.4 * (X_bool['has_path_traversal'] & X_bool['uses_security_scanner']))
        y_dict['path_traversal'] = (path_factors > 0).astype(np.int32)

        # Verify binary labels
        for vuln_type, labels in y_dict.items():
            unique_labels = np.unique(labels)
            if not set(unique_labels).issubset({0, 1}):
                logger.error(f"Invalid labels for {vuln_type}: {unique_labels}")
                raise ValueError(f"Labels for {vuln_type} must be binary (0 or 1)")

        # Add controlled noise to prevent overfitting
        for vuln_type in y_dict:
            labels = y_dict[vuln_type]
            noise_mask = np.random.random(len(labels)) < 0.01  # 1% random noise
            labels[noise_mask] = 1 - labels[noise_mask]  # Flip labels for noise
            y_dict[vuln_type] = labels

        return y_dict

    def save_training_data(self, X: pd.DataFrame, y_dict: Dict[str, np.ndarray]):
        """Save training data with enhanced metadata"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save features
        features_path = self.data_dir / f'features_{timestamp}.csv'
        X.to_csv(features_path, index=False)
        logger.info(f"Saved features to {features_path}")

        # Save labels
        labels_path = self.data_dir / f'labels_{timestamp}.joblib'
        joblib.dump(y_dict, labels_path)
        logger.info(f"Saved labels to {labels_path}")

        # Calculate detailed metadata
        metadata = {
            'timestamp': timestamp,
            'n_samples': len(X),
            'feature_names': list(X.columns),
            'vulnerability_types': list(y_dict.keys()),
            'feature_distributions': X.mean().to_dict(),
            'label_distributions': {k: float(v.mean()) for k, v in y_dict.items()},
            'feature_correlations': X.corr().to_dict(),
            'class_balance': {
                k: {
                    'positive': float(v.sum()),
                    'negative': float(len(v) - v.sum()),
                    'ratio': float(v.sum() / len(v))
                } for k, v in y_dict.items()
            },
            'feature_importance': {
                vuln_type: {
                    feature: float(np.corrcoef(X[feature], y_dict[vuln_type])[0, 1])
                    for feature in self.feature_names
                } for vuln_type in y_dict.keys()
            }
        }

        metadata_path = self.data_dir / f'metadata_{timestamp}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")

    def train_and_evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name, vuln_type):
        """Train and evaluate a single model with comprehensive metrics"""
        start_time = time.time()
        logger.info(f"Training {model_name} for {vuln_type}")

        # Train model
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

        # Calculate confidence scores
        confidence_scores = np.abs(y_prob - 0.5) * 2

        # Calculate feature importance
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_names, model.feature_importances_))
        elif model_name == 'neural_network':
            # For neural networks, use first layer weights as feature importance
            weights = np.abs(model.coefs_[0]).mean(axis=1)
            feature_importance = dict(zip(self.feature_names, weights))
        elif model_name == 'svm' and model.kernel == 'linear':
            feature_importance = dict(zip(self.feature_names, np.abs(model.coef_[0])))

        # Detailed metrics
        metrics = {
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred),
            'training_time': training_time,
            'confidence_mean': float(np.mean(confidence_scores)),
            'confidence_std': float(np.std(confidence_scores)),
            'feature_importance': feature_importance,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'prediction_distribution': {
                'positive_rate': float(y_pred.mean()),
                'high_confidence_rate': float((confidence_scores > 0.8).mean())
            }
        }

        logger.info(f"{model_name} performance metrics for {vuln_type}:")
        logger.info(f"Precision: {metrics['precision']:.3f}")
        logger.info(f"Recall: {metrics['recall']:.3f}")
        logger.info(f"F1 Score: {metrics['f1']:.3f}")
        logger.info(f"Training Time: {metrics['training_time']:.2f}s")

        return model, metrics

    def get_model_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Get specialized model configurations for each vulnerability type"""
        base_configs = {}

        # SQL Injection Models
        base_configs['sql_injection'] = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=4,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                min_samples_split=6,
                min_samples_leaf=4,
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=42
            )
        }

        # XSS Models
        base_configs['xss'] = {
            'random_forest': RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                min_samples_split=4,
                min_samples_leaf=3,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=120,
                learning_rate=0.1,
                max_depth=4,
                subsample=0.7,
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(80, 40),
                activation='relu',
                solver='adam',
                alpha=0.01,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=0.8,
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=42
            )
        }

        # Command Injection Models
        base_configs['command_injection'] = {
            'random_forest': RandomForestClassifier(
                n_estimators=220,
                max_depth=15,
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=180,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.8,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(120, 60),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.2,
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=42
            )
        }

        # Path Traversal Models
        base_configs['path_traversal'] = {
            'random_forest': RandomForestClassifier(
                n_estimators=180,
                max_depth=10,
                min_samples_split=4,
                min_samples_leaf=3,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(90, 45),
                activation='relu',
                solver='adam',
                alpha=0.005,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=42
            )
        }

        # Auth Bypass Models
        base_configs['auth_bypass'] = {
            'random_forest': RandomForestClassifier(
                n_estimators=180,
                max_depth=12,
                min_samples_split=3,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=160,
                learning_rate=0.04,
                max_depth=6,
                subsample=0.8,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=42
            )
        }

        # Rate Limiting Models
        base_configs['rate_limiting'] = {
            'random_forest': RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                min_samples_split=4,
                min_samples_leaf=3,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=120,
                learning_rate=0.1,
                max_depth=4,
                subsample=0.7,
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(80, 40),
                activation='relu',
                solver='adam',
                alpha=0.01,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=0.8,
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=42
            )
        }

        return base_configs

    def train_models(self):
        """Train all models with cross-validation and performance tracking"""
        logger.info("Starting model training process...")

        # Generating training data
        logger.info("Generating training data...")
        X, y_dict = self.generate_training_data(n_samples=10000)

        # Saving training data
        logger.info("Saving training data...")
        self.save_training_data(X, y_dict)
        logger.info("Training data saved successfully")

        # Validate training data
        self.validate_training_data(X, y_dict)

        # Get model configurations
        model_configs = self.get_model_configurations()

        # Initialize performance tracking
        model_performance = {}

        # Train models for each vulnerability type
        for vuln_type in self.vulnerability_types:
            logger.info(f"\nTraining models for {vuln_type}")

            if vuln_type not in self.models:
                self.models[vuln_type] = {}
            if vuln_type not in model_performance:
                model_performance[vuln_type] = {}

            # Create train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_dict[vuln_type],
                test_size=0.2,
                stratify=y_dict[vuln_type],
                random_state=42
            )

            # Train each model type
            for model_name, model in model_configs[vuln_type].items():
                try:
                    logger.info(f"Training {model_name} for {vuln_type}")
                    start_time = time.time()

                    # Fit the model
                    model.fit(X_train, y_train)
                    training_time = time.time() - start_time

                    # Make predictions
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]

                    # Calculate metrics
                    metrics = {
                        'precision': precision_score(y_test, y_pred),
                        'recall': recall_score(y_test, y_pred),
                        'f1': f1_score(y_test, y_pred),
                        'accuracy': accuracy_score(y_test, y_pred),
                        'training_time': training_time,
                        'confidence_mean': float(np.mean(np.abs(y_prob - 0.5) * 2)),
                        'confidence_std': float(np.std(np.abs(y_prob - 0.5) * 2))
                    }

                    # Try to get feature importance
                    try:
                        if hasattr(model, 'feature_importances_'):
                            metrics['feature_importance'] = dict(zip(
                                self.feature_names,
                                model.feature_importances_
                            ))
                        elif model_name == 'neural_network':
                            weights = np.abs(model.coefs_[0]).mean(axis=1)
                            metrics['feature_importance'] = dict(zip(
                                self.feature_names,
                                weights
                            ))
                        elif model_name == 'svm' and model.kernel == 'linear':
                            metrics['feature_importance'] = dict(zip(
                                self.feature_names,
                                np.abs(model.coef_[0])
                            ))
                    except Exception as fe:
                        logger.warning(f"Could not extract feature importance for {model_name}: {str(fe)}")
                        metrics['feature_importance'] = {}

                    # Store model and metrics
                    self.models[vuln_type][model_name] = model
                    model_performance[vuln_type][model_name] = metrics

                    # Log performance
                    logger.info(f"{model_name} performance metrics for {vuln_type}:")
                    logger.info(f"Precision: {metrics['precision']:.3f}")
                    logger.info(f"Recall: {metrics['recall']:.3f}")
                    logger.info(f"F1 Score: {metrics['f1']:.3f}")
                    logger.info(f"Training Time: {metrics['training_time']:.2f}s")

                except Exception as e:
                    logger.error(f"Error training {model_name} for {vuln_type}: {str(e)}")
                    logger.error("Detailed error:", exc_info=True)
                    continue

        return model_performance

    def validate_training_data(self, X: pd.DataFrame, y_dict: Dict[str, np.ndarray]):
        """
        Validate the training data before model training
        """
        # Checking for NaNs
        if X.isna().any().any():
            raise ValueError("Training data contains NaN values")

        # Checking for infinities
        if np.isinf(X.select_dtypes(include=[np.number])).any().any():
            raise ValueError("Training data contains infinite values")

        # Checking data types
        allowed_dtypes = ['int64', 'int32', 'float64', 'float32']
        for col in X.columns:
            if X[col].dtype not in allowed_dtypes:
                raise ValueError(f"Column {col} has unexpected data type: {X[col].dtype}")

        # Checking label distributions
        for vuln_type, labels in y_dict.items():
            if labels.dtype not in ['int32', 'int64']:
                raise ValueError(f"Labels for {vuln_type} must be integer type")

            # Checking label values (binary classification)
            unique_labels = np.unique(labels)
            if not np.array_equal(unique_labels, [0, 1]):
                raise ValueError(f"Labels for {vuln_type} must be binary (0 or 1)")
def train_all_models():
    """Main training function with enhanced logging and error handling"""
    try:
        logger.info("Starting comprehensive model training process")

        # Creating necessary directories
        base_path = Path('/app/src/ml/models')
        base_path.mkdir(parents=True, exist_ok=True)

        model_path = base_path / 'security_models.joblib'
        metadata_path = base_path / 'model_metadata.joblib'
        model_performance_path = base_path / 'model_performance.joblib'

        # Initializing trainer
        trainer = SecurityModelTrainer()
        logger.info("Initialized SecurityModelTrainer")

        try:
            # Train models and get performance metrics
            performance_metrics = trainer.train_models()

            # Save models
            logger.info(f"Saving models to {model_path}")
            joblib.dump(trainer.models, model_path)
            logger.info("Models saved successfully")

            # Save metadata
            metadata = {
                'feature_names': trainer.feature_names,
                'vulnerability_types': trainer.vulnerability_types,
                'n_features': len(trainer.feature_names),
                'training_timestamp': datetime.now().isoformat(),
                'model_configurations': {
                    model_name: str(model.__class__.__name__)
                    for model_name, model in trainer.models.items()
                },
                'training_summary': {
                    vuln_type: {
                        model_name: {
                            'f1_score': metrics['f1'],
                            'precision': metrics['precision'],
                            'recall': metrics['recall']
                        }
                        for model_name, metrics in model_metrics.items()
                    }
                    for vuln_type, model_metrics in performance_metrics.items()
                }
            }
            joblib.dump(metadata, metadata_path)
            logger.info("Metadata saved successfully")

            # Save performance metrics
            joblib.dump(performance_metrics, model_performance_path)
            logger.info("Performance metrics saved successfully")

            # Print comprehensive analysis
            logger.info("\nComparative Analysis of Models:")
            for vuln_type in trainer.vulnerability_types:
                logger.info(f"\nVulnerability Type: {vuln_type}")
                for model_name, metrics in performance_metrics[vuln_type].items():
                    logger.info(f"\n{model_name}:")
                    logger.info(f"Precision: {metrics['precision']:.3f}")
                    logger.info(f"Recall: {metrics['recall']:.3f}")
                    logger.info(f"F1 Score: {metrics['f1']:.3f}")
                    logger.info(f"Training Time: {metrics['training_time']:.2f}s")
                    logger.info(f"Confidence Mean: {metrics['confidence_mean']:.3f}")

                    # Safely check and log feature importance if available
                    if 'feature_importance' in metrics and metrics['feature_importance']:
                        # Get top 5 most important features
                        top_features = sorted(
                            metrics['feature_importance'].items(),
                            key=lambda x: float(x[1]),  # Ensure numerical sorting
                            reverse=True
                        )[:5]
                        logger.info("Top 5 Important Features:")
                        for feature, importance in top_features:
                            logger.info(f"- {feature}: {float(importance):.3f}")

        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            raise

    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise

if __name__ == "__main__":
    logger.info("Starting security model training pipeline")
    train_all_models()
    logger.info("Training pipeline completed")