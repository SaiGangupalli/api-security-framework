import asyncio
import json
import random
import re
from typing import Dict, List
import openai
import logging
import os
from datetime import datetime

from openai import AsyncOpenAI

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
class TestGenerator:
    def __init__(self):
        self.openai_client = AsyncOpenAI()
        self.use_fallback = False  # Track if we should use fallback

    async def generate_openai_test_cases(self, api_details: Dict, risk_assessment: Dict, max_retries: int = 1) -> List[Dict]:
        """Generate test cases using OpenAI"""
        # client = AsyncOpenAI()
        try:
            # Create context-aware prompt
            prompt = create_security_prompt(api_details, risk_assessment)
            logger.debug(f"Prompt: {prompt}")

            for attempt in range(max_retries):
                try:
                    # Call OpenAI API
                    response = await self.openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a security testing expert specializing in API security."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=2000
                    )

                    # Log the raw response
                    logger.debug(f"Response: {response}")
                    logger.debug("Raw OpenAI Response:")
                    logger.debug(response.choices[0].message.content)

                    # Parse and structure the response
                    test_cases = parse_openai_response(response.choices[0].message.content)

                    return test_cases

                except self.openai_client.RateLimitError:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.random()
                        logger.warning(f"Rate limit hit. Waiting {wait_time} seconds before retry.")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error("Max retries reached. Falling back to default test cases.")
                        return []
                except Exception as retry_error:
                    logger.error(f"Error on attempt {attempt + 1}: {str(retry_error)}")
                    if attempt == max_retries - 1:
                        return []
                    await asyncio.sleep((2 ** attempt) + random.random())
        except Exception as e:
            logger.error(f"OpenAI generation error: {str(e)}")
            return []
def create_security_prompt(api_details: Dict, risk_assessment: Dict) -> str:
    """Create context-aware security testing prompt"""
    return f"""
        You are a security testing expert tasked with generating comprehensive security test cases for an API. Based on the risk assessment, create detailed test cases in the following structured format:
        Example#1:
            Test case 1: Name
            Severity (Critical/High/Medium/Low)
            Brief description of the test scenario
            Steps:
            
                Step 1 description
                Step 2 description
                Step 3 description
            Expected Results:
                - Expected HTTP status code or specific response
            Remediation:
                - Specific recommendations to address the vulnerability
        
        Example#2:
            Test Case 2: Name
            Severity (Critical/High/Medium/Low)
            Brief description of the test scenario
            Steps:
            
                Step 1 description
                Step 2 description
                Step 3 description
            Expected Results:
                - Expected HTTP status code or specific response
            Remediation:
                - Specific recommendations to address the vulnerability
            
        API Context:
        - Endpoint: {api_details.get('request', {}).get('uri', '')}
        - Method: {api_details.get('method', '').upper()}
        - Headers: {api_details.get('request', {}).get('headers', {})}
        
        Risk Assessment Insights:
        - SQL Injection Risk: {risk_assessment['sql_injection_risk']} (Severe: {">0.5" if risk_assessment['sql_injection_risk'] > 0.5 else "Moderate"})
        - XSS Risk: {risk_assessment['xss_risk']} (Severe: {">0.5" if risk_assessment['xss_risk'] > 0.5 else "Moderate"})
        - Auth Bypass Risk: {risk_assessment['auth_bypass_risk']} (Severe: {">0.5" if risk_assessment['auth_bypass_risk'] > 0.5 else "Moderate"})
        - Rate Limit Risk: {risk_assessment['rate_limiting_risk']} (Severe: {">0.5" if risk_assessment['rate_limiting_risk'] > 0.5 else "Moderate"})
        
        Guidelines for Test Cases:
        1. Focus on the highest risk areas identified in the risk assessment
        2. Provide concrete, actionable test scenarios
        3. Include specific payloads, expected outcomes, and clear remediation steps
        4. Cover different attack vectors and potential vulnerabilities
        5. Aim to expose potential security weaknesses in the API
        
        Generate at least 3-5 unique security test cases that cover:
        - Authentication and Authorization Bypass
        - Input Validation and Sanitization
        - Data Exposure and Leakage
        - Session Management
        - Rate Limiting and DoS Protection
        
        Each test case should be practical, realistic, and provide clear guidance for improving the API's security posture.
        """

def clean_text(text: str) -> str:
        """Clean markdown artifacts from text"""
        # Remove markdown symbols and extra spaces
        cleaned = text.strip()
        cleaned = cleaned.replace('**', '')  # Remove bold markers
        cleaned = cleaned.replace('`', '')   # Remove code markers
        cleaned = cleaned.replace('  ', ' ') # Remove double spaces
        return cleaned.strip()

# def parse_openai_response(response_text: str) -> List[Dict]:
#     """Parse OpenAI generated test cases into structured format"""
#     test_cases = []
#     sections = response_text.split('### Test Case')[1:]  # Skip the first empty part
#
#     for section in sections:
#         try:
#             # Initialize test case structure
#             test_case = {
#                 "type": "",
#                 "description": "",
#                 "test_cases": [{
#                     "name": "",
#                     "description": "",
#                     "priority": "",
#                     "steps": [],
#                     "expected_results": "",
#                     "remediation": ""
#                 }]
#             }
#
#             # Split into lines and normalize spacing
#             lines = [line.strip() for line in section.split('\n') if line.strip()]
#             current_section = None
#             steps_buffer = []
#             expected_results_buffer = []
#             remediation_buffer = []
#
#             for i, line in enumerate(lines):
#                 # Get test case name from first line
#                 if i == 0 and ':' in line:
#                     name = line.split(':', 1)[1].strip()
#                     test_case["type"] = name
#                     test_case["test_cases"][0]["name"] = name
#                     continue
#
#                 # Handle different section markers
#                 if '**Test Category Name:**' in line or '- **Test Category Name:**' in line:
#                     category = line.split(':**', 1)[1].strip()
#                     test_case["type"] = category
#                     test_case["test_cases"][0]["name"] = category
#
#                 elif '**Severity:**' in line or '- **Severity:**' in line:
#                     severity = line.split(':**', 1)[1].strip()
#                     test_case["test_cases"][0]["priority"] = severity
#
#                 elif '**Brief description' in line or '- **Brief description' in line:
#                     desc = line.split(':', 1)[1].strip()
#                     test_case["description"] = desc.replace('**', '')
#                     test_case["test_cases"][0]["description"] = desc.replace('**', '')
#
#                 # Handle section starts
#                 elif '**Steps:**' in line or '- **Steps:**' in line:
#                     current_section = 'steps'
#                     continue
#
#                 elif '**Expected Results:**' in line or '- **Expected Results:**' in line:
#                     current_section = 'expected_results'
#                     continue
#
#                 elif '**Remediation:**' in line or '- **Remediation:**' in line:
#                     current_section = 'remediation'
#                     continue
#
#                 # Handle section content
#                 elif current_section == 'steps':
#                     if line.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
#                         step = line.split('.', 1)[1].strip()
#                         # Clean up the step
#                         step = step.replace('`', '').strip()
#                         steps_buffer.append(step)
#
#                 elif current_section == 'expected_results':
#                     if line.startswith('-'):
#                         result = line.lstrip('- ').strip()
#                         expected_results_buffer.append(result)
#
#                 elif current_section == 'remediation':
#                     if line.startswith('-'):
#                         rem = line.lstrip('- ').strip()
#                         remediation_buffer.append(rem)
#
#             # Clean up and assign the collected data
#             test_case["test_cases"][0]["steps"] = [s for s in steps_buffer if s]
#             test_case["test_cases"][0]["expected_results"] = ' '.join(expected_results_buffer).replace('`', '').strip()
#             test_case["test_cases"][0]["remediation"] = ' '.join(remediation_buffer).replace('`', '').strip()
#
#             # Only add test cases that have the required fields
#             if (test_case["type"] and
#                     test_case["test_cases"][0]["priority"] and
#                     test_case["test_cases"][0]["steps"]):
#                 test_cases.append(test_case)
#                 logger.debug(f"Added test case: {test_case['type']}")
#                 logger.debug(f"Steps: {len(test_case['test_cases'][0]['steps'])}")
#                 logger.debug(f"Expected Results: {test_case['test_cases'][0]['expected_results']}")
#                 logger.debug(f"Remediation: {test_case['test_cases'][0]['remediation']}")
#
#         except Exception as e:
#             logger.error(f"Error parsing test case: {str(e)}")
#             logger.error(f"Problematic section: {section[:200]}...")
#             continue
#
#     logger.debug(f"Parsed {len(test_cases)} test cases")
#     if test_cases:
#         logger.debug(f"Gen AI Test Cases: {json.dumps(test_cases, indent=2)}")
#     else:
#         logger.debug("No test cases were successfully parsed")
#
#     return test_cases

def parse_openai_response(response_text: str) -> List[Dict]:
    """Parse OpenAI generated test cases into structured format"""
    test_cases = []

    # Find all occurrences of "### Test Case" (with optional number)
    test_case_starts = [m.start() for m in re.finditer(r"### Test Case\s*\d*:\s*(.*)", response_text)]
    test_case_starts.append(len(response_text))  # Add the end of the string
    logger.debug(f"test_case_starts: {test_case_starts}")

    for i in range(len(test_case_starts) - 1):
        start = test_case_starts[i]
        end = test_case_starts[i + 1]
        section = response_text[start:end]

        try:
            # Initialize test case structure
            test_case = {
                "type": "",
                "description": "",
                "test_cases": []
            }

            # Split into lines and normalize spacing
            lines = [line.strip() for line in section.split('\n') if line.strip()]
            current_section = None
            steps_buffer = []
            expected_results_buffer = []
            remediation_buffer = []

            logger.debug(f"lines[0]: {lines[0]}")

            # Extract subsection (if present)
            subsection_match = re.match(r"### Test Case\s*\d*:\s*(.*)", lines[0])

            if subsection_match:
                test_case["type"] = subsection_match.group(1).strip()  # Test Case number
                test_case["test_cases"].append({
                    "name": f"Test {subsection_match.group(1).strip()}",
                    "description": "",
                    "priority": "",
                    "steps": [],
                    "expected_results": "",
                    "remediation": ""
                })

            for i, line in enumerate(lines):
                logger.debug(f"line{i}: {line}")
                # Handle different section markers
                if "**Severity:**" in line:
                    severity = line.split(':**', 1)[1].strip()
                    test_case["test_cases"][-1]["priority"] = severity

                elif "**Brief description" in line:
                    desc = line.split(':', 1)[1].strip()
                    test_case["description"] = desc.replace('**', '')
                    test_case["test_cases"][-1]["description"] = desc.replace('**', '')

                # Handle section starts
                elif "**Steps:**" in line:
                    current_section = 'steps'
                    continue

                elif "**Expected Results:**" in line:
                    current_section = 'expected_results'
                    continue

                elif "**Remediation:**" in line:
                    current_section = 'remediation'
                    continue

                # Handle section content
                elif current_section == 'steps':
                    if line.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                        step = line.split('.', 1)[1].strip()
                        # Clean up the step
                        step = step.replace('`', '').strip()
                        steps_buffer.append(step)

                elif current_section == 'expected_results':
                    if line.startswith('-'):
                        result = line.lstrip('- ').strip()
                        expected_results_buffer.append(result)

                elif current_section == 'remediation':
                    if line.startswith('-'):
                        rem = line.lstrip('- ').strip()
                        remediation_buffer.append(rem)
                    else:
                        remediation_buffer.append(line)

            # Clean up and assign the collected data
            test_case["test_cases"][-1]["steps"] = [s for s in steps_buffer if s]
            test_case["test_cases"][-1]["expected_results"] = ' '.join(expected_results_buffer).replace('`', '').strip()
            test_case["test_cases"][-1]["remediation"] = ' '.join(remediation_buffer).replace('`', '').strip()

            # Only add test cases that have the required fields
            if (test_case["type"] and
                    test_case["test_cases"][-1]["priority"] and
                    test_case["test_cases"][-1]["steps"]):
                test_cases.append(test_case)
                logger.debug(f"Added test case: {test_case['type']}")
                logger.debug(f"Steps: {len(test_case['test_cases'][0]['steps'])}")
                logger.debug(f"Expected Results: {test_case['test_cases'][0]['expected_results']}")
                logger.debug(f"Remediation: {test_case['test_cases'][0]['remediation']}")

        except Exception as e:
            logger.debug(f"Error parsing test case: {str(e)}")
            logger.debug(f"Problematic section: {section[:200]}...")
            continue

    logger.debug(f"Parsed {len(test_cases)} test cases")
    if test_cases:
        logger.debug(f"Gen AI Test Cases: {json.dumps(test_cases, indent=2)}")
    else:
        logger.debug("No test cases were successfully parsed")

    return test_cases

# def parse_openai_response(response_text: str) -> List[Dict]:
#     """Parse OpenAI generated test cases into structured format"""
#     test_cases = []
#
#     # Find all occurrences of "### Test Case" (with optional number)
#     test_case_starts = [m.start() for m in re.finditer(r"### Test Case\s*\d*:\s*(.*)", response_text)]
#     test_case_starts.append(len(response_text))  # Add the end of the string
#
#     for i in range(len(test_case_starts) - 1):
#         start = test_case_starts[i]
#         end = test_case_starts[i + 1]
#         section = response_text[start:end]
#
#         try:
#             # Initialize test case structure
#             test_case = {
#                 "type": "",
#                 "description": "",
#                 "test_cases": []
#             }
#
#             # Split into lines and normalize spacing
#             lines = [line.strip() for line in section.split('\n') if line.strip()]
#             current_section = None
#             steps_buffer = []
#             expected_results_buffer = []
#             remediation_buffer = []
#
#             # Extract subsection (if present)
#             subsection_match = re.match(r"### Test Case\s*\d*:\s*(.*)", lines[0])
#             if subsection_match:
#                 test_case_name = subsection_match.group(1).strip()
#                 test_case["type"] = test_case_name
#                 test_case["test_cases"].append({
#                     "name": test_case_name,
#                     "description": "",
#                     "priority": "",
#                     "steps": [],
#                     "expected_results": "",
#                     "remediation": ""
#                 })
#
#             for i, line in enumerate(lines):
#                 # Handle different section markers
#                 if "**Severity:**" in line:
#                     severity = line.split(':**', 1)[1].strip()
#                     test_case["test_cases"][-1]["priority"] = severity
#
#                 elif "**Brief description**" in line:
#                     desc = line.split(':', 1)[1].strip()
#                     test_case["description"] = desc.replace('**', '')
#                     test_case["test_cases"][-1]["description"] = desc.replace('**', '')
#
#                 # Handle section starts
#                 elif "**Steps:**" in line:
#                     current_section = 'steps'
#                     continue
#
#                 elif "**Expected Results:**" in line:
#                     current_section = 'expected_results'
#                     continue
#
#                 elif "**Remediation:**" in line:
#                     current_section = 'remediation'
#                     continue
#
#                 # Handle section content
#                 elif current_section == 'steps':
#                     if line.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
#                         step = line.split('.', 1)[1].strip()
#                         # Clean up the step
#                         step = step.replace('`', '').strip()
#                         steps_buffer.append(step)
#
#                 elif current_section == 'expected_results':
#                     if line.startswith('-'):
#                         result = line.lstrip('- ').strip()
#                         expected_results_buffer.append(result)
#
#                 elif current_section == 'remediation':
#                     if line.startswith('-'):
#                         rem = line.lstrip('- ').strip()
#                         remediation_buffer.append(rem)
#
#             # Clean up and assign the collected data
#             test_case["test_cases"][-1]["steps"] = [s for s in steps_buffer if s]
#             test_case["test_cases"][-1]["expected_results"] = ' '.join(expected_results_buffer).replace('`', '').strip()
#             test_case["test_cases"][-1]["remediation"] = ' '.join(remediation_buffer).replace('`', '').strip()
#
#             # Only add test cases that have the required fields
#             if (test_case["type"] and
#                     test_case["test_cases"][-1]["priority"] and
#                     test_case["test_cases"][-1]["steps"]):
#                 test_cases.append(test_case)
#
#         except Exception as e:
#             continue
#
#     return test_cases
