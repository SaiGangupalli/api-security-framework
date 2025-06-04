# src/utils/ai_scenario_generator.py
import os
import logging
import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Optional
import openai
from openai import OpenAI
import time

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIScenarioGenerator:
    """Class to generate test scenarios from fraud and security impact stories using AI."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo", base_url: str = None):
        """
        Initialize the AI scenario generator.
        
        Args:
            api_key (str): OpenAI API key or other AI service API key
            model (str): AI model to use (default: gpt-3.5-turbo)
            base_url (str): Custom base URL for AI service (optional)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError("API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize OpenAI client
        if self.base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = OpenAI(api_key=self.api_key)
        
        logger.info(f"AI Scenario Generator initialized with model: {self.model}")
    
    def extract_impact_stories(self, predictions_df: pd.DataFrame, impact_type: str = "both") -> pd.DataFrame:
        """
        Extract stories with fraud and/or security impacts from predictions DataFrame.
        
        Args:
            predictions_df (pd.DataFrame): DataFrame with predictions
            impact_type (str): Type of impact to extract - "security", "fraud", or "both"
        
        Returns:
            pd.DataFrame: Filtered DataFrame with impact stories
        """
        try:
            if predictions_df is None or len(predictions_df) == 0:
                logger.warning("No predictions data available")
                return pd.DataFrame()
            
            # Ensure required columns exist
            required_columns = ['security_prediction', 'fraud_prediction']
            if not all(col in predictions_df.columns for col in required_columns):
                logger.error("Required prediction columns not found in DataFrame")
                return pd.DataFrame()
            
            # Filter based on impact type
            if impact_type.lower() == "security":
                filtered_df = predictions_df[predictions_df['security_prediction'] == 1].copy()
                logger.info(f"Extracted {len(filtered_df)} stories with security impacts")
            elif impact_type.lower() == "fraud":
                filtered_df = predictions_df[predictions_df['fraud_prediction'] == 1].copy()
                logger.info(f"Extracted {len(filtered_df)} stories with fraud impacts")
            else:  # both or any
                filtered_df = predictions_df[
                    (predictions_df['security_prediction'] == 1) | 
                    (predictions_df['fraud_prediction'] == 1)
                ].copy()
                logger.info(f"Extracted {len(filtered_df)} stories with security and/or fraud impacts")
            
            # Add impact labels for clarity
            if len(filtered_df) > 0:
                filtered_df['impact_type'] = 'No Impact'
                filtered_df.loc[
                    (filtered_df['security_prediction'] == 1) & (filtered_df['fraud_prediction'] == 0), 
                    'impact_type'
                ] = 'Security'
                filtered_df.loc[
                    (filtered_df['security_prediction'] == 0) & (filtered_df['fraud_prediction'] == 1), 
                    'impact_type'
                ] = 'Fraud'
                filtered_df.loc[
                    (filtered_df['security_prediction'] == 1) & (filtered_df['fraud_prediction'] == 1), 
                    'impact_type'
                ] = 'Security & Fraud'
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error extracting impact stories: {e}")
            return pd.DataFrame()
    
    def prepare_story_context(self, story_row: pd.Series) -> Dict[str, str]:
        """
        Prepare context from a single story row for AI processing.
        
        Args:
            story_row (pd.Series): Single row from DataFrame
        
        Returns:
            Dict[str, str]: Prepared context
        """
        context = {}
        
        # Basic story information
        context['issue_key'] = str(story_row.get('issue_key', 'Unknown'))
        context['summary'] = str(story_row.get('summary', ''))
        context['description'] = str(story_row.get('description', ''))
        
        # Check for acceptance criteria in various possible column names
        acceptance_criteria_columns = [
            'acceptance_criteria', 'acceptance_criterion', 'ac', 
            'criteria', 'definition_of_done', 'dod'
        ]
        
        acceptance_criteria = ''
        for col in acceptance_criteria_columns:
            if col in story_row.index and pd.notna(story_row.get(col)):
                acceptance_criteria = str(story_row.get(col))
                break
        
        context['acceptance_criteria'] = acceptance_criteria
        
        # Impact information
        context['impact_type'] = str(story_row.get('impact_type', 'Unknown'))
        context['security_probability'] = float(story_row.get('security_probability', 0))
        context['fraud_probability'] = float(story_row.get('fraud_probability', 0))
        
        # Additional context
        context['project_key'] = str(story_row.get('project_key', 'Unknown'))
        context['status'] = str(story_row.get('status', 'Unknown'))
        
        # Matching keywords if available
        context['security_keywords'] = str(story_row.get('security_matching_keywords', ''))
        context['fraud_keywords'] = str(story_row.get('fraud_matching_keywords', ''))
        
        return context
    
    def generate_security_test_scenarios(self, story_context: Dict[str, str]) -> str:
        """
        Generate security test scenarios for a story using AI.
        
        Args:
            story_context (Dict[str, str]): Story context
        
        Returns:
            str: Generated security test scenarios
        """
        prompt = f"""
As a cybersecurity testing expert, analyze this user story and generate comprehensive security test scenarios.

**User Story Information:**
- Issue Key: {story_context['issue_key']}
- Summary: {story_context['summary']}
- Description: {story_context['description']}
- Acceptance Criteria: {story_context['acceptance_criteria']}
- Security Probability: {story_context['security_probability']:.2f}
- Detected Security Keywords: {story_context['security_keywords']}

**Instructions:**
Generate specific, actionable security test scenarios that cover:
1. Authentication and Authorization vulnerabilities
2. Input validation and injection attacks
3. Data exposure and privacy concerns
4. Session management issues
5. API security (if applicable)
6. Access control bypass attempts

**Format your response as:**
## Security Test Scenarios for {story_context['issue_key']}

### High Priority Scenarios:
- [Scenario 1]
- [Scenario 2]
- [Scenario 3]

### Medium Priority Scenarios:
- [Scenario 1]
- [Scenario 2]

### Additional Considerations:
- [Additional security considerations based on the story context]

**Requirements:**
- Be specific to the functionality described in the user story
- Include both positive and negative test cases
- Consider edge cases and boundary conditions
- Reference specific security standards (OWASP, NIST) where relevant
- Provide clear test steps where possible
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert cybersecurity tester with deep knowledge of security testing methodologies, OWASP guidelines, and common vulnerabilities."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating security test scenarios: {e}")
            return f"Error generating security scenarios for {story_context['issue_key']}: {str(e)}"
    
    def generate_fraud_test_scenarios(self, story_context: Dict[str, str]) -> str:
        """
        Generate fraud test scenarios for a story using AI.
        
        Args:
            story_context (Dict[str, str]): Story context
        
        Returns:
            str: Generated fraud test scenarios
        """
        prompt = f"""
As a fraud prevention testing expert, analyze this user story and generate comprehensive fraud test scenarios.

**User Story Information:**
- Issue Key: {story_context['issue_key']}
- Summary: {story_context['summary']}
- Description: {story_context['description']}
- Acceptance Criteria: {story_context['acceptance_criteria']}
- Fraud Probability: {story_context['fraud_probability']:.2f}
- Detected Fraud Keywords: {story_context['fraud_keywords']}

**Instructions:**
Generate specific, actionable fraud test scenarios that cover:
1. Identity verification and impersonation attempts
2. Transaction manipulation and unauthorized changes
3. Account takeover scenarios
4. Payment fraud and financial manipulation
5. Data manipulation and false information
6. Behavioral anomaly detection

**Format your response as:**
## Fraud Test Scenarios for {story_context['issue_key']}

### Critical Fraud Scenarios:
- [Scenario 1]
- [Scenario 2]
- [Scenario 3]

### Common Fraud Attempts:
- [Scenario 1]
- [Scenario 2]

### Advanced Fraud Techniques:
- [Scenario 1]
- [Scenario 2]

### Detection and Prevention Validation:
- [Validation scenarios for fraud detection systems]

**Requirements:**
- Focus on realistic fraud scenarios relevant to the functionality
- Include both automated and manual fraud attempts
- Consider multi-step fraud processes
- Address social engineering aspects where applicable
- Include scenarios for testing fraud detection accuracy
- Provide clear test data requirements
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert fraud prevention analyst with extensive knowledge of fraud patterns, detection methods, and financial crime prevention."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating fraud test scenarios: {e}")
            return f"Error generating fraud scenarios for {story_context['issue_key']}: {str(e)}"
    
    def generate_combined_scenarios(self, story_context: Dict[str, str]) -> str:
        """
        Generate combined security and fraud test scenarios for stories with both impacts.
        
        Args:
            story_context (Dict[str, str]): Story context
        
        Returns:
            str: Generated combined test scenarios
        """
        prompt = f"""
As a comprehensive security and fraud testing expert, analyze this user story that has BOTH security and fraud implications.

**User Story Information:**
- Issue Key: {story_context['issue_key']}
- Summary: {story_context['summary']}
- Description: {story_context['description']}
- Acceptance Criteria: {story_context['acceptance_criteria']}
- Security Probability: {story_context['security_probability']:.2f}
- Fraud Probability: {story_context['fraud_probability']:.2f}
- Security Keywords: {story_context['security_keywords']}
- Fraud Keywords: {story_context['fraud_keywords']}

**Instructions:**
Generate comprehensive test scenarios that address the intersection of security and fraud risks:

**Format your response as:**
## Combined Security & Fraud Test Scenarios for {story_context['issue_key']}

### Critical Combined Scenarios:
- [Scenarios that test both security and fraud aspects simultaneously]

### Security-Focused Test Cases:
- [Primary security vulnerabilities]

### Fraud-Focused Test Cases:
- [Primary fraud risks]

### Cross-Impact Scenarios:
- [How security weaknesses could enable fraud]
- [How fraud attempts could exploit security gaps]

### Integration Testing:
- [End-to-end scenarios covering both domains]

### Compliance and Monitoring:
- [Regulatory compliance scenarios]
- [Monitoring and alerting validation]

**Requirements:**
- Prioritize scenarios that address both security and fraud simultaneously
- Consider the interaction between security controls and fraud prevention
- Include scenarios for testing incident response procedures
- Address data integrity from both security and fraud perspectives
- Consider user experience impact of security/fraud controls
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a senior security and fraud prevention expert with deep understanding of how security vulnerabilities can be exploited for fraudulent activities."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating combined test scenarios: {e}")
            return f"Error generating combined scenarios for {story_context['issue_key']}: {str(e)}"
    
    def process_impact_stories(self, impact_stories_df: pd.DataFrame, 
                             batch_size: int = 5, delay_seconds: float = 1.0) -> pd.DataFrame:
        """
        Process all impact stories and generate AI test scenarios.
        
        Args:
            impact_stories_df (pd.DataFrame): DataFrame with impact stories
            batch_size (int): Number of stories to process in each batch
            delay_seconds (float): Delay between API calls to avoid rate limiting
        
        Returns:
            pd.DataFrame: DataFrame with generated scenarios
        """
        if len(impact_stories_df) == 0:
            logger.warning("No impact stories to process")
            return pd.DataFrame()
        
        logger.info(f"Processing {len(impact_stories_df)} impact stories for AI scenario generation")
        
        results = []
        
        for idx, row in impact_stories_df.iterrows():
            try:
                logger.info(f"Processing story {idx + 1}/{len(impact_stories_df)}: {row.get('issue_key', 'Unknown')}")
                
                # Prepare story context
                story_context = self.prepare_story_context(row)
                
                # Generate scenarios based on impact type
                impact_type = story_context['impact_type']
                
                if impact_type == 'Security':
                    scenarios = self.generate_security_test_scenarios(story_context)
                elif impact_type == 'Fraud':
                    scenarios = self.generate_fraud_test_scenarios(story_context)
                elif impact_type == 'Security & Fraud':
                    scenarios = self.generate_combined_scenarios(story_context)
                else:
                    scenarios = "No specific scenarios generated for this impact type."
                
                # Create result record
                result = {
                    'issue_key': story_context['issue_key'],
                    'project_key': story_context['project_key'],
                    'summary': story_context['summary'],
                    'description': story_context['description'],
                    'acceptance_criteria': story_context['acceptance_criteria'],
                    'impact_type': impact_type,
                    'security_probability': story_context['security_probability'],
                    'fraud_probability': story_context['fraud_probability'],
                    'security_keywords': story_context['security_keywords'],
                    'fraud_keywords': story_context['fraud_keywords'],
                    'generated_scenarios': scenarios,
                    'generation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                results.append(result)
                
                # Add delay to avoid rate limiting
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
                
            except Exception as e:
                logger.error(f"Error processing story {row.get('issue_key', 'Unknown')}: {e}")
                
                # Add error record
                result = {
                    'issue_key': row.get('issue_key', 'Unknown'),
                    'project_key': row.get('project_key', 'Unknown'),
                    'summary': row.get('summary', ''),
                    'description': row.get('description', ''),
                    'acceptance_criteria': '',
                    'impact_type': row.get('impact_type', 'Unknown'),
                    'security_probability': row.get('security_probability', 0),
                    'fraud_probability': row.get('fraud_probability', 0),
                    'security_keywords': row.get('security_matching_keywords', ''),
                    'fraud_keywords': row.get('fraud_matching_keywords', ''),
                    'generated_scenarios': f"Error generating scenarios: {str(e)}",
                    'generation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                results.append(result)
        
        results_df = pd.DataFrame(results)
        logger.info(f"Successfully processed {len(results_df)} stories")
        
        return results_df
    
    def export_scenarios_to_excel(self, scenarios_df: pd.DataFrame, 
                                output_path: str = None) -> str:
        """
        Export generated scenarios to formatted Excel file.
        
        Args:
            scenarios_df (pd.DataFrame): DataFrame with generated scenarios
            output_path (str): Path to save the Excel file
        
        Returns:
            str: Path to the exported Excel file
        """
        if len(scenarios_df) == 0:
            logger.warning("No scenarios to export")
            return None
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/processed/ai_test_scenarios_{timestamp}.xlsx"
        
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Use the enhanced export function from helpers
            from src.utils.helpers import export_predictions_to_excel
            
            # Prepare data for export - scenarios column might be very long
            export_df = scenarios_df.copy()
            
            # Create a summary version for the main sheet
            summary_columns = [
                'issue_key', 'project_key', 'summary', 'impact_type',
                'security_probability', 'fraud_probability', 'generation_timestamp'
            ]
            summary_df = export_df[summary_columns].copy()
            
            # Export using pandas ExcelWriter for multiple sheets
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Summary sheet
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Full scenarios sheet
                export_df.to_excel(writer, sheet_name='Full Scenarios', index=False)
                
                # Separate sheets by impact type
                for impact_type in export_df['impact_type'].unique():
                    if impact_type and impact_type != 'Unknown':
                        impact_df = export_df[export_df['impact_type'] == impact_type]
                        sheet_name = impact_type.replace('&', 'and').replace(' ', '_')[:31]  # Excel sheet name limit
                        impact_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Get workbook for formatting
                workbook = writer.book
                
                # Format all sheets
                for sheet_name in workbook.sheetnames:
                    worksheet = workbook[sheet_name]
                    
                    # Auto-adjust column widths
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        
                        # Set reasonable width limits
                        adjusted_width = min(max_length + 2, 100)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
                    
                    # Enable text wrapping for scenarios column if it exists
                    if 'generated_scenarios' in [cell.value for cell in worksheet[1]]:
                        scenario_col = None
                        for idx, cell in enumerate(worksheet[1], 1):
                            if cell.value == 'generated_scenarios':
                                scenario_col = cell.column_letter
                                break
                        
                        if scenario_col:
                            for row in range(2, worksheet.max_row + 1):
                                cell = worksheet[f'{scenario_col}{row}']
                                cell.alignment = openpyxl.styles.Alignment(wrap_text=True, vertical='top')
                            
                            # Set a larger width for scenarios column
                            worksheet.column_dimensions[scenario_col].width = 80
            
            logger.info(f"AI test scenarios exported to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting scenarios to Excel: {e}")
            return None
    
    def generate_executive_summary(self, scenarios_df: pd.DataFrame) -> str:
        """
        Generate an executive summary of the generated test scenarios.
        
        Args:
            scenarios_df (pd.DataFrame): DataFrame with generated scenarios
        
        Returns:
            str: Executive summary
        """
        if len(scenarios_df) == 0:
            return "No scenarios were generated."
        
        # Calculate statistics
        total_stories = len(scenarios_df)
        security_only = len(scenarios_df[scenarios_df['impact_type'] == 'Security'])
        fraud_only = len(scenarios_df[scenarios_df['impact_type'] == 'Fraud'])
        combined = len(scenarios_df[scenarios_df['impact_type'] == 'Security & Fraud'])
        
        # Get project distribution
        project_counts = scenarios_df['project_key'].value_counts()
        
        prompt = f"""
Generate an executive summary for AI-generated test scenarios based on security and fraud impact analysis.

**Statistics:**
- Total user stories processed: {total_stories}
- Security-only impacts: {security_only}
- Fraud-only impacts: {fraud_only}
- Combined security & fraud impacts: {combined}

**Project Distribution:**
{project_counts.to_string()}

**Average Risk Scores:**
- Average Security Probability: {scenarios_df['security_probability'].mean():.2f}
- Average Fraud Probability: {scenarios_df['fraud_probability'].mean():.2f}

Create a professional executive summary that includes:
1. Overview of testing scope and coverage
2. Risk assessment insights
3. Key testing priorities
4. Resource allocation recommendations
5. Next steps for test execution

Format as a professional business document.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a senior QA manager creating executive summaries for stakeholders."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return f"Error generating executive summary: {str(e)}"


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate AI test scenarios from security/fraud impact predictions')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to predictions Excel file')
    parser.add_argument('--output', type=str,
                        help='Path to save scenarios Excel file')
    parser.add_argument('--impact-type', type=str, default='both',
                        choices=['security', 'fraud', 'both'],
                        help='Type of impact to process')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo',
                        help='AI model to use')
    parser.add_argument('--batch-size', type=int, default=5,
                        help='Batch size for processing')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between API calls in seconds')
    
    args = parser.parse_args()
    
    try:
        # Load predictions data
        logger.info(f"Loading predictions from {args.input}")
        predictions_df = pd.read_excel(args.input)
        
        # Initialize AI scenario generator
        generator = AIScenarioGenerator(model=args.model)
        
        # Extract impact stories
        impact_stories = generator.extract_impact_stories(predictions_df, args.impact_type)
        
        if len(impact_stories) == 0:
            logger.warning("No impact stories found to process")
            return
        
        # Generate scenarios
        scenarios_df = generator.process_impact_stories(
            impact_stories, 
            batch_size=args.batch_size, 
            delay_seconds=args.delay
        )
        
        # Export results
        output_path = generator.export_scenarios_to_excel(scenarios_df, args.output)
        
        # Generate and display executive summary
        summary = generator.generate_executive_summary(scenarios_df)
        print("\n" + "="*80)
        print("EXECUTIVE SUMMARY")
        print("="*80)
        print(summary)
        print("="*80)
        
        if output_path:
            print(f"\nResults exported to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
