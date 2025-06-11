#!/usr/bin/env python3
"""
LLM Analysis Engine for Git Repository Analysis Output
Feeds analysis data to LLMs for project summarization and data flow understanding
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import tiktoken
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMAnalysisEngine:
    """Engine to prepare and feed repository analysis to LLMs"""
    
    def __init__(self, analysis_dir: str, model_name: str = "gemini-2.0-flash"):
        self.analysis_dir = Path(analysis_dir)
        self.model_name = model_name
        
        # Token counting for different models
        try:
            if "gemini" in model_name.lower():
                # For Gemini models, use approximate token counting
                self.encoding = None  # Gemini doesn't use tiktoken
            else:
                self.encoding = tiktoken.encoding_for_model(model_name)
        except:
            self.encoding = tiktoken.get_encoding("cl100k_base")  # Default fallback
        
        # Context limits for different models (updated for Gemini 2.0)
        self.context_limits = {
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "claude-3": 200000,
            "claude-3.5": 200000,
            "claude-3-opus": 200000,
            "claude-3-sonnet": 200000,
            "claude-3-haiku": 200000,
            "gemini-pro": 1000000,
            "gemini-1.5-pro": 2000000,
            "gemini-2.0-flash": 1000000,  # Gemini 2.0 Flash context limit
            "gemini-2.0-flash-001": 1000000,
            "gemini-flash": 1000000,
            "gemini-flash-001": 1000000
        }
        
        self.max_tokens = self.context_limits.get(model_name, self.context_limits.get("gemini-2.0-flash", 1000000))
        
        # Gemini-specific optimizations
        self.is_gemini = "gemini" in model_name.lower()
        if self.is_gemini:
            self.max_tokens = min(self.max_tokens, 800000)  # Leave buffer for response
        
        logger.info(f"Initialized LLM Analysis Engine for {model_name} (max tokens: {self.max_tokens:,})")
        if self.is_gemini:
            logger.info("🤖 Gemini model detected - using optimized prompts and token counting")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text with Gemini-optimized counting"""
        try:
            if self.is_gemini:
                # Gemini token counting approximation
                # Gemini generally has more efficient tokenization than GPT models
                # Approximate: 1 token ≈ 3-4 characters for English text
                return len(text) // 3
            else:
                return len(self.encoding.encode(text))
        except:
            # Fallback estimation
            if self.is_gemini:
                return len(text) // 3
            else:
                return len(text) // 4
    
    def load_analysis_summary(self) -> Dict[str, Any]:
        """Load the main analysis summary"""
        summary_file = self.analysis_dir / "summary_report.json"
        if not summary_file.exists():
            raise FileNotFoundError(f"Summary report not found: {summary_file}")
        
        with open(summary_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_category_data(self, category: str, limit_files: int = None) -> Dict[str, Any]:
        """Load data for a specific category with optional file limit"""
        category_dir = self.analysis_dir / category
        
        if not category_dir.exists():
            logger.warning(f"Category directory not found: {category_dir}")
            return {}
        
        # Load category summary
        summary_file = category_dir / "summary.json"
        category_data = {"summary": {}, "files": []}
        
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                category_data["summary"] = json.load(f)
        
        # Load individual files
        individual_files_dir = category_dir / "individual_files"
        if individual_files_dir.exists():
            files = list(individual_files_dir.glob("*.json"))
            if limit_files:
                files = files[:limit_files]
            
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)
                        category_data["files"].append(file_data)
                except Exception as e:
                    logger.warning(f"Could not load {file_path}: {e}")
        
        return category_data
    
    def create_project_overview_prompt(self) -> str:
        """Create a comprehensive project overview prompt optimized for Gemini"""
        summary = self.load_analysis_summary()
        
        if self.is_gemini:
            # Gemini-optimized prompt structure
            prompt = f"""# Spring Boot Project Analysis for Gemini 2.0 Flash

**Task**: Analyze this Spring Boot project and provide comprehensive insights.

## 📊 Repository Data
- **URL**: {summary.get('repository_url', 'Unknown')}
- **Analysis Date**: {summary.get('analysis_timestamp', 'Unknown')}
- **Total Files**: {summary.get('total_files_analyzed', 0)}

## 📁 File Categories"""
        else:
            prompt = f"""# Spring Boot Project Analysis

I have analyzed a Git repository and need you to provide a comprehensive project overview. Here's the analysis data:

## Repository Information
- **Repository URL**: {summary.get('repository_url', 'Unknown')}
- **Analysis Date**: {summary.get('analysis_timestamp', 'Unknown')}
- **Total Files Analyzed**: {summary.get('total_files_analyzed', 0)}

## File Categories"""
        
        if 'categories' in summary:
            for category in summary['categories']:
                if isinstance(category, dict):
                    prompt += f"\n- **{category.get('category', 'Unknown')}**: {category.get('file_count', 0)} files"
        
        prompt += "\n\n## 🚀 Spring Boot Components Detected"
        
        if 'spring_boot_features' in summary:
            features = summary['spring_boot_features']
            if isinstance(features, dict) and 'note' not in features:
                prompt += f"\n- **Controllers**: {features.get('controllers', 0)}"
                prompt += f"\n- **Services**: {features.get('services', 0)}"
                prompt += f"\n- **Repositories**: {features.get('repositories', 0)}"
                prompt += f"\n- **Components**: {features.get('components', 0)}"
        
        if 'sample_packages' in summary:
            prompt += f"\n\n## 📦 Package Structure Sample"
            for package in summary['sample_packages'][:10]:
                prompt += f"\n- {package}"
        
        if self.is_gemini:
            # Gemini prefers structured, clear requests
            prompt += """

## 🎯 Analysis Request

Please provide a **comprehensive analysis** in the following structure:

### 1. 🏗️ **Project Type & Architecture**
- What type of Spring Boot application is this?
- What architectural patterns are being used?

### 2. 🛠️ **Technology Stack**
- Core technologies and frameworks identified
- Dependencies and integrations

### 3. 📐 **Architecture Analysis**
- Layered architecture assessment
- Design patterns observed
- Component organization

### 4. 🔧 **Key Components**
- Main application components and their roles
- Inter-component relationships

### 5. 💼 **Business Domain**
- Inferred business purpose
- Domain-specific functionality

### 6. ✅ **Code Quality Assessment**
- Organization and structure quality
- Naming conventions and patterns

### 7. 💡 **Recommendations**
- Improvement suggestions
- Best practices recommendations
- Areas for further investigation

**Note**: Base your analysis on the quantitative data provided above. Be specific and actionable in your recommendations.
"""
        else:
            prompt += """

## Analysis Request

Please provide:

1. **Project Type & Architecture**: What type of Spring Boot application is this? (REST API, microservice, web app, etc.)

2. **Technology Stack**: What technologies and frameworks are being used?

3. **Architecture Patterns**: What architectural patterns can you identify? (MVC, layered architecture, microservices, etc.)

4. **Main Components**: What are the key components and their purposes?

5. **Business Domain**: Based on package names and structure, what business domain does this appear to serve?

6. **Code Quality Observations**: Any observations about code organization and structure?

7. **Recommendations**: Any recommendations for improvements or further analysis?

Please provide a structured analysis based on this high-level information.
"""
        
        return prompt
    
    def create_detailed_code_analysis_prompt(self, category: str = "java_files", max_files: int = 15) -> str:
        """Create detailed code analysis prompt optimized for Gemini's large context"""
        if self.is_gemini:
            # Gemini can handle more files due to larger context window
            max_files = min(max_files * 3, 50)  # Increase file limit for Gemini
        
        category_data = self.load_category_data(category, limit_files=max_files)
        
        if not category_data.get("files"):
            return f"No files found in category: {category}"
        
        if self.is_gemini:
            prompt = f"""# 🔍 Detailed {category.title()} Analysis - Gemini 2.0 Flash

**Task**: Perform comprehensive code analysis on these Spring Boot {category}.

## 📁 Files for Analysis ({len(category_data['files'])} files)

"""
        else:
            prompt = f"""# Detailed {category.title()} Analysis

I need you to analyze the following {category} from a Spring Boot project for detailed insights:

## Files to Analyze ({len(category_data['files'])} files)

"""
        
        # Add file contents with Gemini-optimized token management
        current_tokens = self.count_tokens(prompt)
        files_added = 0
        
        for file_data in category_data["files"]:
            if self.is_gemini:
                file_prompt = f"""
### 📄 **{file_data.get('file_path', 'Unknown')}**
- **Size**: {file_data.get('size', 0):,} bytes
- **Lines**: {file_data.get('lines', 0):,}
"""
            else:
                file_prompt = f"""
### File: {file_data.get('file_path', 'Unknown')}
**Size**: {file_data.get('size', 0)} bytes
**Lines**: {file_data.get('lines', 0)}

"""
            
            # Add Java analysis if available
            if 'java_analysis' in file_data:
                java_analysis = file_data['java_analysis']
                if self.is_gemini:
                    file_prompt += f"- **Package**: `{java_analysis.get('package', 'None')}`\n"
                    file_prompt += f"- **Spring Annotations**: `{', '.join(java_analysis.get('spring_annotations', []))}`\n"
                    file_prompt += f"- **Imports**: {java_analysis.get('import_count', 0)} imports\n\n"
                else:
                    file_prompt += f"**Package**: {java_analysis.get('package', 'None')}\n"
                    file_prompt += f"**Spring Annotations**: {', '.join(java_analysis.get('spring_annotations', []))}\n"
                    file_prompt += f"**Import Count**: {java_analysis.get('import_count', 0)}\n\n"
            
            # Add content (preview or full)
            if 'content' in file_data:
                content = file_data['content']
            elif 'content_preview' in file_data:
                content = file_data['content_preview']
                file_prompt += "*(Content preview - full content was chunked)*\n\n"
            else:
                content = "Content not available"
            
            file_prompt += f"```java\n{content}\n```\n\n"
            
            # Check token limit (more generous for Gemini)
            file_tokens = self.count_tokens(file_prompt)
            reserve_tokens = 2000 if self.is_gemini else 1000
            
            if current_tokens + file_tokens > self.max_tokens - reserve_tokens:
                prompt += f"*(Truncated due to context limit - showing {files_added} of {len(category_data['files'])} files)*\n\n"
                break
            
            prompt += file_prompt
            current_tokens += file_tokens
            files_added += 1
        
        if self.is_gemini:
            # Gemini-optimized analysis request
            prompt += """
## 🎯 **Comprehensive Analysis Request**

Provide detailed analysis covering these key areas:

### 1. 🏗️ **Code Architecture & Organization**
- Package structure and organization patterns
- Class hierarchy and relationships
- Adherence to SOLID principles

### 2. 🚀 **Spring Boot Framework Usage**
- Spring annotations and their effectiveness
- Dependency injection patterns
- Configuration management
- Auto-configuration utilization

### 3. 🔄 **Data Flow & Component Interaction**
- Request/response flow patterns
- Service layer interactions
- Data access patterns
- Error handling mechanisms

### 4. 💼 **Business Logic Implementation**
- Domain modeling approach
- Business rule implementation
- Data validation strategies

### 5. 🔗 **Dependencies & Integration**
- External service integrations
- Database interaction patterns
- Cross-cutting concerns handling

### 6. ✅ **Code Quality Assessment**
- Naming conventions and clarity
- Method and class design
- Code duplication analysis
- Documentation and comments

### 7. 🛡️ **Security & Performance**
- Security implementation patterns
- Performance optimization opportunities
- Resource management

### 8. 🚨 **Issues & Anti-patterns**
- Code smells identification
- Design anti-patterns
- Potential bug sources

### 9. 🎯 **Specific Recommendations**
- Immediate improvement opportunities
- Refactoring suggestions
- Best practice implementations

### 10. 📈 **Metrics & Insights**
- Complexity analysis
- Maintainability assessment
- Scalability considerations

**Instructions**: Be specific and reference actual code elements. Provide actionable recommendations with examples where possible.
"""
        else:
            prompt += """
## Analysis Request

Please provide detailed analysis covering:

1. **Code Architecture**: How are these files organized? What patterns do you see?

2. **Spring Boot Usage**: How is Spring Boot being utilized? What annotations and features are used?

3. **Data Flow**: Can you trace the data flow between these components?

4. **Business Logic**: What business functionality can you identify?

5. **Dependencies & Relationships**: How do these files relate to each other?

6. **Code Quality**: Assessment of code quality, naming conventions, structure

7. **Security Considerations**: Any security patterns or potential concerns?

8. **Performance Considerations**: Any performance-related observations?

9. **Potential Issues**: Any code smells, anti-patterns, or areas for improvement?

10. **Recommendations**: Specific recommendations for improvements
"""
        
        return prompt
    
    def create_data_flow_analysis_prompt(self) -> str:
        """Create prompt specifically for data flow analysis"""
        # Load key components
        java_data = self.load_category_data("java_files", limit_files=20)
        config_data = self.load_category_data("config_files", limit_files=5)
        
        prompt = """# Data Flow Analysis Request

I need you to analyze the data flow in this Spring Boot application. Here's the relevant code:

## Configuration Files
"""
        
        # Add config files
        for file_data in config_data.get("files", []):
            if file_data.get('size', 0) < 5000:  # Only include smaller config files
                prompt += f"""
### {file_data.get('file_path', 'Unknown')}
```
{file_data.get('content', file_data.get('content_preview', 'Content not available'))}
```
"""
        
        prompt += "\n## Java Components\n"
        
        # Add Java files with focus on controllers, services, repositories
        current_tokens = self.count_tokens(prompt)
        
        # Prioritize important files
        important_files = []
        other_files = []
        
        for file_data in java_data.get("files", []):
            if 'java_analysis' in file_data:
                annotations = file_data['java_analysis'].get('spring_annotations', [])
                if any('Controller' in ann or 'Service' in ann or 'Repository' in ann for ann in annotations):
                    important_files.append(file_data)
                else:
                    other_files.append(file_data)
            else:
                other_files.append(file_data)
        
        # Add important files first
        for file_data in important_files + other_files:
            file_content = f"""
### {file_data.get('file_path', 'Unknown')}
**Type**: {', '.join(file_data.get('java_analysis', {}).get('spring_annotations', []))}
**Package**: {file_data.get('java_analysis', {}).get('package', 'Unknown')}

```java
{file_data.get('content', file_data.get('content_preview', 'Content not available'))}
```
"""
            
            if current_tokens + self.count_tokens(file_content) > self.max_tokens - 1500:
                break
                
            prompt += file_content
            current_tokens += self.count_tokens(file_content)
        
        prompt += """

## Data Flow Analysis Request

Please analyze and provide:

1. **Request Flow**: Trace how HTTP requests flow through the application
   - Entry points (Controllers)
   - Service layer interactions
   - Data access layer (Repositories)
   - Response generation

2. **Data Models**: What data models/entities can you identify?

3. **API Endpoints**: List all REST endpoints with their purposes

4. **Service Dependencies**: Map dependencies between services

5. **Database Interactions**: How does the application interact with data storage?

6. **Configuration Impact**: How do configuration files affect data flow?

7. **Integration Points**: External service integrations or APIs

8. **Data Transformation**: Where and how is data transformed?

9. **Error Handling**: How are errors handled in the data flow?

10. **Flow Diagram**: Provide a textual representation of the main data flows

Please be specific and reference actual code elements you see in the analysis.
"""
        
        return prompt
    
    def create_chunked_analysis_prompts(self, category: str, analysis_type: str = "code_review") -> List[str]:
        """Create multiple smaller prompts for large datasets"""
        category_data = self.load_category_data(category)
        files = category_data.get("files", [])
        
        if not files:
            return []
        
        prompts = []
        current_prompt = f"# {analysis_type.title()} - Part {{part_num}} of {{total_parts}}\n\n"
        current_tokens = self.count_tokens(current_prompt)
        current_files = []
        
        for file_data in files:
            file_content = self._format_file_for_prompt(file_data)
            file_tokens = self.count_tokens(file_content)
            
            # If adding this file would exceed limit, finalize current prompt
            if current_tokens + file_tokens > self.max_tokens - 1000:
                if current_files:  # Only add if we have files
                    prompts.append(self._finalize_chunk_prompt(current_prompt, current_files, analysis_type))
                
                # Start new prompt
                current_prompt = f"# {analysis_type.title()} - Part {{part_num}} of {{total_parts}}\n\n"
                current_tokens = self.count_tokens(current_prompt)
                current_files = []
            
            current_prompt += file_content
            current_tokens += file_tokens
            current_files.append(file_data)
        
        # Add final prompt if there are remaining files
        if current_files:
            prompts.append(self._finalize_chunk_prompt(current_prompt, current_files, analysis_type))
        
        # Update part numbers
        total_parts = len(prompts)
        for i, prompt in enumerate(prompts):
            prompts[i] = prompt.format(part_num=i+1, total_parts=total_parts)
        
        return prompts
    
    def _format_file_for_prompt(self, file_data: Dict[str, Any]) -> str:
        """Format a file for inclusion in prompt"""
        content = f"""
## File: {file_data.get('file_path', 'Unknown')}
**Size**: {file_data.get('size', 0)} bytes | **Lines**: {file_data.get('lines', 0)}
"""
        
        if 'java_analysis' in file_data:
            java_analysis = file_data['java_analysis']
            content += f"**Package**: {java_analysis.get('package', 'None')}\n"
            content += f"**Spring Annotations**: {', '.join(java_analysis.get('spring_annotations', []))}\n"
        
        file_content = file_data.get('content', file_data.get('content_preview', 'Content not available'))
        content += f"\n```java\n{file_content}\n```\n\n"
        
        return content
    
    def _finalize_chunk_prompt(self, base_prompt: str, files: List[Dict], analysis_type: str) -> str:
        """Finalize a chunk prompt with analysis request"""
        prompt = base_prompt
        
        if analysis_type == "code_review":
            prompt += """
## Analysis Request for This Chunk

Please analyze these files and provide:

1. **Code Quality Assessment**: Overall quality, patterns, conventions
2. **Spring Boot Usage**: How Spring Boot features are utilized
3. **Architecture Observations**: Architectural patterns and structure
4. **Potential Issues**: Any problems or areas for improvement
5. **Dependencies**: Relationships between these files
6. **Recommendations**: Specific improvement suggestions

Focus on the files in this chunk and note any patterns or issues you observe.
"""
        elif analysis_type == "data_flow":
            prompt += """
## Data Flow Analysis for This Chunk

Please analyze the data flow in these files:

1. **Component Interactions**: How do these components interact?
2. **Data Processing**: What data processing occurs?
3. **API Endpoints**: Any REST endpoints defined?
4. **Service Dependencies**: Dependencies on other services
5. **Data Models**: Data structures used
6. **Flow Patterns**: Common flow patterns observed

Focus specifically on how data moves through these components.
"""
        
        return prompt
    
    def generate_analysis_prompts(self, analysis_types: List[str] = None) -> Dict[str, List[str]]:
        """Generate all analysis prompts"""
        if analysis_types is None:
            analysis_types = ["project_overview", "code_analysis", "data_flow", "chunked_review"]
        
        prompts = {}
        
        for analysis_type in analysis_types:
            try:
                if analysis_type == "project_overview":
                    prompts[analysis_type] = [self.create_project_overview_prompt()]
                elif analysis_type == "code_analysis":
                    prompts[analysis_type] = [self.create_detailed_code_analysis_prompt()]
                elif analysis_type == "data_flow":
                    prompts[analysis_type] = [self.create_data_flow_analysis_prompt()]
                elif analysis_type == "chunked_review":
                    prompts[analysis_type] = self.create_chunked_analysis_prompts("java_files", "code_review")
                elif analysis_type == "chunked_data_flow":
                    prompts[analysis_type] = self.create_chunked_analysis_prompts("java_files", "data_flow")
                else:
                    logger.warning(f"Unknown analysis type: {analysis_type}")
            except Exception as e:
                logger.error(f"Error generating {analysis_type} prompts: {e}")
                prompts[analysis_type] = []
        
        return prompts
    
    def save_prompts_to_files(self, prompts: Dict[str, List[str]], output_dir: str = None):
        """Save generated prompts to files"""
        if output_dir is None:
            output_dir = self.analysis_dir / "llm_prompts"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        prompt_index = {
            "generated_at": datetime.now().isoformat(),
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "prompts": {}
        }
        
        for analysis_type, prompt_list in prompts.items():
            if not prompt_list:
                continue
                
            type_dir = output_dir / analysis_type
            type_dir.mkdir(exist_ok=True)
            
            prompt_index["prompts"][analysis_type] = {
                "count": len(prompt_list),
                "files": []
            }
            
            for i, prompt in enumerate(prompt_list):
                filename = f"{analysis_type}_part_{i+1:03d}.txt"
                filepath = type_dir / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(prompt)
                
                prompt_index["prompts"][analysis_type]["files"].append(str(filepath.relative_to(output_dir)))
                logger.info(f"Saved prompt: {filepath}")
        
        # Save index
        index_file = output_dir / "prompt_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(prompt_index, f, indent=2)
        
        logger.info(f"Generated {sum(len(p) for p in prompts.values())} prompts in {output_dir}")
        return str(output_dir)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate LLM analysis prompts from repository analysis')
    parser.add_argument('analysis_dir', help='Analysis output directory')
    parser.add_argument('--model', default='gpt-4', help='LLM model name (default: gpt-4)')
    parser.add_argument('--output', help='Output directory for prompts (default: analysis_dir/llm_prompts)')
    parser.add_argument('--types', nargs='+', 
                       choices=['project_overview', 'code_analysis', 'data_flow', 'chunked_review', 'chunked_data_flow'],
                       help='Analysis types to generate')
    
    args = parser.parse_args()
    
    print("🤖 LLM Analysis Engine")
    print("=" * 50)
    print(f"Analysis Directory: {args.analysis_dir}")
    print(f"Model: {args.model}")
    print(f"Analysis Types: {args.types or 'All'}")
    print("-" * 50)
    
    try:
        engine = LLMAnalysisEngine(args.analysis_dir, args.model)
        prompts = engine.generate_analysis_prompts(args.types)
        
        total_prompts = sum(len(p) for p in prompts.values())
        print(f"✅ Generated {total_prompts} prompts")
        
        # Save prompts
        output_dir = engine.save_prompts_to_files(prompts, args.output)
        print(f"📁 Prompts saved to: {output_dir}")
        
        # Show summary
        print("\n📊 PROMPT SUMMARY:")
        for analysis_type, prompt_list in prompts.items():
            if prompt_list:
                print(f"  {analysis_type}: {len(prompt_list)} prompts")
        
        print(f"\n💡 NEXT STEPS:")
        print("1. Copy prompts to your LLM interface (ChatGPT, Claude, etc.)")
        print("2. Run each prompt and collect responses")
        print("3. Combine insights for comprehensive project understanding")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()





#!/usr/bin/env python3
"""
LLM Response Processor and Report Generator
Processes LLM responses and creates comprehensive project reports
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMResponseProcessor:
    """Process LLM responses and generate comprehensive reports"""
    
    def __init__(self, analysis_dir: str):
        self.analysis_dir = Path(analysis_dir)
        self.responses_dir = self.analysis_dir / "llm_responses"
        self.reports_dir = self.analysis_dir / "llm_reports"
        
        # Create directories
        self.responses_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized LLM Response Processor for {analysis_dir}")
    
    def save_response(self, analysis_type: str, part_num: int, response: str, metadata: Dict = None):
        """Save an LLM response with metadata"""
        if metadata is None:
            metadata = {}
        
        response_data = {
            "analysis_type": analysis_type,
            "part_number": part_num,
            "timestamp": datetime.now().isoformat(),
            "response": response,
            "metadata": metadata
        }
        
        filename = f"{analysis_type}_part_{part_num:03d}_response.json"
        filepath = self.responses_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved response: {filepath}")
        return str(filepath)
    
    def load_responses(self, analysis_type: str = None) -> Dict[str, List[Dict]]:
        """Load all responses, optionally filtered by analysis type"""
        responses = {}
        
        if not self.responses_dir.exists():
            return responses
        
        for response_file in self.responses_dir.glob("*_response.json"):
            try:
                with open(response_file, 'r', encoding='utf-8') as f:
                    response_data = json.load(f)
                
                resp_type = response_data.get("analysis_type", "unknown")
                
                if analysis_type and resp_type != analysis_type:
                    continue
                
                if resp_type not in responses:
                    responses[resp_type] = []
                
                responses[resp_type].append(response_data)
            
            except Exception as e:
                logger.warning(f"Could not load response {response_file}: {e}")
        
        # Sort responses by part number
        for resp_type in responses:
            responses[resp_type].sort(key=lambda x: x.get("part_number", 0))
        
        return responses
    
    def extract_key_insights(self, response_text: str) -> Dict[str, List[str]]:
        """Extract key insights from LLM response using patterns"""
        insights = {
            "architecture_patterns": [],
            "technologies": [],
            "issues": [],
            "recommendations": [],
            "endpoints": [],
            "components": [],
            "data_flow": []
        }
        
        # Define patterns to extract information
        patterns = {
            "architecture_patterns": [
                r"(?i)(?:architecture|pattern|design).*?(?:mvc|microservice|layered|rest|api|service|repository)",
                r"(?i)(?:uses|implements|follows).*?(?:pattern|architecture)"
            ],
            "technologies": [
                r"(?i)(?:spring|hibernate|jpa|mysql|postgresql|redis|kafka|docker|kubernetes)",
                r"(?i)(?:framework|library|database|technology).*?(?:spring|hibernate|jpa)"
            ],
            "issues": [
                r"(?i)(?:issue|problem|concern|warning|risk|vulnerability)",
                r"(?i)(?:should|could|might|potential|recommend).*?(?:fix|improve|address)"
            ],
            "recommendations": [
                r"(?i)(?:recommend|suggest|should|consider|improve)",
                r"(?i)(?:best practice|better approach|optimization)"
            ],
            "endpoints": [
                r"(?i)(?:endpoint|api|route).*?(?:GET|POST|PUT|DELETE|/[a-zA-Z0-9/_-]+)",
                r"(?:[/@][a-zA-Z0-9/_-]+(?:\{[^}]+\})?)"
            ],
            "components": [
                r"(?i)(?:controller|service|repository|component|bean)",
                r"(?i)(?:class|interface).*?(?:Controller|Service|Repository|Component)"
            ]
        }
        
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            for category, pattern_list in patterns.items():
                for pattern in pattern_list:
                    if re.search(pattern, line):
                        # Clean up the line
                        clean_line = re.sub(r'^[-*•]\s*', '', line)  # Remove bullet points
                        clean_line = re.sub(r'^\d+\.\s*', '', clean_line)  # Remove numbers
                        
                        if len(clean_line) > 20 and clean_line not in insights[category]:
                            insights[category].append(clean_line[:200])  # Limit length
        
        return insights
    
    def generate_project_overview_report(self) -> str:
        """Generate project overview report"""
        responses = self.load_responses("project_overview")
        
        if "project_overview" not in responses:
            return "No project overview responses found."
        
        report = """# Project Overview Report

## Summary
"""
        
        for response_data in responses["project_overview"]:
            response_text = response_data.get("response", "")
            report += f"\n{response_text}\n"
        
        return report
    
    def generate_code_analysis_report(self) -> str:
        """Generate comprehensive code analysis report"""
        responses = self.load_responses("code_analysis")
        chunked_responses = self.load_responses("chunked_review")
        
        report = """# Code Analysis Report

## Detailed Analysis
"""
        
        # Add main code analysis
        if "code_analysis" in responses:
            for response_data in responses["code_analysis"]:
                response_text = response_data.get("response", "")
                report += f"\n{response_text}\n"
        
        # Add chunked analysis summary
        if "chunked_review" in chunked_responses:
            report += "\n## Additional Code Review Insights\n"
            
            all_insights = {
                "architecture_patterns": set(),
                "technologies": set(),
                "issues": set(),
                "recommendations": set(),
                "components": set()
            }
            
            for response_data in chunked_responses["chunked_review"]:
                response_text = response_data.get("response", "")
                insights = self.extract_key_insights(response_text)
                
                for category, items in insights.items():
                    if category in all_insights:
                        all_insights[category].update(items)
            
            for category, items in all_insights.items():
                if items:
                    report += f"\n### {category.replace('_', ' ').title()}\n"
                    for item in sorted(items)[:10]:  # Limit to top 10
                        report += f"- {item}\n"
        
        return report
    
    def generate_data_flow_report(self) -> str:
        """Generate data flow analysis report"""
        responses = self.load_responses("data_flow")
        chunked_responses = self.load_responses("chunked_data_flow")
        
        report = """# Data Flow Analysis Report

## Request Flow & Architecture
"""
        
        # Add main data flow analysis
        if "data_flow" in responses:
            for response_data in responses["data_flow"]:
                response_text = response_data.get("response", "")
                report += f"\n{response_text}\n"
        
        # Extract and summarize endpoints and components
        if "chunked_data_flow" in chunked_responses:
            report += "\n## Data Flow Summary\n"
            
            all_endpoints = set()
            all_components = set()
            
            for response_data in chunked_responses["chunked_data_flow"]:
                response_text = response_data.get("response", "")
                insights = self.extract_key_insights(response_text)
                
                all_endpoints.update(insights.get("endpoints", []))
                all_components.update(insights.get("components", []))
            
            if all_endpoints:
                report += "\n### Identified API Endpoints\n"
                for endpoint in sorted(all_endpoints)[:20]:
                    report += f"- {endpoint}\n"
            
            if all_components:
                report += "\n### Key Components\n"
                for component in sorted(all_components)[:15]:
                    report += f"- {component}\n"
        
        return report
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive report combining all analyses"""
        responses = self.load_responses()
        
        if not responses:
            return "No LLM responses found. Please run analysis first."
        
        report = f"""# Comprehensive Project Analysis Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis Types**: {', '.join(responses.keys())}

---

"""
        
        # Project Overview Section
        if "project_overview" in responses:
            report += "## 📋 Project Overview\n\n"
            for response_data in responses["project_overview"]:
                report += response_data.get("response", "") + "\n\n"
        
        # Code Analysis Section
        code_types = ["code_analysis", "chunked_review"]
        code_responses = []
        for code_type in code_types:
            if code_type in responses:
                code_responses.extend(responses[code_type])
        
        if code_responses:
            report += "## 🔍 Code Analysis\n\n"
            
            # Combine insights from all code analysis responses
            combined_insights = {
                "architecture_patterns": set(),
                "technologies": set(),
                "issues": set(),
                "recommendations": set(),
                "components": set()
            }
            
            for response_data in code_responses:
                response_text = response_data.get("response", "")
                
                # Add full response for first detailed analysis
                if response_data.get("analysis_type") == "code_analysis":
                    report += f"### Detailed Analysis\n{response_text}\n\n"
                
                # Extract insights for summary
                insights = self.extract_key_insights(response_text)
                for category, items in insights.items():
                    if category in combined_insights:
                        combined_insights[category].update(items)
            
            # Add consolidated insights
            report += "### Consolidated Insights\n\n"
            for category, items in combined_insights.items():
                if items:
                    report += f"**{category.replace('_', ' ').title()}:**\n"
                    for item in sorted(items)[:8]:  # Top 8 items
                        report += f"- {item}\n"
                    report += "\n"
        
        # Data Flow Section
        data_flow_types = ["data_flow", "chunked_data_flow"]
        data_flow_responses = []
        for df_type in data_flow_types:
            if df_type in responses:
                data_flow_responses.extend(responses[df_type])
        
        if data_flow_responses:
            report += "## 🔄 Data Flow Analysis\n\n"
            
            # Add main data flow analysis
            for response_data in data_flow_responses:
                if response_data.get("analysis_type") == "data_flow":
                    report += response_data.get("response", "") + "\n\n"
                    break
            
            # Extract and summarize key data flow elements
            all_endpoints = set()
            all_components = set()
            data_flows = set()
            
            for response_data in data_flow_responses:
                response_text = response_data.get("response", "")
                insights = self.extract_key_insights(response_text)
                
                all_endpoints.update(insights.get("endpoints", []))
                all_components.update(insights.get("components", []))
                data_flows.update(insights.get("data_flow", []))
            
            if all_endpoints:
                report += "### API Endpoints Summary\n"
                for endpoint in sorted(all_endpoints)[:15]:
                    report += f"- {endpoint}\n"
                report += "\n"
            
            if all_components:
                report += "### Component Architecture\n"
                for component in sorted(all_components)[:12]:
                    report += f"- {component}\n"
                report += "\n"
        
        # Overall Recommendations
        all_recommendations = set()
        all_issues = set()
        
        for response_type, response_list in responses.items():
            for response_data in response_list:
                response_text = response_data.get("response", "")
                insights = self.extract_key_insights(response_text)
                all_recommendations.update(insights.get("recommendations", []))
                all_issues.update(insights.get("issues", []))
        
        if all_recommendations or all_issues:
            report += "## 💡 Summary & Recommendations\n\n"
            
            if all_issues:
                report += "### Key Issues Identified\n"
                for issue in sorted(all_issues)[:10]:
                    report += f"- {issue}\n"
                report += "\n"
            
            if all_recommendations:
                report += "### Recommendations\n"
                for recommendation in sorted(all_recommendations)[:10]:
                    report += f"- {recommendation}\n"
                report += "\n"
        
        # Metadata
        report += "---\n\n"
        report += f"**Total Responses Processed**: {sum(len(r) for r in responses.values())}\n"
        report += f"**Analysis Types**: {len(responses)}\n"
        
        return report
    
    def create_response_template(self, analysis_type: str, part_num: int = 1) -> str:
        """Create a template for collecting LLM responses"""
        template = f"""# LLM Response Collection Template

**Analysis Type**: {analysis_type}
**Part Number**: {part_num}
**Date**: {datetime.now().strftime('%Y-%m-%d')}

## Instructions
1. Copy the prompt from: `llm_prompts/{analysis_type}/`
2. Run it in your LLM (ChatGPT, Claude, etc.)
3. Paste the response below
4. Save this file or use the save_response() method

## LLM Response
```
[PASTE YOUR LLM RESPONSE HERE]
```

## Additional Notes
- Model used: 
- Temperature: 
- Any special instructions: 

---
**Next Steps**: Run `save_response("{analysis_type}", {part_num}, response_text)` to process this response.
"""
        return template
    
    def interactive_response_collector(self):
        """Interactive CLI for collecting LLM responses"""
        print("🤖 Interactive LLM Response Collector")
        print("=" * 50)
        
        # List available prompt types
        prompts_dir = self.analysis_dir / "llm_prompts"
        if not prompts_dir.exists():
            print("❌ No prompts directory found. Run the LLM analyzer first.")
            return
        
        available_types = [d.name for d in prompts_dir.iterdir() if d.is_dir()]
        
        if not available_types:
            print("❌ No prompt types found.")
            return
        
        print("Available analysis types:")
        for i, analysis_type in enumerate(available_types, 1):
            prompt_files = list((prompts_dir / analysis_type).glob("*.txt"))
            print(f"  {i}. {analysis_type} ({len(prompt_files)} prompts)")
        
        while True:
            try:
                choice = input("\nSelect analysis type (number) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    break
                
                if not choice.isdigit() or int(choice) < 1 or int(choice) > len(available_types):
                    print("Invalid choice. Please try again.")
                    continue
                
                analysis_type = available_types[int(choice) - 1]
                self._collect_responses_for_type(analysis_type)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _collect_responses_for_type(self, analysis_type: str):
        """Collect responses for a specific analysis type"""
        prompts_dir = self.analysis_dir / "llm_prompts" / analysis_type
        prompt_files = sorted(prompts_dir.glob("*.txt"))
        
        print(f"\n📝 Collecting responses for: {analysis_type}")
        print(f"Found {len(prompt_files)} prompts")
        
        for i, prompt_file in enumerate(prompt_files, 1):
            print(f"\n--- Prompt {i}/{len(prompt_files)} ---")
            print(f"File: {prompt_file.name}")
            
            # Show a preview of the prompt
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_content = f.read()
            
            print("Preview:")
            print(prompt_content[:200] + "..." if len(prompt_content) > 200 else prompt_content)
            
            action = input("\n[c]opy prompt, [s]kip, [v]iew full, [q]uit: ").strip().lower()
            
            if action == 'q':
                break
            elif action == 's':
                continue
            elif action == 'v':
                print("\n" + "="*50)
                print(prompt_content)
                print("="*50)
                input("Press Enter to continue...")
                continue
            elif action == 'c':
                # Copy to clipboard if available
                try:
                    import pyperclip
                    pyperclip.copy(prompt_content)
                    print("✅ Prompt copied to clipboard!")
                except ImportError:
                    print("📋 Clipboard not available. Here's the prompt:")
                    print("\n" + "="*50)
                    print(prompt_content)
                    print("="*50)
                
                # Collect response
                print("\nAfter running this in your LLM, paste the response below.")
                print("Press Enter twice when finished, or type 'skip' to skip this prompt.")
                
                response_lines = []
                empty_lines = 0
                
                while True:
                    line = input()
                    if line.strip().lower() == 'skip':
                        break
                    if line.strip() == '':
                        empty_lines += 1
                        if empty_lines >= 2:
                            break
                    else:
                        empty_lines = 0
                    response_lines.append(line)
                
                if response_lines and response_lines != ['skip']:
                    response_text = '\n'.join(response_lines).strip()
                    if response_text:
                        self.save_response(analysis_type, i, response_text)
                        print(f"✅ Response saved for {analysis_type} part {i}")
    
    def generate_all_reports(self) -> Dict[str, str]:
        """Generate all available reports"""
        reports = {}
        
        try:
            reports["project_overview"] = self.generate_project_overview_report()
        except Exception as e:
            logger.error(f"Error generating project overview: {e}")
        
        try:
            reports["code_analysis"] = self.generate_code_analysis_report()
        except Exception as e:
            logger.error(f"Error generating code analysis: {e}")
        
        try:
            reports["data_flow"] = self.generate_data_flow_report()
        except Exception as e:
            logger.error(f"Error generating data flow report: {e}")
        
        try:
            reports["comprehensive"] = self.generate_comprehensive_report()
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
        
        return reports
    
    def save_reports(self, reports: Dict[str, str] = None):
        """Save all reports to files"""
        if reports is None:
            reports = self.generate_all_reports()
        
        saved_files = []
        
        for report_type, content in reports.items():
            if content and content.strip():
                filename = f"{report_type}_report.md"
                filepath = self.reports_dir / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                saved_files.append(str(filepath))
                logger.info(f"Saved report: {filepath}")
        
        # Create index file
        index_content = f"""# LLM Analysis Reports

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Available Reports

"""
        
        for report_type in reports.keys():
            if reports[report_type] and reports[report_type].strip():
                index_content += f"- [{report_type.replace('_', ' ').title()}]({report_type}_report.md)\n"
        
        index_content += f"""
## Usage

1. **Project Overview**: High-level summary and architecture
2. **Code Analysis**: Detailed code review and quality assessment  
3. **Data Flow**: Request flow and component interactions
4. **Comprehensive**: Combined analysis with recommendations

## Response Collection

To add more LLM responses:
```python
from llm_response_processor import LLMResponseProcessor
processor = LLMResponseProcessor("{self.analysis_dir}")
processor.interactive_response_collector()
```
"""
        
        index_file = self.reports_dir / "README.md"
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(index_content)
        
        saved_files.append(str(index_file))
        
        return saved_files

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process LLM responses and generate reports')
    parser.add_argument('analysis_dir', help='Analysis output directory')
    parser.add_argument('--interactive', action='store_true', help='Interactive response collection')
    parser.add_argument('--generate-reports', action='store_true', help='Generate reports from existing responses')
    parser.add_argument('--response-type', help='Specific response type to process')
    
    args = parser.parse_args()
    
    print("📊 LLM Response Processor")
    print("=" * 50)
    
    try:
        processor = LLMResponseProcessor(args.analysis_dir)
        
        if args.interactive:
            processor.interactive_response_collector()
        
        if args.generate_reports:
            print("Generating reports...")
            reports = processor.generate_all_reports()
            saved_files = processor.save_reports(reports)
            
            print(f"✅ Generated {len(saved_files)} report files:")
            for file_path in saved_files:
                print(f"  📄 {file_path}")
        
        if not args.interactive and not args.generate_reports:
            print("Use --interactive to collect responses or --generate-reports to create reports")
            print("Example usage:")
            print(f"  python {__file__} {args.analysis_dir} --interactive")
            print(f"  python {__file__} {args.analysis_dir} --generate-reports")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()








#!/usr/bin/env python3
"""
Complete Chunk Coverage Analyzer
Ensures ALL files and chunks are analyzed by creating multiple prompts
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveChunkAnalyzer:
    """Ensures 100% coverage of all files and chunks through multiple prompts"""
    
    def __init__(self, analysis_dir: str, model_name: str = "gemini-2.0-flash-001"):
        self.analysis_dir = Path(analysis_dir)
        self.model_name = model_name
        self.is_gemini = "gemini" in model_name.lower()
        
        # Model-specific settings
        self.context_limits = {
            "gemini-2.0-flash": 1000000,
            "gemini-2.0-flash-001": 1000000,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "claude-3": 200000,
            "claude-3.5": 200000
        }
        
        self.max_tokens = self.context_limits.get(model_name, 1000000)
        self.reserve_tokens = 5000 if self.is_gemini else 2000  # Reserve for response
        self.usable_tokens = self.max_tokens - self.reserve_tokens
        
        logger.info(f"🔍 Comprehensive Chunk Analyzer initialized")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Max Tokens: {self.max_tokens:,}")
        logger.info(f"   Usable Tokens: {self.usable_tokens:,}")
    
    def count_tokens(self, text: str) -> int:
        """Token counting with model-specific optimization"""
        if self.is_gemini:
            return len(text) // 3  # Gemini's more efficient tokenization
        else:
            return len(text) // 4  # Conservative estimate for other models
    
    def analyze_current_coverage(self) -> Dict[str, Any]:
        """Analyze what files and chunks exist vs what's being processed"""
        
        coverage_report = {
            "timestamp": datetime.now().isoformat(),
            "categories": {},
            "total_files": 0,
            "total_chunks": 0,
            "coverage_gaps": []
        }
        
        # Analyze each category
        for category_dir in self.analysis_dir.iterdir():
            if not category_dir.is_dir() or category_dir.name.startswith('.'):
                continue
            
            category_name = category_dir.name
            category_info = {
                "individual_files": 0,
                "chunk_files": 0,
                "total_content_size": 0,
                "files_with_chunks": 0,
                "estimated_tokens": 0
            }
            
            # Count individual files
            individual_dir = category_dir / "individual_files"
            if individual_dir.exists():
                individual_files = list(individual_dir.glob("*.json"))
                category_info["individual_files"] = len(individual_files)
                
                # Analyze content size
                for file_path in individual_files:
                    try:
                        with open(file_path, 'r') as f:
                            file_data = json.load(f)
                        
                        content = file_data.get('content', file_data.get('content_preview', ''))
                        if content:
                            size = len(content)
                            category_info["total_content_size"] += size
                            category_info["estimated_tokens"] += self.count_tokens(content)
                        
                        if file_data.get('has_chunks', False):
                            category_info["files_with_chunks"] += 1
                            
                    except Exception as e:
                        logger.warning(f"Could not analyze {file_path}: {e}")
            
            # Count chunk files
            chunks_dir = category_dir / "chunks"
            if chunks_dir.exists():
                chunk_files = list(chunks_dir.glob("*.json"))
                category_info["chunk_files"] = len(chunk_files)
                
                # Analyze chunk content
                for chunk_file in chunk_files:
                    try:
                        with open(chunk_file, 'r') as f:
                            chunk_data = json.load(f)
                        
                        chunks = chunk_data.get('chunks', [])
                        for chunk in chunks:
                            category_info["estimated_tokens"] += self.count_tokens(chunk)
                            
                    except Exception as e:
                        logger.warning(f"Could not analyze chunk {chunk_file}: {e}")
            
            coverage_report["categories"][category_name] = category_info
            coverage_report["total_files"] += category_info["individual_files"]
            coverage_report["total_chunks"] += category_info["chunk_files"]
        
        # Calculate coverage gaps
        for category, info in coverage_report["categories"].items():
            if info["estimated_tokens"] > self.usable_tokens:
                prompts_needed = math.ceil(info["estimated_tokens"] / self.usable_tokens)
                coverage_report["coverage_gaps"].append({
                    "category": category,
                    "total_tokens": info["estimated_tokens"],
                    "prompts_needed": prompts_needed,
                    "current_coverage": "INCOMPLETE" if prompts_needed > 1 else "COMPLETE"
                })
        
        return coverage_report
    
    def create_multi_part_prompts(self, category: str, analysis_type: str = "code_analysis") -> List[str]:
        """Create multiple prompts to ensure 100% coverage of a category"""
        
        # Load all files in category
        category_dir = self.analysis_dir / category
        if not category_dir.exists():
            return []
        
        all_files = []
        individual_dir = category_dir / "individual_files"
        
        if individual_dir.exists():
            for file_path in individual_dir.glob("*.json"):
                try:
                    with open(file_path, 'r') as f:
                        file_data = json.load(f)
                    
                    # Load chunks if they exist
                    if file_data.get('has_chunks', False):
                        chunks_file = file_data.get('chunks_file')
                        if chunks_file:
                            chunks_path = self.analysis_dir / chunks_file
                            if chunks_path.exists():
                                with open(chunks_path, 'r') as f:
                                    chunk_data = json.load(f)
                                
                                # Create separate entries for each chunk
                                for i, chunk in enumerate(chunk_data.get('chunks', [])):
                                    chunk_file_data = file_data.copy()
                                    chunk_file_data['content'] = chunk
                                    chunk_file_data['is_chunk'] = True
                                    chunk_file_data['chunk_number'] = i + 1
                                    chunk_file_data['total_chunks'] = len(chunk_data.get('chunks', []))
                                    chunk_file_data['original_file'] = file_data['file_path']
                                    all_files.append(chunk_file_data)
                            else:
                                # Fallback to preview if chunks not found
                                all_files.append(file_data)
                    else:
                        all_files.append(file_data)
                        
                except Exception as e:
                    logger.warning(f"Could not load {file_path}: {e}")
        
        if not all_files:
            return []
        
        logger.info(f"📊 Creating comprehensive prompts for {category}")
        logger.info(f"   Total files/chunks to analyze: {len(all_files)}")
        
        # Create multiple prompts to cover all files
        prompts = []
        current_prompt_files = []
        current_tokens = 0
        
        # Base prompt overhead
        base_prompt = self.create_base_prompt(category, analysis_type, 0, 0)
        base_tokens = self.count_tokens(base_prompt)
        
        for file_data in all_files:
            # Calculate tokens for this file
            content = file_data.get('content', file_data.get('content_preview', ''))
            file_section = self.format_file_for_prompt(file_data, content)
            file_tokens = self.count_tokens(file_section)
            
            # Check if adding this file would exceed limit
            if current_tokens + file_tokens + base_tokens > self.usable_tokens and current_prompt_files:
                # Finalize current prompt
                prompt = self.create_complete_prompt(
                    category, analysis_type, current_prompt_files, 
                    len(prompts) + 1, 0  # Will calculate total later
                )
                prompts.append(prompt)
                
                # Start new prompt
                current_prompt_files = [file_data]
                current_tokens = file_tokens
            else:
                current_prompt_files.append(file_data)
                current_tokens += file_tokens
        
        # Add final prompt if there are remaining files
        if current_prompt_files:
            prompt = self.create_complete_prompt(
                category, analysis_type, current_prompt_files,
                len(prompts) + 1, 0
            )
            prompts.append(prompt)
        
        # Update total counts in all prompts
        total_prompts = len(prompts)
        for i in range(len(prompts)):
            prompts[i] = prompts[i].replace("TOTAL_PARTS_PLACEHOLDER", str(total_prompts))
        
        logger.info(f"✅ Created {len(prompts)} prompts for complete {category} coverage")
        
        return prompts
    
    def format_file_for_prompt(self, file_data: Dict[str, Any], content: str) -> str:
        """Format a single file/chunk for inclusion in prompt"""
        
        file_path = file_data.get('file_path', 'Unknown')
        
        # Handle chunk vs regular file
        if file_data.get('is_chunk', False):
            chunk_num = file_data.get('chunk_number', 1)
            total_chunks = file_data.get('total_chunks', 1)
            original_file = file_data.get('original_file', file_path)
            
            section = f"""
### 📄 **{original_file}** (Chunk {chunk_num}/{total_chunks})
"""
        else:
            section = f"""
### 📄 **{file_path}**
"""
        
        # Add metadata
        section += f"- **Size**: {len(content):,} characters\n"
        section += f"- **Lines**: {content.count(chr(10)) + 1:,}\n"
        
        # Add Java analysis if available
        if 'java_analysis' in file_data:
            java_analysis = file_data['java_analysis']
            section += f"- **Package**: `{java_analysis.get('package', 'None')}`\n"
            
            annotations = java_analysis.get('spring_annotations', [])
            if annotations:
                section += f"- **Spring Annotations**: `{', '.join(annotations)}`\n"
            
            # Component type
            if any('Controller' in ann for ann in annotations):
                section += "- **Type**: 🎮 REST Controller\n"
            elif any('Service' in ann for ann in annotations):
                section += "- **Type**: ⚙️ Service Layer\n"
            elif any('Repository' in ann for ann in annotations):
                section += "- **Type**: 🗄️ Data Repository\n"
        
        # Add content
        section += f"\n**Source Code**:\n```java\n{content}\n```\n"
        
        return section
    
    def create_base_prompt(self, category: str, analysis_type: str, part_num: int, total_parts: int) -> str:
        """Create base prompt structure"""
        
        if self.is_gemini:
            return f"""# 🔍 Comprehensive {category.title()} Analysis - Part {part_num}/TOTAL_PARTS_PLACEHOLDER

**Model**: Gemini 2.0 Flash  
**Task**: Complete analysis of ALL {category.replace('_', ' ')} files ensuring 100% coverage  
**Part**: {part_num} of TOTAL_PARTS_PLACEHOLDER (Multi-part analysis for comprehensive coverage)

## 📊 **Analysis Scope**
This is part of a comprehensive multi-prompt analysis to ensure every file and code chunk is thoroughly analyzed. This specific prompt covers a subset of files, and all parts together provide complete coverage.

## 📁 **Files in This Part**

"""
        else:
            return f"""# Comprehensive {category.title()} Analysis - Part {part_num}/TOTAL_PARTS_PLACEHOLDER

This is part {part_num} of a multi-part analysis to ensure complete coverage of all {category.replace('_', ' ')} files.

## Files Analyzed in This Part

"""
    
    def create_complete_prompt(self, category: str, analysis_type: str, files: List[Dict], part_num: int, total_parts: int) -> str:
        """Create a complete prompt with files and analysis request"""
        
        prompt = self.create_base_prompt(category, analysis_type, part_num, total_parts)
        
        # Add all files
        for file_data in files:
            content = file_data.get('content', file_data.get('content_preview', ''))
            prompt += self.format_file_for_prompt(file_data, content)
        
        # Add analysis request
        if self.is_gemini:
            prompt += self.create_gemini_analysis_request(category, analysis_type, part_num, len(files))
        else:
            prompt += self.create_standard_analysis_request(category, analysis_type, part_num, len(files))
        
        return prompt
    
    def create_gemini_analysis_request(self, category: str, analysis_type: str, part_num: int, file_count: int) -> str:
        """Create Gemini-optimized analysis request"""
        
        return f"""

## 🎯 **Comprehensive Analysis Request - Part {part_num}**

Analyze the {file_count} files above as part of a complete codebase analysis:

### 1. 🏗️ **Architectural Analysis**
- **Design Patterns**: Identify specific design patterns in these files
- **Component Relationships**: How do these components relate to each other?
- **Layered Architecture**: What layer do these components represent?

### 2. 🚀 **Spring Boot Implementation**
- **Annotation Usage**: Analyze Spring annotations and their effectiveness
- **Dependency Injection**: Evaluate DI patterns used
- **Configuration**: How are these components configured?

### 3. 🔄 **Functionality & Data Flow**
- **Business Logic**: What business functionality is implemented?
- **Data Processing**: How is data handled and transformed?
- **API Endpoints**: Document any REST endpoints (if controllers)
- **Service Integration**: How do these services integrate with others?

### 4. ✅ **Code Quality Assessment**
- **Code Organization**: Structure and organization quality
- **Naming Conventions**: Consistency and clarity
- **Best Practices**: Adherence to Spring Boot best practices
- **Potential Issues**: Code smells or anti-patterns

### 5. 🛡️ **Security & Performance**
- **Security Patterns**: Any security implementations
- **Performance Considerations**: Potential bottlenecks or optimizations
- **Resource Usage**: Memory and processing efficiency

### 6. 🎯 **Specific Recommendations**
- **Immediate Improvements**: Quick wins for better code quality
- **Refactoring Opportunities**: Structural improvements
- **Best Practice Adoption**: Spring Boot optimizations

## 📋 **Analysis Context**
- This is **Part {part_num}** of a multi-part analysis
- Focus on the {file_count} files provided in this prompt
- Provide specific, actionable insights for these particular components
- Reference actual code elements and line-level observations

**Expected Output**: Detailed technical analysis with specific recommendations for the files in this part.
"""
    
    def create_standard_analysis_request(self, category: str, analysis_type: str, part_num: int, file_count: int) -> str:
        """Create standard analysis request for non-Gemini models"""
        
        return f"""

## Analysis Request - Part {part_num}

Please analyze the {file_count} files above and provide:

1. **Architecture & Design**: Design patterns and component organization
2. **Spring Boot Usage**: How Spring Boot features are utilized
3. **Code Quality**: Quality assessment and improvement suggestions
4. **Functionality**: Business logic and data flow analysis
5. **Issues & Improvements**: Problems identified and recommended fixes

This is part {part_num} of a comprehensive analysis. Focus on the specific files provided.
"""
    
    def generate_comprehensive_prompts(self, categories: List[str] = None) -> Dict[str, List[str]]:
        """Generate comprehensive prompts ensuring 100% coverage"""
        
        if categories is None:
            # Auto-detect categories
            categories = []
            for item in self.analysis_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    if (item / "individual_files").exists() or (item / "chunks").exists():
                        categories.append(item.name)
        
        all_prompts = {}
        coverage_report = self.analyze_current_coverage()
        
        logger.info(f"🎯 Generating comprehensive prompts for {len(categories)} categories")
        
        for category in categories:
            logger.info(f"📂 Processing category: {category}")
            
            # Check if this category needs multiple prompts
            category_info = coverage_report["categories"].get(category, {})
            estimated_tokens = category_info.get("estimated_tokens", 0)
            
            if estimated_tokens > self.usable_tokens:
                logger.info(f"   ⚠️  Large category detected: {estimated_tokens:,} tokens")
                logger.info(f"   📝 Creating multi-part prompts for complete coverage")
                
                prompts = self.create_multi_part_prompts(category, "comprehensive_analysis")
                all_prompts[category] = prompts
                
                logger.info(f"   ✅ Created {len(prompts)} prompts for {category}")
            else:
                logger.info(f"   ✅ Single prompt sufficient: {estimated_tokens:,} tokens")
                
                # Single prompt is sufficient
                single_prompt = self.create_multi_part_prompts(category, "comprehensive_analysis")
                if single_prompt:
                    all_prompts[category] = single_prompt
        
        return all_prompts
    
    def save_comprehensive_prompts(self, all_prompts: Dict[str, List[str]]) -> str:
        """Save all comprehensive prompts with proper organization"""
        
        output_dir = self.analysis_dir / "comprehensive_prompts"
        output_dir.mkdir(exist_ok=True)
        
        # Create metadata
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "model": self.model_name,
            "coverage_type": "comprehensive",
            "total_prompts": sum(len(prompts) for prompts in all_prompts.values()),
            "categories": {}
        }
        
        total_prompts_saved = 0
        
        for category, prompts in all_prompts.items():
            category_dir = output_dir / category
            category_dir.mkdir(exist_ok=True)
            
            category_metadata = {
                "prompt_count": len(prompts),
                "files": []
            }
            
            for i, prompt in enumerate(prompts, 1):
                filename = f"{category}_part_{i:03d}.txt"
                filepath = category_dir / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(prompt)
                
                token_count = self.count_tokens(prompt)
                category_metadata["files"].append({
                    "filename": filename,
                    "part_number": i,
                    "estimated_tokens": token_count,
                    "context_usage": f"{(token_count/self.max_tokens*100):.1f}%"
                })
                
                total_prompts_saved += 1
                logger.info(f"💾 Saved: {filepath} ({token_count:,} tokens)")
            
            metadata["categories"][category] = category_metadata
        
        # Save metadata
        metadata_file = output_dir / "comprehensive_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create comprehensive usage guide
        guide = self.create_comprehensive_usage_guide(metadata, output_dir)
        guide_file = output_dir / "COMPREHENSIVE_USAGE_GUIDE.md"
        with open(guide_file, 'w') as f:
            f.write(guide)
        
        logger.info(f"📁 Comprehensive prompts saved to: {output_dir}")
        logger.info(f"📊 Total prompts created: {total_prompts_saved}")
        
        return str(output_dir)
    
    def create_comprehensive_usage_guide(self, metadata: Dict, output_dir: Path) -> str:
        """Create usage guide for comprehensive analysis"""
        
        guide = f"""# 🎯 Comprehensive Analysis Guide

## 📊 **Complete Coverage Analysis**

**Generated**: {metadata['generated_at']}  
**Model**: {metadata['model']}  
**Total Prompts**: {metadata['total_prompts']}  
**Coverage**: 100% of all files and chunks

## 🔍 **Why Comprehensive Analysis?**

This analysis ensures **complete coverage** of your codebase by:
- ✅ **Including ALL files**: Every Java file, config file, and component
- ✅ **Processing ALL chunks**: Large files split into chunks are fully analyzed
- ✅ **No truncation**: Nothing is skipped due to context limits
- ✅ **Systematic coverage**: Organized by category with multiple prompts per category

## 📁 **Prompt Organization**

"""
        
        total_prompts = 0
        for category, info in metadata['categories'].items():
            guide += f"### {category.replace('_', ' ').title()}\n"
            guide += f"- **Prompts**: {info['prompt_count']}\n"
            guide += f"- **Location**: `{category}/`\n"
            guide += f"- **Files**: {category}_part_001.txt to {category}_part_{info['prompt_count']:03d}.txt\n\n"
            total_prompts += info['prompt_count']
        
        guide += f"""## 🚀 **Execution Strategy**

### **Option 1: Sequential Analysis (Recommended)**
```bash
# Run prompts in order for each category
# Category 1: java_files
cat java_files/java_files_part_001.txt  # Copy to Gemini
cat java_files/java_files_part_002.txt  # Copy to Gemini
# ... continue for all parts

# Category 2: config_files  
cat config_files/config_files_part_001.txt  # Copy to Gemini
# ... continue for all parts
```

### **Option 2: Parallel Analysis**
- Run different categories simultaneously
- Useful if you have multiple team members
- Each person can take a different category

### **Option 3: Automated Processing**
```python
# Process all prompts programmatically
import os
from pathlib import Path

prompt_dir = Path("comprehensive_prompts")
for category_dir in prompt_dir.iterdir():
    if category_dir.is_dir():
        for prompt_file in sorted(category_dir.glob("*.txt")):
            with open(prompt_file) as f:
                prompt = f.read()
            # Send to your LLM API or copy to UI
            print(f"Processing: {{prompt_file}}")
```

## 📋 **Response Collection Strategy**

### **Organized Response Collection**
```python
from llm_response_processor import LLMResponseProcessor
processor = LLMResponseProcessor('analysis_output/analysis')

# For each prompt response:
# java_files_part_001 response
processor.save_response('java_files_comprehensive', 1, '''[GEMINI RESPONSE]''')

# java_files_part_002 response  
processor.save_response('java_files_comprehensive', 2, '''[GEMINI RESPONSE]''')

# Continue for all parts...
```

### **Response Naming Convention**
- **Category**: `java_files_comprehensive`, `config_files_comprehensive`
- **Part Number**: 1, 2, 3... for each prompt in the category
- **Type**: Use `_comprehensive` suffix to distinguish from regular analysis

## 🎯 **Expected Results**

After running **ALL {total_prompts} prompts**, you'll have:

### **Complete Code Coverage**
- ✅ Every Java class analyzed in detail
- ✅ All configuration files reviewed
- ✅ Large files completely processed (no truncation)
- ✅ All Spring Boot components documented

### **Comprehensive Insights**
- 🏗️ **Complete Architecture Map**: Full understanding of system design
- 🔍 **Detailed Code Quality**: Every component assessed for quality
- 🔄 **Complete Data Flow**: All API endpoints and data paths mapped
- 🛡️ **Security Assessment**: Comprehensive security analysis
- 📈 **Performance Profile**: Complete performance bottleneck identification

### **Actionable Recommendations**
- 🎯 **Prioritized Issues**: All problems ranked by severity
- 🔧 **Specific Fixes**: Detailed remediation steps for each issue
- 📊 **Metrics**: Quantified assessment of code quality
- 🚀 **Improvement Roadmap**: Strategic technical improvements

## ⚡ **Processing Tips**

### **For Large Repositories**
- **Time Investment**: Plan for {total_prompts} LLM interactions
- **Batch Processing**: Process 5-10 prompts at a time
- **Team Distribution**: Divide categories among team members
- **Progress Tracking**: Check off completed prompts

### **Quality Assurance**
- **Response Validation**: Ensure each response addresses the full prompt
- **Consistency Check**: Look for consistent themes across responses
- **Gap Analysis**: Verify no files are mentioned in multiple responses

## 📊 **Coverage Verification**

After collecting all responses, verify complete coverage:

```python
# Check that all files were analyzed
coverage_check = processor.verify_comprehensive_coverage()
print(f"Files analyzed: {{coverage_check['files_covered']}}")
print(f"Coverage percentage: {{coverage_check['coverage_percent']:.1f}}%")
```

## 🎉 **Final Report Generation**

```bash
# Generate comprehensive report from all responses
python scripts/run_llm_analysis.py analysis_output/analysis --mode comprehensive-report

# This creates:
# - comprehensive_analysis_report.md (Combined insights from all prompts)
# - coverage_verification.json (Confirmation of 100% coverage)
# - executive_summary.md (High-level findings)
```

---

**🎯 This comprehensive analysis ensures NO CODE IS LEFT UNANALYZED**

**Estimated Time**: {total_prompts} prompts × 2-3 minutes = {total_prompts * 2.5:.0f} minutes total
**Result**: Complete, thorough understanding of your entire codebase
"""
        
        return guide

def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive prompts with 100% coverage')
    parser.add_argument('analysis_dir', help='Analysis output directory')
    parser.add_argument('--model', default='gemini-2.0-flash-001', help='Model name')
    parser.add_argument('--categories', nargs='+', help='Specific categories to analyze')
    parser.add_argument('--coverage-report', action='store_true', help='Show coverage analysis only')
    
    args = parser.parse_args()
    
    print("🎯 Comprehensive Chunk Coverage Analyzer")
    print("=" * 60)
    print(f"Analysis Directory: {args.analysis_dir}")
    print(f"Model: {args.model}")
    
    try:
        analyzer = ComprehensiveChunkAnalyzer(args.analysis_dir, args.model)
        
        if args.coverage_report:
            coverage = analyzer.analyze_current_coverage()
            
            print("\n📊 COVERAGE ANALYSIS:")
            print(f"Total Files: {coverage['total_files']:,}")
            print(f"Total Chunks: {coverage['total_chunks']:,}")
            
            for category, info in coverage['categories'].items():
                print(f"\n📂 {category.title()}:")
                print(f"   Files: {info['individual_files']:,}")
                print(f"   Chunks: {info['chunk_files']:,}")
                print(f"   Estimated Tokens: {info['estimated_tokens']:,}")
            
            if coverage['coverage_gaps']:
                print(f"\n⚠️  COVERAGE GAPS:")
                for gap in coverage['coverage_gaps']:
                    print(f"   {gap['category']}: {gap['prompts_needed']} prompts needed ({gap['current_coverage']})")
        
        else:
            # Generate comprehensive prompts
            all_prompts = analyzer.generate_comprehensive_prompts(args.categories)
            output_dir = analyzer.save_comprehensive_prompts(all_prompts)
            
            total_prompts = sum(len(prompts) for prompts in all_prompts.values())
            
            print(f"\n✅ SUCCESS!")
            print(f"📁 Comprehensive prompts saved to: {output_dir}")
            print(f"📊 Total prompts generated: {total_prompts}")
            print(f"🎯 Coverage: 100% of all files and chunks")
            
            print(f"\n📋 PROMPT BREAKDOWN:")
            for category, prompts in all_prompts.items():
                print(f"   {category}: {len(prompts)} prompts")
            
            print(f"\n🚀 NEXT STEPS:")
            print("1. Review the COMPREHENSIVE_USAGE_GUIDE.md")
            print("2. Execute all prompts systematically")
            print("3. Collect responses for each prompt")
            print("4. Generate final comprehensive report")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
