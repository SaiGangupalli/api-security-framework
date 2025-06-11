#!/usr/bin/env python3
"""
Git Repository Analyzer for Java Spring Boot Projects
Fetches code from GitLab and prepares it for Gen AI analysis
"""

import os
import git
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any
import argparse
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GitRepoAnalyzer:
    def __init__(self, repo_url: str, output_dir: str = "analysis_output"):
        self.repo_url = repo_url
        self.output_dir = Path(output_dir)
        self.repo_dir = self.output_dir / "cloned_repo"
        self.analysis_dir = self.output_dir / "analysis"
        
        # File extensions to analyze for Java Spring Boot projects
        self.java_extensions = {'.java', '.xml', '.properties', '.yml', '.yaml'}
        self.config_files = {'pom.xml', 'build.gradle', 'application.properties', 
                           'application.yml', 'application.yaml'}
        
        # Max file size for processing (1MB)
        self.max_file_size = 1024 * 1024
        
    def setup_directories(self):
        """Create necessary directories for analysis"""
        directories = [
            self.output_dir,
            self.repo_dir,
            self.analysis_dir,
            self.analysis_dir / "java_files",
            self.analysis_dir / "config_files",
            self.analysis_dir / "chunks",
            self.analysis_dir / "metadata"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def clone_repository(self, branch: str = "main"):
        """Clone the GitLab repository with detailed error reporting"""
        try:
            if self.repo_dir.exists():
                logger.info(f"Removing existing repository directory: {self.repo_dir}")
                shutil.rmtree(self.repo_dir)
                logger.info("✅ Existing directory removed")
            
            logger.info(f"🔄 Cloning repository: {self.repo_url}")
            logger.info(f"🔄 Target branch: {branch}")
            logger.info(f"🔄 Target directory: {self.repo_dir}")
            
            # Try cloning with more detailed error handling
            try:
                repo = git.Repo.clone_from(
                    self.repo_url, 
                    self.repo_dir, 
                    branch=branch,
                    progress=lambda op_code, cur_count, max_count, message: 
                        logger.info(f"Cloning progress: {message or 'Processing...'}")
                )
                logger.info("✅ Repository cloned successfully")
                
                # Verify the clone
                if not self.repo_dir.exists():
                    raise Exception("Clone appeared successful but directory doesn't exist")
                    
                # Check if it's a valid git repository
                if not (self.repo_dir / '.git').exists():
                    raise Exception("Cloned directory is not a valid git repository")
                    
                # Count files in repository
                total_items = len(list(self.repo_dir.rglob('*')))
                logger.info(f"✅ Repository contains {total_items} items")
                
            except git.exc.GitCommandError as git_error:
                logger.error(f"Git command failed: {git_error}")
                logger.error("Common causes:")
                logger.error("1. Repository URL is incorrect")
                logger.error("2. Branch doesn't exist")
                logger.error("3. No access permissions")
                logger.error("4. Network connectivity issues")
                logger.error("5. Git not installed or not in PATH")
                
                # Try to get more specific error info
                if "Authentication failed" in str(git_error):
                    logger.error("🔐 Authentication issue - check your Git credentials")
                elif "not found" in str(git_error).lower():
                    logger.error("🔍 Repository or branch not found")
                elif "permission denied" in str(git_error).lower():
                    logger.error("🚫 Permission denied - check repository access")
                
                raise git_error
                
        except Exception as e:
            logger.error(f"❌ Repository cloning failed: {e}")
            logger.error(f"Repository URL: {self.repo_url}")
            logger.error(f"Branch: {branch}")
            raise
    
    def scan_repository(self) -> Dict[str, List[Path]]:
        """Scan repository and categorize files"""
        file_categories = {
            'java_files': [],
            'config_files': [],
            'test_files': [],
            'resource_files': [],
            'other_files': []
        }
        
        logger.info("Scanning repository for relevant files...")
        
        for file_path in self.repo_dir.rglob('*'):
            if file_path.is_file() and file_path.stat().st_size <= self.max_file_size:
                relative_path = file_path.relative_to(self.repo_dir)
                
                # Skip hidden files and directories
                if any(part.startswith('.') for part in relative_path.parts):
                    continue
                
                # Skip build directories
                if any(part in ['target', 'build', 'node_modules'] for part in relative_path.parts):
                    continue
                
                # Categorize files
                if file_path.suffix == '.java':
                    if 'test' in str(file_path).lower():
                        file_categories['test_files'].append(file_path)
                    else:
                        file_categories['java_files'].append(file_path)
                elif file_path.name in self.config_files or file_path.suffix in {'.xml', '.properties', '.yml', '.yaml'}:
                    file_categories['config_files'].append(file_path)
                elif file_path.suffix in {'.sql', '.json', '.txt', '.md'}:
                    file_categories['resource_files'].append(file_path)
                else:
                    file_categories['other_files'].append(file_path)
        
        # Log statistics
        for category, files in file_categories.items():
            logger.info(f"Found {len(files)} {category}")
        
        return file_categories
    
    def extract_file_content(self, file_path: Path) -> Dict[str, Any]:
        """Extract content and metadata from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            relative_path = file_path.relative_to(self.repo_dir)
            
            return {
                'file_path': str(relative_path),
                'full_path': str(file_path),
                'content': content,
                'size': len(content),
                'lines': len(content.splitlines()),
                'extension': file_path.suffix,
                'is_test': 'test' in str(file_path).lower(),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")
            return None
    
    def chunk_content(self, content: str, chunk_size: int = 4000, overlap: int = 200) -> List[str]:
        """Split content into overlapping chunks for Gen AI processing"""
        if len(content) <= chunk_size:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            
            # Try to break at natural boundaries (line breaks)
            if end < len(content):
                # Look for the last newline before the end
                last_newline = content.rfind('\n', start, end)
                if last_newline != -1 and last_newline > start:
                    end = last_newline + 1
            
            chunks.append(content[start:end])
            
            if end >= len(content):
                break
                
            start = end - overlap
        
        return chunks
    
    def analyze_java_structure(self, file_content: Dict[str, Any]) -> Dict[str, Any]:
        """Basic analysis of Java file structure"""
        content = file_content['content']
        
        # Extract basic Java elements
        analysis = {
            'package': None,
            'imports': [],
            'classes': [],
            'interfaces': [],
            'annotations': [],
            'methods': [],
            'spring_annotations': []
        }
        
        lines = content.splitlines()
        
        for line in lines:
            line = line.strip()
            
            # Package declaration
            if line.startswith('package '):
                analysis['package'] = line.replace('package ', '').replace(';', '').strip()
            
            # Imports
            elif line.startswith('import '):
                import_stmt = line.replace('import ', '').replace(';', '').strip()
                analysis['imports'].append(import_stmt)
            
            # Class declarations
            elif 'class ' in line and ('public' in line or 'private' in line or 'protected' in line):
                analysis['classes'].append(line)
            
            # Interface declarations
            elif 'interface ' in line:
                analysis['interfaces'].append(line)
            
            # Spring annotations
            elif line.startswith('@') and any(spring_ann in line for spring_ann in 
                ['Controller', 'Service', 'Repository', 'Component', 'RestController', 
                 'Autowired', 'RequestMapping', 'GetMapping', 'PostMapping']):
                analysis['spring_annotations'].append(line)
            
            # General annotations
            elif line.startswith('@'):
                analysis['annotations'].append(line)
        
        return analysis
    
    def process_files(self, file_categories: Dict[str, List[Path]]):
        """Process all files and create analysis outputs"""
        logger.info("Processing files for analysis...")
        
        all_files_data = []
        
        for category, files in file_categories.items():
            category_data = []
            
            for file_path in files:
                file_content = self.extract_file_content(file_path)
                if file_content is None:
                    continue
                
                # Add Java-specific analysis for Java files
                if file_path.suffix == '.java':
                    file_content['java_analysis'] = self.analyze_java_structure(file_content)
                
                # Create chunks for large files
                if file_content['size'] > 2000:
                    chunks = self.chunk_content(file_content['content'])
                    file_content['chunks'] = chunks
                    file_content['chunk_count'] = len(chunks)
                
                category_data.append(file_content)
                all_files_data.append(file_content)
            
            # Save category-specific data
            if category_data:
                output_file = self.analysis_dir / f"{category}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(category_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {len(category_data)} files to {output_file}")
        
        # Save combined analysis
        combined_file = self.analysis_dir / "combined_analysis.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(all_files_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved combined analysis with {len(all_files_data)} files")
        
        return all_files_data
    
    def create_summary_report(self, all_files_data: List[Dict[str, Any]]):
        """Create a summary report of the analysis"""
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'repository_url': self.repo_url,
            'total_files_analyzed': len(all_files_data),
            'file_statistics': {},
            'spring_boot_features': {
                'controllers': 0,
                'services': 0,
                'repositories': 0,
                'components': 0,
                'configuration_files': 0
            },
            'packages': set(),
            'dependencies': set()
        }
        
        # Gather statistics
        for file_data in all_files_data:
            extension = file_data.get('extension', 'unknown')
            summary['file_statistics'][extension] = summary['file_statistics'].get(extension, 0) + 1
            
            # Analyze Java files for Spring Boot features
            if 'java_analysis' in file_data:
                java_analysis = file_data['java_analysis']
                
                if java_analysis['package']:
                    summary['packages'].add(java_analysis['package'])
                
                for annotation in java_analysis['spring_annotations']:
                    if 'Controller' in annotation:
                        summary['spring_boot_features']['controllers'] += 1
                    elif 'Service' in annotation:
                        summary['spring_boot_features']['services'] += 1
                    elif 'Repository' in annotation:
                        summary['spring_boot_features']['repositories'] += 1
                    elif 'Component' in annotation:
                        summary['spring_boot_features']['components'] += 1
                
                for import_stmt in java_analysis['imports']:
                    if 'springframework' in import_stmt:
                        summary['dependencies'].add(import_stmt)
        
        # Convert sets to lists for JSON serialization
        summary['packages'] = list(summary['packages'])
        summary['dependencies'] = list(summary['dependencies'])
        
        # Save summary report
        summary_file = self.analysis_dir / "summary_report.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summary report saved to {summary_file}")
        return summary
    
    def run_analysis(self, repo_branch: str = "main"):
        """Run the complete analysis pipeline with detailed error reporting"""
        step = "initialization"
        try:
            logger.info("=" * 60)
            logger.info("Starting Git repository analysis...")
            logger.info(f"Repository URL: {self.repo_url}")
            logger.info(f"Branch: {repo_branch}")
            logger.info(f"Output directory: {self.output_dir}")
            logger.info("=" * 60)
            
            # Step 1: Setup directories
            step = "directory setup"
            logger.info(f"STEP 1: {step}")
            self.setup_directories()
            logger.info("✅ Directories created successfully")
            
            # Step 2: Clone repository
            step = "repository cloning"
            logger.info(f"STEP 2: {step}")
            self.clone_repository(repo_branch)
            logger.info("✅ Repository cloned successfully")
            
            # Step 3: Verify repository contents
            step = "repository verification"
            logger.info(f"STEP 3: {step}")
            if not self.repo_dir.exists():
                raise Exception(f"Repository directory not found: {self.repo_dir}")
            
            repo_files = list(self.repo_dir.rglob('*'))
            logger.info(f"Repository contains {len(repo_files)} total items")
            
            # Step 4: Scan and categorize files
            step = "file scanning"
            logger.info(f"STEP 4: {step}")
            file_categories = self.scan_repository()
            
            total_files = sum(len(files) for files in file_categories.values())
            if total_files == 0:
                logger.warning("⚠️  No files found to analyze!")
                logger.info("Repository structure:")
                for item in list(self.repo_dir.iterdir())[:10]:  # Show first 10 items
                    logger.info(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
            
            logger.info("✅ File scanning completed")
            
            # Step 5: Process files
            step = "file processing"
            logger.info(f"STEP 5: {step}")
            all_files_data = self.process_files(file_categories)
            logger.info("✅ File processing completed")
            
            # Step 6: Create summary
            step = "summary generation"
            logger.info(f"STEP 6: {step}")
            summary = self.create_summary_report(all_files_data)
            logger.info("✅ Summary report generated")
            
            logger.info("=" * 60)
            logger.info("🎉 Analysis completed successfully!")
            logger.info(f"📁 Results saved in: {self.analysis_dir}")
            logger.info(f"📊 Total files analyzed: {len(all_files_data)}")
            logger.info("=" * 60)
            
            return {
                'success': True,
                'output_directory': str(self.analysis_dir),
                'files_processed': len(all_files_data),
                'summary': summary
            }
            
        except Exception as e:
            logger.error("=" * 60)
            logger.error(f"❌ Analysis failed at step: {step}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            
            # Additional debugging info
            if step == "repository cloning":
                logger.error(f"Repository URL: {self.repo_url}")
                logger.error(f"Target directory: {self.repo_dir}")
                logger.error("Please check:")
                logger.error("1. Repository URL is correct and accessible")
                logger.error("2. You have proper Git credentials configured")
                logger.error("3. The branch name exists")
                
            elif step == "file scanning":
                if self.repo_dir.exists():
                    logger.error("Repository was cloned but no analyzable files found")
                    logger.error("Repository contents:")
                    try:
                        for item in list(self.repo_dir.iterdir())[:20]:
                            logger.error(f"  - {item.name}")
                    except:
                        logger.error("Could not list repository contents")
            
            # Print full traceback for debugging
            import traceback
            logger.error("Full traceback:")
            logger.error(traceback.format_exc())
            logger.error("=" * 60)
            
            return {
                'success': False,
                'error': str(e),
                'step_failed': step,
                'error_type': type(e).__name__
            }

def main():
    parser = argparse.ArgumentParser(description='Analyze Git repository for Gen AI processing')
    parser.add_argument('repo_url', help='GitLab repository URL')
    parser.add_argument('--branch', default='main', help='Git branch to analyze (default: main)')
    parser.add_argument('--output', default='analysis_output', help='Output directory (default: analysis_output)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--test-git', action='store_true', help='Test Git installation')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Test Git installation if requested
    if args.test_git:
        test_git_installation()
        return
    
    # Validate repository URL
    if not args.repo_url.strip():
        print("❌ Repository URL cannot be empty")
        exit(1)
    
    if not (args.repo_url.startswith('http://') or args.repo_url.startswith('https://') or args.repo_url.startswith('git@')):
        print("❌ Repository URL should start with http://, https://, or git@")
        print(f"Provided URL: {args.repo_url}")
        exit(1)
    
    print(f"🚀 Starting analysis...")
    print(f"📍 Repository: {args.repo_url}")
    print(f"🌿 Branch: {args.branch}")
    print(f"📁 Output: {args.output}")
    print("-" * 50)
    
    analyzer = GitRepoAnalyzer(args.repo_url, args.output)
    result = analyzer.run_analysis(args.branch)
    
    print("\n" + "=" * 60)
    if result['success']:
        print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"📁 Output directory: {result['output_directory']}")
        print(f"📊 Files processed: {result['files_processed']}")
        print("\n📋 Generated files:")
        output_dir = Path(result['output_directory'])
        if output_dir.exists():
            for file in output_dir.glob('*.json'):
                size = file.stat().st_size
                print(f"   - {file.name} ({size:,} bytes)")
    else:
        print("❌ ANALYSIS FAILED!")
        print(f"💥 Error: {result['error']}")
        print(f"📍 Failed at step: {result.get('step_failed', 'unknown')}")
        print(f"🔧 Error type: {result.get('error_type', 'unknown')}")
        print("\n🔍 Troubleshooting tips:")
        print("   1. Check if Git is installed: git --version")
        print("   2. Verify repository URL and access permissions")
        print("   3. Try with --debug flag for more details")
        print("   4. Use --test-git to test Git installation")
        exit(1)

def test_git_installation():
    """Test if Git is properly installed and configured"""
    print("🧪 Testing Git installation...")
    
    try:
        import git
        print("✅ GitPython library is installed")
        
        # Test Git executable
        git_version = git.cmd.Git().version()
        print(f"✅ Git version: {git_version}")
        
        # Test basic Git functionality
        print("✅ Git is working properly")
        
        # Check Git configuration
        try:
            git_config = git.cmd.Git()
            user_name = git_config.config('--get', 'user.name')
            user_email = git_config.config('--get', 'user.email')
            print(f"✅ Git user: {user_name} <{user_email}>")
        except:
            print("⚠️  Git user not configured (might cause issues with some repositories)")
            print("   Configure with: git config --global user.name 'Your Name'")
            print("   Configure with: git config --global user.email 'your.email@example.com'")
        
    except ImportError:
        print("❌ GitPython library not installed")
        print("   Install with: pip install GitPython")
    except Exception as e:
        print(f"❌ Git test failed: {e}")
        print("   Make sure Git is installed and in your PATH")
    
    print("\n🔧 If you're having issues:")
    print("   1. Install Git: https://git-scm.com/downloads")
    print("   2. Install GitPython: pip install GitPython")
    print("   3. Configure Git credentials for private repositories")

if __name__ == "__main__":
    main()









# Git Repository Analysis Project Structure

## Recommended Folder Structure

```
git-repo-analyzer/
├── README.md
├── requirements.txt
├── config/
│   ├── analysis_config.json
│   └── file_patterns.json
├── src/
│   ├── __init__.py
│   ├── git_analyzer.py          # Main analysis script
│   ├── file_processor.py        # File processing utilities
│   ├── chunking_strategies.py   # Content chunking for Gen AI
│   └── utils/
│       ├── __init__.py
│       ├── file_utils.py
│       └── logging_config.py
├── analysis_output/             # Generated during analysis
│   ├── cloned_repo/            # Cloned repository
│   └── analysis/               # Analysis results
│       ├── java_files.json
│       ├── config_files.json
│       ├── test_files.json
│       ├── chunks/
│       ├── metadata/
│       └── summary_report.json
├── scripts/
│   ├── run_analysis.py         # Entry point script
│   └── batch_analysis.py       # For multiple repositories
└── tests/
    ├── __init__.py
    ├── test_analyzer.py
    └── sample_data/
```

## Installation and Setup

### 1. Create Virtual Environment

```bash
# Create project directory
mkdir git-repo-analyzer
cd git-repo-analyzer

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

Create `requirements.txt`:

```txt
GitPython==3.1.40
requests==2.31.0
python-dotenv==1.0.0
tiktoken==0.5.2
beautifulsoup4==4.12.2
langchain==0.1.0
langchain-community==0.0.10
python-magic==0.4.27
chardet==5.2.0
pathspec==0.12.1
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Configuration Files

#### `config/analysis_config.json`
```json
{
  "chunk_settings": {
    "default_chunk_size": 4000,
    "overlap_size": 200,
    "min_chunk_size": 100
  },
  "file_settings": {
    "max_file_size_mb": 1,
    "excluded_directories": [
      "target",
      "build",
      "node_modules",
      ".git",
      ".idea",
      ".vscode"
    ],
    "included_extensions": [
      ".java",
      ".xml",
      ".properties",
      ".yml",
      ".yaml",
      ".sql",
      ".json"
    ]
  },
  "spring_boot_patterns": {
    "controller_annotations": [
      "@Controller",
      "@RestController"
    ],
    "service_annotations": [
      "@Service",
      "@Component"
    ],
    "repository_annotations": [
      "@Repository"
    ],
    "configuration_files": [
      "application.properties",
      "application.yml",
      "application.yaml",
      "pom.xml",
      "build.gradle"
    ]
  },
  "gen_ai_settings": {
    "context_window_size": 8000,
    "preserve_code_structure": true,
    "include_comments": true,
    "extract_documentation": true
  }
}
```

#### `config/file_patterns.json`
```json
{
  "java_patterns": {
    "class_pattern": "^\\s*(public|private|protected)?\\s*class\\s+\\w+",
    "interface_pattern": "^\\s*(public|private|protected)?\\s*interface\\s+\\w+",
    "method_pattern": "^\\s*(public|private|protected)\\s+.*\\s+\\w+\\s*\\(",
    "annotation_pattern": "^\\s*@\\w+"
  },
  "spring_patterns": {
    "rest_endpoint": "@(GetMapping|PostMapping|PutMapping|DeleteMapping|RequestMapping)",
    "dependency_injection": "@(Autowired|Inject|Resource)",
    "configuration": "@(Configuration|EnableAutoConfiguration|ComponentScan)"
  }
}
```

## Usage Examples

### Basic Usage

```bash
# Clone and analyze a repository
python git_analyzer.py https://gitlab.com/your-org/your-spring-boot-repo.git

# Specify branch and output directory
python git_analyzer.py https://gitlab.com/your-org/repo.git --branch develop --output my_analysis
```

### Advanced Usage with Configuration

```python
from git_analyzer import GitRepoAnalyzer

# Initialize analyzer with custom settings
analyzer = GitRepoAnalyzer(
    repo_url="https://gitlab.com/your-org/your-repo.git",
    output_dir="analysis_results"
)

# Run analysis
result = analyzer.run_analysis(repo_branch="main")

if result['success']:
    print(f"Analysis completed: {result['files_processed']} files processed")
    print(f"Results saved to: {result['output_directory']}")
else:
    print(f"Analysis failed: {result['error']}")
```

## Key Features

### 1. **Repository Cloning**
- Clones GitLab repositories
- Supports different branches
- Handles authentication (configure Git credentials)

### 2. **File Categorization**
- Java source files
- Configuration files (XML, Properties, YAML)
- Test files
- Resource files

### 3. **Content Chunking**
- Splits large files for Gen AI processing
- Configurable chunk sizes with overlap
- Preserves code structure boundaries

### 4. **Spring Boot Analysis**
- Identifies Spring annotations
- Extracts REST endpoints
- Analyzes dependency injection patterns
- Configuration file parsing

### 5. **Gen AI Preparation**
- JSON output format
- Metadata extraction
- Context preservation
- Structured data for LLM consumption

## Output Files

### `java_files.json`
Contains all Java source files with:
- File content and metadata
- Package and import analysis
- Spring annotations detection
- Method and class extraction

### `config_files.json`
Configuration files including:
- application.properties/yml
- pom.xml or build.gradle
- Other configuration files

### `summary_report.json`
High-level analysis including:
- File statistics
- Spring Boot component counts
- Package structure
- Dependency analysis

## Authentication for Private Repositories

### Using Git Credentials
```bash
# Set up Git credentials
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# For GitLab, use personal access token
git config --global credential.helper store
```

### Using Environment Variables
Create a `.env` file:
```bash
GITLAB_TOKEN=your_personal_access_token
GITLAB_USERNAME=your_username
```

## Troubleshooting

### Common Issues

1. **Repository Access Denied**
   - Ensure you have proper GitLab access tokens
   - Check repository URL format

2. **Large File Processing**
   - Adjust `max_file_size` in configuration
   - Files larger than 1MB are skipped by default

3. **Memory Issues**
   - Reduce chunk sizes for large repositories
   - Process files in batches

### Logging

The script provides detailed logging. To increase verbosity:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Next Steps for Gen AI Integration

After running the analysis, you can:

1. **Feed to LLM**: Use the JSON outputs directly with your Gen AI model
2. **Vector Database**: Index the chunks for similarity search
3. **Code Understanding**: Use the structured data for code comprehension tasks
4. **Documentation**: Generate documentation from the analyzed code
5. **Code Review**: Identify patterns and potential improvements
