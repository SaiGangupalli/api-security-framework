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
    def __init__(self, repo_url: str, output_dir: str = "analysis_output", 
                 max_file_size_mb: int = 1, batch_size: int = 50, 
                 max_chunk_size: int = 2000):
        self.repo_url = repo_url
        self.output_dir = Path(output_dir)
        self.repo_dir = self.output_dir / "cloned_repo"
        self.analysis_dir = self.output_dir / "analysis"
        
        # Memory optimization settings
        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert MB to bytes
        self.batch_size = batch_size  # Process files in batches
        self.max_chunk_size = max_chunk_size  # Smaller chunks to reduce memory
        self.max_files_in_memory = 20  # Limit files held in memory
        
        # File extensions to analyze for Java Spring Boot projects
        self.java_extensions = {'.java', '.xml', '.properties', '.yml', '.yaml'}
        self.config_files = {'pom.xml', 'build.gradle', 'application.properties', 
                           'application.yml', 'application.yaml'}
        
        logger.info(f"Memory settings: max_file_size={max_file_size_mb}MB, "
                   f"batch_size={batch_size}, max_chunk_size={max_chunk_size}")
        
    def setup_directories(self):
        """Create necessary directories for analysis"""
        directories = [
            self.output_dir,
            self.repo_dir,
            self.analysis_dir,
            self.analysis_dir / "java_files",
            self.analysis_dir / "config_files", 
            self.analysis_dir / "chunks",
            self.analysis_dir / "metadata",
            self.analysis_dir / "test_files",
            self.analysis_dir / "resource_files"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        # Create placeholder files to ensure directories are visible
        placeholder_files = [
            self.analysis_dir / "java_files" / ".gitkeep",
            self.analysis_dir / "config_files" / ".gitkeep",
            self.analysis_dir / "chunks" / ".gitkeep",
            self.analysis_dir / "metadata" / ".gitkeep"
        ]
        
        for placeholder in placeholder_files:
            placeholder.write_text("# Placeholder file to maintain directory structure\n")
    
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
        """Extract content and metadata from a file with memory optimization"""
        try:
            # Check file size first
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                logger.warning(f"Skipping large file {file_path}: {file_size/1024/1024:.1f}MB")
                return None
            
            # Read file in chunks for large files
            content = ""
            if file_size > 100 * 1024:  # 100KB threshold for chunked reading
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    while True:
                        chunk = f.read(8192)  # Read 8KB chunks
                        if not chunk:
                            break
                        content += chunk
                        # Prevent excessive memory usage
                        if len(content) > self.max_file_size:
                            logger.warning(f"File {file_path} too large, truncating")
                            break
            else:
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
        except MemoryError:
            logger.error(f"Memory error reading file {file_path}")
            return None
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")
            return None
    
    def chunk_content(self, content: str, chunk_size: int = None, overlap: int = 100) -> List[str]:
        """Split content into overlapping chunks with memory optimization"""
        if chunk_size is None:
            chunk_size = self.max_chunk_size
            
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
            
            chunk = content[start:end]
            chunks.append(chunk)
            
            if end >= len(content):
                break
                
            start = end - overlap
            
            # Memory protection: if too many chunks, break
            if len(chunks) > 100:  # Limit chunks per file
                logger.warning("Too many chunks for file, truncating")
                break
        
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
        """Process files in batches with memory optimization and proper file creation"""
        logger.info("Processing files for analysis with memory optimization...")
        
        all_categories_summary = []
        
        # Process each category separately to manage memory
        for category, files in file_categories.items():
            if not files:
                logger.info(f"No files found for category: {category}")
                # Create empty file for the category
                empty_file = self.analysis_dir / f"{category}.json"
                with open(empty_file, 'w', encoding='utf-8') as f:
                    json.dump([], f, indent=2)
                logger.info(f"Created empty file: {empty_file}")
                continue
                
            logger.info(f"Processing {len(files)} {category}...")
            
            # Process files in batches
            category_file_path = self.analysis_dir / f"{category}.json"
            chunks_dir = self.analysis_dir / "chunks" / category
            chunks_dir.mkdir(exist_ok=True)
            
            # Initialize the JSON file with proper array structure
            all_category_data = []
            processed_count = 0
            chunk_file_count = 0
            
            for i in range(0, len(files), self.batch_size):
                batch = files[i:i + self.batch_size]
                batch_num = i//self.batch_size + 1
                total_batches = (len(files) + self.batch_size - 1)//self.batch_size
                logger.info(f"Processing {category} batch {batch_num}/{total_batches} ({len(batch)} files)")
                
                batch_data = []
                
                for file_path in batch:
                    try:
                        file_content = self.extract_file_content(file_path)
                        if file_content is None:
                            continue
                        
                        # Add Java-specific analysis for Java files
                        if file_path.suffix == '.java':
                            file_content['java_analysis'] = self.analyze_java_structure(file_content)
                        
                        # Handle chunking and save chunks separately
                        if file_content['size'] > 1000:  # Threshold for chunking
                            chunks = self.chunk_content(file_content['content'])
                            file_content['chunk_count'] = len(chunks)
                            
                            # Save chunks to separate files
                            if chunks:
                                chunk_file_count += 1
                                chunk_file = chunks_dir / f"file_{chunk_file_count:04d}_chunks.json"
                                chunk_data = {
                                    'source_file': file_content['file_path'],
                                    'chunk_count': len(chunks),
                                    'chunks': chunks
                                }
                                with open(chunk_file, 'w', encoding='utf-8') as f:
                                    json.dump(chunk_data, f, indent=2, ensure_ascii=False)
                                
                                # Reference chunk file instead of storing content
                                file_content['chunks_file'] = str(chunk_file.relative_to(self.analysis_dir))
                                del file_content['content']  # Remove large content
                            else:
                                file_content['chunks'] = []
                        
                        batch_data.append(file_content)
                        processed_count += 1
                        
                    except MemoryError:
                        logger.error(f"Memory error processing {file_path}, skipping")
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing {file_path}: {e}")
                        continue
                
                # Add batch data to category collection
                all_category_data.extend(batch_data)
                
                # Clear batch data from memory
                del batch_data
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Log progress
                if processed_count > 0 and processed_count % 50 == 0:
                    logger.info(f"  Processed {processed_count}/{len(files)} files in {category}")
            
            # Save complete category data
            with open(category_file_path, 'w', encoding='utf-8') as f:
                json.dump(all_category_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Saved {processed_count} files to {category_file_path}")
            logger.info(f"✅ Created {chunk_file_count} chunk files in {chunks_dir}")
            
            # Add to summary
            all_categories_summary.append({
                'category': category,
                'file_count': processed_count,
                'chunk_files': chunk_file_count,
                'output_file': str(category_file_path.relative_to(self.analysis_dir))
            })
            
            # Clear category data from memory
            del all_category_data
            import gc
            gc.collect()
        
        # Create combined summary (lightweight)
        self.create_combined_summary(all_categories_summary)
        
        return all_categories_summary
    
    def create_combined_summary(self, categories_summary: List[Dict[str, Any]]):
        """Create a combined summary file"""
        combined_summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'repository_url': self.repo_url,
            'categories': categories_summary,
            'total_files': sum(cat['file_count'] for cat in categories_summary),
            'total_chunk_files': sum(cat['chunk_files'] for cat in categories_summary),
            'output_structure': {
                'main_files': [cat['output_file'] for cat in categories_summary],
                'chunk_directories': ['chunks/' + cat['category'] for cat in categories_summary if cat['chunk_files'] > 0],
                'metadata_files': ['summary_report.json', 'combined_summary.json']
            }
        }
        
        combined_file = self.analysis_dir / "combined_summary.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(combined_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Created combined summary: {combined_file}")
        return combined_summary
    
    def create_summary_report(self, summary_data: List[Dict[str, Any]]):
        """Create a memory-efficient summary report"""
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'repository_url': self.repo_url,
            'total_files_analyzed': sum(item['file_count'] for item in summary_data),
            'categories': summary_data,
            'memory_optimized': True,
            'settings': {
                'max_file_size_mb': self.max_file_size / (1024 * 1024),
                'batch_size': self.batch_size,
                'max_chunk_size': self.max_chunk_size
            },
            'output_files': {}
        }
        
        # Document all output files that were created
        for category_info in summary_data:
            category = category_info['category']
            main_file = self.analysis_dir / f"{category}.json"
            chunks_dir = self.analysis_dir / "chunks" / category
            
            summary['output_files'][category] = {
                'main_file': f"{category}.json",
                'file_count': category_info['file_count'],
                'chunk_files': category_info['chunk_files'],
                'chunks_directory': f"chunks/{category}" if category_info['chunk_files'] > 0 else None,
                'exists': main_file.exists()
            }
        
        # Add basic statistics from existing JSON files
        try:
            # Read a sample of java files to get Spring Boot info
            java_file = self.analysis_dir / "java_files.json"
            if java_file.exists() and java_file.stat().st_size > 0:
                spring_features = {'controllers': 0, 'services': 0, 'repositories': 0, 'components': 0}
                packages = set()
                
                # Parse safely to avoid memory issues
                try:
                    with open(java_file, 'r', encoding='utf-8') as f:
                        java_data = json.load(f)
                        
                    for file_info in java_data[:50]:  # Sample first 50 files
                        if 'java_analysis' in file_info:
                            java_analysis = file_info['java_analysis']
                            
                            if java_analysis.get('package'):
                                packages.add(java_analysis['package'])
                            
                            for annotation in java_analysis.get('spring_annotations', []):
                                if 'Controller' in annotation:
                                    spring_features['controllers'] += 1
                                elif 'Service' in annotation:
                                    spring_features['services'] += 1
                                elif 'Repository' in annotation:
                                    spring_features['repositories'] += 1
                                elif 'Component' in annotation:
                                    spring_features['components'] += 1
                    
                    summary['spring_boot_features'] = spring_features
                    summary['sample_packages'] = list(packages)[:20]  # Limit to 20 packages
                    
                except (json.JSONDecodeError, MemoryError) as e:
                    logger.warning(f"Could not parse java_files.json for Spring analysis: {e}")
                    summary['spring_boot_features'] = {'note': 'Analysis failed due to file size/format'}
            else:
                summary['spring_boot_features'] = {'note': 'No Java files found'}
        
        except Exception as e:
            logger.warning(f"Could not analyze Spring Boot features: {e}")
            summary['spring_boot_features'] = {'note': 'Analysis skipped due to memory constraints'}
        
        # Save summary report
        summary_file = self.analysis_dir / "summary_report.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Memory-optimized summary report saved to {summary_file}")
        
        # Also create a simple file listing for easy reference
        file_listing = self.analysis_dir / "file_listing.txt"
        with open(file_listing, 'w', encoding='utf-8') as f:
            f.write("Git Repository Analysis - Generated Files\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Main Analysis Files:\n")
            for category_info in summary_data:
                category = category_info['category']
                f.write(f"  📄 {category}.json ({category_info['file_count']} files)\n")
                if category_info['chunk_files'] > 0:
                    f.write(f"     └── chunks/{category}/ ({category_info['chunk_files']} chunk files)\n")
            
            f.write(f"\nSummary Files:\n")
            f.write(f"  📊 summary_report.json\n")
            f.write(f"  📋 combined_summary.json\n")
            f.write(f"  📝 file_listing.txt\n")
            
            f.write(f"\nTotal Files Analyzed: {summary['total_files_analyzed']}\n")
            f.write(f"Analysis Date: {summary['analysis_timestamp']}\n")
        
        logger.info(f"✅ Created file listing: {file_listing}")
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
            summary_data = self.process_files(file_categories)
            logger.info("✅ File processing completed")
            
            # Step 6: Create summary
            step = "summary generation"
            logger.info(f"STEP 6: {step}")
            summary = self.create_summary_report(summary_data)
            logger.info("✅ Summary report generated")
            
            total_files = sum(item['file_count'] for item in summary_data)
            
            logger.info("=" * 60)
            logger.info("🎉 Analysis completed successfully!")
            logger.info(f"📁 Results saved in: {self.analysis_dir}")
            logger.info(f"📊 Total files analyzed: {total_files}")
            logger.info("=" * 60)
            
            return {
                'success': True,
                'output_directory': str(self.analysis_dir),
                'files_processed': total_files,
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
    
    # Memory optimization options
    parser.add_argument('--max-file-size', type=int, default=1, help='Maximum file size in MB (default: 1)')
    parser.add_argument('--batch-size', type=int, default=50, help='Files to process per batch (default: 50)')
    parser.add_argument('--max-chunk-size', type=int, default=2000, help='Maximum chunk size for large files (default: 2000)')
    
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
    
    print(f"🚀 Starting analysis with memory optimization...")
    print(f"📍 Repository: {args.repo_url}")
    print(f"🌿 Branch: {args.branch}")
    print(f"📁 Output: {args.output}")
    print(f"💾 Memory settings: max_file_size={args.max_file_size}MB, batch_size={args.batch_size}, chunk_size={args.max_chunk_size}")
    print("-" * 50)
    
    analyzer = GitRepoAnalyzer(
        args.repo_url, 
        args.output,
        max_file_size_mb=args.max_file_size,
        batch_size=args.batch_size,
        max_chunk_size=args.max_chunk_size
    )
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
        print("   1. Reduce memory usage: --max-file-size 0.5 --batch-size 25")
        print("   2. Use smaller chunks: --max-chunk-size 1000")
        print("   3. Check available system memory")
        print("   4. Try with --debug flag for more details")
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
