#!/usr/bin/env python3
"""
GitLab Repository Fetcher and Processor for Spring Boot Analysis
Updated for clean folder structure without numbers
"""

import os
import json
import subprocess
import shutil
from pathlib import Path
import gitlab
from dotenv import load_dotenv
from tqdm import tqdm
import chardet
import tiktoken

class GitLabRepoFetcher:
    def __init__(self, config_path="config/analysis-config.json"):
        """Initialize the GitLab fetcher with configuration"""
        load_dotenv("config/.env")
        
        self.config = self.load_config(config_path)
        self.gitlab_url = os.getenv('GITLAB_URL')
        self.access_token = os.getenv('GITLAB_TOKEN')
        self.project_id = os.getenv('PROJECT_ID')
        
        if not all([self.gitlab_url, self.access_token, self.project_id]):
            raise ValueError("Missing required GitLab credentials in .env file")
        
        # Initialize GitLab client
        self.gl = gitlab.Gitlab(self.gitlab_url, private_token=self.access_token)
        self.project = self.gl.projects.get(self.project_id)
        
        # Setup clean paths
        self.raw_repo_path = Path("source-code/raw-repository")
        self.processed_path = Path("source-code/processed-files")
        self.chunks_path = Path("analysis-output/code-chunks")
        self.output_path = Path("analysis-output")
        
        # File extensions to analyze
        self.java_extensions = {'.java', '.xml', '.yml', '.yaml', '.properties', '.gradle', '.md'}
        
        # Initialize token encoder for chunking
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Create default config
            default_config = {
                "max_file_size_mb": 5,
                "chunk_size_tokens": 3000,
                "overlap_tokens": 200,
                "exclude_patterns": [
                    "*.class", "*.jar", "*.war", "target/*", 
                    ".git/*", "node_modules/*", "*.log"
                ]
            }
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config

    def clone_repository(self):
        """Clone the GitLab repository"""
        print("🔄 Cloning repository...")
        
        # Clean existing directory
        if self.raw_repo_path.exists():
            shutil.rmtree(self.raw_repo_path)
        
        # Ensure parent directory exists
        self.raw_repo_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Clone repository
        clone_url = f"{self.gitlab_url}/{self.project.path_with_namespace}.git"
        
        try:
            subprocess.run([
                'git', 'clone', 
                f"https://oauth2:{self.access_token}@{clone_url.replace('https://', '')}",
                str(self.raw_repo_path)
            ], check=True, capture_output=True, text=True)
            print(f"✅ Repository cloned to {self.raw_repo_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error cloning repository: {e.stderr}")
            raise

    def detect_file_encoding(self, file_path):
        """Detect file encoding"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except:
            return 'utf-8'

    def is_text_file(self, file_path):
        """Check if file is a text file"""
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(8192)
                return b'\x00' not in chunk
        except:
            return False

    def filter_relevant_files(self):
        """Filter and process relevant Spring Boot files"""
        print("🔍 Filtering relevant files...")
        
        relevant_files = []
        
        for root, dirs, files in os.walk(self.raw_repo_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(
                d.startswith(pattern.rstrip('/*')) 
                for pattern in self.config['exclude_patterns']
                if '/' in pattern
            )]
            
            for file in files:
                file_path = Path(root) / file
                
                # Check file extension
                if file_path.suffix not in self.java_extensions:
                    continue
                
                # Check file size
                if file_path.stat().st_size > self.config['max_file_size_mb'] * 1024 * 1024:
                    continue
                
                # Check if it's a text file
                if not self.is_text_file(file_path):
                    continue
                
                relevant_files.append(file_path)
        
        print(f"✅ Found {len(relevant_files)} relevant files")
        return relevant_files

    def copy_and_clean_files(self, relevant_files):
        """Copy relevant files to processed directory with cleaning"""
        print("🧹 Cleaning and copying files...")
        
        # Clean processed directory
        if self.processed_path.exists():
            shutil.rmtree(self.processed_path)
        self.processed_path.mkdir(parents=True)
        
        processed_files = []
        
        for file_path in tqdm(relevant_files, desc="Processing files"):
            try:
                # Maintain directory structure
                relative_path = file_path.relative_to(self.raw_repo_path)
                dest_path = self.processed_path / relative_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Read with proper encoding
                encoding = self.detect_file_encoding(file_path)
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read()
                
                # Basic cleaning
                content = content.replace('\r\n', '\n')  # Normalize line endings
                content = '\n'.join(line.rstrip() for line in content.split('\n'))  # Remove trailing spaces
                
                # Write cleaned content
                with open(dest_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                processed_files.append({
                    'original_path': str(file_path),
                    'processed_path': str(dest_path),
                    'relative_path': str(relative_path),
                    'size_bytes': len(content.encode('utf-8')),
                    'lines': len(content.split('\n'))
                })
                
            except Exception as e:
                print(f"⚠️ Error processing {file_path}: {e}")
        
        return processed_files

    def categorize_and_chunk_files(self, processed_files):
        """Categorize files by type and create chunks"""
        print("📂 Categorizing and chunking files...")
        
        # Create category directories
        categories = {
            'controllers': self.chunks_path / 'controllers',
            'services': self.chunks_path / 'services', 
            'repositories': self.chunks_path / 'repositories',
            'entities': self.chunks_path / 'entities',
            'configurations': self.chunks_path / 'configurations',
            'other': self.chunks_path / 'other'
        }
        
        for category_path in categories.values():
            category_path.mkdir(parents=True, exist_ok=True)
        
        chunk_metadata = []
        
        for file_info in tqdm(processed_files, desc="Categorizing and chunking"):
            try:
                # Determine category based on file path and content
                category = self.determine_file_category(file_info)
                
                with open(file_info['processed_path'], 'r', encoding='utf-8') as f:
                    content = f.read()
                
                chunks = self.chunk_content(content)
                
                for i, chunk in enumerate(chunks):
                    safe_filename = Path(file_info['relative_path']).stem.replace(' ', '_')
                    chunk_filename = f"{safe_filename}_chunk_{i+1:03d}.txt"
                    chunk_path = categories[category] / chunk_filename
                    
                    with open(chunk_path, 'w', encoding='utf-8') as f:
                        f.write(f"# File: {file_info['relative_path']}\n")
                        f.write(f"# Category: {category}\n")
                        f.write(f"# Chunk: {i+1}/{len(chunks)}\n")
                        f.write("# " + "="*60 + "\n\n")
                        f.write(chunk)
                    
                    chunk_metadata.append({
                        'original_file': file_info['relative_path'],
                        'category': category,
                        'chunk_file': str(chunk_path),
                        'chunk_number': i + 1,
                        'total_chunks': len(chunks),
                        'tokens': len(self.encoding.encode(chunk))
                    })
            
            except Exception as e:
                print(f"⚠️ Error chunking {file_info['relative_path']}: {e}")
        
        # Save chunk metadata
        with open(self.chunks_path / 'chunk_metadata.json', 'w') as f:
            json.dump(chunk_metadata, f, indent=2)
        
        return chunk_metadata

    def determine_file_category(self, file_info):
        """Determine file category based on path and naming conventions"""
        file_path = file_info['relative_path'].lower()
        
        if any(keyword in file_path for keyword in ['controller', 'rest', 'endpoint']):
            return 'controllers'
        elif any(keyword in file_path for keyword in ['service', 'business']):
            return 'services'
        elif any(keyword in file_path for keyword in ['repository', 'dao', 'data']):
            return 'repositories'
        elif any(keyword in file_path for keyword in ['entity', 'model', 'domain']):
            return 'entities'
        elif any(keyword in file_path for keyword in ['config', 'configuration', 'application']):
            return 'configurations'
        else:
            return 'other'

    def chunk_content(self, content, max_tokens=None):
        """Split content into chunks for AI processing"""
        if max_tokens is None:
            max_tokens = self.config['chunk_size_tokens']
        
        tokens = self.encoding.encode(content)
        
        if len(tokens) <= max_tokens:
            return [content]
        
        chunks = []
        overlap = self.config['overlap_tokens']
        
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            if end >= len(tokens):
                break
            
            start = end - overlap
        
        return chunks

    def generate_analysis_summary(self, processed_files, chunk_metadata):
        """Generate summary of the analysis"""
        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Count categories
        category_counts = {}
        for chunk in chunk_metadata:
            category = chunk['category']
            category_counts[category] = category_counts.get(category, 0) + 1
        
        summary = {
            'repository_info': {
                'name': self.project.name,
                'description': self.project.description,
                'url': self.project.web_url,
                'last_activity': self.project.last_activity_at
            },
            'processing_summary': {
                'total_files_processed': len(processed_files),
                'total_chunks_created': len(chunk_metadata),
                'category_distribution': category_counts,
                'file_types': {}
            },
            'files': processed_files,
            'chunks': chunk_metadata
        }
        
        # Count file types
        for file_info in processed_files:
            ext = Path(file_info['relative_path']).suffix
            summary['processing_summary']['file_types'][ext] = \
                summary['processing_summary']['file_types'].get(ext, 0) + 1
        
        # Save summary
        with open(self.output_path / 'processing-summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary

    def run_full_process(self):
        """Run the complete fetching and processing pipeline"""
        print("🚀 Starting GitLab repository analysis...")
        
        try:
            # Step 1: Clone repository
            self.clone_repository()
            
            # Step 2: Filter relevant files
            relevant_files = self.filter_relevant_files()
            
            # Step 3: Copy and clean files
            processed_files = self.copy_and_clean_files(relevant_files)
            
            # Step 4: Categorize and create chunks
            chunk_metadata = self.categorize_and_chunk_files(processed_files)
            
            # Step 5: Generate summary
            summary = self.generate_analysis_summary(processed_files, chunk_metadata)
            
            print("✅ Repository processing completed successfully!")
            print(f"📊 Processed {len(processed_files)} files")
            print(f"📦 Created {len(chunk_metadata)} chunks")
            print(f"📁 Output saved to analysis-output/")
            
            return summary
            
        except Exception as e:
            print(f"❌ Error during processing: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    fetcher = GitLabRepoFetcher()
    summary = fetcher.run_full_process()
