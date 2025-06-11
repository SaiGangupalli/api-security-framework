import os
import git
import requests
from pathlib import Path
import json
import tiktoken
from typing import List, Dict, Tuple

class GitLabRepoAnalyzer:
    def __init__(self, gitlab_url: str, project_id: str, access_token: str):
        self.gitlab_url = gitlab_url.rstrip('/')
        self.project_id = project_id
        self.access_token = access_token
        self.headers = {'PRIVATE-TOKEN': access_token}
        self.local_repo_path = None
        
    def clone_repository(self, local_path: str = "./temp_repo"):
        """Clone the GitLab repository locally"""
        try:
            # Construct clone URL with token
            clone_url = f"https://oauth2:{self.access_token}@{self.gitlab_url.replace('https://', '')}/{self.project_id}.git"
            
            # Remove existing directory if it exists
            if os.path.exists(local_path):
                import shutil
                shutil.rmtree(local_path)
            
            # Clone the repository
            repo = git.Repo.clone_from(clone_url, local_path)
            self.local_repo_path = local_path
            print(f"Repository cloned successfully to {local_path}")
            return repo
            
        except Exception as e:
            print(f"Error cloning repository: {e}")
            return None
    
    def get_java_files(self) -> List[Dict]:
        """Get all Java files from the repository"""
        if not self.local_repo_path:
            print("Repository not cloned yet")
            return []
        
        java_files = []
        repo_path = Path(self.local_repo_path)
        
        for java_file in repo_path.rglob("*.java"):
            try:
                with open(java_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                relative_path = java_file.relative_to(repo_path)
                java_files.append({
                    'path': str(relative_path),
                    'full_path': str(java_file),
                    'content': content,
                    'size': len(content),
                    'lines': len(content.split('\n'))
                })
            except Exception as e:
                print(f"Error reading {java_file}: {e}")
        
        return java_files
    
    def get_config_files(self) -> List[Dict]:
        """Get configuration files"""
        if not self.local_repo_path:
            return []
        
        config_extensions = ['.yml', '.yaml', '.properties', '.xml']
        config_files = []
        repo_path = Path(self.local_repo_path)
        
        for ext in config_extensions:
            for config_file in repo_path.rglob(f"*{ext}"):
                if 'target' in str(config_file) or '.git' in str(config_file):
                    continue
                    
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    relative_path = config_file.relative_to(repo_path)
                    config_files.append({
                        'path': str(relative_path),
                        'content': content,
                        'type': ext
                    })
                except Exception as e:
                    print(f"Error reading {config_file}: {e}")
        
        return config_files

    def split_large_files(self, files: List[Dict], max_tokens: int = 3000) -> List[Dict]:
        """Split large files into smaller chunks for AI analysis"""
        encoding = tiktoken.get_encoding("cl100k_base")
        processed_files = []
        
        for file_info in files:
            content = file_info['content']
            tokens = encoding.encode(content)
            
            if len(tokens) <= max_tokens:
                processed_files.append(file_info)
            else:
                # Split into chunks
                lines = content.split('\n')
                current_chunk = []
                current_tokens = 0
                chunk_number = 1
                
                for line in lines:
                    line_tokens = len(encoding.encode(line))
                    
                    if current_tokens + line_tokens > max_tokens and current_chunk:
                        # Save current chunk
                        chunk_content = '\n'.join(current_chunk)
                        chunk_info = file_info.copy()
                        chunk_info['content'] = chunk_content
                        chunk_info['path'] = f"{file_info['path']}_chunk_{chunk_number}"
                        chunk_info['is_chunk'] = True
                        chunk_info['original_path'] = file_info['path']
                        processed_files.append(chunk_info)
                        
                        # Start new chunk
                        current_chunk = [line]
                        current_tokens = line_tokens
                        chunk_number += 1
                    else:
                        current_chunk.append(line)
                        current_tokens += line_tokens
                
                # Add remaining chunk
                if current_chunk:
                    chunk_content = '\n'.join(current_chunk)
                    chunk_info = file_info.copy()
                    chunk_info['content'] = chunk_content
                    chunk_info['path'] = f"{file_info['path']}_chunk_{chunk_number}"
                    chunk_info['is_chunk'] = True
                    chunk_info['original_path'] = file_info['path']
                    processed_files.append(chunk_info)
        
        return processed_files
    
    def save_analysis_data(self, output_dir: str = "./analysis_output"):
        """Save all extracted data for analysis"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all files
        java_files = self.get_java_files()
        config_files = self.get_config_files()
        
        # Split large files
        java_files_split = self.split_large_files(java_files)
        config_files_split = self.split_large_files(config_files)
        
        # Save to JSON files
        with open(f"{output_dir}/java_files.json", 'w', encoding='utf-8') as f:
            json.dump(java_files_split, f, indent=2, ensure_ascii=False)
        
        with open(f"{output_dir}/config_files.json", 'w', encoding='utf-8') as f:
            json.dump(config_files_split, f, indent=2, ensure_ascii=False)
        
        # Create summary
        summary = {
            'total_java_files': len(java_files),
            'total_config_files': len(config_files),
            'java_files_after_split': len(java_files_split),
            'config_files_after_split': len(config_files_split),
            'project_structure': self.get_project_structure()
        }
        
        with open(f"{output_dir}/summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Analysis data saved to {output_dir}")
        return summary
    
    def get_project_structure(self) -> Dict:
        """Analyze project structure"""
        if not self.local_repo_path:
            return {}
        
        structure = {
            'controllers': [],
            'services': [],
            'repositories': [],
            'models': [],
            'config': [],
            'dto': [],
            'utils': []
        }
        
        repo_path = Path(self.local_repo_path)
        
        for java_file in repo_path.rglob("*.java"):
            relative_path = str(java_file.relative_to(repo_path))
            
            if 'controller' in relative_path.lower():
                structure['controllers'].append(relative_path)
            elif 'service' in relative_path.lower():
                structure['services'].append(relative_path)
            elif 'repository' in relative_path.lower():
                structure['repositories'].append(relative_path)
            elif 'model' in relative_path.lower() or 'entity' in relative_path.lower():
                structure['models'].append(relative_path)
            elif 'config' in relative_path.lower():
                structure['config'].append(relative_path)
            elif 'dto' in relative_path.lower():
                structure['dto'].append(relative_path)
            elif 'util' in relative_path.lower():
                structure['utils'].append(relative_path)
        
        return structure

# Usage example
def main():
    # Configure your GitLab details
    GITLAB_URL = "https://gitlab.com"  # or your private GitLab instance
    PROJECT_ID = "your-project-id"     # can be numeric ID or namespace/project-name
    ACCESS_TOKEN = "your-access-token"  # GitLab personal access token
    
    # Initialize analyzer
    analyzer = GitLabRepoAnalyzer(GITLAB_URL, PROJECT_ID, ACCESS_TOKEN)
    
    # Clone repository
    repo = analyzer.clone_repository("./my_spring_boot_repo")
    
    if repo:
        # Save analysis data
        summary = analyzer.save_analysis_data("./analysis_output")
        print("Summary:", json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
