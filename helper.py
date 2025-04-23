# src/data_processing/jira_connector.py - Updated with project list filtering

def extract_user_stories(self, project_keys=None, max_results=None):
    """
    Extract user stories from specified projects.
    
    Args:
        project_keys (list): List of project keys to extract stories from.
                            If None, extract from all accessible projects.
        max_results (int): Maximum number of results to fetch per project.
                          If None, fetch all available user stories.
    
    Returns:
        pandas.DataFrame: DataFrame containing user stories
    """
    if not project_keys:
        projects = self.get_all_projects()
        project_keys = [project['key'] for project in projects]
        logger.info(f"No specific projects provided. Will extract from all {len(project_keys)} projects.")
    else:
        logger.info(f"Extracting user stories from {len(project_keys)} specified projects: {', '.join(project_keys)}")
    
    all_stories = []
    
    for project_key in project_keys:
        logger.info(f"Extracting user stories from project: {project_key}")
        
        # JQL query to fetch user stories
        jql_query = f'project = {project_key} AND issuetype = "Story"'
        
        try:
            # Process stories in batches to handle large projects
            start_at = 0
            batch_size = 100
            total = None
            
            while total is None or start_at < total:
                # Build request with JQL query
                payload = {
                    "jql": jql_query,
                    "startAt": start_at,
                    "maxResults": batch_size,
                    "fields": [
                        "summary", 
                        "description", 
                        "created", 
                        "status", 
                        "assignee", 
                        "reporter", 
                        "labels", 
                        "priority", 
                        "components"
                    ]
                }
                
                response = self.session.post(
                    f"{self.base_url}/rest/api/2/search",
                    data=json.dumps(payload)
                )
                response.raise_for_status()
                
                result = response.json()
                issues = result.get('issues', [])
                
                if not issues:
                    break
                
                # Update total for pagination
                if total is None:
                    total = result.get('total', 0)
                
                for issue in issues:
                    fields = issue.get('fields', {})
                    story_data = {
                        'project_key': project_key,
                        'issue_key': issue.get('key'),
                        'summary': fields.get('summary', ''),
                        'description': fields.get('description', '') or "",
                        'status': fields.get('status', {}).get('name', '') if fields.get('status') else "",
                        'created_date': fields.get('created', ''),
                        'assignee': fields.get('assignee', {}).get('displayName', '') if fields.get('assignee') else "Unassigned",
                        'reporter': fields.get('reporter', {}).get('displayName', '') if fields.get('reporter') else "Unknown",
                        'priority': fields.get('priority', {}).get('name', '') if fields.get('priority') else "None",
                        'labels': ", ".join(fields.get('labels', [])) if fields.get('labels') else "",
                        'components': ", ".join([c.get('name', '') for c in fields.get('components', [])]) if fields.get('components') else ""
                    }
                    all_stories.append(story_data)
                
                start_at += len(issues)
                
                if max_results and start_at >= max_results:
                    break
                
            logger.info(f"Extracted {start_at} user stories from project {project_key}")
            
        except Exception as e:
            logger.error(f"Error fetching user stories from project {project_key}: {e}")
            continue
    
    if not all_stories:
        logger.warning("No user stories were found.")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_stories)
    logger.info(f"Total user stories extracted: {len(df)}")
    
    return df





# Updated parse_arguments function in main.py to add a project list file option

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Jira User Story Security and Fraud Analysis')
    
    parser.add_argument('--extract', action='store_true', 
                        help='Extract user stories from Jira')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze existing data without extraction')
    parser.add_argument('--train', action='store_true',
                        help='Train or update the ML model')
    parser.add_argument('--predict', action='store_true',
                        help='Make predictions on existing data')
    parser.add_argument('--projects', nargs='+', type=str,
                        help='List of Jira project keys to extract (comma-separated or space-separated)')
    parser.add_argument('--project-file', type=str,
                        help='Path to file containing list of Jira project keys (one per line)')
    parser.add_argument('--input', type=str,
                        help='Input file path (for analyze, train, or predict modes)')
    parser.add_argument('--output', type=str,
                        help='Output file path (optional)')
    parser.add_argument('--feedback', type=str,
                        help='Path to feedback file for model updating')
    parser.add_argument('--nltk-data', type=str, default='C:/nltk_data',
                        help='Path to NLTK data directory (default: C:/nltk_data)')
    
    return parser.parse_args()








# Add this function to main.py

def read_project_keys_from_file(file_path):
    """
    Read Jira project keys from a file.
    
    Args:
        file_path (str): Path to the file containing project keys
        
    Returns:
        list: List of project keys
    """
    try:
        with open(file_path, 'r') as f:
            # Read lines and strip whitespace, filter out empty lines
            project_keys = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Successfully read {len(project_keys)} project keys from {file_path}")
        return project_keys
    except Exception as e:
        logger.error(f"Failed to read project keys from {file_path}: {e}")
        return []







# Updated extract_jira_data function in main.py

def extract_jira_data(project_keys=None, project_file=None, output_path=None):
    """
    Extract user story data from Jira.
    
    Args:
        project_keys (list): List of project keys to extract stories from
        project_file (str): Path to file containing project keys
        output_path (str): Path to save the extracted data
        
    Returns:
        str: Path to the extracted data file
    """
    logger.info("Starting Jira data extraction")
    
    # If a project file is provided, read project keys from it
    if project_file:
        file_project_keys = read_project_keys_from_file(project_file)
        if file_project_keys:
            # If project_keys is also provided, merge the lists
            if project_keys:
                logger.info(f"Merging {len(project_keys)} command line projects with {len(file_project_keys)} file projects")
                project_keys = list(set(project_keys + file_project_keys))
            else:
                project_keys = file_project_keys
    
    # Check if we have comma-separated project keys as a single string
    if project_keys and len(project_keys) == 1 and ',' in project_keys[0]:
        project_keys = [key.strip() for key in project_keys[0].split(',')]
    
    # Log the final project list
    if project_keys:
        logger.info(f"Will extract data from {len(project_keys)} projects: {', '.join(project_keys)}")
    else:
        logger.info("No project keys specified. Will extract from all accessible projects.")
    
    jira = JiraConnector()
    
    # Extract user stories
    user_stories_df = jira.extract_user_stories(project_keys)
    
    if user_stories_df.empty:
        logger.warning("No user stories extracted from Jira")
        return None
    
    # Export to Excel
    export_path = jira.export_to_excel(user_stories_df, output_path)
    
    if not export_path:
        logger.error("Failed to export user stories to Excel")
        return None
    
    logger.info(f"Successfully extracted {len(user_stories_df)} user stories to {export_path}")
    return export_path













# Update this part in the main function of main.py

def main():
    """Main function to orchestrate the workflow."""
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    setup_directories()
    
    # Parse command line arguments
    args = parse_arguments()
    
    if not any([args.extract, args.analyze, args.train, args.predict]):
        logger.info("No action specified. Running complete workflow.")
        args.extract = args.analyze = args.train = args.predict = True
    
    # Extract data from Jira
    if args.extract:
        data_path = extract_jira_data(args.projects, args.project_file, args.output)
        if not data_path:
            logger.error("Data extraction failed. Exiting.")
            return
    else:
        data_path = args.input or EXCEL_EXPORT_PATH
    
    # Rest of the function remains the same...
    # ...
