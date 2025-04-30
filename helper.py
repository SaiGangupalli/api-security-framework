# Jira Security & Fraud Analysis System

A comprehensive solution for analyzing Jira user stories to identify potential security and fraud impacts using machine learning and natural language processing.

## Project Overview

This system extracts user stories from Jira, preprocesses the data, and analyzes them for potential security and fraud impacts using a continuously trained machine learning model. The system provides detailed reporting, visualization, and automated email notifications.

## Key Features

- **Jira Integration**: Extract user stories from specified Jira projects using Bearer token authentication
- **Machine Learning Analysis**: Identify security and fraud impacts using Random Forest models
- **Keyword Matching**: Find security and fraud indicators using advanced word matching
- **Detailed Excel Reports**: Generate professionally formatted Excel outputs with charts
- **Email Reporting**: Send beautiful HTML email reports with analysis results
- **Program Impact Analysis**: Identify which test programs have the most security and fraud impacts
- **Continuous Learning**: Model improves over time through feedback and retraining

## Project Structure

```
project/
├── config/               # Configuration settings
├── data/                 # Data storage
│   ├── processed/        # Processed datasets
│   └── raw/              # Raw data from Jira
├── models/               # ML model code and storage
│   └── trained_models/   # Saved model files
├── src/                  # Source code
│   ├── data_processing/  # Data extraction and preprocessing
│   ├── model_training/   # ML training code
│   └── utils/            # Helper utilities
├── logs/                 # Log files
├── templates/            # Email templates
├── main.py               # Main application script
└── requirements.txt      # Python dependencies
```

## Setup and Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the root directory with your configuration:
   ```
   # Jira API Configuration
   JIRA_URL=https://your-jira-instance.atlassian.net
   JIRA_API_TOKEN=your_bearer_token
   
   # Email Configuration
   SMTP_SERVER=smtp.gmail.com
   SMTP_PORT=587
   EMAIL_USERNAME=your_email@gmail.com
   EMAIL_PASSWORD=your_app_password
   SENDER_EMAIL=your_email@gmail.com
   DEFAULT_RECIPIENTS=recipient1@example.com,recipient2@example.com
   ```

5. Prepare NLTK data (if needed):
   ```
   python nltk_setup.py --create-minimal ./nltk_minimal_data
   ```

## Usage

### Basic Usage

Run the complete pipeline:

```bash
python main.py
```

This will extract data from Jira, preprocess it, train the model, analyze the user stories, and generate reports.

### Specific Operations

Extract user stories from specific projects:
```bash
python main.py --extract --projects PROJ1 PROJ2
```

Use a text file containing project keys:
```bash
python main.py --extract --project-file projects.txt
```

Analyze existing data:
```bash
python main.py --analyze --input path/to/data.xlsx
```

Train or update the model:
```bash
python main.py --train --input path/to/data.xlsx
```

Train with feedback:
```bash
python main.py --train --input path/to/data.xlsx --feedback path/to/feedback.xlsx
```

Make predictions on existing data:
```bash
python main.py --predict --input path/to/data.xlsx
```

### Email Reporting

Send email report with analysis results:
```bash
python main.py --predict --input path/to/data.xlsx --email-report
```

Specify custom recipients:
```bash
python main.py --predict --input path/to/data.xlsx --email-report --email-recipients person1@example.com person2@example.com
```

Customize email subject:
```bash
python main.py --predict --input path/to/data.xlsx --email-report --email-subject "Security Analysis for Project XYZ"
```

### Excel Customization

Control which columns appear in the Excel output:
```bash
python main.py --predict --input data.xlsx --exclude-columns combined_text initial_security_flag --hide-columns security_keyword_count fraud_keyword_count
```

Load Excel configuration from a file:
```bash
python main.py --predict --input data.xlsx --excel-config excel_config.json
```

## Key Components

### 1. Jira Connector

The `JiraConnector` connects to Jira using Bearer token authentication to extract user stories. It supports:
- Extraction from specific projects
- Pagination for large projects
- Project list filtering

### 2. Text Preprocessing

The system employs advanced text preprocessing to prepare data for analysis:
- HTML and markup removal
- Special character and stopword removal
- Lemmatization and tokenization
- Similar word matching for keywords

### 3. Machine Learning Models

Two Random Forest classifiers are used:
- Security impact detection model
- Fraud impact detection model

The models use TF-IDF vectorization and additional metadata features to make predictions.

### 4. Excel Reporting

The system generates professional Excel reports with:
- Formatted data tables
- Auto-filtering for easy data exploration
- Color coding for impact labels
- Summary statistics
- Pivot tables for program-level analysis

### 5. Email Reporting

Automated email reports include:
- Overall summary statistics
- Program impact pivot table (top 20 programs)
- Model performance metrics
- Confusion matrices visualization
- Top security and fraud stories
- Excel file attachments

### 6. Pivot Table Analysis

The system analyzes which programs have the most security and fraud impacts:
- Counts security and fraud impacts by program
- Distinguishes between security-only, fraud-only, and both impacts
- Calculates total impacts correctly (no double-counting)
- Sorts programs by total impact count

## Continuous Learning Flow

The system improves over time through this feedback loop:

1. **Initial Prediction**:
   - Model makes predictions on user stories
   - Excel output shows predictions and matching keywords

2. **User Feedback**:
   - Expert reviews predictions in feedback template
   - Marks correct/incorrect predictions
   - Saves as feedback.xlsx

3. **Model Retraining**:
   - System combines original data with feedback
   - Trains new model using the corrected labels
   - Updates keyword lists based on feature importance

4. **Improved Predictions**:
   - Each feedback cycle makes the model more accurate
   - Keyword lists grow to include newly identified important terms

## Output Files

The system generates several output files:

- `jira_user_stories.xlsx`: Raw data extracted from Jira
- `processed_user_stories.xlsx`: Preprocessed data ready for analysis
- `security_fraud_predictions_[timestamp].xlsx`: User stories with security and fraud predictions
- `feedback_template_[timestamp].xlsx`: Template for collecting expert feedback
- `program_impact_analysis.xlsx`: Pivot table analysis of program impacts
- Visualizations in `reports/figures/`: Model performance and feature importance charts

## Extending the System

You can extend this system by:

1. Adding new feature extraction methods in `feature_engineering.py`
2. Implementing additional ML algorithms in `ml_model.py`
3. Creating new preprocessing steps in `excel_processor.py`
4. Enhancing the keyword lists in `settings.py`
5. Customizing the email template in `templates/email_template.html`

## Requirements

- Python 3.8+
- Jira API access
- SMTP server access for email reporting
- See `requirements.txt` for Python package dependencies
