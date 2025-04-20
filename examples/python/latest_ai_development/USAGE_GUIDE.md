# Latest AI Development with CrewAI and Cognee: Usage Guide

This guide explains how to set up and run the Latest AI Development project that integrates CrewAI with Cognee memory tools.

## Prerequisites

Before running the application, make sure you have:

1. Python 3.10+ installed
2. Required dependencies installed
3. API keys configured

## Installation

### 1. Clone the Repository

If you haven't already cloned the repository, do so:

```bash
git clone <repository-url>
cd cognee/examples/python/latest_ai_development
```

### 2. Set Up a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -e .  # Install the package in development mode
```

Or install the specific dependencies:

```bash
pip install crewai[tools] cognee>=0.1.34 s3fs>=2025.3.2
```

### 4. Configure Environment Variables

Create or update a `.env` file in the root directory with the following:

```
MODEL=gpt-4o
OPENAI_API_KEY=your-openai-api-key
```

For proper Cognee functionality, you might need to set up Cognee-specific variables. The current setup uses the OPENAI_API_KEY for Cognee operations as well.

## Running the Application

There are several ways to run the application:

### 1. Using the CrewAI CLI

```bash
crewai run
```

This command runs the crew defined in `main.py` using the default settings.

### 2. Using Python Directly

```bash
python -m latest_ai_development.main
```

Or:

```bash
python test_cognee_tools.py  # To test just the Cognee tools integration
```

## Known Issues and Troubleshooting

### UserNotFoundError

If you encounter a `UserNotFoundError: Failed to retrieve default user: default_user@example.com (Status code: 404)` error, this indicates that the Cognee service is not properly configured with user authentication.

This error occurs because Cognee expects a user account to be set up, but the default user does not exist. Despite this error, the CrewAI workflow will continue to run, but the Cognee memory operations will fail.

To resolve this issue, you would need to:
1. Set up a proper user account within the Cognee service
2. Update the authentication configuration

## Project Structure

- `src/latest_ai_development/tools/`: Contains the Cognee tool integrations
- `src/latest_ai_development/crew.py`: Defines the agents and their tools
- `src/latest_ai_development/config/`: Contains YAML configuration files for agents and tasks
- `test_cognee_tools.py`: A test script to verify Cognee tool functionality

## Next Steps

After running the application, check the generated `report.md` file for the output from the Reporting Analyst agent.

## Example Commands

Here are some common commands you might use:

```bash
# Run the crew
crewai run

# Test the Cognee tools
python test_cognee_tools.py

# Train the crew (requires additional parameters)
crewai train <iterations> <filename>

# Replay a specific task
crewai replay <task-id>
```
