import os
import requests
from pathlib import Path
import argparse

def read_markdown_file(file_path: str) -> str:
    """Read contents of a markdown file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def get_script_from_claude(wiki_content: str, issues_content: str, api_key: str) -> str:
    """Make request to Claude API with markdown contents"""
    
    # Claude API endpoint
    url = "https://api.anthropic.com/v1/messages"
    
    # Construct prompt with both documents
    prompt = f"""I need you to build me a Python script. Attached is `home.md` that provides a high-level system requirement:

{wiki_content}

Second is another document called `issues.md` that describes the list of requirements, features, and bugs related to this script:

{issues_content}

Consider the contents of these documents and provide me a Python script that fulfills the requirements."""

    # Make API request
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
        "X-API-Key": api_key,
    }
    
    data = {
        "model": "claude-3-7-sonnet-20250219",
        "max_tokens": 4000,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code != 200:
        print(f"Response content: {response.text}")  # Add detailed error reporting
        raise Exception(f"API request failed with status {response.status_code}")
        
    return response.json()["content"][0]["text"]

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate Python script from requirements using Claude API')
    parser.add_argument('--wiki', required=True, help='Path to the wiki markdown file')
    parser.add_argument('--issues', required=True, help='Path to the issues markdown file')
    parser.add_argument('--output', default='classify-and-move.py', help='Output path for the generated script (default: classify-and-move.py)')
    
    args = parser.parse_args()
    
    # Get API key from environment variable
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise Exception("CLAUDE_API_KEY environment variable not set")
    
    # Read markdown files
    wiki_content = read_markdown_file(args.wiki)
    issues_content = read_markdown_file(args.issues)
    
    # Get script from Claude
    script = get_script_from_claude(wiki_content, issues_content, api_key)
    
    # Write script to new file
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(script)
        
    print(f"Script written to {output_path}")

if __name__ == "__main__":
    main()


