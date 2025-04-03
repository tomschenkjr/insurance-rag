import requests
import base64
import json
from typing import List, Dict
from pathlib import Path
import argparse
import subprocess
import os
import shutil

class GitHubWikiReader:
    def __init__(self, owner: str, repo: str):
        """
        Initialize the GitHub Wiki reader
        
        Args:
            owner (str): Repository owner/organization name
            repo (str): Repository name
        """
        self.owner = owner
        self.repo = repo
        self.wiki_url = f"https://github.com/{owner}/{repo}.wiki.git"
        self.api_url = f"https://api.github.com/repos/{owner}/{repo}"

    def clone_wiki(self, clone_dir: str = "wiki_clone") -> None:
        """
        Clone the wiki repository
        
        Args:
            clone_dir (str): Directory to clone the wiki into
        """
        # Create the clone directory if it doesn't exist
        Path(clone_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            # Clone the wiki repository
            subprocess.run(["git", "clone", self.wiki_url, clone_dir], check=True)
            print(f"Successfully cloned wiki to {clone_dir}")
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to clone wiki: {str(e)}")

    def get_wiki_pages(self, clone_dir: str = "wiki_clone") -> List[Dict]:
        """
        Get a list of all wiki pages in the repository
        
        Args:
            clone_dir (str): Directory where wiki is cloned
            
        Returns:
            List[Dict]: List of wiki pages with their metadata
        """
        wiki_dir = Path(clone_dir)
        if not wiki_dir.exists():
            self.clone_wiki(clone_dir)
        
        # Get all .md files in the wiki directory
        pages = []
        for md_file in wiki_dir.glob("*.md"):
            pages.append({
                "title": md_file.stem,
                "path": str(md_file)
            })
        
        return pages

    def get_wiki_content(self, page_path: str) -> str:
        """
        Get the content of a specific wiki page
        
        Args:
            page_path (str): Path to the wiki page file
            
        Returns:
            str: Content of the wiki page
        """
        try:
            with open(page_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Failed to read wiki content: {str(e)}")

    def save_wiki_to_markdown(self, page_name: str, page_path: str, output_path: str = "wiki.md") -> None:
        """
        Save the wiki content to a local markdown file
        
        Args:
            page_name (str): Name of the wiki page
            page_path (str): Path to the wiki page file
            output_path (str): Path where to save the markdown file
        """
        content = self.get_wiki_content(page_path)
        
        # Ensure the directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Write content to file
        with open(output_path, 'w', encoding='utf-8') as f:
            # Add title as header
            f.write(f"# {page_name}\n\n")
            f.write(content)
        
        print(f"Wiki content saved to {output_path}")

    def cleanup(self, clone_dir: str = "wiki_clone") -> None:
        """
        Clean up the cloned wiki directory
        
        Args:
            clone_dir (str): Directory to clean up
        """
        try:
            if Path(clone_dir).exists():
                shutil.rmtree(clone_dir)
                print(f"Cleaned up {clone_dir}")
        except Exception as e:
            print(f"Warning: Failed to clean up {clone_dir}: {str(e)}")

    def get_issues(self) -> List[Dict]:
        """
        Get all issues from the repository
        
        Returns:
            List[Dict]: List of issues with their metadata and content
        """
        issues_url = f"{self.api_url}/issues"
        issues = []
        page = 1
        
        while True:
            response = requests.get(
                issues_url,
                params={
                    "state": "all",
                    "per_page": 100,
                    "page": page,
                    "sort": "created",
                    "direction": "desc"
                }
            )
            
            if response.status_code != 200:
                error_msg = response.json().get('message', 'Unknown error')
                raise Exception(f"Failed to fetch issues: {response.status_code} - {error_msg}")
            
            page_issues = response.json()
            if not page_issues:
                break
                
            issues.extend(page_issues)
            page += 1
            
        return issues

    def save_issues_to_markdown(self, output_path: str = "issues.md") -> None:
        """
        Save all issues to a markdown file
        
        Args:
            output_path (str): Path where to save the issues markdown file
        """
        issues = self.get_issues()
        
        # Ensure the directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Write content to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# GitHub Issues\n\n")
            
            for issue in issues:
                # Write issue header
                f.write(f"## Issue #{issue['number']}: {issue['title']}\n\n")
                
                # Write issue metadata
                f.write(f"**State:** {issue['state']}\n")
                f.write(f"**Created:** {issue['created_at']}\n")
                if issue['closed_at']:
                    f.write(f"**Closed:** {issue['closed_at']}\n")
                f.write(f"**Author:** {issue['user']['login']}\n\n")
                
                # Write issue body
                f.write("### Description\n\n")
                f.write(f"{issue['body']}\n\n")
                
                # Add separator between issues
                f.write("---\n\n")
        
        print(f"Issues saved to {output_path}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='GitHub Wiki and Issues Reader')
    parser.add_argument('--owner', required=True, help='Repository owner/organization name')
    parser.add_argument('--repo', required=True, help='Repository name')
    parser.add_argument('--output-dir', default='wiki_pages', 
                       help='Directory to save wiki pages and issues (default: wiki_pages)')
    parser.add_argument('--clone-dir', default='wiki_clone',
                       help='Directory to clone wiki into (default: wiki_clone)')
    
    args = parser.parse_args()

    # Create WikiReader instance with command line arguments
    wiki_reader = GitHubWikiReader(args.owner, args.repo)

    try:
        # Get and save wiki pages
        wiki_pages = wiki_reader.get_wiki_pages(args.clone_dir)
        print("Available wiki pages:")
        
        for page in wiki_pages:
            page_name = page['title']
            page_path = page['path']
            print(f"- {page_name}")
            safe_filename = page_name.replace(" ", "_").lower() + ".md"
            output_path = f"{args.output_dir}/{safe_filename}"
            wiki_reader.save_wiki_to_markdown(page_name, page_path, output_path)
        
        # Get and save issues
        print("\nFetching issues...")
        issues_path = f"{args.output_dir}/issues.md"
        wiki_reader.save_issues_to_markdown(issues_path)
            
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Clean up the cloned wiki directory
        wiki_reader.cleanup(args.clone_dir)

if __name__ == "__main__":
    main()