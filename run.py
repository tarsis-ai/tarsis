#!/usr/bin/env python
"""
Entry point for running Tarsis.

This script properly sets up the Python path and runs the Tarsis server.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Check for required dependencies
try:
    from dotenv import load_dotenv
    import uvicorn
except ImportError as e:
    print(f"""
❌ Error: Missing required dependencies

{e}

Please make sure you have activated your virtual environment and installed dependencies:

    # Activate virtual environment
    source venv/bin/activate  # Linux/macOS
    # or
    venv\\Scripts\\activate  # Windows

    # Install dependencies
    pip install -r requirements.txt

Then try running again:
    python run.py
""")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configure logging before importing application modules
from tarsis.logging_config import configure_logging
configure_logging()

# Import and run the application
from tarsis.main import app


def main():
    """Run the Tarsis server."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    Tarsis Agent v0.3                         ║
║          AI-Powered GitHub Issue Implementation              ║
╚══════════════════════════════════════════════════════════════╝

Architecture: Custom agentic loop with tool calling
Pattern: Recursive agent with multi-provider LLM support

📋 Configuration:
   - LLM Provider: {}
   - Model: {}
   - GitHub Repo: {}/{}

🌐 Starting server...
    """.format(
        os.getenv("LLM_PROVIDER", "ollama"),
        os.getenv("LLM_MODEL_ID", "default"),
        os.getenv("GITHUB_REPO_OWNER", "not-set"),
        os.getenv("GITHUB_REPO_NAME", "not-set")
    ))

    # Check for required environment variables
    required_vars = ["GITHUB_TOKEN", "GITHUB_REPO_OWNER", "GITHUB_REPO_NAME"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"⚠️  Warning: Missing environment variables: {', '.join(missing_vars)}")
        print("   Please set them in your .env file\n")

    # Get host and port from environment or use defaults
    host = os.getenv("TARSIS_HOST", "127.0.0.1")
    port = int(os.getenv("TARSIS_PORT", "8000"))

    print(f"🚀 Server starting on http://{host}:{port}")
    print(f"📖 API docs available at http://{host}:{port}/docs")
    print(f"💚 Health check: http://{host}:{port}/health\n")

    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )


if __name__ == "__main__":
    main()
