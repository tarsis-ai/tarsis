# Tarsis

**AI-Powered GitHub Issue Implementation Assistant**

Tarsis is an autonomous AI agent that implements GitHub issues using a custom recursive agent architecture with tool calling. Simply comment `/implement` on an issue, and Tarsis will autonomously analyze, plan, code, validate, and create a pull request.

## Features

- **Autonomous Implementation**: Recursive agent loop with multi-step planning and execution
- **Multi-LLM Support**: Works with Anthropic Claude, Ollama (local), and Google Gemini
- **Intelligent Code Discovery**: Hybrid file discovery combining filesystem scanning, code search (ripgrep), and LLM reasoning
- **Advanced Multi-Tier Validation**: 5-tier validation system (tests → static analysis → linting → syntax → dependency validation)
- **Local Repository Operations**: Full local clone management with branch operations and lifecycle management
- **Conventional Commits**: AI-powered commit message generation with validation and formatting
- **Tool-Based Architecture**: 20 production tools for GitHub operations, file management, code search, validation, and local git operations
- **Extensible Design**: Modular prompt system and plugin-style tool architecture

## Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd tarsis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install ripgrep (required for code search)
sudo dnf install ripgrep  # Fedora/CentOS
# or
sudo apt install ripgrep  # Ubuntu/Debian
# or: brew install ripgrep  # macOS
```

### Configuration

Create a `.env` file (see `.env.example` for reference):

```bash
# GitHub credentials
GITHUB_TOKEN=ghp_your_token_here
GITHUB_REPO_OWNER=your-username
GITHUB_REPO_NAME=your-repo

# LLM Provider (choose one)
# Option 1: Ollama (Local, Free)
LLM_PROVIDER=ollama
LLM_MODEL_ID=qwen2.5-coder:7b
OLLAMA_BASE_URL=http://localhost:11434

# Option 2: Anthropic Claude
# LLM_PROVIDER=anthropic
# LLM_MODEL_ID=claude-3-5-sonnet-20241022
# ANTHROPIC_API_KEY=sk-ant-xxx

# Option 3: Google Gemini
# LLM_PROVIDER=google
# LLM_MODEL_ID=gemini-2.5-flash
# GOOGLE_API_KEY=xxx
```

### Running Tarsis

```bash
# Start the server
python run.py

# The webhook endpoint will be available at:
# http://localhost:8000/webhook
```

Configure your GitHub repository to send issue comment webhooks to this endpoint.

### Usage

1. Create or open a GitHub issue
2. Comment `/implement` on the issue
3. Tarsis will:
   - Analyze the issue requirements
   - Discover relevant files in your codebase
   - Create a feature branch
   - Implement the changes
   - Run validation (tests, linting, type checking)
   - Create a pull request with the implementation

## Architecture

Tarsis uses a **recursive agent loop** with tool calling, inspired by modern AI coding assistants:

```
GitHub Webhook → AgentTask (recursive loop) → LLM Provider (Claude/Ollama/Gemini)
                       ↓
                 Tool Executor
                       ↓
    ┌──────────────────┼──────────────────────┐
    ↓                  ↓                      ↓
GitHub Tools     File Operations     Code Search & Discovery
(6 tools)           (4 tools)              (4 tools)
    ↓                  ↓                      ↓
Task Tools      Validation Tools      Local Git Operations
(2 tools)           (1 tool)              (3 tools)
```

The agent autonomously plans and executes tasks using 20 production tools across multiple categories.

## Current Status (v0.2.0)

- ✅ Core agentic architecture (recursive agent loop)
- ✅ 20 production tools (GitHub, file operations, code search, validation, local git operations)
- ✅ Multi-provider LLM support (Anthropic, Ollama, Google)
- ✅ Intelligent file discovery and code search (hybrid approach with ripgrep)
- ✅ **Advanced multi-tier validation system** (5 tiers: tests → static analysis → linting → syntax → dependency)
- ✅ **Local repository clone management** with lifecycle management and branch operations
- ✅ **Conventional commits support** with AI-powered message generation and validation
- ✅ Local git operations (file rename, symlink creation, local file modifications)
- ✅ Dependency and import validation
- ✅ Fully functional for basic to advanced tasks

## Tech Stack

- **Python 3.10+** with asyncio
- **FastAPI** - Web framework for webhooks
- **Anthropic/Ollama/Google** - LLM providers
- **ripgrep** - Fast code search
- **GitHub API** - Repository operations

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with the modern agentic architecture pattern
- Inspired by AI coding assistants like Claude Code, Cursor, Cline, and Aider
- Uses tool calling for autonomous task execution

---

Happy coding! 🚀
