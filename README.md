# Build & Connect Event - Building with AI Agents

## Goals

1. Build an AI Agent end-to-end using:
    * CrewAI for multi-agent orchestration
    * MCP servers for tool calling
2. Get an understanding of how AI Agents work in practice
3. Learn techniques for developing using AI Assisted tools
4. Walk away with something that can be extended for your own projects


## Pre-requisites

1. We will be using python and virtual environments

```bash
$ which python3
/opt/homebrew/bin/python3
$ pip3 install virtualenv
$ virtualenv .venv -p python313
```

2. We will be using Docker for MCP server hosting. 

3. It will be useful to have an AI-assisted coding tool, like Cursor, Claude Code, etc.

## Syllabus

1. Hello World with Crew AI
  1. Setup dependencies and walk through Crew API
  2. Learn about Tools and integrate a Web Search tool

2. MCP Server Integration
  1. Intro to MCP
  2. Learn where to find MCP servers
  3. Integrate Crew AI Agent with MCP Server

3. Multi-agent setup
  1. Intro to orchestration and task management
  2. Multi-agents with Crew AI