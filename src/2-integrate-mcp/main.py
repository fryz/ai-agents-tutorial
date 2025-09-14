# src/latest_ai_development/crew.py
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, MCPServerAdapter
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

@CrewBase
class BDAnalystCrew():
  """BD Analyst crew"""

  agents: List[BaseAgent]
  tasks: List[Task]
  
  # MCP server configuration for Notion running in Docker Desktop MCP Toolkit
  mcp_server_params = {
    "url": "http://localhost:8000/sse",
    "transport": "sse"
  }
  
  # Connection timeout for MCP server
  mcp_connect_timeout = 60

  @agent
  def analyst(self) -> Agent:
    return Agent(
      config=self.agents_config['analyst'], # type: ignore[index]
      verbose=True,
      tools=[SerperDevTool()] + self.get_mcp_tools()  # Add MCP tools back
    )

  @task
  def research_task(self) -> Task:
    return Task(
      config=self.tasks_config['research_task'], # type: ignore[index]
    )

  @task
  def reporting_task(self) -> Task:
    return Task(
      config=self.tasks_config['reporting_task'], # type: ignore[index]
    )

  @crew
  def crew(self) -> Crew:
    """Creates the LatestAiDevelopment crew"""
    return Crew(
      agents=self.agents, # Automatically created by the @agent decorator
      tasks=self.tasks, # Automatically created by the @task decorator
      process=Process.sequential,
      verbose=True,
    )


if __name__ == "__main__":
  crew = BDAnalystCrew()
  inputs = {
    "company_name": "Salesforce"
  }
  crew.crew().kickoff(inputs=inputs)