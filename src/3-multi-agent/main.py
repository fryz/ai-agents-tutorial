# src/latest_ai_development/crew.py
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, MCPServerAdapter
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from pydantic import BaseModel


class ResearchTask(BaseModel):
  company_name: str
  research_topic: str
  feedback: str = ""


class ResearchPlan(BaseModel):
  tasks: List[ResearchTask]
  completed_tasks: List[ResearchTask]
  incomplete_tasks: List[ResearchTask]


def filter_out_notion_tools(tools):
  return [tool for tool in tools if "API-" not in tool.name ]


@CrewBase
class CompanyResearchCrew():
  """Company Research crew"""

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
  def orchestrator(self) -> Agent:
    return Agent(
      config=self.agents_config['orchestrator'], # type: ignore[index]
      verbose=True,
      tools=filter_out_notion_tools(self.get_mcp_tools()),
      allow_delegation=True
    )

  @agent
  def analyst(self) -> Agent:
    return Agent(
      config=self.agents_config['analyst'], # type: ignore[index]
      verbose=True,
      tools=[SerperDevTool()] + filter_out_notion_tools(self.get_mcp_tools())  # Add MCP tools back
    )
  
  @agent
  def account_executive(self) -> Agent:
    return Agent(
      config=self.agents_config['account_executive'], # type: ignore[index]
      verbose=True,
      tools=[SerperDevTool()] + filter_out_notion_tools(self.get_mcp_tools())  # Add MCP tools back
    )

  @agent
  def reporter(self) -> Agent:
    return Agent(
      config=self.agents_config['reporter'], # type: ignore[index]
      verbose=True,
      tools=self.get_mcp_tools()  # Add MCP tools back
    )

  @task
  def planning_task(self) -> Task:
    return Task(
      config=self.tasks_config['planning_task'], # type: ignore[index]
      output_model=ResearchPlan
    )

  @task
  def research_task(self) -> Task:
    return Task(
      config=self.tasks_config['research_task'], # type: ignore[index]
      output_model=ResearchTask
    )
  
  @task
  def feedback_task(self) -> Task:
    return Task(
      config=self.tasks_config['feedback_task'], # type: ignore[index]
      context=[self.research_task()],
      output_model=ResearchTask
    )

  @task
  def review_task(self) -> Task:
    return Task(
      config=self.tasks_config['review_task'], # type: ignore[index]
      context=[self.planning_task(), self.research_task(), self.feedback_task()],
      output_model=ResearchPlan,
    )

  @task
  def reporting_task(self) -> Task:
    return Task(
      config=self.tasks_config['reporting_task'], # type: ignore[index]
      context=[self.planning_task(), self.review_task(), self.research_task(), self.feedback_task()]
    )

  @crew
  def crew(self) -> Crew:
    """Creates the LatestAiDevelopment crew"""
    return Crew(
      agents=self.agents, # Automatically created by the @agent decorator
      tasks=self.tasks, # Automatically created by the @task decorator
      process=Process.hierarchical,
      manager_llm="gpt-4o",
      verbose=True,
    )


if __name__ == "__main__":
  crew = CompanyResearchCrew()

  inputs = {
    "company_name": "Salesforce",
    "research_topic": "fill this in"
  }
  crew.crew().kickoff(inputs=inputs)