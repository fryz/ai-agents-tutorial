# src/latest_ai_development/crew.py
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

@CrewBase
class BDAnalystCrew():
  """BD Analyst crew"""

  agents: List[BaseAgent]
  tasks: List[Task]

  @agent
  def analyst(self) -> Agent:
    return Agent(
      config=self.agents_config['analyst'], # type: ignore[index]
      verbose=True,
      tools=[SerperDevTool()]
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
      output_file=f'output/report.md' # This is the file that will be contain the final report.
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