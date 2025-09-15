from enum import Enum
from typing import List
from uuid import UUID, uuid4
from crewai.agent import Agent
from crewai.flow.flow import Flow, listen, or_, router, start
from crewai.flow.persistence import persist
from crewai_tools import MCPServerAdapter
from crewai_tools.tools import SerperDevTool
from pydantic import BaseModel, Field


MCP_SERVER_PARAMS = {
  "url": "http://localhost:8000/sse",
  "transport": "sse"
}

class InitialResearchTopics(BaseModel):
  research_topics: List[str] = []

class ResearchTask(BaseModel):
  task_id: UUID = Field(default_factory=uuid4)
  research_topic: str = ""
  research_results: str | None = None
  research_citations: List[str] | None = None
  is_complete: bool = False
  feedback: str | None = None
  feedback_counter: int = 0


class ResearchPlan(BaseModel):
  research_tasks: List[ResearchTask] = []

class ResearchFlowState(BaseModel):
  company_name: str = ""
  research_plan: ResearchPlan | None = None
  completed_tasks: List[ResearchTask] = []
  report: str | None = None
  current_task: ResearchTask | None = None


def get_next_uncompleted_task(state) -> ResearchTask | None:
  completed_tasks = [task.task_id for task in state.completed_tasks]
  for task in state.research_plan.research_tasks:
    if task.task_id in completed_tasks:
      continue
    return task
  return None


class ResearchFlow(Flow[ResearchFlowState]):

  @start()
  def create_plan(self):
    planning_agent = Agent(
      role = "Planner",
      goal = f"Create a plan for researching the company {self.state.company_name}",
      backstory = "You are an experienced planner who is responsible for creating a plan for researching a company.",
    )

    query = """
    You are a planner who is responsible for creating a plan for researching {self.state.company_name}.
    Your goal is to develop a plan that will produce a research document on a company that will be used by an Account Executive to prepare for a sales call. 
    The plan should touch on the following items: 
    * Researching the company
        - Company description
        - Company goals
        - Company performance
        - Company competitors
        - Company market position
        - Company financials
        - Company news
    * Looking at the research that is prepared and providing feedback on the completeness, correctness and relevance of the research
    * Producing a report based off the research and feedback in markdown format

    """

    result = planning_agent.kickoff(query, response_format=InitialResearchTopics)

    self.state.research_plan = ResearchPlan() 
    for research_topic in result.pydantic.research_topics:
      self.state.research_plan.research_tasks.append(ResearchTask(research_topic=research_topic))
    self.state.current_task = self.state.research_plan.research_tasks[0]

    return "execute"

  @listen("research")
  def research_task(self):

    with MCPServerAdapter(
      MCP_SERVER_PARAMS,
      connect_timeout=60
    ) as mcp_tools:
      agent = Agent(
        role = "Researcher",
        goal = f"Research the company {self.state.company_name} and provide a report on the company",
        backstory = "You are a researcher who is responsible for researching a company and providing a report on the company",
        tools = [SerperDevTool()] + mcp_tools,
      )

      query = f"""
          Conduct a thorough research of {self.state.company_name}'s on topic {self.state.current_task.research_topic} and prepare a brief that will be used by an Account Executive to present to a potential client.
          To perform this task, you have access to tools that allow you to search the web and download information from specific web resources.

          Output should be a list of 10 bullet points of the most relevant information about {self.state.company_name}'s on the topic of {self.state.current_task.research_topic}, along with a bulleted list of the web resources that you used to gather the information.
      """

      result = agent.kickoff(query)
      print(result)
      self.state.current_task.research_results = result.raw

    # Finally, clear out the feedback
    self.state.current_task.feedback = None
    return "execute"

  @listen("review")
  def review_task(self):
    with MCPServerAdapter(
      MCP_SERVER_PARAMS,
      connect_timeout=60
    ) as mcp_tools:
      agent = Agent(
        role = "Reviewer",
        goal = f"Review the research results for the company {self.state.company_name} and provide feedback on the research",
        backstory = "You are a reviewer who is responsible for reviewing the research results for a company and providing feedback on the research",
        tools = [SerperDevTool()] + mcp_tools,
      )
      
      query = f"""
      Review the following information and providefeedback about the quality, completeness, and correctness of the information. 
      To perform this task, you have access to tools that allow you to search the web and download information from specific web resources.

      Your output should be feedback about the quality, completeness, and correctness of the information. 
      If there is no feedback, return "No feedback".

      Information: 
      {self.state.current_task.research_results}
      """

      result = agent.kickoff(query)
      self.state.current_task.feedback = result.raw
      print(result)

    return "execute"

  @listen("report")
  def report_task(self):
    with MCPServerAdapter(
      MCP_SERVER_PARAMS,
      connect_timeout=60
    ) as mcp_tools:
      agent = Agent(
        role = "Reporter",
        goal = f"Report the research results for the company {self.state.company_name} and provide a report on the company",
        backstory = "You are a reporter who is responsible for reporting the research results for a company and providing a report on the company",
        tools = mcp_tools,
      )

      query = """
      Review the content you received and expand on it to create a detailed report that will be used by an Account Executive to present to a potential client.
      Make sure the report is detailed and contains all the relevant information about $COMPANY_NAME
      
      After creating the report, use the Notion MCP Tool (API-post-page) to create a new document in Notion with the report content.
      
      IMPORTANT: The basic page creation worked! Now try to add content using the API-patch-page tool.
      
      STEP 1: Create the page with just the title:
      {
        "parent": {"page_id": "26d1d37e9a10806b807aec4b2c9f6d62"},
        "properties": {
          "title": [{"text": {"content": "$COMPANY_NAME Comprehensive Report (2025)"}}]
        }
      }
      
      STEP 2: Use API-patch-page to add content. The correct schema for API-patch-page is:
      {
        "page_id": "[page_id_from_step_1]",
        "properties": {
          "title": [{"text": {"content": "$COMPANY_NAME Comprehensive Report (2025)"}}]
        }
      }
      
      STEP 3: Use API-patch-block-children to add content blocks. NOTE: This tool only supports "paragraph" and "bulleted_list_item" types, NOT heading blocks.
      
      Use markdown-style headers within paragraph content:
      {
        "block_id": "[page_id_from_step_1]",
        "children": [
          {
            "type": "paragraph",
            "paragraph": {
              "rich_text": [{"type": "text", "text": {"content": "## Executive Summary\nYour content here"}}]
            }
          },
          {
            "type": "paragraph",
            "paragraph": {
              "rich_text": [{"type": "text", "text": {"content": "## Market Position\nYour content here"}}]
            }
          }
        ]
      }
      
      IMPORTANT: The API-patch-block-children tool only supports paragraph and bulleted_list_item blocks. Use markdown headers (##, ###) within the paragraph content to create section headers.
      
      If none of the tools work for adding content, create the page with just the title and include the full report content in the expected_output instead.      
      """

      result = agent.kickoff(query)
      self.state.report = result.raw
    return "execute"

  @router(or_(create_plan, research_task, review_task, report_task, "execute"))
  def orchestrator(self):


    # If we are done with research move to reporting
    if self.state.current_task is None:
      if self.state.report is None:
        return "report"
      else:
        return "stop"

    # Starting the research task
    if self.state.current_task.feedback_counter == 0 and self.state.current_task.research_results is None:
      return "research"

    # We received feedback from the reviewer - send back to the analyst
    if self.state.current_task.feedback_counter < 3 and self.state.current_task.feedback is not None:
      self.state.current_task.feedback_counter += 1

      # if there is no feedback, mark as complete and review the plan
      if self.state.current_task.feedback == "No feedback":
        self.state.current_task.is_complete = True
        self.state.completed_tasks.append(self.state.current_task)
        self.state.current_task = get_next_uncompleted_task(self.state)
        return "execute"
        
      return "research"

    # If there isn't feedback, send to the reviewer for feedback
    if self.state.current_task.feedback_counter < 3 and self.state.current_task.feedback is None:
      return "review"

    # If we've received feedback 3 times, mark as complete and review the plan
    elif self.state.current_task.feedback_counter >= 3:
      self.state.current_task.is_complete = True
      self.state.completed_tasks.append(self.state.current_task)
      self.state.current_task = get_next_uncompleted_task(self.state)
      return "execute"


def run_flow():
  company_name = "Palantir"
  flow = ResearchFlow()
  result = flow.kickoff(inputs={"company_name": company_name})
  return result


if __name__ == "__main__":
  run_flow()
