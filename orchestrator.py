from typing import Dict, Any

from analytics_agent import AnalyticsAgent
from validator_agent import ValidatorAgent
from web_search import WebSearchAgent
from memory import ConversationMemory
from planner_agent import PlannerAgent


class Orchestrator:
    def __init__(self, dataset_dir: str = "datasets"):
        self.memory = ConversationMemory()
        self.analytics_agent = AnalyticsAgent(dataset_dir=dataset_dir)
        self.validator = ValidatorAgent()
        self.web_agent = WebSearchAgent(max_results=5)
        self.planner = PlannerAgent()

    def handle_question(self, question: str) -> Dict[str, Any]:
        self.memory.add("user", question)

        available_columns = self.analytics_agent.get_available_columns()
        available_years = self.analytics_agent.get_available_years()

        plan = self.planner.make_plan(
            question=question,
            available_years=available_years,
            available_columns=available_columns
        )

        route = plan.get("route", "web")

        print("\n========================")
        print("QUESTION:", question)
        print("PLAN:", plan)
        print("ROUTE:", route)
        print("========================\n")

        if route == "web":
            result = self.web_agent.answer(question)
            result["plan"] = plan
            self.memory.add(
                "assistant",
                result["final_answer"],
                {"route": "web", "plan": plan}
            )
            return result

        # First analytics attempt
        analytics_result = self.analytics_agent.answer_with_plan(question, plan)
        validation = self.validator.validate(question, analytics_result)

        trace = analytics_result.get("trace", [])

        trace.insert(0, {
            "iteration": "planner",
            "action": "llm_plan",
            "observation": plan
        })

        trace.append({
            "iteration": "validation",
            "action": "validator_check",
            "observation": validation
        })

        analytics_result["validation"] = validation
        analytics_result["trace"] = trace
        analytics_result["plan"] = plan
        analytics_result["retry_performed"] = False

        # If validation passes, append visible validation paragraph
        if validation.get("passed", False):
            validation_paragraph = validation.get("validation_paragraph", "")

            if validation_paragraph:
                analytics_result["final_answer"] += "\n\n" + validation_paragraph

            self.memory.add(
                "assistant",
                analytics_result.get("final_answer", ""),
                {
                    "route": "analytics",
                    "plan": plan,
                    "validation": validation,
                    "retry_performed": False
                }
            )

            return analytics_result

        # If validation fails, retry once
        retry_result = self.analytics_agent.answer_with_plan(question, plan)
        retry_validation = self.validator.validate(question, retry_result)

        retry_trace = retry_result.get("trace", [])

        retry_trace.insert(0, {
            "iteration": "planner_retry",
            "action": "reuse_llm_plan",
            "observation": plan
        })

        retry_trace.append({
            "iteration": "validation_retry",
            "action": "validator_check_after_retry",
            "observation": retry_validation
        })

        retry_result["validation"] = retry_validation
        retry_result["trace"] = trace + retry_trace
        retry_result["plan"] = plan
        retry_result["retry_performed"] = True

        retry_validation_paragraph = retry_validation.get("validation_paragraph", "")

        if retry_validation_paragraph:
            retry_result["final_answer"] += "\n\n" + retry_validation_paragraph

        # Add a short note explaining that retry occurred
        retry_result["final_answer"] += (
            "\n\nRetry Note: The validator requested a retry because the first analysis "
            "was incomplete or suspicious. The system reran the analytics agent using the same structured plan."
        )

        self.memory.add(
            "assistant",
            retry_result.get("final_answer", ""),
            {
                "route": "analytics",
                "plan": plan,
                "validation": retry_validation,
                "retry_performed": True
            }
        )

        return retry_result