from __future__ import annotations

from local_lifestyle_agent.config import Settings
from local_lifestyle_agent.llm_client import LLMClient
from local_lifestyle_agent.adapters.google_places import GooglePlacesAdapter
from local_lifestyle_agent.planner import Planner
from local_lifestyle_agent.executor import Executor
from local_lifestyle_agent.evaluator import Evaluator
from local_lifestyle_agent.orchestrator import Orchestrator, RunContext
from local_lifestyle_agent.renderer import Renderer


def main():
    settings = Settings.load(interactive=True)

    llm = LLMClient(api_key=settings.openai_api_key, model=settings.openai_model)
    places = GooglePlacesAdapter(api_key=settings.google_places_api_key)

    planner = Planner(llm)
    executor = Executor(places)
    evaluator = Evaluator(min_rating=4.0)
    orch = Orchestrator(planner, executor, evaluator)
    renderer = Renderer()

    user_prompt = "I want afternoon tea on Sunday afternoon in Seattle. Prefer quiet vibe, within 30 minutes."
    out = orch.run(user_prompt, ctx=RunContext(max_tool_calls=6, max_iterations=3))

    if out["plan"]:
        md = renderer.to_markdown(out["intent"], out["plan"]).markdown
        print(md)
    else:
        print("No solution found.")
        if out["eval_report"]:
            print(out["eval_report"].model_dump())


if __name__ == "__main__":
    main()
