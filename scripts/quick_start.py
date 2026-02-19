#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Lifestyle Agent - Quick Start Example

Before using, ensure:
1. Dependencies installed: pip install -r requirements.txt
2. Environment variables set:
   export OPENAI_API_KEY="sk-..."
   export GOOGLE_PLACES_API_KEY="AIza..."

Usage:
1. Interactive input: python scripts/quick_start.py
2. Command line argument: python scripts/quick_start.py "Find brunch in Seattle"
"""

import sys
import os
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from local_lifestyle_agent.orchestrator import Orchestrator
from local_lifestyle_agent.planner import Planner
from local_lifestyle_agent.executor import Executor
from local_lifestyle_agent.evaluator import Evaluator
from local_lifestyle_agent.llm_client import LLMClient
from local_lifestyle_agent.adapters.google_places import GooglePlacesAdapter
from local_lifestyle_agent.infrastructure.config import Config


def main():
    print("Local Lifestyle Agent - Quick Start\n")
    
    # 1. Load configuration
    print("Loading configuration...")
    try:
        config = Config.load()
        print("   Configuration loaded successfully")
        print(f"   - OpenAI Model: {config.openai_model}")
        print(f"   - Cache: {'Enabled' if config.cache_enabled else 'Disabled'}")
        print(f"   - Log Level: {config.log_level}")
    except Exception as e:
        print(f"   Configuration loading failed: {e}")
        print("\nPlease set environment variables:")
        print("   export OPENAI_API_KEY='sk-...'")
        print("   export GOOGLE_PLACES_API_KEY='AIza...'")
        return
    
    # 2. Initialize components
    print("\nInitializing components...")
    
    from local_lifestyle_agent.infrastructure.logger import StructuredLogger
    logger = StructuredLogger("orchestrator", log_level=config.log_level)
    
    llm_client = LLMClient(api_key=config.openai_api_key, logger=logger)
    places_adapter = GooglePlacesAdapter(api_key=config.google_places_api_key, logger=logger)
    
    planner = Planner(llm_client, logger=logger)
    executor = Executor(places_adapter, logger=logger)
    evaluator = Evaluator(min_rating=4.0, logger=logger)
    
    orchestrator = Orchestrator(planner, executor, evaluator, logger=logger)
    print("   Components initialized")
    
    # 3. Get user query
    print("\nPreparing search...")
    
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
        print(f"   Query (command line): {user_query}")
    else:
        print("\nEnter your query (e.g., Find brunch in Seattle on Sunday morning)")
        print("Or press Ctrl+C to exit")
        try:
            user_query = input("\nYour query: ").strip()
            if not user_query:
                print("Query cannot be empty")
                return
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            return
    
    print(f"\nStarting search...")
    print(f"   Query: {user_query}")
    
    try:
        result = orchestrator.run(user_query)
    except Exception as e:
        print(f"\nExecution failed: {e}")
        return
    
    # 4. Display results
    print("\n" + "="*70)
    
    if "error" in result:
        error = result["error"]
        print("Error occurred")
        print(f"   Error code: {error.error_code}")
        print(f"   Error message: {error.error_message}")
        print(f"   Request ID: {error.request_id}")
        if error.retry_after:
            print(f"   Tip: Retry after {error.retry_after} seconds")
    
    elif result.get("plan"):
        plan = result["plan"]
        print("Found recommendations!\n")
        
        print("Primary recommendation:")
        print(f"   Name: {plan.primary.name}")
        print(f"   Address: {plan.primary.address}")
        if plan.primary.rationale:
            print(f"   Rationale:")
            for reason in plan.primary.rationale:
                print(f"      - {reason}")
        
        if plan.backups:
            print(f"\nBackup options ({len(plan.backups)}):")
            for i, backup in enumerate(plan.backups[:3], 1):
                print(f"\n   {i}. {backup.name}")
                print(f"      Address: {backup.address}")
        
        if plan.schedule:
            print(f"\nSchedule:")
            if "arrive_at" in plan.schedule:
                print(f"   Arrival: {plan.schedule['arrive_at']}")
            if "leave_at" in plan.schedule:
                print(f"   Departure: {plan.schedule['leave_at']}")
        
        if plan.tips:
            print(f"\nTips:")
            for tip in plan.tips[:3]:
                print(f"   - {tip}")
        
        candidates = result.get("candidates", [])
        print(f"\nStatistics:")
        print(f"   Candidates: {len(candidates)}")
        print(f"   Request ID: {result.get('request_id')}")
        
        # Display cost summary
        cost_summary = result.get("cost_summary")
        if cost_summary:
            print(f"\nCost Summary:")
            print(f"   LLM (gpt-4o-mini):")
            print(f"      Tokens: {cost_summary['llm']['total_tokens']:,} ({cost_summary['llm']['prompt_tokens']:,} prompt + {cost_summary['llm']['completion_tokens']:,} completion)")
            print(f"      Cost: ${cost_summary['llm']['cost_usd']:.6f}")
            print(f"   Google Places API:")
            print(f"      Calls: {cost_summary['google_places']['api_calls']}")
            print(f"      Cost: ${cost_summary['google_places']['cost_usd']:.6f}")
            print(f"   Total Cost: ${cost_summary['total_cost_usd']:.6f}")
    
    else:
        print("No suitable recommendations found")
        eval_report = result.get("eval_report")
        if eval_report:
            print(f"   Reason: {eval_report.hard_violations}")
            if eval_report.replan_suggestions:
                print(f"   Suggestions: {eval_report.replan_suggestions}")
    
    print("="*70)
    print("\nDone!")
    print("\nTips:")
    print("   - Run again: python scripts/quick_start.py")
    print("   - Command line mode: python scripts/quick_start.py \"Your query here\"")
    print("   - See USAGE_GUIDE.md for more information")


if __name__ == "__main__":
    main()
