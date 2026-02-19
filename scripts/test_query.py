#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Lifestyle Agent - Simple Test Script

Quickly test different queries without complex configuration.

Usage:
    python scripts/test_query.py "Your query here"
    
Examples:
    python scripts/test_query.py "Find brunch in Seattle"
    python scripts/test_query.py "Find coffee shop in Portland"
    python scripts/test_query.py "Find dinner in San Francisco"
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
    # Check command line arguments
    if len(sys.argv) < 2:
        print("❌ Error: Please provide query content")
        print("\nUsage:")
        print('    python scripts/test_query.py "Your query here"')
        print("\nExamples:")
        print('    python scripts/test_query.py "Find brunch in Seattle"')
        print('    python scripts/test_query.py "Find coffee shop in Portland"')
        return
    
    user_query = " ".join(sys.argv[1:])
    
    print(f"🔍 Test Query: {user_query}\n")
    
    # Load configuration
    try:
        config = Config.load()
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        print("\nPlease ensure environment variables are set:")
        print("   export OPENAI_API_KEY='sk-...'")
        print("   export GOOGLE_PLACES_API_KEY='AIza...'")
        return
    
    # Initialize components (silent mode)
    from local_lifestyle_agent.infrastructure.logger import StructuredLogger
    logger = StructuredLogger("test", log_level="ERROR")  # Only show errors
    
    llm_client = LLMClient(api_key=config.openai_api_key, logger=logger)
    places_adapter = GooglePlacesAdapter(api_key=config.google_places_api_key, logger=logger)
    
    planner = Planner(llm_client, logger=logger)
    executor = Executor(places_adapter, logger=logger)
    evaluator = Evaluator(min_rating=4.0, logger=logger)
    
    orchestrator = Orchestrator(planner, executor, evaluator, logger=logger)
    
    # Run query
    print("⏳ Processing...\n")
    try:
        result = orchestrator.run(user_query)
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        return
    
    # Display results
    print("="*70)
    
    if "error" in result:
        error = result["error"]
        print("❌ Error occurred")
        print(f"   {error.error_message}")
    
    elif result.get("plan"):
        plan = result["plan"]
        intent = result.get("intent")
        
        print("✅ Found recommendations!\n")
        
        # Display recognized intent
        if intent:
            print(f"📝 Recognized Intent:")
            print(f"   Activity Type: {intent.activity_type}")
            print(f"   City: {intent.city}")
            print(f"   Party Size: {intent.party_size}")
            print(f"   Budget: {intent.budget_level}")
            print()
        
        # Primary option
        print(f"🎯 Primary: {plan.primary.name}")
        print(f"   Address: {plan.primary.address}")
        
        # Backup options
        if plan.backups:
            print(f"\n📋 Backups ({len(plan.backups)} total):")
            for i, backup in enumerate(plan.backups[:3], 1):
                print(f"   {i}. {backup.name}")
        
        # Statistics
        candidates = result.get("candidates", [])
        print(f"\n📊 Found {len(candidates)} candidate venues")
        
        # Cost statistics
        cost_summary = result.get("cost_summary")
        if cost_summary:
            print(f"\n💰 Cost Statistics:")
            print(f"   LLM: ${cost_summary['llm']['cost_usd']:.6f} ({cost_summary['llm']['total_tokens']:,} tokens)")
            print(f"   Google Places: ${cost_summary['google_places']['cost_usd']:.6f} ({cost_summary['google_places']['api_calls']} calls)")
            print(f"   Total: ${cost_summary['total_cost_usd']:.6f}")
    
    else:
        print("⚠️  No suitable recommendations found")
        eval_report = result.get("eval_report")
        if eval_report and eval_report.hard_violations:
            print(f"   Reason: {', '.join(eval_report.hard_violations)}")
    
    print("="*70)


if __name__ == "__main__":
    main()
