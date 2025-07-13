#!/usr/bin/env python3
"""
Quick setup and run script for logits analysis
"""

import os
import sys
import argparse
from pathlib import Path

def setup_environment():
    """Set up the environment using uv"""
    print("Setting up environment with uv...")
    
    # Install dependencies
    os.system("uv sync")
    
    print("Environment setup complete!")

def run_analysis(models=None, data_file=None):
    """Run the entropy analysis"""
    from analyzer import LogitsAnalyzer
    
    analyzer = LogitsAnalyzer()
    
    try:
        if models:
            model_list = models.split(',')
            results = analyzer.run_full_analysis(model_list, data_file)
        else:
            results = analyzer.run_full_analysis(data_file=data_file)
        
        print("\\nAnalysis completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return None
            analyzer.run_full_analysis(data_file=data_file, generate_predictions=generate_predictions)
    except Exception as e:
        print(f"Analysis failed: {e}")
        print("\\nIf you're running this for the first time, the model paths might need to be adjusted.")
        print("Check config.py and update the model paths to match your system.")

def main():
    parser = argparse.ArgumentParser(description="LLM Logits Analysis Tool")
    parser.add_argument("--setup", action="store_true", help="Set up the environment")
    parser.add_argument("--run", action="store_true", help="Run analysis")
    parser.add_argument("--models", type=str, help="Comma-separated list of models to analyze (baseline,sft,rl)")
    parser.add_argument("--data", type=str, help="Path to data file")
    parser.add_argument("--all", action="store_true", help="Run complete pipeline (setup + sample + analysis)")
    parser.add_argument("--no-predictions", action="store_true", help="Skip prediction generation (only analyze logits)")
    
    args = parser.parse_args()
    
    generate_predictions = not args.no_predictions
    
    if args.all:
        setup_environment()
        run_analysis(args.models, args.data, generate_predictions)
    else:
        if args.setup:
            setup_environment()
        
        if args.run:
            run_analysis(args.models, args.data, generate_predictions)
    
    if not any([args.setup, args.run, args.all]):
        parser.print_help()

if __name__ == "__main__":
    main()
