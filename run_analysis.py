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

def create_sample_data():
    """Create sample data for testing"""
    from data_processor import create_sample_data
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    create_sample_data("data/queries.jsonl", num_samples=20)
    print("Sample data created!")

def run_analysis(models=None, data_file=None):
    """Run the logits analysis"""
    from analyzer import LogitsAnalyzer
    
    analyzer = LogitsAnalyzer()
    
    try:
        if models:
            model_list = models.split(',')
            analyzer.run_full_analysis(model_list, data_file)
        else:
            analyzer.run_full_analysis(data_file=data_file)
    except Exception as e:
        print(f"Analysis failed: {e}")
        print("\\nIf you're running this for the first time, the model paths might need to be adjusted.")
        print("Check config.py and update the model paths to match your system.")

def main():
    parser = argparse.ArgumentParser(description="LLM Logits Analysis Tool")
    parser.add_argument("--setup", action="store_true", help="Set up the environment")
    parser.add_argument("--create-sample", action="store_true", help="Create sample data")
    parser.add_argument("--run", action="store_true", help="Run analysis")
    parser.add_argument("--models", type=str, help="Comma-separated list of models to analyze (baseline,sft,rl)")
    parser.add_argument("--data", type=str, help="Path to data file")
    parser.add_argument("--all", action="store_true", help="Run complete pipeline (setup + sample + analysis)")
    
    args = parser.parse_args()
    
    if args.all:
        setup_environment()
        create_sample_data()
        run_analysis(args.models, args.data)
    else:
        if args.setup:
            setup_environment()
        
        if args.create_sample:
            create_sample_data()
        
        if args.run:
            run_analysis(args.models, args.data)
    
    if not any([args.setup, args.create_sample, args.run, args.all]):
        parser.print_help()

if __name__ == "__main__":
    main()
