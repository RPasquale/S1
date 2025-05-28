"""
Startup script for the Data Extraction API

This script initializes and starts the data extraction system with proper configuration.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import fastapi
        import requests
        import git
        import huggingface_hub
        import schedule
        import bs4
        print("✓ All required packages are available")
        return True
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        return False

def install_requirements():
    """Install requirements from the data extraction requirements file"""
    requirements_file = Path("data_extraction_requirements.txt")
    
    if not requirements_file.exists():
        print("✗ Requirements file not found")
        return False
    
    print("Installing data extraction requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False

def setup_directories():
    """Create necessary directories"""
    directories = [
        "extracted_data",
        "extracted_data/github",
        "exported_data",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def run_initial_extraction():
    """Run initial data extraction"""
    print("Running initial data extraction...")
    try:
        from data_extraction_api import DataExtractionAPI
        
        api = DataExtractionAPI()
        count = api.run_extraction_cycle()
        print(f"✓ Initial extraction complete. Extracted {count} items.")
        return True
    except Exception as e:
        print(f"✗ Error during initial extraction: {e}")
        return False

def start_api_server():
    """Start the FastAPI server"""
    print("Starting Data Extraction API Server on port 6000...")
    try:
        import uvicorn
        from data_extraction_server import app
        
        uvicorn.run(
            "data_extraction_server:app",
            host="127.0.0.1",
            port=6000,
            reload=True,
            log_level="info"
        )
    except Exception as e:
        print(f"✗ Error starting server: {e}")

def start_scheduler():
    """Start the background scheduler"""
    print("Starting background data extraction scheduler...")
    try:
        from data_extraction_api import start_scheduled_extraction
        start_scheduled_extraction()
    except Exception as e:
        print(f"✗ Error starting scheduler: {e}")

def main():
    """Main startup function"""
    print("=" * 60)
    print("Data Extraction API Startup")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("data_extraction_api.py").exists():
        print("✗ Please run this script from the S1 directory")
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Check requirements
    if not check_requirements():
        print("\nInstalling missing requirements...")
        if not install_requirements():
            print("✗ Failed to install requirements. Please install manually.")
            sys.exit(1)
        
        # Check again after installation
        if not check_requirements():
            print("✗ Requirements still missing after installation.")
            sys.exit(1)
    
    # Run initial extraction
    if len(sys.argv) > 1 and sys.argv[1] == "--skip-initial":
        print("Skipping initial extraction...")
    else:
        run_initial_extraction()
    
    # Start the appropriate service
    if len(sys.argv) > 1:
        if sys.argv[1] == "server":
            start_api_server()
        elif sys.argv[1] == "scheduler":
            start_scheduler()
        elif sys.argv[1] == "both":
            # Start scheduler in background and then start server
            import threading
            scheduler_thread = threading.Thread(target=start_scheduler)
            scheduler_thread.daemon = True
            scheduler_thread.start()
            time.sleep(2)  # Give scheduler time to start
            start_api_server()
        else:
            print("Usage: python start_data_extraction.py [server|scheduler|both|--skip-initial]")
    else:
        print("\nWhat would you like to start?")
        print("1. API Server only")
        print("2. Background Scheduler only") 
        print("3. Both API Server and Scheduler")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            start_api_server()
        elif choice == "2":
            start_scheduler()
        elif choice == "3":
            import threading
            scheduler_thread = threading.Thread(target=start_scheduler)
            scheduler_thread.daemon = True
            scheduler_thread.start()
            time.sleep(2)
            start_api_server()
        else:
            print("Invalid choice. Starting API server by default.")
            start_api_server()

if __name__ == "__main__":
    main()
