#!/usr/bin/env python3
"""
Streamlit Frontend Startup Script
Expert Matching System
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Start the Streamlit frontend application"""
    print("🎨 Starting Expert Matching System Frontend...")
    print("📍 App will be available at: http://localhost:3000")
    print("🔄 Press Ctrl+C to stop the app")
    print("-" * 50)
    
    try:
        # Get the path to the streamlit app
        app_path = Path(__file__).parent / "streamlit_app.py"
        
        # Start Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port=3000",
            "--server.address=0.0.0.0",
            "--browser.gatherUsageStats=false"
        ])
        
    except KeyboardInterrupt:
        print("\n⏹️ Frontend stopped by user")
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 