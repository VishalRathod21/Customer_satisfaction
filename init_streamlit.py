import os
import subprocess
from pathlib import Path

def setup_environment():
    """Set up the environment for Streamlit Cloud."""
    print("Setting up environment for Streamlit Cloud...")
    
    # Create necessary directories
    zenml_dir = Path(".zen")
    zenml_dir.mkdir(exist_ok=True)
    
    # Initialize ZenML if not already initialized
    if not (zenml_dir / "config.yaml").exists():
        print("Initializing ZenML...")
        subprocess.run(["python", "init_zenml.py"], check=True)
    
    print("✅ Environment setup complete!")

if __name__ == "__main__":
    setup_environment()
