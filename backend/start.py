"""
Simple startup script for SunShift Backend
Run this file to start the backend server
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    print("☀️  Starting SunShift Backend...")
    print()
    
    # Get the directory of this script
    backend_dir = Path(__file__).parent
    os.chdir(backend_dir)
    
    # Check if virtual environment exists
    venv_path = backend_dir / "venv"
    if not venv_path.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✓ Virtual environment created")
        print()
    
    # Determine the correct Python executable in venv
    if sys.platform == "win32":
        python_exe = venv_path / "Scripts" / "python.exe"
        pip_exe = venv_path / "Scripts" / "pip.exe"
    else:
        python_exe = venv_path / "bin" / "python"
        pip_exe = venv_path / "bin" / "pip"
    
    # Install dependencies
    print("Installing dependencies (this may take a few minutes)...")
    print("-" * 60)
    subprocess.run([str(pip_exe), "install", "-r", "requirements.txt"], check=True)
    print("-" * 60)
    print("✓ Dependencies installed")
    print()
    
    # Check for .env file
    env_file = backend_dir / ".env"
    env_example = backend_dir / ".env.example"
    
    if not env_file.exists() and env_example.exists():
        print("⚠️  No .env file found. Copying from .env.example...")
        with open(env_example, 'r') as src:
            with open(env_file, 'w') as dst:
                dst.write(src.read())
        print("Please edit .env file with your API keys (optional)")
        print()
    
    print("✓ Setup complete!")
    print()
    print("Starting FastAPI server on http://localhost:8000")
    print("Press Ctrl+C to stop")
    print()
    
    # Run the server
    try:
        subprocess.run([str(python_exe), "main.py"])
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        sys.exit(0)

if __name__ == "__main__":
    main()
