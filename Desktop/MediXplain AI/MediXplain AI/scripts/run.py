import sys
import os
import subprocess

def run_pipeline():
    """
    Run the complete MediXplain AI pipeline.
    """
    print("Starting MediXplain AI Pipeline...")

    # Step 1: Download data
    print("\nStep 1: Downloading PIMA Diabetes Dataset...")
    subprocess.run([sys.executable, 'data/download.py'])

    # Step 2: Preprocess data
    print("\nStep 2: Preprocessing data...")
    subprocess.run([sys.executable, 'data/preprocess.py'])

    # Step 3: Train model
    print("\nStep 3: Training model...")
    subprocess.run([sys.executable, 'models/train.py'])

    # Step 4: Start backend server
    print("\nStep 4: Starting backend server...")
    print("Backend server will run on http://localhost:5000")
    print("Open frontend/index.html in your browser to use the application.")
    print("Press Ctrl+C to stop the server.")

    subprocess.run([sys.executable, 'backend/app.py'])

if __name__ == "__main__":
    run_pipeline()
