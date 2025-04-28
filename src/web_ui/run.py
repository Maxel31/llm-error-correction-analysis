"""
Run script for the web UI visualization.
"""
import os
import argparse
from pathlib import Path
from app import app

def main():
    """Run the Flask application."""
    project_root = Path(__file__).parent.parent.parent.absolute()
    
    parser = argparse.ArgumentParser(description="Run the web UI for activation visualization")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(project_root / "data" / "results"),
        help="Directory containing result files"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the server on"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the server on"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode"
    )
    
    args = parser.parse_args()
    
    os.environ["RESULTS_DIR"] = args.results_dir
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Warning: Results directory '{args.results_dir}' does not exist.")
        print("Creating directory...")
        results_dir.mkdir(parents=True, exist_ok=True)
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )

if __name__ == "__main__":
    main()
