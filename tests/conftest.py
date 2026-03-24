"""pytest configuration and fixtures for the project.

This file ensures the src directory is in the Python path,
allowing imports of the package to work correctly.
"""

import sys
from pathlib import Path

# Add src to Python path so tests can import the package
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
