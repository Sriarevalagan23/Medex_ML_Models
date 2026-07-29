import sys
from pathlib import Path

# Add project root to sys.path so modules in the repository root can be imported by Vercel Lambda
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from medex_ml_api.app import app
