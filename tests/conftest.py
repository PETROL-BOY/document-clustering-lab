import sys
from pathlib import Path

# Ensure backend and backend/app are on sys.path
root = Path(__file__).resolve().parent.parent
backend_path = root / "backend"
app_path = root / "backend" / "app"

for p in [str(backend_path), str(app_path)]:
    if p not in sys.path:
        sys.path.insert(0, p)
