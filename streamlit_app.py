"""
Streamlit Community Cloud default entrypoint.
Redirects execution to app.py.
"""
import runpy
from pathlib import Path

app_path = Path(__file__).resolve().parent / "app.py"
runpy.run_path(str(app_path), run_name="__main__")
