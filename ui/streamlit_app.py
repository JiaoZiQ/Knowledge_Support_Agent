from pathlib import Path
import runpy


ROOT_APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"
runpy.run_path(str(ROOT_APP), run_name="__main__")
