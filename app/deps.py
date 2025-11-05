import os
from pathlib import Path
from dotenv import load_dotenv

def load_env():
    script_dir = Path(__file__).resolve().parent.parent
    env_file = script_dir / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=str(env_file))
    else:
        load_dotenv()
