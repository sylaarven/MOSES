"""Example MOSES configuration. Adjust paths and database settings as needed.

Set MOSES_PSQL_PASSWORD in your environment before using PostgreSQL.
Do not save real credentials in this file.
"""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# Input and working directories; environment variables can override defaults.
PATH_CSV = os.environ.get('MOSES_PATH_CSV', str(PROJECT_DIR / 'input_data'))
BASE_DIR = os.environ.get('MOSES_BASE_DIR', str(PROJECT_DIR / 'build_input'))
FN_XLSX = os.environ.get('MOSES_FN_XLSX', str(PROJECT_DIR / 'input_data' / 'input_levels.xlsx'))

# Example database settings. Configure these for your own installation.
DATABASE = os.environ.get('MOSES_DATABASE', 'moses')
SCHEMA = os.environ.get('MOSES_SCHEMA', 'lp_input_example')
PSQL_USER = os.environ.get('MOSES_PSQL_USER', 'postgres')
PSQL_PASSWORD = os.environ.get('MOSES_PSQL_PASSWORD', '')
PSQL_PORT = int(os.environ.get('MOSES_PSQL_PORT', '5432'))
PSQL_HOST = os.environ.get('MOSES_PSQL_HOST', 'localhost')
