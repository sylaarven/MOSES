# MOSES
MOSES is a collaboration project between Artelys and University of Geneva (Energy Efficiency Group) which is funded by SFOE. The main objective is to assess the Impact of the European Market Design Measures on the Swiss Electricity System.


## Project files

- `call_model_moses_project.py`: model entry point.
- `call_model_moses_project_nuclear_zero.py`: alternative model entry point.
- `model_loop_modifier_moses_project.py`: scenario modifications.
- `grimsel/`: bundled GRIMSEL model code.
- `input_data/`: model input tables.
- `config_local.py`: portable example configuration, with environment variable overrides.
- `full_model_test_moses_project.sh`: example SLURM submission script from the original computing environment.

## Preparing the input data

The demand table is stored as `input_data/profdmnd.csv.gz` to keep the file small enough for GitHub. Restore it after cloning, from the repository root:

```sh
python3 -c "import gzip, shutil; src=gzip.open('input_data/profdmnd.csv.gz', 'rb'); dst=open('input_data/profdmnd.csv', 'wb'); shutil.copyfileobj(src, dst); src.close(); dst.close()"
```

This restores the original CSV exactly. The extracted file is excluded from Git.

## Configuration and execution

Review `config_local.py` and adjust its paths and database settings for your environment. Set `MOSES_PSQL_PASSWORD` in your environment if PostgreSQL is used; do not commit database credentials. Input data defaults to the repository's `input_data` directory.

The scripts require a compatible Python environment, the imported scientific Python packages (including NumPy, pandas and Pyomo), and the CPLEX solver configured by GRIMSEL. The original cluster script references Python 3.8.2. A validated dependency lock file is not included.

Before running, update the output path `sc_out` in the model entry point and any environment-specific paths in the SLURM script. The model scripts modify some input CSV files during execution; keep a clean copy of the inputs.

The model has not been run or validated as part of this repository upload.
