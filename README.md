# MOSES
MOSES is a collaboration project between Artelys and the University of Geneva (Energy Efficiency Group) that is funded by SFOE. The main objective is to assess the Impact of the European Market Design Measures on the Swiss Electricity System.


## Project files

- `call_model_moses_project.py`: call model file.
- `call_model_moses_project_nuclear_zero.py`: call model file.
- `model_loop_modifier_moses_project.py`: scenario modifications.
- `grimsel/`: the core model code of GRIMSEL.
- `input_data/`: input data.
- `config_local.py`: configuration.
- `full_model_test_moses_project.sh`: example SLURM submission script to run the model on HPC. 

## Configuration and execution

Review `config_local.py` and adjust its paths and database settings for your environment. Set `MOSES_PSQL_PASSWORD` in your environment. Input data defaults to the repository's `input_data` directory.

The scripts require a compatible Python environment (see requirements). In this project, we used CPLEX as a solver; other solvers may be used, which can be configured under the grimsel folder.
