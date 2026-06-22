# DBRA Experiment Repository

This folder contains all experiments related to the DBRA paper. The `baseline` directory includes methods such as ITDS and ESMA; however, ESMA was not used in the paper because it does not satisfy both transferability and targeted attack properties.

> Due to GitHub limitations, the pre-trained models and partitioned datasets have been uploaded to [datasets](https://huggingface.co/datasets/qwdugbk/DBRA) and [models](https://huggingface.co/qwdugbk/DBRA-models). Place the corresponding model files in the folders named after each experiment and adjust the paths as needed.

If you have questions, please contact: zecliu@whu.edu.cn

## Directory Overview

- `observation_experiment`: contains cross-model transfer experiments.
- `attack`: contains the main attack experiments, ablation studies, parameter analysis, and defense experiments.
- `confs`: contains configuration files for devices, attack settings, etc. The default configurations match the main experiments.
- `utils`: contains model loading, dataset loading, and other utilities. Because finetuning algorithms store models differently than models trained by PFLlib, you may need to adjust the loader in `utils/pfl_dataset_utils.py`.

## Before Running

Make sure you have checked and updated path settings for logs, datasets, and model files in each script before running.

## Run Instructions

### Cross-model transfer validation experiments

Run one of the following files directly:

- `observation_experiment/transfer_experiments_cifar100.py`
- `observation_experiment/transfer_experiments_miniimg.py`

### Main attack experiments

Run the appropriate script for the dataset:

- `attack/attack_executor_{dataset}.py`

### Parameter analysis experiments

1. Update dataset paths in `confs/data_conf.py`.
2. Update model paths in `utils/load_utils.py`.
3. For attacker parameter experiments, adjust parameters in `confs/implantation_confs.py`.
4. For ablation analysis and attacker data experiments, run:
   - `attack/dbra_parallel.py`
   - `attack/ablation_parallel.py`
5. For victim dataset condition analysis, run:
   - `attack/attack_executor_cifar10.py`

### Comparison experiments with Adversary Sample

1. Configure the defense method in `attack/attack_all_dbra.py` (default is JPEG compression around line 152).
2. Start the baseline experiments:
   - Navigate to `baseline/ITDS_main/`
   - For parameter analysis, run `attack_fl_param_analysis.py` (ITDS is very slow).
   - To use pre-tested parameters directly, run `attack_fl.py`.

### Finetuning experiments

1. Modify the model loading method and path as described above.
2. Run `attack/attack_finetuned_model.py`.
