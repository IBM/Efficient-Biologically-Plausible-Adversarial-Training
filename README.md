# **Efficient Biologically Plausible Adversarial Training**
This repository is the official implementation of the
anonymous NeurIPS 2023 submission 
"Efficient Biologically Adversarial Training".

<br/>

## **Details on the code organization**
* `adv_robustness_env.yaml` (yaml): recommend environment for this repository (versions might change depending on your installations)
* `configs` (folder):
    * `table_x` (subfolder, where x can be 1, 2, 3, or 4): contains the yaml files with the different final tuned configs used for the results in Table x of the paper
* `data` (folder): contains the different datasets extracted after running a config (not uploaded)
* `logs` (folder): contains all logs from individual runs of final tuned configs provided the option `save_performances_values is True in the config
    * `net_models` (subfolder): contains the saved trained pytorch networks
    * `results` (subfolder): contains the testing and adversarial accuracy performances for every epoch
* `utils` (folder):
    * `aux_functions.py` (script): contains a progression bar function and the function to load the specified datasets
    * `net_models.py` (script): contains the class instantiating the network models
    * `train_and_test_functions.py` (script): contains training (per epoch) and testing related functions
* `run_main_config.py` (script): runs a specified tuned config
* `main_config.py` (script): contains the main used for the sweep called by the script `run_main_config.py`
* `deploy_config.py` (script): runs all the results reported in the paper by calling the script `run_main_config.py` for each config in configs/table_x

<br/>

## **Install Python packages**
All the needed Python libraries can be installed with conda by running:
```
$ conda env create -f adv_robustness_env.yml
```

## **Running the tuned configs**
To run all the tuned configs for the several implemented methods, run:
```
$ python3 deploy_config.py
```

To run a specific tuned configs, run:
```
$ python3 run_main_config.py
```
and change the `config_name` with the different possible config names in the folder configs.

You can also changes the parameters of each config in the folder `configs` (e.g. to run for more epochs) or add your own.

<br/>

