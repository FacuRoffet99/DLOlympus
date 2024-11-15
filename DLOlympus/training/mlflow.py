import mlflow, os
import numpy as np
from datetime import datetime


def mlflow_log(path, hyperparameters, metrics, experiment_name):
    '''
    Log files to MLFlow.

    Args:
        path(str): Folder containing all the files to log.
        hyperparameters (dict): Dictionary containing the names and values of the hyperparameters of the model.
        metrics (dict): Dictionary containing the names and values of the resulting metrics.
        experiment_name (str): Name of the MLFlow experiment to log the files to.
    '''

    mlflow.set_experiment(experiment_name=experiment_name)

    with mlflow.start_run(run_name=datetime.now().strftime('%d/%m/%Y %H:%M')) as run:
        
        # Get run id
        run_id = run.info.run_id
        print(f'Logging started for run {run_id}')

        # Log hyperparameters
        for hp in hyperparameters.keys():
            if isinstance(hyperparameters[hp], list):
                mlflow.log_param(hp, [t.__class__.__name__ for t in hyperparameters[hp]])
            else:
                mlflow.log_param(hp, hyperparameters[hp])
        print('Hyperparameters logged!')

        # Log metrics
        for m in metrics:
            mlflow.log_metric(m, metrics[m])
        print('Metrics logged!')

        # Log artifacts
        for f in [p for p in os.listdir(path) if os.path.isfile(os.path.join(path, f))]:
            mlflow.log_artifact(path+f, '')
        print('Artifacts logged!')
