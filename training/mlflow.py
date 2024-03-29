import mlflow
import numpy as np
from datetime import datetime
from .singlelabelclassification import get_predictions_table, fastai2onnx
from .plots import plot_confusion_matrix, plot_metrics, plot_losses


def mlflow_train(learn, hyperparameters, callbacks, metrics_names, path, experiment_name, with_onnx=False):

    mlflow.set_experiment(experiment_name=experiment_name)

    with mlflow.start_run(run_name=datetime.now().strftime('%d/%m/%Y %H:%M')) as run:

        # Get run id
        run_id = run.info.run_id

        # Log hyperparameters
        for hp in hyperparameters.keys():
            if isinstance(hyperparameters[hp], list):
                mlflow.log_param(hp, [t.__class__.__name__ for t in hyperparameters[hp]])
            else:
                mlflow.log_param(hp, hyperparameters[hp])

        # Train model
        learn.fine_tune(hyperparameters['EPOCHS'], base_lr=hyperparameters['LR'], cbs=callbacks)

        # Create and log figures
        try:
            _ = plot_losses(learn, path)
            _ = plot_metrics(learn, metrics_names, path)
            probs, ground_truths = learn.get_preds(ds_idx=1)
            predictions = np.argmax(probs, axis=1)
            _ = plot_confusion_matrix(ground_truths, predictions, learn.dls.vocab, path)
            mlflow.log_artifact(f'{path}losses.png', '')
            mlflow.log_artifact(f'{path}metrics.png', '')
            mlflow.log_artifact(f'{path}confusion.png', '')
        except Exception:
            print('ERROR: the figures were not properly created or logged') 

        # Create and log tables
        try:
            train_table = get_predictions_table(learn, learn.dls.train)
            valid_table = get_predictions_table(learn, learn.dls.valid)
            train_table.to_csv(f'{path}train_table.csv', index=False)
            valid_table.to_csv(f'{path}valid_table.csv', index=False)
            mlflow.log_artifact(f'{path}train_table.csv', '')
            mlflow.log_artifact(f'{path}valid_table.csv', '')
        except Exception:
            print('ERROR: the tables were not properly created or logged') 

        # Log metrics
        results = learn.validate()
        metrics = results[1:]
        for name, met in zip(metrics_names, metrics):
            mlflow.log_metric(name, met)

        if with_onnx:
            # Get input size for the model
            height, width = hyperparameters['IMG_SIZE']
            # Convert FastAI model to ONNX model and export it
            fastai2onnx(learn, path, height, width)
            # Log model
            mlflow.log_artifact(f'{path}model.onnx', '')
        
        # Export fastai model
        learn.export(f'{path}model.pkl')
        # Log model
        mlflow.log_artifact(f'{path}model.pkl', '')

        return train_table, valid_table
