import mlflow 
from datetime import date
from sklearn.metrics import log_loss
from utils.names import MLFLOW_TRACKING_URI



def run_mlflow(model, model_name : str, y_test, y_pred_prob) -> None:
        
        
    mlflow_uri =  MLFLOW_TRACKING_URI
    run_name = model_name + '_' + str(date.today())

    mlflow.set_experiment('Cirrhosis classification')
    mlflow.set_tracking_uri(mlflow_uri)
    loss = log_loss(y_test, y_pred_prob)


    with mlflow.start_run(run_name = run_name): 
        
        mlflow.log_metric('log_loss' ,loss)  

        mlflow.log_params( dict({param: value for param, value in model.get_params().items()}))

        mlflow.end_run() 