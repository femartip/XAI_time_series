import torch
from tqdm import tqdm
from torcheval.metrics import BinaryAccuracy, MulticlassAccuracy
import argparse
import Utils.conv_model as conv_model
import numpy as np
import logging
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
import joblib
import pandas as pd

from Utils.plotting import plot_metrics
from Utils.load_data import load_dataset, load_dataset_labels
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

from sktime.classification.kernel_based._rocket_classifier import RocketClassifier

def test_miniRocket(X,y,model):
    accuracy = accuracy_score(y, model.predict(X))
    return {"test_acc": accuracy}
def train_miniRocket(X_train, y_train, X_val, y_val):
    model = RocketClassifier(
        rocket_transform="rocket",
        use_multivariate='no'
    )

    model.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    val_accuracy = accuracy_score(y_val, model.predict(X_val))
    return model, {"train_acc": train_accuracy, "val_acc": val_accuracy}

def test_decision_tree(X, y, model):
    accuracy = accuracy_score(y, (model.predict(X) > 0.5).astype(int))  
    return {"test_acc": accuracy}

def train_decision_tree(X_train, y_train, X_val, y_val):
    model = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, random_state=SEED)
    model.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, (model.predict(X_train) > 0.5).astype(int))
    val_accuracy = accuracy_score(y_val, (model.predict(X_val) > 0.5).astype(int))
    return model, {"train_acc": train_accuracy, "val_acc": val_accuracy}

def test_logistic_regression(X, y, model):
    accuracy = accuracy_score(y, (model.predict(X) > 0.5).astype(int))
    return {"test_acc": accuracy}

def train_logistic_regression(X_train, y_train, X_val, y_val):
    model = LogisticRegression(solver='liblinear', random_state=SEED)
    model.fit(X_train, y_train)
    model.predict(X_train)
    train_accuracy = accuracy_score(y_train, (model.predict(X_train) > 0.5).astype(int))
    val_accuracy = accuracy_score(y_val, (model.predict(X_val) > 0.5).astype(int))
    return model, {"train_acc": train_accuracy, "val_acc": val_accuracy}

def test_knn(X, y, model):
    accuracy = accuracy_score(y, model.predict(X))
    return {"test_acc": accuracy}

def train_knn(X_train, y_train, X_val, y_val):
    model = KNeighborsTimeSeriesClassifier(n_neighbors=5, weights="distance", metric='dtw')
    model.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    val_accuracy = accuracy_score(y_val, model.predict(X_val))
    return model, {"train_acc": train_accuracy, "val_acc": val_accuracy}

def test_conv_model(X, y, model, plot=False):
    num_classes = len(set(y))
    if num_classes == 2:
        X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        X = X.unsqueeze(1)
        y = torch.tensor(y, dtype=torch.float32).to(DEVICE)
    else:
        X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        X = X.unsqueeze(1)
        y = torch.tensor(y, dtype=torch.long).to(DEVICE)
    model.eval()
    with torch.no_grad():
        output = model(X)
        
        if num_classes == 2:
            output = torch.sigmoid(input=output)
            output = output.squeeze(1)
            metric = BinaryAccuracy()
        else:
            output = torch.softmax(input=output, dim=1)
            metric = MulticlassAccuracy(num_classes=num_classes)
        metric.update(output, y)
        if plot:
            print(f'Test Accuracy: {metric.compute()}')
        return {"test_acc": metric.compute()}

def train_conv_model(X,y, X_val, y_val, plot=False):
    learning_rate = 0.01
    epochs = 100
    batch_size = 1024

    train_metrics = []
    train_losses = []
    val_metrics = []
    val_losses = []

    input_size = X.shape[1]
    logging.info("Input size: " + str(input_size))
    num_classes = len(set(y))
    logging.info("Number of classes: " + str(num_classes))
    logging.debug(f"y: {y}")

    if num_classes == 2:
        X_val = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
        X_val = X_val.unsqueeze(1)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    else:
        X_val = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
        X_val = X_val.unsqueeze(1)
        y_val = torch.tensor(y_val, dtype=torch.long).to(DEVICE)
    
    if num_classes == 2:
        model = conv_model.ConvClassifier(num_classes=1)
        model.train()
        model.to(DEVICE)
        criterion = torch.nn.BCEWithLogitsLoss()
        metric = BinaryAccuracy()
        metric_val = BinaryAccuracy()
    else:
        model = conv_model.ConvClassifier(num_classes=num_classes)
        model.train()
        model.to(DEVICE)
        criterion = torch.nn.CrossEntropyLoss()
        metric = MulticlassAccuracy(num_classes=num_classes)
        metric_val = MulticlassAccuracy(num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            X_batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(DEVICE)
            X_batch = X_batch.unsqueeze(1)
            logging.debug("X_batch shape: " + str(X_batch.shape))
            
            X_batch = X_batch.to(DEVICE)
            if num_classes == 2:
                y_batch = torch.tensor(y[i:i+batch_size], dtype=torch.float32).to(DEVICE)
                y_batch = y_batch.unsqueeze(1)
                y_batch = y_batch.clip(0,1)
            else:
                y_batch = torch.tensor(y[i:i+batch_size], dtype=torch.long).to(DEVICE)
            
            logging.debug("y_batch shape: " + str(y_batch.shape))

            optimizer.zero_grad()
            output = model(X_batch)
            logging.debug("Output shape: " + str(output.shape))            
        
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            output = torch.sigmoid(output).squeeze(1) if num_classes == 2 else output 
            y_batch = y_batch.squeeze(1) if num_classes == 2 else y_batch
            metric.update(output, y_batch)

            if i % 1 == 0:
                train_losses.append(loss.item())
                train_metrics.append(metric.compute())
                if plot:
                    print(f'Epoch {epoch}, Loss: {loss.item()}, Accuracy: {metric.compute()}')
                with torch.no_grad():
                    logging.debug("Validation shape: " + str(X_val.shape))
                    val_output = model(X_val.clone().to(DEVICE)) #   torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
                    val_output = torch.squeeze(val_output, 1)
                    val_loss = criterion(val_output, y_val.clone().to(DEVICE))   #torch.tensor(y_val, dtype=torch.float32)
                    metric_val.update(val_output, y_val)
                    val_losses.append(val_loss.item())
                    val_metrics.append(metric_val.compute())
                    if plot:
                        print(f'Validation Loss: {val_loss.item()}, Accuracy: {metric_val.compute()}')

    final_metrics = {"train_acc": metric.compute(), "val_acc": metric_val.compute()}
    if plot:
        print(f'Final Accuracy: {metric.compute()}')
        print(f'Final Validation Accuracy: {metric_val.compute()}')
        plot_metrics(train_metrics, train_losses, val_metrics, val_losses)

    return model, final_metrics    
    
def save_pytorch_model(model, model_path='model.pth'):
    torch.save(model.state_dict(), model_path)

def save_sklearn_model(model, model_path='model.pkl'):
    joblib.dump(model, model_path)


def save_model(model, model_path: str, model_type: str):
    if model_path and model_type == 'cnn':
        save_pytorch_model(model, model_path)
    elif model_path:
        save_sklearn_model(model, model_path)
    else:
        logging.info("Model not saved")

def train_model(dataset_name: str, model_type:str, normalized: bool):
    extra = ("_normalized" if normalized else "")

    X_train = load_dataset(dataset_name=dataset_name, data_type="TRAIN" + extra)
    y_train = load_dataset_labels(dataset_name=dataset_name, data_type="TRAIN" + extra)
    X_val = load_dataset(dataset_name=dataset_name, data_type='VALIDATION' + extra)
    y_val = load_dataset_labels(dataset_name=dataset_name, data_type='VALIDATION'+ extra)
    X_test = load_dataset(dataset_name=dataset_name, data_type='TEST' + extra)
    y_test = load_dataset_labels(dataset_name=dataset_name, data_type='TEST' + extra)

    if model_type == 'cnn':
        model, metrics = train_conv_model(X_train, y_train, X_val, y_val)
        test_metric = test_conv_model(X_test, y_test, model)
        metrics.update(test_metric)
        metrics = {key: value.item() if isinstance(value, torch.Tensor) else value for key, value in metrics.items()}
        return model, metrics
    elif model_type == 'miniRocket':
        model, metrics = train_miniRocket(X_train, y_train, X_val, y_val)
        test_metric = test_miniRocket(X_test, y_test, model)
        metrics.update(test_metric)
        metrics = {key: value.item() if isinstance(value, torch.Tensor) else value for key, value in metrics.items()}
        return model, metrics

    elif model_type == 'decision-tree':
        model, metrics = train_decision_tree(X_train, y_train, X_val, y_val)
        test_metrics = test_decision_tree(X_test, y_test, model)
        metrics.update(test_metrics)
        return model, metrics

    elif model_type == 'logistic-regression':
        model, metrics = train_logistic_regression(X_train, y_train, X_val, y_val)
        test_metrics = test_logistic_regression(X_test, y_test, model)
        metrics.update(test_metrics)
        return model, metrics
    elif model_type == 'knn':
        model, metrics = train_knn(X_train, y_train, X_val, y_val)
        test_metrics = test_knn(X_test, y_test, model)
        metrics.update(test_metrics)
        return model, metrics
    else:
        raise ValueError("Model type not supported")



def old():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', help='Dataset to use, supported: Chinatown, ECG200, ItalyPowerDemand')
    parser.add_argument('--normalized', action='store_true', help='True or False')
    parser.add_argument('--model_type', type=str, help='Type of model to train. Supported: cnn, decision-tree, logistic-regression')
    parser.add_argument('--save_model', action='store_true', help='Save the model in results file')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if args.datasets is not None:
        datasets = args.datasets
    else:
        datasets = [x for x in os.listdir("./data/") if os.path.isdir(f"./data/{x}")]
    strange_results = []
    for dataset in datasets:
        model, metrics = train_model(dataset, args.model_type, args.normalized)
        print(f"Train accuracy: {metrics['train_acc']}, Validation accuracy: {metrics['val_acc']}, Test accuracy: {metrics['test_acc']}")
        logging.info("Model trained")
        if metrics["train_acc"] == 0:
            strange_results.append(dataset)

        if args.save_model:
            model_path = f"models/{dataset}/cnn_norm.pth" if args.normalized else f"models/{dataset}/cnn.pth"
            model_csv = f"results/{dataset}/models.csv"
            if os.path.exists(model_csv):
                model_df = pd.read_csv(model_csv, header=0)
            else:
                model_df = pd.DataFrame(columns=["model_type", "train_acc", "val_acc", "test_acc"])

            if args.model_type not in model_df["model_type"].unique(): 
                model_df.loc[len(model_df)] = [args.model_type, metrics["train_acc"], metrics["val_acc"], metrics["test_acc"]]
                model_df.to_csv(model_csv, index=False)
                save_model(model, model_path, args.model_type)
            else:
                model_df[model_df["model_type"] == args.model_type] = [args.model_type, metrics["train_acc"], metrics["val_acc"], metrics["test_acc"]]
                model_df.to_csv(model_csv, index=False)
                save_model(model, model_path, args.model_type)
            print("Model saved")

    print("Datasets with 0 training accuracy:")
    print(" ".join(strange_results))


def new():
    from generate_user_survey.configurations import selected_datasets_to_be_in_survey
    #datasets = selected_datasets_to_be_in_survey()
    datasets = sorted([x for x in os.listdir("./results/") if os.path.isdir(f"./data/{x}") if x.split("/")[-1] not in ["global_results.ipynb","results.csv","__pycache__"]])
    print(datasets)
    model_type = "miniRocket"
    normalized = True
    strange_results = []
    do_save_model = True
    for dataset in tqdm(datasets):
        print(dataset)
        model, metrics = train_model(dataset, model_type, normalized)
        print(f"Train accuracy: {metrics['train_acc']}, Validation accuracy: {metrics['val_acc']}, Test accuracy: {metrics['test_acc']}")
        logging.info("Model trained")
        if metrics["train_acc"] == 0:
            strange_results.append(dataset)

        if do_save_model:
            model_folder = f"models/{dataset}"
            os.makedirs(model_folder, exist_ok=True)
            model_path = f"{model_folder}/{model_type}{'_norm' if normalized else ''}.pkl"
            csv_folder = f"results/{dataset}"
            os.makedirs(csv_folder, exist_ok=True)
            model_csv = f"{csv_folder}/models.csv"
            if os.path.exists(model_csv):
                model_df = pd.read_csv(model_csv, header=0)
            else:
                model_df = pd.DataFrame(columns=["model_type", "train_acc", "val_acc", "test_acc"])

            if model_type not in model_df["model_type"].unique():
                model_df.loc[len(model_df)] = [model_type, metrics["train_acc"], metrics["val_acc"], metrics["test_acc"]]
            else:
                model_df[model_df["model_type"] == model_type] = [model_type, metrics["train_acc"], metrics["val_acc"], metrics["test_acc"]]

            model_df.to_csv(model_csv, index=False)
            save_model(model, model_path, model_type)
            print("Model saved")

    print("Datasets with 0 training accuracy:")
    print(" ".join(strange_results))


def my_code():
    from generate_user_survey.configurations import selected_datasets_to_be_in_survey
    datasets = selected_datasets_to_be_in_survey()
    for dataset in datasets:
        dataset_name = dataset
        x_test = load_dataset(dataset_name, data_type="TEST_normalized")
        y_test = load_dataset_labels(dataset_name, data_type="TEST_normalized")
        model_path = f"models/{dataset_name}/miniRocket_norm.pkl"
        from Utils.load_models import model_batch_classify
        preds = model_batch_classify(model_path, x_test, 2)
        print(np.array(preds).tolist().count(0), np.array(preds).tolist().count(1))
        print("Accuracy:", accuracy_score(y_test, preds))

if __name__ == '__main__':
    new()
    my_code()



    #test_conv_model(x_test,y_test,model, True)

    #train_miniRocket(x_test, y_test, x_test,y_test)