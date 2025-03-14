import torch
from torcheval.metrics import BinaryAccuracy, MulticlassAccuracy
import argparse
import Utils.conv_model as conv_model
import numpy as np
import logging
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

from Utils.plotting import plot_metrics
from Utils.load_data import load_dataset, load_dataset_labels

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

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

def test_conv_model(X, y, model, plot=False):
    X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    X = X.unsqueeze(1)
    y = torch.tensor(y, dtype=torch.float32).to(DEVICE)
    model.eval()
    with torch.no_grad():
        output = model(X)
        output = torch.squeeze(output, 1)
        if len(set(y)) == 2:
            metric = BinaryAccuracy()
        else:
            metric = MulticlassAccuracy()
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

    X_val = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    X_val = X_val.unsqueeze(1)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    input_size = X.shape[1]
    logging.info("Input size: " + str(input_size))
    num_classes = len(set(y))
    logging.info("Number of classes: " + str(num_classes))
    logging.debug(f"y: {y}")

    model = conv_model.ConvClassifier()
    model.train()
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if num_classes == 2:
        criterion = torch.nn.BCELoss()
        metric = BinaryAccuracy()
        metric_val = BinaryAccuracy()
    else:
        criterion = torch.nn.CrossEntropyLoss()
        metric = MulticlassAccuracy()
        metric_val = MulticlassAccuracy()
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            X_batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(DEVICE)
            X_batch = X_batch.unsqueeze(1)
            logging.debug("X_batch shape: " + str(X_batch.shape))
            logging.debug(f"X_batch: {X_batch}")
            X_batch = X_batch.to(DEVICE)
            y_batch = torch.tensor(y[i:i+batch_size], dtype=torch.float32).to(DEVICE)
            #y_batch = y_batch.unsqueeze(1)
            logging.debug("y_batch shape: " + str(y_batch.shape))

            optimizer.zero_grad()
            output = model(X_batch)
            output = torch.squeeze(output, 1)
            logging.debug("Output shape: " + str(output.shape))
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            #scheduler.step(loss)
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
    else:
        raise ValueError("Model type not supported")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help='Dataset to use, supported: Chinatown, ECG200, ItalyPowerDemand')
    parser.add_argument('--normalized', action='store_true', help='True or False')
    parser.add_argument('--model_type', type=str, help='Type of model to train. Supported: cnn, decision-tree, logistic-regression')
    parser.add_argument('--model_file_name', type=str, help='Path to save the model')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if args.dataset_name not in ['Chinatown', 'ECG200', 'ItalyPowerDemand']:
        logging.error("Dataset not supported")
        exit(1)

    model, metrics = train_model(args.dataset_name, args.model_type, args.normalized)
    print(f"Train accuracy: {metrics['train_acc']}, Validation accuracy: {metrics['val_acc']}, Test accuracy: {metrics['test_acc']}")
    logging.info("Model trained")
    if args.model_file_name:
        save_model(model, args.model_file_name, args.model_type)
    