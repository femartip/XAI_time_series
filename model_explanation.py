import torch
from torcheval.metrics import BinaryAccuracy
import argparse
import logging
import numpy as np
from sklearn.model_selection import train_test_split

from Utils.load_data import load_dataset, load_dataset_labels
from ORSalgorithm.ORSalgorithm import ORSalgorithm
from conv_model import ConvClassifier
from Utils.plotting import plot_metrics

logging.basicConfig(level=logging.INFO)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""
Given an complex AI model M_AI for TS classification and an unsupervised TS simplification algorithm  S ( like the one we used in the sketch paper). The idea is the following:
Main goal
Try to select a set of simplified examples X from which we are able to explain M_AI to humans.
In order to find a good set X, we propose to test the representativeness of X on learning a simple model M_ML from ), and analyse its fidelity wrt to D. 
In a similar way we did with boolean functions.
Steps:
Given a dataset D
Given M_AI
Loop: Extract prototypes X from D. Learn M_ML from S(X), and test the performance wrt S(D) and M_AI until a certain level of accuracy is reached.
Test the capability of the selected S(X) with humans for teaching M_AI.
"""

def simplify_ts(time_series:np.ndarray, blackbox_model_path:str, k:int, alpha:float):
    """
    Create a simplified version (X) of the time series (D) using the ORS algorithm given a blackbox model (M_AI).
    """
    prototypes, confidences, prot_pred_labels = ORSalgorithm(time_series, blackbox_model_path, k=k, alpha=alpha)      #Extract prototypes from blackbox model
    prototypes, confidences, prot_pred_labels = np.array(prototypes), np.array(confidences), np.array(prot_pred_labels)

    logging.info(f"Prototypes shape: {prototypes.shape}")
    logging.info(f"Confidences shape: {confidences.shape}")
    logging.info(f"Average confidence of the prototypes: {np.mean(confidences)}")

    return prototypes, confidences, prot_pred_labels

def train_machine_teaching_model(prototypes:np.ndarray, prot_labels:np.ndarray, orig_data:np.ndarray, orig_labels:np.ndarray):
    """
    Learn a machine teaching model (M_ML) from the prototypes S(x).
    """

    X_train = prototypes
    y_train = prot_labels
    X_val = torch.tensor(orig_data, dtype=torch.float32).to(DEVICE)
    X_val = X_val.unsqueeze(1)
    y_val = torch.tensor(orig_labels, dtype=torch.float32).to(DEVICE)

    learning_rate = 0.01
    epochs = 100
    batch_size = 1024

    m_ml = ConvClassifier()
    m_ml.train()
    m_ml.to(DEVICE)
    optimizer = torch.optim.Adam(m_ml.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()
    metric = BinaryAccuracy()
    metric_val = BinaryAccuracy()

    train_metrics = []
    train_losses = []
    val_metrics = []
    val_losses = []

    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            X_batch = torch.tensor(X_train[i:i+batch_size], dtype=torch.float32).to(DEVICE)
            X_batch = X_batch.unsqueeze(1)
            logging.info("X_batch shape: " + str(X_batch.shape))
            logging.debug(f"X_batch: {X_batch}")
            X_batch = X_batch.to(DEVICE)
            y_batch = torch.tensor(y_train[i:i+batch_size], dtype=torch.float32).to(DEVICE)
            logging.info("y_batch shape: " + str(y_batch.shape))

            optimizer.zero_grad()
            output = m_ml(X_batch)
            output = torch.squeeze(output, 1)
            logging.info("Output shape: " + str(output.shape))
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            #scheduler.step(loss)
            metric.update(output, y_batch)

            if i % 1 == 0:
                train_losses.append(loss.item())
                train_metrics.append(metric.compute())
                print(f'Epoch {epoch}, Loss: {loss.item()}, Accuracy: {metric.compute()}')
                with torch.no_grad():
                    logging.info("Validation shape: " + str(X_val.shape))
                    val_output = m_ml(X_val)
                    val_output = torch.squeeze(val_output, 1)
                    val_loss = criterion(val_output, y_val)
                    metric_val.update(val_output, y_val)
                    val_losses.append(val_loss.item())
                    val_metrics.append(metric_val.compute())
                    print(f'Validation Loss: {val_loss.item()}, Accuracy: {metric_val.compute()}')

    print(f'Final Accuracy: {metric.compute()}')
    print(f'Final Validation Accuracy: {metric_val.compute()}')

    plot_metrics(train_metrics, train_losses, val_metrics, val_losses)

    return m_ml
        

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_name", type=str, default="Chinatown", help="Name of the dataset, can be either Chinatown, ECG200 or ItalyEnergy")
    args.add_argument("--blackbox_model_path", type=str, help="Path to the pytorch model")
    args.add_argument("--k", type=int, default=10000, help="Number of prototypes to extract")
    args.add_argument("--alpha", type=float, default=0.02, help="Alpha value for the ORS algorithm")
    args.add_argument("--save_results", type=bool, default=False, help="Save the results to a file")
    args = args.parse_args()

    time_series = load_dataset(args.dataset_name, data_type="TRAIN")
    labels_ts = load_dataset_labels(args.dataset_name, data_type="TRAIN")

    prototypes, confidences, prot_pred_labels = simplify_ts(time_series, args.blackbox_model_path, k=args.k, alpha=args.alpha)
    
    if args.save_results:
        np.save(f"./data/{args.dataset_name}/{args.dataset_name}_PROTOTYPES_{args.alpha}.npy", prototypes)
        np.save(f"./data/{args.dataset_name}/{args.dataset_name}_CONFIDENCES_{args.alpha}.npy", confidences)

    mt_model = train_machine_teaching_model(prototypes, labels_ts, time_series, labels_ts)
