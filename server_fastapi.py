import numpy as np
from fastapi import FastAPI, Query, UploadFile, File, Form, HTTPException, WebSocket
from pythonServer.classifyTimeSeries import _classify
from fastapi.middleware.cors import CORSMiddleware
#from pythonServer.getConfidence import get_confidence
from pythonServer.simplification import simplify_ts_by_alpha
#from pythonServer.generateCF import generate_native_cf, generate_subseq_cf
from pythonServer.getTimeSeries import get_time_series
import shutil
import os
from pathlib import Path
import uuid
import asyncio
import json
import pandas as pd
import random

from simplifications import get_OS_simplification, get_RDP_simplification, get_bottom_up_simplification, get_VW_simplification, get_LSF_simplification
from Utils.metrics import calculate_mean_loyalty, calculate_kappa_loyalty, calculate_complexity, score_simplicity, calculate_percentage_agreement
from Utils.load_models import model_batch_classify, load_pytorch_model, batch_classify_pytorch_model # type: ignore
from Utils.load_data import load_dataset, load_dataset_labels

from typing import Any

app = FastAPI()

# Where do we accept calls from
origins = [
    "",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def convert_time_series_str_list_float(time_series: str) -> np.ndarray[Any, np.dtype[np.float64]]:
    if time_series is None:
        return np.array([], dtype=float)
    time_series = time_series.replace("[", "").replace("]", "")
    time_series_array = time_series.split(",")
    time_series_array = [float(val) for val in time_series_array]
    time_series_array = np.array(time_series_array)
    return time_series_array


# Security improvments. DO NOT MAKE A FILE BASED ON THE ENTERED NAME!!!
# instead make a map between model name and some basic numbering system.
@ app.post("/reciveDataset")
async def reciveData(file: UploadFile):
    with open(f"pythonServer/utils/csvData/{file.filename}", "wb") as f:
        contents = await file.read()  # read the file
        f.write(contents)
    return {"filename": file.filename}


@ app.post("/reciveModel")
async def reciveModel(file: UploadFile):
    with open(f"pythonServer/KerasModels/models/{file.filename}", "wb") as f:
        contents = await file.read()  # read the file
        f.write(contents)

    return {"filename": file.filename}

@ app.get('/simplification')
async def get_simplification(simp_algo: str, time_series: str, alpha: float):
    if alpha < 0:
        time_series_array = convert_time_series_str_list_float(time_series)
        return time_series_array
    """
    we want to find a counterfactual of the index item to make it positive
    @return A counterfactual time series. For now we only change one time series
    """
    time_series_array = convert_time_series_str_list_float(time_series)

    time_series_array = simplify_ts_by_alpha(algo=simp_algo,alpha=alpha,time_series=time_series_array)
    return time_series_array

@ app.get('/confidence')
async def confidence(time_series: str = Query(None, description=''), dataset_name: str = Query(None, description='')):
    time_series_array = convert_time_series_str_list_float(time_series)
    model_confidence = 1
    return str(model_confidence)

@ app.get('/getClass')
async def get_class(time_series: str = Query(None, description=''), dataset_name: str = Query(None, description='')):
    if time_series == "[0,0]":
        return 0
    time_series_array = convert_time_series_str_list_float(time_series)
    class_of_ts = _classify(dataset_name=dataset_name,
                            time_series=time_series_array)
    print(class_of_ts, type(class_of_ts))
    print("Class:",class_of_ts)
    class_of_ts = str(class_of_ts)
    return class_of_ts


@ app.get('/getTS')
async def get_ts(dataset_name: str = Query(None, description='Name of domain'), index: int = Query(None, description='Index of entry in train data')):
    dataset_name = dataset_name
    time_series = get_time_series(dataset_name=dataset_name,data_type="TEST_normalized", index=index).flatten().tolist()
    return time_series

@ app.get("/")
async def welcome():
    return "Welcom home", 200


"""
Need to execute the upload in background
"""
active_tasks: dict[str, dict] = {}

async def score_different_alphas(task_id: str, dataset_name: str, datset_type: str, model_path: str) -> pd.DataFrame:
    """
    Evaluate the impact of different alpha values on loyalty, kappa, and complexity.
    """
    diff_alpha_values = np.arange(0,1,0.01)     #Bigger steps of alpha skew the results
    df = pd.DataFrame(columns=["Type","Alpha", "Percentage Agreement", "Kappa Loyalty", "Complexity", "Num Segments"])
    all_time_series = load_dataset(dataset_name, data_type=datset_type)
    
    labels = load_dataset_labels(dataset_name, data_type=datset_type)
    num_classes = len(set(labels))
    
    # Algorithms grow in time complexity with the number of time series. If there are too many, we will stratify the data to get 100 samples
    if np.shape(all_time_series)[0] > 100:
        real_shape = np.shape(all_time_series)
        porcentage_data = 100/real_shape[0]
        rand_idx = random.sample(range(real_shape[0]), 100)
        all_time_series = [all_time_series[i] for i in rand_idx]

    predicted_classes_original = get_model_predictions(model_path, all_time_series, num_classes)    #type: ignore
    total_iterations = len(diff_alpha_values)
    
    for i, alpha in enumerate(diff_alpha_values):
        active_tasks[task_id].update({"status": "processing","progress": 20 + (i / total_iterations) * 80,"message": "Processing... Do not refresh or quit this page."})
        await asyncio.sleep(0.1)  
        all_simplifications_OS = get_OS_simplification(time_series=all_time_series, alpha=alpha)    #type: ignore

        #Step 2 get model predictions
        batch_simplified_ts = [ts.line_version for ts in all_simplifications_OS]
        predicted_classes_simplifications_OS = get_model_predictions(model_path, batch_simplified_ts, num_classes) 

        # Step 3 calculate loyalty and complexity
        #mean_loyalty_OS = calculate_mean_loyalty(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_OS)
        kappa_loyalty_OS = calculate_kappa_loyalty(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_OS, num_classes=num_classes)
        percentage_agreement_OS = calculate_percentage_agreement(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_OS)
        complexity_OS = calculate_complexity(batch_simplified_ts=all_simplifications_OS)
        num_segments_OS = np.mean([(len(ts.x_pivots) - 1) for ts in all_simplifications_OS])
        row = ["OS", alpha, percentage_agreement_OS, kappa_loyalty_OS, complexity_OS, num_segments_OS]
        df.loc[len(df)] = row

        all_simplifications_RDP = get_RDP_simplification(time_series=all_time_series, epsilon=alpha)    #type: ignore

        #Step 2 get model predictions
        batch_simplified_ts = [ts.line_version for ts in all_simplifications_RDP]
        predicted_classes_simplifications_RDP = get_model_predictions(model_path, batch_simplified_ts, num_classes)   

        # Step 3 calculate loyalty and complexity
        #mean_loyalty_RDP = calculate_mean_loyalty(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_RDP)
        kappa_loyalty_RDP = calculate_kappa_loyalty(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_RDP, num_classes=num_classes)
        percentage_agreement_RDP = calculate_percentage_agreement(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_RDP)
        complexity_RDP = calculate_complexity(batch_simplified_ts=all_simplifications_RDP)
        num_segments_RDP = np.mean([(len(ts.x_pivots) - 1) for ts in all_simplifications_RDP])
        row = ["RDP", alpha, percentage_agreement_RDP, kappa_loyalty_RDP, complexity_RDP, num_segments_RDP]
        df.loc[len(df)] = row

        all_simplifications_BU = get_bottom_up_simplification(time_series=all_time_series, max_error=alpha) #type: ignore

        # Step 2 get model predictions
        batch_simplified_ts = [ts.line_version for ts in all_simplifications_BU]
        predicted_classes_simplifications_BU = get_model_predictions(model_path, batch_simplified_ts, num_classes)  # I will say this and all_time_series_OS are the same, but just in case

        # Step 3 calculate loyalty and complexity
        #mean_loyalty_BU = calculate_mean_loyalty(pred_class_original=predicted_classes_original,pred_class_simplified=predicted_classes_simplifications_BU)
        kappa_loyalty_BU = calculate_kappa_loyalty(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_BU, num_classes=num_classes)
        percentage_agreement_BU = calculate_percentage_agreement(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_BU)
        complexity_BU = calculate_complexity(batch_simplified_ts=all_simplifications_BU)
        num_segments_BU = np.mean([ts.num_real_segments for ts in all_simplifications_BU])
        row = ["BU", alpha, percentage_agreement_BU, kappa_loyalty_BU, complexity_BU, num_segments_BU]
        df.loc[len(df)] = row

        all_simplifications_VW = get_VW_simplification(time_series=all_time_series,alpha=alpha) #type: ignore

        # Step 2 get model predictions
        batch_simplified_ts = [ts.line_version for ts in all_simplifications_VW]
        predicted_classes_simplifications_VW = get_model_predictions(model_path, batch_simplified_ts, num_classes)  # I will say this and all_time_series_OS are the same, but just in case

        # Step 3 calculate loyalty and complexity
        #mean_loyalty_VW = calculate_mean_loyalty(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_VW)
        kappa_loyalty_VW = calculate_kappa_loyalty(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_VW, num_classes=num_classes)
        percentage_agreement_VW = calculate_percentage_agreement(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_VW)
        complexity_VW = calculate_complexity(batch_simplified_ts=all_simplifications_VW)
        num_segments_VW = np.mean([(len(ts.x_pivots) - 1) for ts in all_simplifications_VW])
        row = ["VW", alpha, percentage_agreement_VW, kappa_loyalty_VW, complexity_VW, num_segments_VW]
        df.loc[len(df)] = row

    return df

def get_model_predictions(model_path: str, batch_of_TS: list[list[float]], num_classes: int) -> list[int]:
    predicted_classes = model_batch_classify(model_path, batch_of_TS, num_classes) 
    return predicted_classes

async def get_loyalty_alphas_csv(task_id: str, dataset: str, dataset_type: str, model_path: str, model_type: str) -> None:
    active_tasks[task_id].update({"status": "processing","progress": 20,"message": "Processing..."})
    try:
        #df, time_dict = score_different_alphas_mp(dataset, datset_type=dataset_type, model_path=model_path)
        df = await score_different_alphas(task_id, dataset, datset_type=dataset_type, model_path=model_path)
        df.to_csv(f"results/{dataset}/{model_type}_alpha_complexity_loyalty.csv", index=False)
        active_tasks[task_id].update({"status": "completed","progress": 100,"message": "Processing completed successfully! You should be able to chose them in the home page."})
    except Exception as e:
        active_tasks[task_id].update({"status": "error","progress": 0,"message": f"Processing failed: {str(e)}"})

"""
async def test_progress_bar(task_id: str, dataset: str, dataset_type: str, model_path: str, model_type: str) -> None:
    active_tasks[task_id].update({"status": "processing", "progress": 20, "message": "Processing..."})
    
    for i in range(5):
        progress = 20 + (i * 15)
        active_tasks[task_id].update({
            "status": "processing",
            "progress": progress,
            "message": f"Test step {i+1} of 5"
        })
        print(f"📝 Test progress {i+1}: {active_tasks[task_id]}")
        await asyncio.sleep(2)  # Wait 2 seconds between updates
    
    
    active_tasks[task_id].update({"status": "completed", "progress": 100, "message": "Processing completed successfully!"})
"""

@app.websocket("/ws/progress/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await websocket.accept()
    print(f"Task: {task_id}")
    print(f"Active tasks: {list(active_tasks.keys())}")
    try:
        while True:
            if task_id in active_tasks:
                #await websocket.send_text(json.dumps(active_tasks[task_id]))
                task_data = active_tasks[task_id]
                print(f"Sending progress: {task_data}")
                await websocket.send_text(json.dumps(task_data))

                if task_data["status"] in ["completed", "error"]:
                    print(f"Task {task_id} finished with status: {task_data['status']}")
                    await asyncio.sleep(1)  
                    break
            else:
                print("Error in websocket")
                await websocket.send_text(json.dumps({
                    "status": "error",
                    "progress": 0,
                    "message": "Task not found"
                }))

                break
            await asyncio.sleep(1)  # Send updates every second
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print(f"WebSocket disconnected for task: {task_id}")

@app.post("/upload")
async def upload_files( model_file: UploadFile = File(...), dataset_file: UploadFile = File(...), dataset_name: str = Form(...)):
    task_id = str(uuid.uuid4())
    active_tasks[task_id] = {"status": "starting","progress": 0,"message": "Initializing upload... Do not refresh or quit this page."}
    print("Initializing upload...")
    data_path = Path(os.path.join("data", dataset_name))
    model_path = Path(os.path.join("models", dataset_name))
    results_path = Path(os.path.join("results", dataset_name))
    
    if data_path.exists() or model_path.exists():
        raise HTTPException(status_code=400, detail="Dataset already exists")
    if not model_file.filename:
        raise HTTPException(status_code=400, detail="Model file must have a filename")
    if not dataset_file.filename:
        raise HTTPException(status_code=400, detail="Dataset file must have a filename")
    
    active_tasks[task_id].update({"status": "uploading","progress": 10,"message": "Uploading files... Do not refresh or quit this page."})
    await asyncio.sleep(0.1) 
    print("Uploading files")

    data_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)    
    results_path.mkdir(parents=True, exist_ok=True)
    
    model_type = model_file.filename.split(".")[0]
    model_file_format = model_file.filename.split(".")[1]
    model_file_name = dataset_name + "_TEST." + model_file_format
    model_file_path = model_path.joinpath(model_file_name)
    model_file_path_str = str(model_file_path)
    with open(model_file_path, "wb") as buffer:
            shutil.copyfileobj(model_file.file, buffer)
    
    dataset_file_format = dataset_file.filename.split(".")[1]
    dataset_file_name = dataset_name + "_TEST." + dataset_file_format
    dataset_file_path = data_path.joinpath(dataset_file_name)
    with open(dataset_file_path, "wb") as buffer:
            shutil.copyfileobj(dataset_file.file, buffer)

    print("Starting background task")
    asyncio.create_task(get_loyalty_alphas_csv(task_id, dataset_name, "TEST", model_file_path_str, model_type))
    #asyncio.create_task(test_progress_bar(task_id, dataset_name, "TEST", model_file_path_str, model_type))
    
    return {"message": "Upload started", "task_id": task_id}

