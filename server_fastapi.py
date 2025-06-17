import numpy as np
from fastapi import FastAPI, Query, UploadFile, File, Form, HTTPException, WebSocket
from pythonServer.classifyTimeSeries import _classify
from fastapi.middleware.cors import CORSMiddleware
#from pythonServer.getConfidence import get_confidence
from pythonServer.simplification import simplify_ts_by_alpha
#from pythonServer.generateCF import generate_native_cf, generate_subseq_cf
from pythonServer.getTimeSeries import get_time_series
from evaluation import score_different_alphas, score_different_alphas_mp
import shutil
import os
from pathlib import Path
import uuid
import asyncio
import json


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

async def get_loyalty_alphas_csv(task_id: str, dataset: str, dataset_type: str, model_path: str, model_type: str) -> None:
    active_tasks[task_id].update({"status": "processing","progress": 20,"message": "Processing..."})
    try:
        #df, time_dict = score_different_alphas_mp(dataset, datset_type=dataset_type, model_path=model_path)
        df, time_dict = score_different_alphas(dataset, datset_type=dataset_type, model_path=model_path)
        df.to_csv(f"results/{dataset}/{model_type}_alpha_complexity_loyalty.csv", index=False)
        active_tasks[task_id].update({"status": "completed","progress": 100,"message": "Processing completed successfully!"})
    except Exception as e:
        active_tasks[task_id].update({"status": "error","progress": 0,"message": f"Processing failed: {str(e)}"})


@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await websocket.accept()
    try:
        while True:
            if task_id in active_tasks:
                await websocket.send_text(json.dumps(active_tasks[task_id]))
            await asyncio.sleep(1)  # Send updates every second
    except:
        pass

@app.post("/upload")
async def upload_files( model_file: UploadFile = File(...), dataset_file: UploadFile = File(...), dataset_name: str = Form(...)):
    task_id = str(uuid.uuid4())
    active_tasks[task_id] = {"status": "starting","progress": 0,"message": "Initializing upload..."}
    data_path = Path(os.path.join("data", dataset_name))
    model_path = Path(os.path.join("models", dataset_name))
    
    if data_path.exists() or model_path.exists():
        raise HTTPException(status_code=400, detail="Dataset already exists")
    if not model_file.filename:
        raise HTTPException(status_code=400, detail="Model file must have a filename")
    if not dataset_file.filename:
        raise HTTPException(status_code=400, detail="Dataset file must have a filename")
    
    active_tasks[task_id].update({"status": "uploading","progress": 10,"message": "Uploading files"})

    data_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)    
    
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

    
    asyncio.create_task(get_loyalty_alphas_csv(task_id, dataset_name, "TEST", model_file_path_str, model_type))
    
    return {"message": "Upload started", "task_id": task_id}

