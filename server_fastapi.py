import numpy as np
from fastapi import FastAPI, Query, UploadFile, File
from pythonServer.classifyTimeSeries import _classify
from Utils.load_data import get_time_series
from fastapi.middleware.cors import CORSMiddleware
from pythonServer.getConfidence import get_confidence
from pythonServer.simplification import simplify_ts
from pythonServer.generateCF import generate_native_cf, generate_subseq_cf
import shutil


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

    time_series_array = simplify_ts(algo=simp_algo,alpha=alpha,time_series=time_series_array)
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
    time_series = get_time_series(dataset_name=dataset_name,data_type="TEST", instance_nr=index).flatten().tolist()
    return time_series

@ app.get("/")
async def welcome():
    return "Welcom home", 200
