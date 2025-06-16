import React, { useEffect, useState } from 'react';
import DraggableGraph from './DraggableData';
import axios from 'axios';
import BasicExample from "./MyProgressBar";


export const TrainSetting = ({ datasetName, instanceNumber, simpMethod, alphaValue }) => {

    const addr = "localhost"
    const port = "8000"
    const url_and_port = 'http://' + addr + ':' + port + '/'
    const make_full_url = (endpoint) => {
        return url_and_port + endpoint
    };

    const simp_mode = simpMethod
    const color_class_map = {
        "0": "rgba(0,100,255,0.5)",
        "1": "rgba(217,2,250,0.5)"
    }


    const updateColor = (timeSeries, colorSet, dataSetName) => {
        if (!timeSeries) {
            return;
        }
        axios.get(make_full_url('getClass'), {
            params: {
                time_series: JSON.stringify(timeSeries),
                dataset_name: dataSetName
            }
        })
            .then((res) => {
                // Change the color based on the response
                colorSet(color_class_map[res.data]);


            })
            .catch((error) => {
                console.error('Error:', error);
            });
    };

    const updateData = (dataSet, setData) => {
        if (Array.isArray(dataSet)) {
            setData([...dataSet]);
        } else if (dataSet) {
            console.error('Error: updateData was called with a non-array value');
        } else {
            return;
        }
    }

    // To recall the orginal data
    const [lineColorOrg, setLineColorOrg] = useState(
        "rgba(159,159,171,0.25)"
    )
    const [dataSetOriginal, setDataSetOriginal] = useState(
        null // Replace with python call
    )
    const getOrgData = () => {
        console.log(make_full_url('getTS'))
        axios.get(make_full_url('getTS'), {
            params: {
                dataset_name: datasetName,
                index: instanceNumber,//73=0 171=1// Convert dataSet to a JSON string
            }
        })
            .then((res) => {
                console.log(res.data);
                setDataSetOriginal(res.data);
            })
            .catch((error) => {
                console.error('Error getting org:', error);
            });
    };

    useEffect(() => {
        getOrgData();
    }, [datasetName, instanceNumber]);





    // Movable data
    const [dataSetCurr, setDataSetCurr] = useState(null);
    const [lineColorCurr, setLineColorCurr] = useState("rgba(159,159,171,0.25)");
    useEffect(() => { updateColor(dataSetCurr, setLineColorCurr,datasetName); }, [dataSetCurr, datasetName]);
    useEffect(() => { updateData(dataSetOriginal, setDataSetCurr); }, [dataSetOriginal]); // If change original, update moveable also

    // Simplification data
    const [dataSetSimp, setDataSetSimp] = useState(null)
    const [lineColorSimp, setLineColorSimp] = useState("rgba(159,159,171,0.25)");

    const getSimpData = (dataSetCurr,simp_mode, alpha, dataset_name) => {
        if (!dataSetCurr || !simp_mode) {
            return;
        }
        axios.get(make_full_url('simplification'), {
            params: {
                time_series: JSON.stringify(dataSetCurr),// Convert dataSet to a JSON string
                simp_algo: simp_mode,
                alpha: alpha,
                dataset_name: dataset_name
            }
        })
            .then((res) => {

                console.log("Simp from python:", res.data);

                // Display new counterfactual data
                setDataSetSimp([...res.data]);


            })
            .catch((error) => {
                console.error('Error:', error);
            });
    };
    useEffect(() => { getSimpData(dataSetCurr,simp_mode, alphaValue, datasetName); }, [dataSetCurr, simp_mode, alphaValue, datasetName]);
    useEffect(() => { updateColor(dataSetSimp, setLineColorSimp, datasetName) }, [dataSetSimp, datasetName]);



    const reset = () => {
        if (dataSetOriginal) {
            updateData(dataSetOriginal, setDataSetCurr)
        }
    }
    const updateConfidence = (setConfidence, timeseries, datasetName) => {
        if (!timeseries || !datasetName) {
            return;
        }
        axios.get(make_full_url('confidence'), {
            params: {
                time_series: JSON.stringify(timeseries),// Convert dataSet to a JSON string
                dataset_name: datasetName
            }
        })
            .then((res) => {
                console.log(res.data)
                // Display new counterfactual data
                const confidence = parseFloat(res.data) * 100; // Percentage
                const confidence_one_dec = Math.round(confidence * 10) / 10; // one decimal
                setConfidence(confidence_one_dec);


            })
            .catch((error) => {
                console.error('Error:', error);
            });
    };

    const [confidence, setConfidence] = useState(100);
    useEffect(() => {
        updateConfidence(setConfidence, dataSetCurr, datasetName);
    }, [dataSetCurr, datasetName]);

    const button_show = true
    return (
        <div>
            {false && <BasicExample currValue={confidence} />}
            <DraggableGraph dataSetCurrent={dataSetCurr} setDataCurrent={setDataSetCurr}
                dataSetOriginal={dataSetOriginal} updateData={updateData} dataSetSimp={dataSetSimp}
                lineColorCurr={lineColorCurr} lineColorOrg={lineColorOrg} lineColorSimp={lineColorSimp}
            />
            {(button_show) ?
            <button className={"button"} onClick={reset} >RESET TO PROTOTYPE</button> :
            <div/>}
        </div>
    );
};

