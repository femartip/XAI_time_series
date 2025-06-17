import React, { useEffect, useState } from "react";
import "./styles.css";
import { TrainSetting } from "./TrainSetting";
import { ModelUploadComponent } from "./ModelManager"
import { DatasetUploadComponent } from "./DatasetManager"
import ImportPage from "./ImportPage";

const Header = () => (
    <div className="header">
        <h3>The Prototype (a typical example) is now Interactive.</h3>
        <h3>Click and drag on the black dots to change the Interactive time series.</h3>
        <h3>When you are done, CLOSE this window. </h3>
    </div>
);

export default () => {
    const [datasetName, setDatasetName] = useState(null);
    const [instanceNumber, setInstanceNumber] = useState(0);
    const [simpMethod, setSimpMethod] = useState("RDP");
    const [alphaValue, setAlphaValue] = useState(0)
    const [currentPage, setCurrentPage] = useState('home');

    const setDatasetNameFunc = (name) => {
        setDatasetName(name);
    }
    const setInstanceNumberFunc = (number) => {
        setInstanceNumber(number);
    }

    const setSimplificationMethod = (name) => {
        setSimpMethod(name);
    }

    const setAlphaValueFunc = (number) => {
        setAlphaValue(number);
    }

    return (
        <div className="App">
            {currentPage === 'home' ? (
                <>
                    {/* Add navigation button */}
                    <div style={{ padding: '10px' }}>
                        <button onClick={() => setCurrentPage('import')}>
                            Go to Import Page
                        </button>
                    </div>

                    <div className="float-container">
                        <div className="float-right">
                            <h3> Select Dataset</h3>
                            <select name="cars" id="cars" defaultValue={"Chinatown"} onInputCapture={(event) => { setDatasetNameFunc(event.target.value) }}>
                                <option value="Chinatown" >Chinatown</option>
                                <option value="ItalyPowerDemand">ItalyPowerDemand</option>
                                <option value="ECG200">ECG200</option>
                            </select>
                        </div>

                        <div className="float-right">
                            <h3>Select instance number</h3>
                            <input type="number" defaultValue={instanceNumber} onInput={(event) => {
                                setInstanceNumberFunc(event.target.value)
                            }} />
                        </div>
                        <div className="float-right">
                            <h3> Select Simplification method</h3>
                            <select name="cars" id="cars" defaultValue={"RDP"} onInputCapture={(event) => { setSimplificationMethod(event.target.value) }}>
                                <option value="RDP" >RDP</option>
                                <option value="VW">VW</option>
                                <option value="OS">OS</option>
                                <option value="Bottom-up">BU</option>
                                <option value={"LSF"}>LSF</option>
                            </select>
                        </div>
                        <div className="float-right">
                            <h3>Select alpha value</h3>
                            <input type="number" defaultValue={instanceNumber} onInput={(event) => {
                                setAlphaValueFunc(event.target.value)
                            }} />
                        </div>
                    </div>
                    <div className="InteractiveTool">
                        {(datasetName) ?
                            <TrainSetting datasetName={datasetName} instanceNumber={instanceNumber} simpMethod={simpMethod} alphaValue={alphaValue} /> :
                            <div />}
                    </div>
                </>
            ) : (
                /* Import Page with back button */
                <div>
                    <button onClick={() => setCurrentPage('home')} style={{ margin: '10px' }}>
                        Back to Home
                    </button>
                    <ImportPage />
                </div>
            )}
        </div>
    );

};


