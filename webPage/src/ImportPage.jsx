import React, { useState, useEffect } from 'react';
import './ImportPage.css';

const ImportPage = () => {
    const [modelFile, setModelFile] = useState(null);
    const [datasetFile, setDatasetFile] = useState(null);
    const [datasetName, setDatasetName] = useState('');
    const [isUploading, setIsUploading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [statusMessage, setStatusMessage] = useState('');
    const [taskId, setTaskId] = useState(null);

    useEffect(() => {
        if (taskId) {
            console.log("Connecting to", taskId)
            const ws = new WebSocket(`ws://localhost:8000/ws/progress/${taskId}`);
            console.log("Connected")
            ws.onmessage = (event) => {
                console.log("Message", event.data)
                const data = JSON.parse(event.data);
                setProgress(data.progress);
                setStatusMessage(data.message);

                if (data.status === 'completed') {
                    setIsUploading(false);
                    console.log("Completed")
                    ws.close();
                } else if (data.status === 'error') {
                    setIsUploading(false);
                    console.log("Error")
                    ws.close();
                }
            };

            return () => {
                console.log("Cleaning Websoket")
                ws.close()
            };
        }
    }, [taskId]);

    const handleFileSelect = (event, type) => {
        const file = event.target.files[0];
        if (type === 'model') {
            setModelFile(file);
        } else {
            setDatasetFile(file);
        }
    };

    const handleSubmit = async () => {
        if (modelFile && datasetFile && datasetName.trim()) {
            setIsUploading(true);
            setProgress(0);
            setStatusMessage('Starting upload...');
            console.log("Starting upload...")

            const formData = new FormData();
            formData.append('model_file', modelFile);
            formData.append('dataset_file', datasetFile);
            formData.append('dataset_name', datasetName);

            try {
                const response = await fetch('http://localhost:8000/upload', {
                    method: 'POST',
                    body: formData,
                });

                const result = await response.json();

                if (response.ok) {
                    setTaskId(result.task_id);
                } else {
                    alert(result.detail || 'Upload failed');
                    setStatusMessage("Error")
                    console.log("Error")
                    setIsUploading(false);
                }
            } catch (error) {
                alert('Upload failed');
                setStatusMessage("Error")
                console.log("Error")
                setIsUploading(false);
            }
        }
    };

    return (
        <div className="import-page">
            <h1>📁 Import Files</h1>

            <div className="upload-grid">
                <div className="upload-card">
                    <h3>🤖 Model File</h3>
                    <input
                        type="file"
                        accept=".pth,.pkl"
                        onChange={(e) => handleFileSelect(e, 'model')}
                        className="file-input"
                    />
                    <p>.pth or .pkl files</p>
                    {modelFile && <div className="file-selected">✅ {modelFile.name}</div>}
                </div>

                <div className="upload-card">
                    <h3>📊 Dataset File</h3>
                    <input
                        type="file"
                        accept=".npy"
                        onChange={(e) => handleFileSelect(e, 'dataset')}
                        className="file-input"
                    />
                    <p>.npy files only</p>
                    {datasetFile && <div className="file-selected">✅ {datasetFile.name}</div>}
                </div>
            </div>

            <div className="dataset-name-section">
                <h3>📝 Dataset Name</h3>
                <input
                    type="text"
                    value={datasetName}
                    onChange={(e) => setDatasetName(e.target.value)}
                    placeholder="Enter dataset name..."
                    className="dataset-name-input"
                />
            </div>
            {isUploading && (
                <div className="progress-section">
                    <div className="progress-bar">
                        <div
                            className="progress-fill"
                            style={{ width: `${progress}%` }}
                        ></div>
                    </div>
                    <p className="progress-message">{statusMessage}</p>
                </div>
            )}
            {!isUploading && statusMessage && (
                <div className={`status-alert ${statusMessage.includes('successfully') ? 'status-success pulse' :
                    statusMessage.includes('Error') || statusMessage.includes('failed') ? 'status-error' : ''}`}>
                    {statusMessage.includes('successfully') ? '🎉 ' : statusMessage.includes('Error') ? '❌ ' : ''}
                    {statusMessage}
                </div>
            )}
            <button
                className="upload-btn"
                onClick={handleSubmit}
                disabled={!modelFile || !datasetFile || !datasetName.trim() || isUploading}
            >
                {isUploading ? 'Processing...' : '🚀 Upload Files'}
            </button>
        </div>
    );
};

export default ImportPage;
