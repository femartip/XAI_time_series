import React, { useState } from 'react';
import axios from 'axios';

export function DatasetUploadComponent({ setDatasetNameFunc }) {
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    console.log(file)
    if (!file) {
      return
    }
    setSelectedFile(file);
    // Create an object of formData
    const formData = new FormData();
    // Update the formData object
    formData.append(
      "file",
      file,
      file.name
    );




    try {
      // Replace with your API endpoint
      const response = await axios.post('http://localhost:8000/reciveDataset', formData);
      console.log('File uploaded successfully:', response.data);
      setDatasetNameFunc(file.name)

    } catch (error) {
      console.error('Error uploading file:', error.message);
    }
  };


  return (
    <div>
      <h3>Upload a Dataset</h3>
      <input
        type="file" onChange={(event) => { handleFileChange(event) }} />
      {selectedFile ? <p>Selected file: {selectedFile.name} </p> : null}
    </div>
  );
}