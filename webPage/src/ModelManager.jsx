import React, { useState } from 'react';
import axios from 'axios';

export function ModelUploadComponent({ setModelNameFunc }) {
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
    console.log("Files" + file.name)


    try {
      // Replace with your API endpoint
      const response = await axios.post('http://localhost:8000/reciveModel', formData);
      console.log('File uploaded successfully:', response.data);
      setModelNameFunc(file.name)

    } catch (error) {
      console.error('Error uploading file:', error.message);
    }
  };

  return (
    <div>
      <h3>Upload a keras model</h3>
      <input type="file" onChange={handleFileChange} />
      {selectedFile && (
        <p>Selected file: {selectedFile.name}</p>
      )}
    </div>
  );
}
