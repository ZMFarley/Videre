import { useState, type ChangeEvent } from 'react'
import axios from 'axios';
import loadingSpinner from "./loading_spinner.svg"
type Prediction = {
    class: number;
    probReal: number;
    probAI: number;
};

export default function UploadFile() {
    /* Use State Section */
    const [file, setFile] = useState<File | null>(null);
    const [image, setImage] = useState<string | null>(null);
    const [page, setPage] = useState<"submit" | "loading" | "analysis">("submit");
    const [response, setResponse] = useState<Prediction | null>(null);
    /* Image Handling Function Section */
    function handleImageChange(e: ChangeEvent<HTMLInputElement>) {
        if(e.target.files){
            setFile(e.target.files[0]);
            setImage(URL.createObjectURL(e.target.files[0]));
        }
    }
    
    async function uploadImage(){
        setPage("loading")
        if (file){
            const formData = new FormData();
            formData.append("file", file);
            const response = await axios.post("http://localhost:8000/predict", formData,
                { headers: {"Content-Type": "multipart/form-data"}}
            )
            setResponse({class: response.data.class, probReal: response.data.probability_real, probAI: response.data.probability_ai});
            setPage("analysis");
            setImage(null);
        }
    }   

    return (
        <div>
            {page === "submit" && (
                <div className="center">
                    <h1 className="title">Videre: An AI Image Detection Tool</h1>
                    <h2>This tool will intake whatever image desired, and determine if it is AI generated or not!</h2>
                    {/*Input section for image*/}
                    <div className="center">
                        {image && (<img alt="Preview image" src={image} width="500" height="auto"/>)}
                        <input type="file" accept = "image/*" onChange={handleImageChange}/>
                    </div>
            
                    <button className="startingButton" onClick={uploadImage}>Click To Submit to the detector</button>
                </div> )}


                {page === "loading" && (
                <div className="center">
                    {image && (<img alt="loading spinner" src={loadingSpinner} width="500" height="500"/>)}
                </div> )}

                {page === "analysis"  && (
                <div className="center">
                    <h1 className="title">Results</h1>
                    {/*Output section for results*/}
                    <div className="center">
                        {response?.class === 0  && <h2>The model has predicted this image is REAL</h2>}
                        {response?.class === 1  && <h2>The model has predicted this image is FAKE</h2>}
                        {response?.class === 0  && <h3>with {(response?.probReal * 100).toFixed(2)}% certainty</h3>}
                        {response?.class === 1  && <h3>with {(response?.probAI * 100).toFixed(2)}% certainty</h3>}

                    </div>  
            
                    <button className="startingButton" onClick={() => setPage("submit")}>Click to test another image!</button>
                </div> )}

        </div>
    );
    
}