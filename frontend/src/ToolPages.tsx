import { useState, type ChangeEvent } from 'react'
import axios from 'axios';
import loadingSpinner from "./loading_spinner.svg"
/* Type to hold prediction from model from API request */
type Prediction = {
    class: number;
    probReal: number;
    probAI: number;
};

export default function UploadFile() {
    /* Use State Section */
    const [file, setFile] = useState<File | null>(null); /* State to hold image file */
    const [image, setImage] = useState<string | null>(null); /* State to hold image file */
    const [page, setPage] = useState<"submit" | "loading" | "analysis">("submit"); /* State to hold current active page for display*/
    const [response, setResponse] = useState<Prediction | null>(null); /* State to hold output of classifier from API request */
    /* Image Handling Function Section */
    function handleImageChange(e: ChangeEvent<HTMLInputElement>) {
        //Accepts image from file explorer, stores its url for later use
        if(e.target.files){
            setFile(e.target.files[0]);
            setImage(URL.createObjectURL(e.target.files[0]));
        }
    }
    
    /* API POST request to send image and recieve prediction */
    async function uploadImage(){
        //Update page to prevent reinput of image
        setPage("loading")
        if (file){
            const formData = new FormData();
            formData.append("file", file);
            const response = await axios.post("http://localhost:8000/predict", formData,
                { headers: {"Content-Type": "multipart/form-data"}}
            )
            // Adjust api response for proper intake and display 
            setResponse({class: response.data.class, probReal: response.data.probability_real, probAI: response.data.probability_ai});
            //Update page after analyization and prevent image from remaining upon return to original screen
            setPage("analysis");
            setImage(null);
        }
    }   

    return (
        <div>
            {/*Section for image input and submission to model*/}
            {page === "submit" && (
                <div className="center">
                    <h1 className="title">AI Image Detection Tool</h1>
                    <h2>This tool will intake whatever image desired, and determine if it is AI generated or not!</h2>
                    {/*Input section for image*/}
                    <div className="center">
                        {/*Prompt user to input image, and display it on the screen*/}
                        {image && (<img alt="Preview image" src={image} width="500" height="auto"/>)}
                        <input type="file" accept = "image/*" onChange={handleImageChange}/>
                    </div>
            
                    <button className="startingButton" onClick={uploadImage}>Click To Submit to the detector</button>
                </div> )}

                {/*Loading screen to prevent additional inputs while the client is waiting for response*/}
                {page === "loading" && (
                <div className="center">
                    {/*SVG sourced from https://magecdn.com/tools/svg-loaders/cog05/*/}
                    {image && (<img alt="loading spinner" src={loadingSpinner} width="500" height="500"/>)}
                </div> )}
                
                {/*Section for model prediction output*/}
                {page === "analysis"  && (
                <div className="center">
                    <h1 className="title">Results</h1>
                    {/*Output section for results*/}
                    <div className="center">
                        {/*Conditionals to output model predictions*/}
                        {response?.class === 0  && <h2>The model has predicted this image is NOT AI Generated</h2>}
                        {response?.class === 1  && <h2>The model has predicted this image is AI Generated</h2>}
                        {response?.class === 0  && <h3>with {(response?.probReal * 100).toFixed(2)}% certainty</h3>}
                        {response?.class === 1  && <h3>with {(response?.probAI * 100).toFixed(2)}% certainty</h3>}

                    </div>  
            
                    <button className="startingButton" onClick={() => setPage("submit")}>Click to test another image!</button>
                </div> )}

        </div>
    );
    
}