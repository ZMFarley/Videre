import { useState, type ChangeEvent } from 'react'
import axios from 'axios';
export default function UploadFile() {
    /* Use State Section */
    const [file, setFile] = useState<File | null>(null);
    const [image, setImage] = useState<string | null>(null);
    const [page, setPage] = useState<"submit" | "loading" | "analysis">("submit");
    /* Image Handling Function Section */
    function handleImageChange(e: ChangeEvent<HTMLInputElement>) {
        if(e.target.files){
            setFile(e.target.files[0]);
            setImage(URL.createObjectURL(e.target.files[0]));
        }
    }
    
    async function uploadImage(){
        if (file){
            const formData = new FormData();
            formData.append("file", file);
            const response = await axios.post("http://localhost:8000/upload-image", formData,
                { headers: {"Content-Type": "multipart/form-data"}}
            )
            console.log(response)
            setPage("loading");
        }
    }   

    return  <div className="center">
        <h1 className="title">Videre: An AI Image Detection Tool</h1>
        <h2>This tool will intake whatever image desired, and determine if it is AI generated or not!</h2>
            {/*Input section for image*/}
            <div className="center">
                {image && (<img alt="Preview image" src={image}/>)}
                <input type="file" accept = "image/*" onChange={handleImageChange}/>
            </div>
            
        <button className="startingButton" onClick={uploadImage}>Click To Submit to the detector</button>
            </div>;
}