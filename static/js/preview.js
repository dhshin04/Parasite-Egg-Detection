/* Preview Image - read file input & create new image in HTML */
// Called when new file input OR form submitted and file upload is cleared
function preview(event) {
    console.log('Some change')
    const fileInputs = event.target.files;

    // There is new file input
    if (fileInputs.length > 0) {
        const fileInput = fileInputs[0];
        const fileReader = new FileReader();

        // Callback function to execute after fileReader.readAsDataURL
        // Onload is parsed first because file-read must happen after the onload is defined for onload to properly run
        fileReader.onload = (e) => {
            document.getElementById(
            "img-preview"
            ).innerHTML = `<img id="preview" src="${e.target.result}" alt="Image Preview" />`;
            document.getElementById("fec").style.visibility = "hidden";
        };

        // FileReader reads --> callback function above is executed after
        fileReader.readAsDataURL(fileInput);
    } else {
        // No New File Input - form submitted and file upload is cleared
        document.getElementById("fec").style.visibility = "visible";
    }
}

document
.getElementById("fec-image1")
.addEventListener("change", preview);

document
.getElementById("fec-image2")
.addEventListener("change", preview);