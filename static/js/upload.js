const fileForm = document.getElementById("file");
fileForm.addEventListener('change', upload_file);

let file;
let extension;
let data;

function getExtension() {
    file = fileForm.files[0]
    return file.name.split('.')[1]
}

function read_csv() {
    Papa.parse(
        file,
        {
            worker: true,
            dynamicTyping: true,
            complete: results => {
                data = JSON.stringify(results.data);
            }
        });
    };

function read_json() {
    const blob = new Blob([file], {type:"application/json"});
    const reader = new FileReader();
    reader.readAsText(blob);
    reader.onload = () => {data = reader.result}
}

function upload_file() {
    extension = getExtension();
    console.log(extension);

    if (extension === 'json') {
        read_json();
    } else if (extension === 'csv') {
        read_csv();
    }
}