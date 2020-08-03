const fileForm = document.getElementById("file");
fileForm.addEventListener('change', parse)

let data;
let extension;

function getExtension(file) {
    return file.name.split('.')[1]
}

function parse() {
    let file = fileForm.files[0];
    let extension = getExtension(file);

    if (extension !== 'json') {
        Papa.parse(
            file,
            {
                worker: true,
                dynamicTyping: true,
                complete: results => {
                    data = results;
                    console.log(data);
                }
            });
    }
    else {
        data = file;
    };
};