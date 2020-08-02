const myForm = document.getElementById('myForm')
const inputFile = document.getElementById('inputFile')

let dataset = ''

myForm.addEventListener('submit', e => {
    e.preventDefault();

    const endpoint = '/upload_dataset';
    const formData = new FormData();
    formData.append('inputFile', inputFile.files[0]);

    fetch(endpoint, {
        method: 'POST',
        body: formData
    }).catch(console.error);

    dataset = inputFile.files[0];

});