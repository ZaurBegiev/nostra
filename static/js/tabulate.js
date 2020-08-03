import extension from "upload_data_from_file.js"
var table = new Tabulator("#example-table", {
    height:311,
    layout:"fitColumns",
    autoColumns:true,
    placeholder:"Awaiting Data, Please Load File",
});

document.getElementById("file-load-trigger").addEventListener("click", function(){
    if (extension !== 'json') {
    table.setData(data.data);
    } else {
        table.setData(data)
    }
});