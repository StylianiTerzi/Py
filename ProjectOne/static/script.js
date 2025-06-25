var socket = io();

socket.on("message", function(data) {
    const chatBox = document.getElementById("chat-box");
    const messageElement = document.createElement("p");
    messageElement.innerHTML = `<strong>${data.from}:</strong> ${data.text}`;
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
});


function sendMessage() {
    const msg = document.getElementById("message").value;
    if (msg.trim() !== "") {
        socket.send(msg);
        document.getElementById("message").value = "";
    }
}

function uploadFile() {
    const fileInput = document.getElementById("file-input");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select a file first.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    //console.log("Uploading:", file.name);

    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert("Error: " + data.error);
        } else {
            document.getElementById("file-info").innerHTML =
                `File: ${data.filename} <br> Size: ${data.size} <br> Type: ${data.type}`;
        }
    })
    .catch(error => console.error("Error:", error));
}

