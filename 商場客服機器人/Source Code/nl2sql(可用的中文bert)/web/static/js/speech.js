let recognizer;
let transcription = '';
let recordedBlobs;
let isRecording = false;
let mediaRecorder = null;

const input = document.querySelector('#text-input');
const button = document.querySelector('#send-button');
const voiceButton = document.querySelector('#voice-button');
const status_text = document.querySelector('#status');
const chatHistory = document.querySelector('.chat-history');

document.querySelector('.message-time').innerHTML = new Date().toLocaleTimeString();

button.addEventListener('click', (event) => {    
  event.preventDefault();
  const question = input.value;
  if (question.trim() === '') return;
  input.value = '';
  sendMessageByUser(question);
  sendMessageByRobot(question );
  
});

voiceButton.addEventListener('click', (event) => {
  event.preventDefault();
  if (!isRecording) {
    $('#voice-button').find('i').addClass('fas fa-microphone-recording');
    // $('#voice-button').find('i').removeClass('fas fa-microphone');
    startRecording();
    isRecording = true;
  }else{
    // document.getElementById("voice-button").textContent = "Record";
    $('#voice-button').find('i').removeClass('fas fa-microphone-recording');
    $('#voice-button').find('i').addClass('fas fa-microphone');
    mediaRecorder.stop();
    isRecording = false;
  }
});

document.addEventListener('keydown', (event) => {
  if (event.code !== 'Enter') return;
  event.preventDefault();
  button.click();
});

const startRecording = () => {
  navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
      window.stream = stream;
      const gumAudio = document.querySelector('audio#gum');
      gumAudio.srcObject = stream;

      recordedBlobs = [];
      try {
          mediaRecorder = new MediaRecorder(window.stream);
          mediaRecorder.ondataavailable = handleDataAvailable;
          mediaRecorder.start();
      } catch (e) {
          console.error('创建MediaRecorder时异常:', e);
      }
    })
    .catch(error => {
      console.error(error);
      status_text.innerText = 'Could not start recording';
    });
};

function handleDataAvailable(event) {
  if (event.data && event.data.size > 0) {      
      recordedBlobs.push(event.data);
      const formData = new FormData();
      formData.append('audio', new Blob(recordedBlobs, { type: 'audio/webm' }));
      fetch('/speech-to-text', { method: 'POST', body: formData })
        .then(response => response.json())
        .then(data => {            
          sendMessageByUser(data.transcription);
          sendMessageByRobot(data.transcription);            
      });      
  }
}

function sendMessageByUser(question){  
  let messageSent = `
    <div class="message message-sent">
      <div class="message-sender-avatar">
      <img src="/static/img/user.png">
      </div>
      <div class="message-content">
        <div class="message-info">
          <div class="message-sender">顧客</div>
          <div class="message-time">${new Date().toLocaleTimeString()}</div>
        </div>
        <div class="message-body">${question}</div>
      </div>
    </div>
  `;
  chatHistory.insertAdjacentHTML('beforeend', messageSent);
}

function sendMessageByRobot(question) {
  const json_data = {
    question:question
  };

  messageSent = `
    <div class="message message-received">
      <div class="message-sender-avatar">
        <img src="/static/img/customer-service.png">
      </div>
      <div class="message-content">
        <div class="message-info">
          <div class="message-sender">客服</div>
          <div class="message-time">${new Date().toLocaleTimeString()}</div>
        </div>
        <div class="message-body">馬上為您查詢, 請稍後</div>
      </div>
    </div>
  `;
  chatHistory.insertAdjacentHTML('beforeend', messageSent);

  fetch('/get-db-data', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(json_data)
  })
  .then(response => response.json())
  .then(data => {
    console.log( data.data);
    console.log( data.cols);
    let message = "";
    if (data.data.length > 0) {
      let tableRows = '';
      let tdRows = '';
      let thRows = '';
      for (let i = 0; i < data.data.length; i++) {
        for (let j = 0; j < data.data[i].length; j++) {
          tdRows += `<td>${data.data[i][j]}</td>`
        }
        tableRows += `<tr>` + tdRows + `</tr>`;
        tdRows = '';
      }
      for (let i = 0; i < data.cols.length; i++) {
        thRows += `<th>` + data.cols[i] + `</th>`;
      }
      const tableHTML = `
                        <table>
                          <thead>
                            <tr>
                              ${thRows}
                            </tr>
                          </thead>
                          <tbody>
                            ${tableRows}
                          </tbody>
                        </table>
                      `;
      const messageSent = `
        <div class="message message-received">
          <div class="message-sender-avatar">
            <img src="/static/img/customer-service.png">
          </div>
          <div class="message-content">
            <div class="message-info">
              <div class="message-sender">客服</div>
              <div class="message-time">${new Date().toLocaleTimeString()}</div>
            </div>
            <div class="message-body">${tableHTML}</div>
          </div>
        </div>
      `;
      chatHistory.insertAdjacentHTML('beforeend', messageSent);
    } else {
      
    }
  })
  .catch(error => console.error(error));
}