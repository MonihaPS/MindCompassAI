// Global variables
let mediaRecorder;
let recordedChunks = [];
let stream;
let recordingStartTime;
let timerInterval;

// Tab switching
function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });

    // Remove active class from all buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');
    event.target.classList.add('active');

    // Hide results when switching tabs
    hideResults();
}

// Text prediction
async function predictText() {
    const text = document.getElementById('text-input').value.trim();

    if (!text) {
        showError('Please enter some text');
        return;
    }

    showLoading();
    hideError();

    try {
        const response = await fetch('/predict/text', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text })
        });

        const data = await response.json();

        if (response.ok) {
            displayResults(data, 'text');
        } else {
            showError(data.error || 'Prediction failed');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        hideLoading();
    }
}

// Audio prediction
async function predictAudio() {
    const fileInput = document.getElementById('audio-file');
    const file = fileInput.files[0];

    if (!file) {
        showError('Please select an audio file');
        return;
    }

    showLoading();
    hideError();

    const formData = new FormData();
    formData.append('audio', file);

    try {
        const response = await fetch('/predict/audio', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            displayResults(data, 'audio');
        } else {
            showError(data.error || 'Prediction failed');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        hideLoading();
    }
}

// Video prediction
async function predictVideo() {
    const fileInput = document.getElementById('video-file');
    const file = fileInput.files[0];

    if (!file) {
        showError('Please select a video file');
        return;
    }

    showLoading();
    hideError();

    const formData = new FormData();
    formData.append('video', file);

    try {
        const response = await fetch('/predict/fusion', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            displayResults(data, 'fusion');
        } else {
            showError(data.error || 'Prediction failed');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        hideLoading();
    }
}

// Live recording functions
async function startRecording() {
    try {
        // Request camera and microphone access
        stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: true
        });

        // Show preview
        const preview = document.getElementById('preview');
        preview.srcObject = stream;

        // Setup MediaRecorder
        const options = {
            mimeType: 'video/webm;codecs=vp8,opus'
        };

        // Fallback for browsers that don't support webm
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options.mimeType = 'video/webm';
        }

        mediaRecorder = new MediaRecorder(stream, options);
        recordedChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                recordedChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = () => {
            stopTimer();
            document.getElementById('recording-indicator').classList.add('hidden');
            document.getElementById('analyze-btn').classList.remove('hidden');
            updateStatus('✅ Recording saved! Click "Analyze Recording" to get results.');
        };

        // Start recording
        mediaRecorder.start();
        recordingStartTime = Date.now();
        startTimer();

        // Update UI
        document.getElementById('start-btn').classList.add('hidden');
        document.getElementById('stop-btn').classList.remove('hidden');
        document.getElementById('recording-indicator').classList.remove('hidden');
        updateStatus('🔴 Recording... Speak naturally and show your facial expressions');

    } catch (error) {
        showError('Failed to access camera/microphone: ' + error.message);
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();

        // Stop all tracks
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }

        // Update UI
        document.getElementById('stop-btn').classList.add('hidden');
        document.getElementById('start-btn').classList.remove('hidden');
    }
}

async function analyzeRecording() {
    if (recordedChunks.length === 0) {
        showError('No recording available');
        return;
    }

    showLoading();
    hideError();

    // Create blob from recorded chunks
    const blob = new Blob(recordedChunks, { type: 'video/webm' });

    // Create form data
    const formData = new FormData();
    formData.append('video', blob, 'recording.webm');

    try {
        const response = await fetch('/predict/fusion', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            displayResults(data, 'fusion');
            // Reset recording
            recordedChunks = [];
            document.getElementById('analyze-btn').classList.add('hidden');
            updateStatus('');
        } else {
            showError(data.error || 'Prediction failed');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        hideLoading();
    }
}

// Timer functions
function startTimer() {
    timerInterval = setInterval(() => {
        const elapsed = Date.now() - recordingStartTime;
        const seconds = Math.floor(elapsed / 1000);
        const minutes = Math.floor(seconds / 60);
        const secs = seconds % 60;

        document.getElementById('recording-time').textContent =
            `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }, 1000);
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
    }
}

// Display results
function displayResults(data, type) {
    hideResults();

    const resultsDiv = document.getElementById('results');
    resultsDiv.classList.remove('hidden');

    // Display based on type
    if (type === 'fusion' && data.fusion) {
        displayFusionResult(data.fusion);

        // Show individual modality results if available
        if (data.video) displayModalityResult(data.video, 'video');
        if (data.audio) displayModalityResult(data.audio, 'audio');
        if (data.text) displayModalityResult(data.text, 'text');

    } else if (type === 'text' && data.text) {
        displayModalityResult(data.text, 'text');
    } else if (type === 'audio' && data.audio) {
        displayModalityResult(data.audio, 'audio');
    } else if (type === 'video' && data.video) {
        displayModalityResult(data.video, 'video');
    }

    // Show XAI explanations
    displayXAI(data);
}

function displayFusionResult(fusion) {
    const fusionDiv = document.getElementById('fusion-result');
    fusionDiv.classList.remove('hidden');

    const emotion = fusion.emotion || fusion.predicted_emotion;
    const confidence = fusion.confidence || 0;

    document.getElementById('fusion-emotion').innerHTML =
        `<span class="emotion-label">${emotion}</span>`;

    document.getElementById('fusion-confidence').innerHTML =
        `<div class="confidence-label">Confidence: ${(confidence * 100).toFixed(1)}%</div>
         <div class="progress-bar">
             <div class="progress-fill" style="width: ${confidence * 100}%"></div>
         </div>`;
}

function displayModalityResult(result, modality) {
    const resultDiv = document.getElementById(`${modality}-result`);
    resultDiv.classList.remove('hidden');

    const emotion = result.emotion || result.predicted_emotion;
    const confidence = result.confidence || 0;

    document.getElementById(`${modality}-emotion`).textContent = emotion;
    document.getElementById(`${modality}-confidence`).textContent =
        `${(confidence * 100).toFixed(1)}% confident`;
}

function displayXAI(data) {
    const xaiSection = document.getElementById('xai-section');
    const xaiContent = document.getElementById('xai-content');

    xaiSection.classList.remove('hidden');

    let html = '';

    // 1. Text SHAP Visualization
    if (data.text && data.text.xai) {
        html += `<div class="xai-card">
            <div class="xai-modality-header">📝 Text Influence Analysis (SHAP)</div>
            <p class="info-text" style="font-size: 0.85em; margin-bottom: 10px; background: transparent; padding: 0;">
                Highlights show which words most influenced the <b>${data.text.emotion}</b> prediction.
            </p>
            <div class="shap-container">
                ${renderShapTokens(data.text.xai, data.text.emotion)}
            </div>
            ${data.text.reasoning ? `<div class="xai-reasoning">"${data.text.reasoning}"</div>` : ''}
        </div>`;
    }

    // 2. Audio Reasoning
    if (data.audio && data.audio.reasoning) {
        html += `<div class="xai-card">
            <div class="xai-modality-header">🎤 Audio Reasoning</div>
            <div class="xai-reasoning">${data.audio.reasoning}</div>
        </div>`;
    }

    // 3. Video Reasoning
    if (data.video && data.video.reasoning) {
        html += `<div class="xai-card">
            <div class="xai-modality-header">🎥 Video Reasoning</div>
            <div class="xai-reasoning">${data.video.reasoning}</div>
        </div>`;
    }

    // 4. Fusion Logic (Synthesis)
    if (data.synthesis && data.synthesis.description) {
        html += `<div class="xai-card">
            <div class="xai-modality-header">🧩 Final Logic</div>
            <p>${data.synthesis.description}</p>
        </div>`;
    }

    xaiContent.innerHTML = html;
}

function renderShapTokens(xaiData, targetEmotion) {
    if (!xaiData || !xaiData.tokens || !xaiData.values) return 'Analysis data unavailable.';

    // Unified emotion order matching config.py
    const emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'];
    const emoIndex = emotions.indexOf(targetEmotion.toLowerCase());

    if (emoIndex === -1) return `Click a modality to see details.`;

    let html = '';
    xaiData.tokens.forEach((token, i) => {
        const value = xaiData.values[i][emoIndex];
        const absVal = Math.abs(value);
        const opacity = Math.min(absVal * 15, 0.9); // Scale for visibility

        let className = 'shap-token';
        let color = 'transparent';

        if (value > 0.005) {
            className += ' shap-positive';
            color = `rgba(255, 0, 0, ${opacity})`;
        } else if (value < -0.005) {
            className += ' shap-negative';
            color = `rgba(0, 0, 255, ${opacity})`;
        }

        const cleanToken = token.replace('Ġ', ' ').replace('Ċ', ' ').trim();
        if (!cleanToken && token !== ' ') return;

        html += `<span class="${className}" 
                       style="background-color: ${color}"
                       title="Impact: ${value.toFixed(4)}">
                    ${cleanToken || '&nbsp;'}
                 </span>`;
    });

    return html;
}

// UI helper functions
function showLoading() {
    document.getElementById('loading').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loading').classList.add('hidden');
}

function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');
}

function hideError() {
    document.getElementById('error').classList.add('hidden');
}

function hideResults() {
    document.getElementById('results').classList.add('hidden');
    document.querySelectorAll('.result-card').forEach(card => {
        card.classList.add('hidden');
    });
    document.getElementById('xai-section').classList.add('hidden');
}

function updateStatus(message) {
    document.getElementById('recording-status').textContent = message;
}
