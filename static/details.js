// ============================
// DOM Elements
// ============================
const loadingState = document.getElementById('loadingState');
const detailsContent = document.getElementById('detailsContent');
const errorState = document.getElementById('errorState');
const previewImage = document.getElementById('previewImage');
const emotionBadge = document.getElementById('emotionBadge');
const detectedEmotion = document.getElementById('detectedEmotion');
const confidenceFill = document.getElementById('confidenceFill');
const confidenceValue = document.getElementById('confidenceValue');
const probBars = document.getElementById('probBars');
const metadataGrid = document.getElementById('metadataGrid');
const infoGrid = document.getElementById('infoGrid');

// ============================
// Emotion Colors
// ============================
const emotionColors = {
    happy: 'hsl(140, 70%, 55%)',
    sad: 'hsl(220, 70%, 60%)',
    angry: 'hsl(0, 75%, 60%)'
};

// ============================
// Load Result Data
// ============================
async function loadResultData() {
    try {
        const response = await fetch(`/api/result/${resultId}`);

        if (!response.ok) {
            throw new Error('Result not found');
        }

        const data = await response.json();
        displayResultDetails(data);

    } catch (error) {
        console.error('Error loading result:', error);
        showError();
    }
}

// ============================
// Display Functions
// ============================
function displayResultDetails(data) {
    // Hide loading, show content
    loadingState.style.display = 'none';
    detailsContent.style.display = 'block';

    // Image Preview
    previewImage.src = `/uploads/${data.filename}`;
    previewImage.alt = data.original_filename;

    // Emotion Badge
    const emotion = data.prediction.emotion;
    emotionBadge.textContent = emotion.toUpperCase();
    emotionBadge.className = `emotion-badge ${emotion}`;
    emotionBadge.style.background = `${emotionColors[emotion]}20`;
    emotionBadge.style.color = emotionColors[emotion];

    // Detected Emotion
    detectedEmotion.textContent = emotion.toUpperCase();
    detectedEmotion.style.color = emotionColors[emotion];

    // Confidence
    const confidence = data.prediction.confidence * 100;
    confidenceValue.textContent = `${confidence.toFixed(1)}%`;

    // Animate confidence bar
    setTimeout(() => {
        confidenceFill.style.width = `${confidence}%`;
    }, 100);

    // Probability Distribution
    displayProbabilityBars(data.prediction.probabilities);

    // Metadata
    displayMetadata(data.metadata);

    // Processing Info
    displayProcessingInfo(data);
}

function displayProbabilityBars(probabilities) {
    const sortedProbs = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

    probBars.innerHTML = sortedProbs.map(([emotion, prob]) => {
        const percentage = (prob * 100).toFixed(1);
        const color = emotionColors[emotion] || 'hsl(260, 85%, 60%)';

        return `
            <div class="prob-bar">
                <div class="prob-bar-header">
                    <span class="prob-bar-label">${emotion}</span>
                    <span class="prob-bar-value">${percentage}%</span>
                </div>
                <div class="prob-bar-fill-container">
                    <div class="prob-bar-fill" style="width: 0%; background: ${color};" data-width="${percentage}"></div>
                </div>
            </div>
        `;
    }).join('');

    // Animate bars
    setTimeout(() => {
        document.querySelectorAll('.prob-bar-fill').forEach(bar => {
            const width = bar.dataset.width;
            bar.style.width = `${width}%`;
        });
    }, 200);
}

function displayMetadata(metadata) {
    const metadataItems = [
        { label: 'Filename', value: metadata.filename },
        { label: 'File Size', value: metadata.size },
        { label: 'Resolution', value: metadata.resolution },
        { label: 'Width', value: `${metadata.width}px` },
        { label: 'Height', value: `${metadata.height}px` },
        { label: 'Format', value: metadata.format || 'N/A' },
        { label: 'Color Mode', value: metadata.mode || 'N/A' }
    ];

    metadataGrid.innerHTML = metadataItems.map(item => `
        <div class="metadata-item">
            <span class="metadata-label">${item.label}</span>
            <span class="metadata-value">${item.value}</span>
        </div>
    `).join('');
}

function displayProcessingInfo(data) {
    const uploadDate = new Date(data.upload_time);
    const formattedDate = uploadDate.toLocaleString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });

    const infoItems = [
        { label: 'Upload Time', value: formattedDate },
        { label: 'Result ID', value: data.id },
        { label: 'Model', value: 'ImprovedEmotionCNN' },
        { label: 'Classes', value: data.prediction.all_emotions.join(', ') },
        { label: 'Processing Status', value: '✓ Complete' }
    ];

    infoGrid.innerHTML = infoItems.map(item => `
        <div class="info-item">
            <span class="info-label">${item.label}</span>
            <span class="info-value">${item.value}</span>
        </div>
    `).join('');
}

function showError() {
    loadingState.style.display = 'none';
    errorState.style.display = 'block';
}

// ============================
// Initialize
// ============================
document.addEventListener('DOMContentLoaded', () => {
    loadResultData();
});
