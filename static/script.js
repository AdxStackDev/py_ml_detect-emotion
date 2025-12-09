// ============================
// State Management
// ============================
let selectedFiles = [];

// ============================
// DOM Elements
// ============================
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const selectedFilesContainer = document.getElementById('selectedFiles');
const processBtn = document.getElementById('processBtn');
const processingOverlay = document.getElementById('processingOverlay');
const processingStatus = document.getElementById('processingStatus');
const progressFill = document.getElementById('progressFill');
const resultsSection = document.getElementById('resultsSection');
const resultsGrid = document.getElementById('resultsGrid');
const clearBtn = document.getElementById('clearBtn');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');

// ============================
// Event Listeners
// ============================
browseBtn.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
processBtn.addEventListener('click', processImages);
clearBtn.addEventListener('click', clearResults);
newAnalysisBtn.addEventListener('click', resetUpload);

// Drag and drop
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');

    const files = Array.from(e.dataTransfer.files).filter(file =>
        file.type.startsWith('image/')
    );

    if (files.length > 0) {
        addFiles(files);
    }
});

// ============================
// File Handling
// ============================
function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    addFiles(files);
}

function addFiles(files) {
    files.forEach(file => {
        if (!selectedFiles.find(f => f.name === file.name && f.size === file.size)) {
            selectedFiles.push(file);
        }
    });

    renderSelectedFiles();
    updateProcessButton();
}

function removeFile(index) {
    selectedFiles.splice(index, 1);
    renderSelectedFiles();
    updateProcessButton();
}

function renderSelectedFiles() {
    if (selectedFiles.length === 0) {
        selectedFilesContainer.innerHTML = '';
        return;
    }

    selectedFilesContainer.innerHTML = selectedFiles.map((file, index) => `
        <div class="file-item">
            <div class="file-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                    <circle cx="8.5" cy="8.5" r="1.5"/>
                    <polyline points="21 15 16 10 5 21"/>
                </svg>
            </div>
            <div class="file-info">
                <div class="file-name">${file.name}</div>
                <div class="file-size">${formatFileSize(file.size)}</div>
            </div>
            <button class="file-remove" onclick="removeFile(${index})">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="18" y1="6" x2="6" y2="18"/>
                    <line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
            </button>
        </div>
    `).join('');
}

function updateProcessButton() {
    processBtn.disabled = selectedFiles.length === 0;
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

// ============================
// Image Processing
// ============================
async function processImages() {
    if (selectedFiles.length === 0) return;

    // Show processing overlay
    processingOverlay.classList.add('active');
    processingStatus.textContent = `Processing 0 of ${selectedFiles.length} images`;
    progressFill.style.width = '0%';

    // Create FormData
    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('images', file);
    });

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Upload failed');
        }

        const data = await response.json();

        // Animate progress
        progressFill.style.width = '100%';
        processingStatus.textContent = `Processed ${data.count} of ${selectedFiles.length} images`;

        // Wait a moment for the animation
        await new Promise(resolve => setTimeout(resolve, 500));

        // Hide processing overlay
        processingOverlay.classList.remove('active');

        // Display results
        displayResults(data.results);

        // Reset upload
        selectedFiles = [];
        fileInput.value = '';
        renderSelectedFiles();
        updateProcessButton();

    } catch (error) {
        console.error('Error processing images:', error);
        alert('Error processing images. Please try again.');
        processingOverlay.classList.remove('active');
    }
}

// ============================
// Results Display
// ============================
function displayResults(results) {
    resultsSection.style.display = 'block';

    resultsGrid.innerHTML = results.map(result => {
        const emotion = result.prediction.emotion;
        const confidence = (result.prediction.confidence * 100).toFixed(1);

        return `
            <div class="result-card">
                <img src="/uploads/${result.filename}" alt="${result.original_filename}" class="result-image">
                <div class="result-content">
                    <div class="result-emotion ${emotion}">${emotion}</div>
                    <div class="result-filename">${result.original_filename}</div>
                    <div class="result-confidence">
                        Confidence: <strong>${confidence}%</strong>
                    </div>
                    <a href="/details/${result.id}" class="btn-details">View Details</a>
                </div>
            </div>
        `;
    }).join('');

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function clearResults() {
    resultsGrid.innerHTML = '';
    resultsSection.style.display = 'none';
}

function resetUpload() {
    clearResults();
    selectedFiles = [];
    fileInput.value = '';
    renderSelectedFiles();
    updateProcessButton();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ============================
// Utility Functions
// ============================
// Make removeFile available globally
window.removeFile = removeFile;
