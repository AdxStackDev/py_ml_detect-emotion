# Emotion Detection Web Interface

A stunning, modern web interface for the Emotion Detection AI system with batch processing capabilities.

## Features

### 🎨 **Premium Design**
- Modern, dark-themed UI with vibrant gradients
- Smooth animations and micro-interactions
- Glassmorphism effects and dynamic backgrounds
- Fully responsive design for all devices

### 📤 **Smart Upload System**
- Drag & drop support for easy file upload
- Batch processing for multiple images
- Real-time file preview and management
- Support for JPG, PNG, GIF formats
- Maximum file size: 16MB per image

### 🧠 **AI-Powered Analysis**
- Real-time emotion detection using PyTorch CNN
- Three emotion classes: Happy, Sad, Angry
- Confidence scores and probability distribution
- Processing speed: <100ms per image

### 📊 **Comprehensive Results**
- Grid layout for batch results
- Individual result cards with emotion badges
- Confidence percentages
- Detailed view for each analysis

### 🔍 **Detailed Analysis Page**
- Large image preview
- Emotion detection results with confidence meter
- Probability distribution visualization
- Complete image metadata (size, resolution, format)
- Processing information and timestamps

## Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Model is Available**
   Make sure `sad_happy_angry.pth` is in the project root directory.

## Usage

### Starting the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

### Using the Interface

1. **Upload Images**
   - Click "Browse Files" or drag & drop images onto the upload zone
   - Multiple images can be selected for batch processing
   - Review selected files before processing

2. **Process Images**
   - Click "Process Images" button
   - Watch the real-time progress indicator
   - Results appear automatically in a grid layout

3. **View Results**
   - Each result card shows:
     - Thumbnail of the analyzed image
     - Detected emotion with color-coded badge
     - Confidence percentage
     - "View Details" button
   
4. **Detailed Analysis**
   - Click "View Details" on any result card
   - See full-size image preview
   - View complete probability distribution
   - Access all image metadata
   - Review processing information

### API Endpoints

#### Upload and Process Images
```
POST /api/upload
Content-Type: multipart/form-data
Body: images (multiple files)

Response:
{
  "success": true,
  "count": 2,
  "results": [
    {
      "id": "uuid",
      "filename": "unique_filename.jpg",
      "original_filename": "photo.jpg",
      "upload_time": "2025-12-09T16:50:00",
      "metadata": {
        "filename": "photo.jpg",
        "size": "45.2 KB",
        "resolution": "640 x 480",
        "format": "JPEG"
      },
      "prediction": {
        "emotion": "happy",
        "confidence": 0.942,
        "probabilities": {
          "happy": 0.942,
          "sad": 0.035,
          "angry": 0.023
        }
      }
    }
  ]
}
```

#### Get Specific Result
```
GET /api/result/{result_id}

Response: Single result object
```

#### Get Processing History
```
GET /api/history

Response:
{
  "results": [...]
}
```

## Project Structure

```
detectEmotions/
├── app.py                 # Flask backend server
├── templates/
│   ├── index.html        # Main upload interface
│   └── details.html      # Detailed analysis page
├── static/
│   ├── style.css         # Premium styling
│   ├── script.js         # Main page logic
│   └── details.js        # Details page logic
├── uploads/              # Uploaded images (auto-created)
├── results.json          # Processing history (auto-created)
└── sad_happy_angry.pth   # Trained model
```

## Technology Stack

### Backend
- **Flask** - Lightweight web framework
- **PyTorch** - Deep learning framework
- **Pillow** - Image processing

### Frontend
- **HTML5** - Semantic markup
- **CSS3** - Modern styling with custom properties
- **Vanilla JavaScript** - No framework dependencies
- **Fetch API** - Asynchronous requests

### Design Features
- CSS Grid & Flexbox for layouts
- CSS Custom Properties for theming
- CSS Animations & Transitions
- Gradient backgrounds
- Glassmorphism effects
- Responsive design patterns

## Model Information

- **Architecture**: ImprovedEmotionCNN
- **Input Size**: 48x48 grayscale images
- **Classes**: angry, happy, sad
- **Accuracy**: ~94.2%
- **Processing Speed**: <100ms per image

## Browser Support

- Chrome/Edge (recommended)
- Firefox
- Safari
- Opera

## Performance

- Optimized image loading
- Lazy rendering for large batches
- Efficient DOM manipulation
- Smooth 60fps animations
- Progressive enhancement

## Security Features

- File type validation
- File size limits (16MB)
- Unique filename generation
- Safe file storage
- Input sanitization

## Future Enhancements

- [ ] Export results to CSV/JSON
- [ ] Batch download of processed images
- [ ] Real-time webcam analysis
- [ ] More emotion classes
- [ ] User authentication
- [ ] Result sharing capabilities
- [ ] Advanced filtering and search
- [ ] Performance analytics dashboard

## Troubleshooting

### Model Not Loading
- Ensure `sad_happy_angry.pth` exists in the project root
- Check that the model was trained with the same architecture

### Upload Fails
- Verify file size is under 16MB
- Ensure file is a valid image format
- Check server logs for errors

### Slow Processing
- Reduce batch size
- Check system resources
- Consider using GPU acceleration

## License

This project is part of the Emotion Detection AI system.

## Credits

- Design: Modern UI/UX principles
- Framework: Flask
- ML Framework: PyTorch
- Fonts: Inter (Google Fonts)
