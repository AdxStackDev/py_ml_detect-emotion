# Emotion Detection using CNN (PyTorch) - Improved Version

A comprehensive PyTorch-based Convolutional Neural Network for detecting emotions from grayscale facial images. This improved version includes validation splits, class balancing, early stopping, comprehensive metrics, and visualization tools.

---

## 📂 Project Structure

```
detectEmotions/
├── emotion_dataset/           # Training dataset
│   ├── angry/                # 515 images (22.6%)
│   ├── happy/                # 1,006 images (44.2%)
│   └── sad/                  # 757 images (33.2%)
│
├── train_improved.py         # ✨ NEW: Enhanced training script
├── inference_improved.py     # ✨ NEW: Enhanced inference script
├── analyze_model.py          # ✨ NEW: Model analysis utility
│
├── emotions.py               # Original training script (fixed)
├── detect_emotion.py         # Original inference script
│
├── emotion_model.pth         # Trained model weights (improved)
├── best_emotion_model.pth    # Best checkpoint with metadata
├── sad_happy_angry.pth       # Original model weights
│
├── training_results/         # Training outputs
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── per_class_accuracy.png
│   └── model_info.txt
│
├── analysis_results/         # Analysis outputs
│   └── sample_predictions.png
│
├── requirements.txt          # ✨ NEW: Python dependencies
├── report.html              # Cyberpunk-themed analysis report
└── README.md                # This file
```

---

## 🎯 Detected Emotion Classes

- **angry** 😠
- **happy** 😊
- **sad** 😢

---

## 🛠️ Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
```

**Dependencies:**
- Python 3.8+
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- Pillow >= 9.0.0
- matplotlib >= 3.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- seaborn >= 0.12.0

---

## 🚀 Quick Start

### 1. Train the Model (Improved Version)

```bash
python train_improved.py
```

**Features:**
- ✅ 80/20 train/validation split
- ✅ Class-weighted loss for imbalance handling
- ✅ Early stopping (patience=10)
- ✅ Learning rate scheduling
- ✅ Comprehensive metrics (accuracy, precision, recall, F1)
- ✅ Confusion matrix and learning curves
- ✅ Model checkpointing (saves best model)
- ✅ Data augmentation (flip, rotation, brightness)
- ✅ Batch normalization and dropout

**Output:**
- `emotion_model.pth` - Final model weights
- `best_emotion_model.pth` - Best checkpoint with metadata
- `training_results/` - Visualizations and metrics

### 2. Run Inference (Improved Version)

**Single image:**
```bash
python inference_improved.py --image path/to/image.png
```

**Multiple images:**
```bash
python inference_improved.py --images img1.png img2.png img3.png
```

**Show probability distribution:**
```bash
python inference_improved.py --image test.png --show-probs
```

**Custom model and classes:**
```bash
python inference_improved.py --model custom_model.pth --classes angry happy sad neutral
```

**Default (uses test images):**
```bash
python inference_improved.py
```

### 3. Analyze the Model

```bash
python analyze_model.py
```

**Analysis includes:**
- Dataset distribution and class imbalance
- Model architecture and parameter count
- Model file size
- Inference speed (FPS)
- Sample predictions visualization

---

## 📊 Model Architecture

### ImprovedEmotionCNN

```
Input: 1×48×48 grayscale image
  ↓
Conv2D (1→32) + BatchNorm + ReLU + MaxPool → [32, 24, 24]
  ↓
Conv2D (32→64) + BatchNorm + ReLU + MaxPool → [64, 12, 12]
  ↓
Conv2D (64→128) + BatchNorm + ReLU + MaxPool → [128, 6, 6]
  ↓
Flatten → 4,608 features
  ↓
FC (4,608→256) + ReLU + Dropout(0.5)
  ↓
FC (256→128) + ReLU + Dropout(0.5)
  ↓
FC (128→3) → Output logits
```

**Parameters:** ~1.2M  
**Model Size:** ~5 MB  
**Expected Accuracy:** 75-85% (with improvements)

---

## 🔧 What's Improved?

### Critical Fixes

1. **✅ Fixed filename mismatch bug** in `emotions.py`
   - Was saving as `sad_happy_angry1.pth` but loading `sad_happy_angry.pth`

2. **✅ Added train/validation split (80/20)**
   - Prevents overfitting
   - Enables proper evaluation

3. **✅ Class-weighted loss**
   - Handles class imbalance (happy: 44%, sad: 33%, angry: 23%)
   - Prevents bias toward majority class

4. **✅ Comprehensive metrics tracking**
   - Accuracy, precision, recall, F1-score
   - Confusion matrix
   - Per-class accuracy

### Major Enhancements

5. **✅ Early stopping**
   - Stops training when validation loss stops improving
   - Saves best model checkpoint

6. **✅ Learning rate scheduling**
   - Reduces LR when validation loss plateaus
   - Improves convergence

7. **✅ Improved architecture**
   - Added 3rd convolutional layer
   - Batch normalization for stable training
   - Dropout for regularization

8. **✅ Data augmentation**
   - Random horizontal flip
   - Random rotation (±10°)
   - Brightness jittering

9. **✅ Visualization**
   - Training/validation loss curves
   - Accuracy curves
   - Confusion matrix heatmap
   - Per-class accuracy bar chart

10. **✅ Error handling**
    - Try/except blocks for file I/O
    - Graceful error messages
    - Input validation

11. **✅ Command-line interface**
    - Flexible inference options
    - Probability display
    - Custom model/class support

12. **✅ Documentation**
    - requirements.txt
    - Comprehensive README
    - Inline code comments

---

## 📈 Expected Performance

### Original Model
- Accuracy: 60-75%
- Issues: Class imbalance, no validation, overfitting

### Improved Model
- Accuracy: 75-85%
- Balanced predictions across classes
- Better generalization
- Robust to variations

---

## 🎓 Usage Examples

### Training with Custom Settings

Edit `train_improved.py` to customize:
- `EPOCHS` - Maximum training epochs (default: 50)
- `batch_size` - Batch size (default: 32)
- `patience` - Early stopping patience (default: 10)
- `dropout_rate` - Dropout probability (default: 0.5)
- Learning rate, optimizer, etc.

### Inference Examples

```bash
# Basic inference
python inference_improved.py --image crying.png

# Show probabilities
python inference_improved.py --image boy.png --show-probs

# Batch inference
python inference_improved.py --images *.png

# Use original model
python inference_improved.py --model sad_happy_angry.pth --image test.png
```

---

## 🚀 Potential Use Cases

- **Mental health monitoring** - Track emotional trends over time
- **Customer service** - Analyze sentiment from video calls
- **Education** - Monitor student engagement in online classes
- **Gaming** - Adaptive difficulty based on player emotions
- **Marketing** - A/B test content via emotional reactions
- **Accessibility** - Emotion-aware assistive technologies

---

## 📌 Future Improvements

### High Priority
- [ ] Collect more data for minority classes (angry)
- [ ] Add more emotion classes (surprise, fear, disgust, neutral)
- [ ] Implement test-time augmentation
- [ ] Cross-validation for robust evaluation

### Medium Priority
- [ ] Transfer learning with pre-trained models (ResNet, EfficientNet)
- [ ] Ensemble methods for better accuracy
- [ ] Real-time webcam inference
- [ ] Web interface (Flask/Streamlit)

### Low Priority
- [ ] Export to ONNX for deployment
- [ ] Mobile optimization (TensorFlow Lite)
- [ ] Multi-face detection and tracking
- [ ] Temporal smoothing for video

---

## 📜 License

Free to use, modify, and share for educational purposes. ✨

---

## 🙏 Acknowledgments

- Dataset augmented with brightness and rotation variants
- Built with PyTorch and torchvision
- Visualization using matplotlib and seaborn

---

**Happy Learning! 😊**

For questions or issues, please check the `report.html` for detailed analysis.
