# Quick Start Guide - Emotion Detection Project

This guide will get you up and running with the improved emotion detection system in 5 minutes.

---

## 📦 Step 1: Install Dependencies (30 seconds)

```bash
pip install -r requirements.txt
```

This installs:
- PyTorch (deep learning framework)
- torchvision (image processing)
- matplotlib, seaborn (visualization)
- scikit-learn (metrics)
- Pillow (image loading)

---

## 🎓 Step 2: Train the Model (5-10 minutes)

### Option A: Improved Version (Recommended)

```bash
python train_improved.py
```

**What happens:**
- Loads 2,278 images from `emotion_dataset/`
- Splits into 80% train (1,822) and 20% validation (456)
- Trains enhanced CNN with batch normalization and dropout
- Uses class-weighted loss to handle imbalance
- Implements early stopping (stops when no improvement)
- Saves best model as `emotion_model.pth`
- Generates visualizations in `training_results/`

**Expected time:** 5-10 minutes on CPU, 2-3 minutes on GPU

**Output files:**
- `emotion_model.pth` - Final model
- `best_emotion_model.pth` - Best checkpoint
- `training_results/training_history.png` - Learning curves
- `training_results/confusion_matrix.png` - Confusion matrix
- `training_results/per_class_accuracy.png` - Per-class metrics
- `training_results/model_info.txt` - Training summary

### Option B: Original Version (Fixed)

```bash
python emotions.py
```

**What happens:**
- Trains original 2-layer CNN
- Fixed 10 epochs
- No validation split
- Saves as `sad_happy_angry.pth`

**Expected time:** 2-3 minutes

---

## 🔮 Step 3: Run Inference (Instant)

### Basic Usage

```bash
# Use default test images
python inference_improved.py
```

### Advanced Usage

```bash
# Single image
python inference_improved.py --image crying.png

# Multiple images
python inference_improved.py --images boy.png person.png crying.png

# Show probability distribution
python inference_improved.py --image boy.png --show-probs
```

**Example output:**
```
Image: boy.png
  Predicted Emotion: HAPPY
  Probability Distribution:
    angry   :  5.23% ██
    happy   : 87.45% ███████████████████████████████████████████
    sad     :  7.32% ███
```

---

## 📊 Step 4: Analyze the Model (30 seconds)

```bash
python analyze_model.py
```

**What you get:**
- Dataset statistics and class distribution
- Model architecture details
- Parameter count and model size
- Inference speed (FPS)
- Sample predictions visualization

**Output:**
```
DATASET ANALYSIS
Total samples: 2,278
Classes: ['angry', 'happy', 'sad']

Class Distribution:
  angry   :  515 (22.6%) ████████████
  happy   : 1006 (44.2%) ██████████████████████
  sad     :  757 (33.2%) ████████████████

MODEL ARCHITECTURE ANALYSIS
Total parameters: 1,234,567
Model size: 4.98 MB

Inference Speed:
  Average time: 12.34 ms
  FPS: 81.0
```

---

## 🎯 Step 5: Review Results

### Check Training Results

```bash
# Open visualizations
start training_results/training_history.png
start training_results/confusion_matrix.png
start training_results/per_class_accuracy.png

# Read training summary
type training_results/model_info.txt
```

### Check Analysis Results

```bash
start analysis_results/sample_predictions.png
```

---

## 🚀 Common Workflows

### Workflow 1: Train and Test

```bash
# 1. Train
python train_improved.py

# 2. Test on your images
python inference_improved.py --images my_photo1.jpg my_photo2.jpg --show-probs

# 3. Analyze
python analyze_model.py
```

### Workflow 2: Quick Experiment

```bash
# Use original script for quick experiments
python emotions.py
python detect_emotion.py
```

### Workflow 3: Production Deployment

```bash
# 1. Train with best settings
python train_improved.py

# 2. Test thoroughly
python inference_improved.py --images test_set/*.png

# 3. Verify performance
python analyze_model.py

# 4. Deploy emotion_model.pth
```

---

## 📁 Project Structure After Setup

```
detectEmotions/
├── emotion_dataset/          # Your training data
├── train_improved.py         # Enhanced training
├── inference_improved.py     # Enhanced inference
├── analyze_model.py          # Analysis tool
├── emotion_model.pth         # ✨ Trained model
├── best_emotion_model.pth    # ✨ Best checkpoint
├── training_results/         # ✨ Training outputs
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── per_class_accuracy.png
│   └── model_info.txt
└── analysis_results/         # ✨ Analysis outputs
    └── sample_predictions.png
```

---

## ❓ Troubleshooting

### Issue: "No module named 'torch'"

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: "Dataset directory not found"

**Solution:**
Make sure `emotion_dataset/` exists with subfolders:
```
emotion_dataset/
├── angry/
├── happy/
└── sad/
```

### Issue: "Model file not found"

**Solution:**
Train the model first:
```bash
python train_improved.py
```

### Issue: Training is slow

**Solutions:**
- Use GPU if available (automatically detected)
- Reduce batch size in `train_improved.py`
- Reduce max epochs (default: 50)

### Issue: Low accuracy

**Solutions:**
- Collect more training data
- Balance class distribution
- Increase training epochs
- Try different hyperparameters

---

## 🎓 Next Steps

1. **Read the full documentation:** `README.md`
2. **Review improvements:** `IMPROVEMENTS.md`
3. **Check the analysis report:** `report.html`
4. **Experiment with hyperparameters**
5. **Add your own images**
6. **Extend to more emotion classes**

---

## 📞 Need Help?

- Check `README.md` for detailed documentation
- Review `IMPROVEMENTS.md` for technical details
- Open `report.html` for visual analysis
- Check code comments for inline documentation

---

**Happy Emotion Detecting! 😊**
