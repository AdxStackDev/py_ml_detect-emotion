
# Emotion Detection using CNN (PyTorch)

This repository contains a simple Convolutional Neural Network (CNN) for training and predicting basic human emotions from grayscale face images.  
It demonstrates how to build, train, and use an emotion detection model using PyTorch and torchvision.

---

## 📂 Project Structure

```
emotion_dataset/    # Folder with subfolders: angry/, happy/, sad/
crying.png          # Example image for prediction
boy.png             # Example image for prediction
person.png          # Example image for prediction
train.py            # (Your training code)
predict.py          # (Your inference code)
sad_happy_angry.pth # Trained model weights
README.md
```

---

## 📌 Classes

The model currently predicts:
- **angry**
- **happy**
- **sad**

---

## 🛠️ Requirements

- Python 3.x
- PyTorch
- torchvision
- Pillow (PIL)
- matplotlib (optional, for training plots)

Install dependencies:
```bash
pip install torch torchvision pillow matplotlib
```

---

## 🚀 How to Train

1. Place your training images in `emotion_dataset/` with subfolders for each emotion label.  
   Example:
   ```
   emotion_dataset/
     angry/
       img1.jpg
       img2.jpg
     happy/
       img3.jpg
       img4.jpg
     sad/
       img5.jpg
       img6.jpg
   ```

2. Run the training script:
   ```bash
   python train.py
   ```

   This will:
   - Load the images
   - Train the CNN for `EPOCHS` (currently 10)
   - Save the model as `sad_happy_angry.pth`

---

## 🤖 How to Predict

1. Place the images you want to predict (e.g., `crying.png`, `boy.png`, `person.png`) in the project folder.

2. Run the prediction script:
   ```bash
   python predict.py
   ```

   The script will:
   - Load the saved model
   - Apply the same transforms
   - Predict the emotion class for each image
   - Print the results to the console

---

## ⚙️ Notes

- The model uses grayscale 48x48 images — matching many standard emotion datasets.
- Make sure your dataset classes match the `class_names` list in both `train.py` and `predict.py`.
- The images are normalized to `(-1, 1)` with `mean=0.5`, `std=0.5`.

---

## 📌 TODO

- Improve the CNN architecture for better accuracy.
- Add a validation/test split.
- Add accuracy/loss plots.
- Try with a larger or public dataset like FER-2013.
- Experiment with more emotion classes.

---

## 📜 License

Feel free to use, modify, and share for educational purposes. ✨

---

**Happy Learning! 😊**
