"""
Test ONNX Model Inference
Simple script to test the exported ONNX model
"""

import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

def test_onnx_model(model_path='emotion_model.onnx', image_path='boy.png'):
    """Test ONNX model with a sample image"""
    
    print("="*60)
    print("ONNX MODEL INFERENCE TEST")
    print("="*60)
    
    # Load ONNX model
    print("\n[1/4] Loading ONNX model...")
    try:
        session = ort.InferenceSession(model_path)
        print(f"  OK - Model loaded: {model_path}")
    except Exception as e:
        print(f"  ERROR - Failed to load model: {e}")
        return
    
    # Print model info
    print("\n[2/4] Model information...")
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name
    output_shape = session.get_outputs()[0].shape
    
    print(f"  Input name: {input_name}")
    print(f"  Input shape: {input_shape}")
    print(f"  Output name: {output_name}")
    print(f"  Output shape: {output_shape}")
    
    # Prepare image
    print(f"\n[3/4] Processing image: {image_path}...")
    try:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        image = Image.open(image_path).convert("L")
        img_tensor = transform(image).unsqueeze(0).numpy()
        print(f"  OK - Image processed, shape: {img_tensor.shape}")
    except Exception as e:
        print(f"  ERROR - Failed to process image: {e}")
        return
    
    # Run inference
    print("\n[4/4] Running inference...")
    try:
        outputs = session.run(None, {input_name: img_tensor})
        predictions = outputs[0][0]
        
        # Apply softmax
        exp_preds = np.exp(predictions - np.max(predictions))
        probabilities = exp_preds / exp_preds.sum()
        
        # Get predicted class
        class_names = ['angry', 'happy', 'sad']
        predicted_idx = np.argmax(probabilities)
        predicted_emotion = class_names[predicted_idx]
        confidence = probabilities[predicted_idx]
        
        print("  OK - Inference complete")
        
    except Exception as e:
        print(f"  ERROR - Inference failed: {e}")
        return
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nImage: {image_path}")
    print(f"Predicted Emotion: {predicted_emotion.upper()}")
    print(f"Confidence: {confidence:.2%}")
    
    print("\nProbability Distribution:")
    for i, emotion in enumerate(class_names):
        bar = "#" * int(probabilities[i] * 50)
        print(f"  {emotion:8s}: {probabilities[i]:6.2%} {bar}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    # Test with default image
    test_onnx_model('emotion_model.onnx', 'boy.png')
    
    # You can also test with other images:
    # test_onnx_model('emotion_model.onnx', 'crying.png')
    # test_onnx_model('emotion_model.onnx', 'person.png')
