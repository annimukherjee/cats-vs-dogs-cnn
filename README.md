# CNN Cat vs Dog Classifier

Code: 

## Problem Statement
The objective of this project is to design and train a **Convolutional Neural Network (CNN)** that can classify images of cats and dogs. This is a binary classification problem where the model must output whether the given input image belongs to the "Cat" class or the "Dog" class.

## Dataset
- Source: [Kaggle's Cats vs Dogs dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog) (`./dataset/`).
- Structure:
  - **training_set/** → used for training & validation with an **80/20 split**
  - **test_set/** → unseen images for model evaluation
- Preprocessing:
  - Images resized to **128×128 pixels**.
  - **Data Augmentation**: Random flips, rotations, and zoom applied on training data.
  - **Normalization**: Pixel values scaled to range `[0, 1]`.

## Approach
1. **Model Architecture**:  
   - Input Layer (128×128×3)  
   - Conv2D → ReLU → MaxPooling  
   - Conv2D → ReLU → MaxPooling  
   - Flatten → Dense(128, ReLU) → Dense(1, Sigmoid)  
   - Total: ~ a few hundred thousand trainable parameters.
   
2. **Training**:
   - Optimizer: `Adam`
   - Loss: `Binary Crossentropy`
   - Metrics: `Accuracy`
   - Epochs: 25
   - Batch Size: 32

3. **Evaluation**:
   - Performance measured on validation and independent test set.
   - Visualization of **training/validation accuracy and loss curves**.
   - Random test image predictions with predicted vs. actual labels.

## Results

- **Training Accuracy**: ~95% (after ~25 epochs)  
- **Validation Accuracy**: ~92% (balanced generalization)  
- **Test Accuracy**: ~91–93% depending on random splits  
- Predictions on unseen images were largely correct, with occasional misclassifications in edge cases (low-light or ambiguous images).

## Challenges & Learnings
- **Data Quality**: Some images contained noise or ambiguous labeling, which affected predictions.  
- **Overfitting**: Initially, the model showed signs of overfitting; adding **data augmentation** and normalization improved generalization.  
- **Hyperparameter Sensitivity**: Small changes in batch size, learning rate, or augmentation intensity influenced validation accuracy.  
- **Key Learning**: Well-structured preprocessing and augmentation can significantly enhance model performance on small to medium datasets.

## Future Improvements

- Use deeper architectures (e.g., VGG16, ResNet) with transfer learning.
- Introduce regularization techniques like Dropout or L2 penalties.
- Experiment with larger image sizes for finer feature extraction.
- Deploy the model as a web app for interactive classification.

---

### References & Inspiration
- [Kaggle Cats vs Dogs Tutorial](https://www.kaggle.com/code/tongpython/nattawut-5920421014-cat-vs-dog-dl/script)  
- [Becoming Human Blog](https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8)  
- [CampusX YouTube CNN Tutorial](https://www.youtube.com/watch?v=0K4J_PTgysc)