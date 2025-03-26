# Open-Set Classification with MNIST

by Liz Gokhvat and Noa Levy

## Overview
This project explores Open-Set Recognition (OSR) using the MNIST dataset. OSR addresses the critical challenge in computer vision where models must not only correctly classify known classes but also identify inputs from unknown classes never seen during training.

Traditional classification models operate under the closed-set assumption (all test samples belong to classes seen during training). However, real-world applications frequently encounter samples from unknown classes, requiring more robust solutions.

## Models Architecture

### Baseline MNIST Model
A simple classifier consisting of:
- 2 fully connected layers (784→128→10)
- SoftMax normalization
- Trained with standard Negative Log Likelihood (NLL) loss

### OSR Model
A more sophisticated architecture designed to handle out-of-distribution samples:

#### Architecture
- **Convolutional Layers**: 6 layers with increasing channel depth (16→32→64→128→256→512)
- **Pooling**: Max pooling with 3×3 kernel and stride 2 in layers 1, 3, and 5
- **Fully Connected Layers**: 5 layers gradually reducing dimensions to 10 outputs
- The forward pass returns both classification probabilities and embedding vectors

#### Multi-task Learning Approach
1. **Classification Loss**: Negative Log Likelihood for standard digit classification
2. **Triplet Loss**: Creates distinct clusters for each digit class, enabling the model to determine if a sample is too far from known class centroids

#### Unknown Class Detection
- During training, centroids are calculated for each class using average embeddings
- A threshold distance determines if a sample belongs to a known class or should be classified as "unknown" (class 10)
- After experimenting with different thresholds (average distance, maximum distance), a balanced threshold (average of the two) was chosen to optimize detection of unknown samples while minimizing misclassification of known digits
- The final prediction is an 11-class output (digits 0-9 plus "unknown")

## Hyperparameters and Training Details

### Baseline Model
- **Training Data**: 20% of MNIST training data
- **Optimizer**: Adam
- **Learning Rate**: 0.002
- **Epochs**: 50

### OSR Model
- **Training Data**: 20% of MNIST training data + 25% augmented samples
- **Augmentation**: RandomAffine with rotation limited to (-20°, 20°) to preserve digit orientation
- **Optimizer**: Adam
- **Learning Rate**: 0.0003
- **Epochs**: 60
- **Triplet Loss Margin**: 1
- **Weight Decay**: 1e-5


## Inference Process
1. The sample is processed through the model to obtain both classification output and embedding
2. The highest probability class is identified from the 10 MNIST classes
3. Euclidean distance is calculated between the sample's embedding and the centroid of the predicted class
4. If the distance exceeds the threshold, the sample is reclassified as "unknown" (class 10)
5. Final prediction is normalized with SoftMax

## Running the Project
To run this project:

1. Open the notebook `Open Set Classification with MNIST.ipynb` in Google Colab
2. For evaluation mode (default):
   - The notebook is set to evaluation mode by default (`eval_mode = True`)
   - Upload the pretrained models (`baseline_model.pth` and `model_osr.pth`) to the notebook environment.
   - Run the entire notebook to see the pretrained model performance
3. For training mode:
   - Change the `eval_mode` parameter to `False` at the beginning of the notebook
   - Run the entire notebook to train the models from scratch
   - It is strongly recommended to use GPU runtime for training `(Runtime → Change runtime type → GPU)` because training will be significantly faster on GPU compared to CPU

## Conclusion
This project demonstrates an effective approach to Open-Set Recognition with the MNIST dataset. By combining traditional classification with distance-based unknown detection, the model can identify samples from classes not seen during training while maintaining high accuracy on known classes.
