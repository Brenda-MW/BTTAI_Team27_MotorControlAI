```markdown
# Model 1: Feature-Based Dense Network

## Architecture
```matlab
% Define CNN Architecture with additional complexity
layers = [
   featureInputLayer(size(XTrain, 2), "Name", "input") % Direct feature inputs

   % First dense block
   fullyConnectedLayer(128, "Name", "fc1") % Increase number of neurons
   reluLayer("Name", "relu1")
   dropoutLayer(0.4, "Name", "dropout1") % Moderate dropout

   % Second dense block
   fullyConnectedLayer(256, "Name", "fc2") % Deeper layer
   reluLayer("Name", "relu2")
   dropoutLayer(0.5, "Name", "dropout2")

   % Third dense block
   fullyConnectedLayer(128, "Name", "fc3") % Back to fewer neurons
   reluLayer("Name", "relu3")
   dropoutLayer(0.3, "Name", "dropout3")

   % Output layer
   fullyConnectedLayer(numel(categories(YTrain)), "Name", "output_fc")
   softmaxLayer("Name", "softmax")
];

% Define training options with modifications
opts = trainingOptions("adam", ...
   "InitialLearnRate", 0.0005, ... % Slower learning rate to stabilize
   "MaxEpochs", 150, ... % Increase epochs for more learning
   "MiniBatchSize", 32, ... % Larger batch size for better gradient estimates
   "Shuffle", "every-epoch", ...
   "ValidationData", {XValidation, YValidation}, ...
   "ValidationFrequency", 30, ...
   "Verbose", false, ...
   "Metrics", "accuracy", ...
   "Plots", "training-progress");
```

## Results
- **Training Accuracy**: 53.125%
- **Validation Accuracy**: 50%

## Observations
Model 1 treated extracted audio features as flat vectors and used fully connected layers. Despite moderate dropout to prevent overfitting, the dense network struggled to capture the complexity of the audio features. This may be due to the lack of structure in the feature representation, causing the model to miss critical temporal or spectral relationships.

---

# Model 2: Convolutional Neural Network

## Architecture
```matlab
% Define CNN Architecture
layers = [
   imageInputLayer([size(features{1}, 1), size(features{1}, 2), 1], "Name", "input")

   % First convolutional block
   convolution2dLayer(3, 16, "Padding", "same", "Name", "conv1")
   reluLayer("Name", "relu1")
   maxPooling2dLayer(2, "Stride", 2, "Name", "maxpool1")

   % Second convolutional block
   convolution2dLayer(3, 32, "Padding", "same", "Name", "conv2")
   reluLayer("Name", "relu2")
   maxPooling2dLayer(2, "Stride", 2, "Name", "maxpool2")

   % Third convolutional block
   convolution2dLayer(3, 64, "Padding", "same", "Name", "conv3")
   reluLayer("Name", "relu3")
   maxPooling2dLayer(1, "Stride", 1, "Name", "maxpool3")

   % Fully connected layers
   flattenLayer("Name", "flatten")
   fullyConnectedLayer(128, "Name", "fc1")
   reluLayer("Name", "relu4")
   dropoutLayer(0.6, "Name", "dropout")
   fullyConnectedLayer(numel(categories(YTrain)), "Name", "fc2")
   softmaxLayer("Name", "softmax")
];

% Define training options
opts = trainingOptions("adam", ...
   "InitialLearnRate", 0.0005, ...
   "MaxEpochs", 50, ...
   "MiniBatchSize", 16, ...
   "Shuffle", "every-epoch", ...
   "ValidationData", {XValidation, YValidation}, ...
   "ValidationFrequency", 30, ...
   "Verbose", false, ...
   "Metrics", "accuracy", ...
   "Plots", "training-progress");
```

## Results
- **Training Accuracy**: 96.875%
- **Validation Accuracy**: 50%

## Observations
Model 2 adapted audio features into a 2D matrix format, allowing convolutional layers to effectively capture temporal and spectral patterns. The convolutional blocks, coupled with max pooling, helped extract localized and hierarchical features. However, despite achieving high training accuracy, the model suffered from overfitting, with no improvement in validation accuracy. Dropout layers (rate: 0.6) were insufficient to bridge the gap, highlighting the need for additional regularization or data augmentation.

---

# Key Comparisons and Findings

### Model Performance
- **Model 1**: Struggled to learn effectively, with only 53.125% training accuracy. The dense architecture failed to capture the complexity of audio features.
- **Model 2**: Achieved a high training accuracy (96.875%) but overfitted, as validation accuracy stagnated at 50%.

### Input Representation
- **Model 1**: Treated features as flat vectors, missing relationships between features.
- **Model 2**: Treated audio features as 2D matrices, enabling convolutional layers to detect patterns, significantly improving training performance.

### Overfitting
Model 2’s overfitting highlighted the need for techniques like:
1. Data augmentation to increase training set diversity.
2. Regularization, such as L2 weight penalties or lower dropout rates.
3. Cross-validation for better generalization evaluation.

---

# Conclusion
Choosing the right model architecture and input representation is crucial for effectively handling complex data like audio signals. While Model 1 lacked the capacity to learn meaningful patterns, Model 2 leveraged convolutional layers to significantly improve training performance. Addressing overfitting remains the key challenge moving forward, requiring strategies like data augmentation and improved regularization. Overall, this project emphasized the importance of aligning model design with the underlying data structure.
