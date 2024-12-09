
<h2>Deep Learning Approach</h2>
This project milestone focused on utilizing deep-learning approaches to model our motor control data and evaluate how effectively different neural network-based models can predict motor faults. Neural networks are a popular supervised learning model that identifies nonlinear relationships and learns complex patterns between features and labels. Given that the data generated from our three-phase induction motor is time-series data with routine motor vibration signals, we chose to utilize recurrent neural networks that effectively recognize long-term patterns and dependencies between our generated features and our categorical label. <br><br>

Our first model created a long short-term memory (LSTM) neural network from scratch and fine-tuned the number of LSTM layers, the number of hidden units within each LSTM layer, and the dropout rate which affects the regularization of our neural network. LSTMs are beneficial because they are well-suited for analyzing time-series data and handling long sequences of data but they are also very computationally intensive, their results can be less intuitive to interpret, and they can be very prone to overfitting to the training data. Our second model utilized a convolutional neural network (CNN) and tuned the learning rate and the number of CNN layers. While CNN is effective at feature learning and is robust despite noise and distortion in the input data, it also has a high computational cost and requires large amounts of training data to achieve high accuracy. In our case, because our data is generated from our motor system, it is possible that we didn’t have enough robust data to achieve desirable results. 

<h2>LSTM Results:</h2> Our initial LSTM model had the following configuration: <br>
layers = [ ...<br>
   sequenceInputLayer(numFeatures)<br>
   lstmLayer(100, 'OutputMode', 'last')<br> 
   fullyConnectedLayer(numClasses)<br>
   softmaxLayer<br>
   classificationLayer];<br><br>
   
It achieved the following results: a training accuracy of 100% and a validation accuracy of 50.00% after 150 epochs. The large gap between training and validation accuracy initially indicates the model is overfitting on the training data and that we should increase the amount of regularization while training our model. Therefore, in our final LSTM model, we utilize a __ dropout rate to reduce the amount of overfitting and include several more LSTM layers with more hidden layers to improve our model’s validation accuracy. The precision score of our model was 42.85%, meaning that of the motors that the model predicted had faults, only 42.85% were actually broken. The recall score is 60.0%, indicating that of the motors that are broken, the model correctly predicted 60% of them.<br>

Given our problem statement to predict motor faults before they occur to reduce reactive maintenance costs, we should evaluate our results in their business context. We continued to train this initial model with the goal of reducing the amount of overfitting at the cost of training accuracy, in order to increase the scalability and reliability of our model. Additionally, since a false negative, where the model incorrectly predicts a broken rotor as healthy, is likely more costly and dangerous than a false positive, we trained our model intending to prioritize its recall score.

<h2>Best LSTM Model:</h2>
Our best LSTM model had the following configuration: <br>
layers = [ ...<br>
   sequenceInputLayer(numFeatures)<br>
   lstmLayer(100, OutputMode="sequence")<br>
   lstmLayer(100, OutputMode="sequence")<br>
   lstmLayer(100, OutputMode="sequence")<br>
   lstmLayer(100, OutputMode="sequence")<br>
   lstmLayer(100, OutputMode="last")<br>
   fullyConnectedLayer(numClasses)<br>
   softmaxLayer<br>
   classificationLayer];<br><br>
It was a deep LSTM model with five LSTM layers, each with 100 hidden units and no dropout rate regularization. It achieved a training accuracy of 100% and a validation accuracy of 62.50%, though this number ranged from 50-62.5% validation accuracy across different training sessions. This data once again indicates that the model is overfitting on our training data. Additionally, as we trained our model utilizing a different number of LSTM layers, hidden layers, and the dropout rate for regularization, we found that a higher regularization led to lower validation and training accuracies. This could be the result of having insufficient data for the model to correctly predict the relationship between our generated features and classification label. <br><br>
<img width="635" alt="Screenshot 2024-12-06 at 2 52 49 AM" src="https://github.com/user-attachments/assets/40d1e444-c907-40da-938b-f96f21bc09ee"><br>
From the confusion matrix, we can see that the model achieved a precision score of 60% and a recall score of 90%. This means that of the motors that the model predicted to be broken, its predictions were correct just over half the time. Additionally, of the rotors that were actually broken, it correctly predicted 90% of them. Therefore, while the validation accuracy was somewhat low, LSTM had a high recall score which is important in our business context where we are trying to prioritize predictive maintenance and identifying truly broken rotors which can be expensive to repair and potentially dangerous. Additionally, when it comes to scalability LSTMs can be very well-suited for time-series data and may be more appropriate when there is more time-series data available, albeit at the cost of computation intensity. 
<img width="327" alt="Screenshot 2024-12-06 at 2 53 17 AM" src="https://github.com/user-attachments/assets/4fe98041-81d9-4409-b562-64098d58c72e">

---

<h2>CNN Modeling</h2>

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

![MATLAB Training and testing Loss FOR MODEL 1](https://github.com/Brenda-MW/BTTAI_Team27_MotorControlAI/blob/main/EvaluateModels/CNN_DeepLearning/Control-V-2.png)


![MATLAB Confusion Matrix FOR MODEL 1](https://github.com/Brenda-MW/BTTAI_Team27_MotorControlAI/blob/main/EvaluateModels/CNN_DeepLearning/Control-V.png)

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

![MATLAB Training and testing Loss FOR MODEL 2](https://github.com/Brenda-MW/BTTAI_Team27_MotorControlAI/blob/main/EvaluateModels/CNN_DeepLearning/Control-V-3.png)

![MATLAB Confusion Matrix FOR MODEL 2](https://github.com/Brenda-MW/BTTAI_Team27_MotorControlAI/blob/main/EvaluateModels/CNN_DeepLearning/Control-V-4.png)

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
