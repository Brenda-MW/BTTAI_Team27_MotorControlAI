
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
From the confusion matrix, we can see that the model achieved a precision score of 60% and a recall score of 90%. This means that of the motors that the model predicted to be broken, its predictions were correct just over half the time. Additionally, of the rotors that were actually broken, it correctly predicted 90% of them. Therefore, while the validation accuracy was somewhat low, LSTM had a high recall score which is important in our business context where we are trying to prioritize predictive maintenance and identifying truly broken rotors which can be expensive to repair and potentially dangerous. Additionally, when it comes to scalability LSTMs can be very well-suited for time-series data and may be more appropriate when there is more time-series data available, albeit at the cost of computation intensity. 
