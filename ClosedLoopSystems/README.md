<h1>Closed-Loop Simulink Models</h1>

This project milestone focused on placing our models in a closed-loop system in Simulink that is fed a continuous signal generated from the vibration signals of a motor. Placing these signals in a dynamic simulation environment where the data input isn’t a flat vector but a time-stamped input timetable has several advantages. First, this allows us to see how the model reacts to signal data over a period of time. Importantly, this will allow us to measure the model’s latency and observe how long the model delays before reacting to a change in signal. Secondly, this allows us to compare the efficiency of several different models by observing how quickly they’re able to respond to signal changes. Lastly, the Simulink environment’s offered scope block allows us to observe both the signal’s value and the model’s prediction probabilities for each signal. <br><br>

KNN Model: <br>
The final simulated environment had the following configurations: <br>

<img width="551" alt="Screenshot 2024-12-06 at 3 07 49 AM" src="https://github.com/user-attachments/assets/89d5169a-2608-4da3-bd5f-49e408ba97e8">

The signal input data was passed through our KNN model which outputted both a label class and the model’s probability predictions. The label data was sent to an output port, the output predictions, and the scope block. The scores, on the other hand, were sent to the scope block to be viewed and the display block which outputs the final signal’s predictions. For the KNN model, the signal begins as the vibration signals generated from a healthy rotor, which the model correctly predicts with 100% certainty, and then moves to simulating the signals of broken rotor 1 after 16 seconds, broken rotor 2 after another 16 signals, and onwards up to broken rotor 4. Each time, the model correctly predicts the true class label with 100% certainty and changes as soon as the signal changes, indicating no latency between changing signals. <br><br>
 
<img width="615" alt="Screenshot 2024-12-06 at 3 09 02 AM" src="https://github.com/user-attachments/assets/9dc42f54-1e72-425f-82c2-dfb125339a4c"><br>

LSTM Model: <br><br>
 <img width="440" alt="Screenshot 2024-12-06 at 3 09 21 AM" src="https://github.com/user-attachments/assets/52ceb111-e8da-4467-bcad-d2740cea09ab">

The LSTM model, under the same simulink environment configuration, had very different projected predictions in response to the signal outputs. During the first 15 seconds when the input signal was emulating a healthy rotor, the model predicted varying probabilities for each class label. After 16 seconds, the input signal simulates broken rotor 2 and the LSTM model’s change to predict the reflected change. However, unlike KNN which had instantaneous changes, the LSTM model takes a second to adjust to the new input signals and therefore we see a number of different fluctuations in probabilities. We can then conclude that LSTM’s are much more uncertain when classifying motor input signals and additionally they have a considerable latency time in comparison to KNN. 

