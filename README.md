
# Autonomous Vehicle Motion Forecasting

Given the first five seconds of a vehicle's location, the next six 
seconds of a vehicle's location were predicted using machine learning 
algorithms (Kalman Filters and LSTM)


## Roadmap

- Task Description and Background: This project uses the Argoverse 2 motion forecasting dataset that contains over 250,000 different scenarios from 6 different cities each recorded for 11 seconds at a frequency of 10hz. The data is given in pairs of (x, y) cartesian coordinates. The task requires a model to predict the location of a tracked object k steps into the future. Specially, the model must learn from the initial 5 seconds of the car’s location and predict the next 6 seconds. Successfully solving this task can have a great impact in our daily lives by improving the safety and reliability of autonomous driving. It is important to have a prediction model since there are potential instances where autonomous cars fail to recognize their surroundings using purely computer vision. It can act as a tool to insure safety as well as accuracy.

- Exploratory Data Analysis: The dataset consists of 233659 data points from six different cities. It is divided by 203816 training and 29843 testing data points. Each training data consists of the features and labels which are the previous 50 steps and the next 60 steps, respectively. On the other hand, each testing data only has the previous 50 steps. It will be the model's goal to produce the next 60 steps of the testing data. This forms an input dimension of 50 by 2 and an output dimension of 60 by 2. Each row consists of one (x,y) coordinate. Since the data is collected on frequency of 10Hz, each second consists of 10 points.

- Machine Learning Model: Four models were used in this project: Kalman Filter, LSTM (2 different variations), and Kalman Filter with Data Knowledge. The Kalman filter model uses the previous time steps to predict the next time step. Thus, no prior training is required. The LSTM deep learning model required a window shifting technique to feed and learn from the data. The Kalman filter with data knowledge is a custom Kalman filter model that uses prior data as a prior for future predictions.

- Experiment Design and Results: The cost function for this project was the mean squared error. The Kalman filter with data knowledge performed the best with an average MSE of 145.28 on the test set. The LSTM received an MSE of 166.83 while the vanilla Kalman filter received an MSE of 162.87.

- Discussion and Future Work: The lack of success in the LSTM was partly due to the computational restraints. Given the limited amount time, it was difficult to tune the hyperparameters and experiment with more deep learning models. Aside from this setback, I found the results of the Kalman filter quite surprising. In the future, I would like to implement an extended Kalman filter with an LSTM. This may allow for better predictions when the roads are curved since standard Kalman filters struggle with non linear systems.


## Optimizations

The deep learning model uses the Adam optimizer instead of the traditional stochastic gradient descent. The Adam optimizer has an adaptive learning rate with a momentum algorithm for a better convergence. Momentum helps find the direction of convergence by accumulating the gradients from past steps. For LSTM Model 1, the optimal learning rate turned out to be around 0.001. This learning rate was large enough to converge in 20 epochs but not large enough to miss the local minima. For LSTM Model 2, the optimal learning rate turned out to be around 0.01. Since this model took a longer time for training, only 10 epochs were used. Even with ten epochs, the momentum from the Adam optimizer helped quickly achieve the local minima.


There were two different approaches to using the city locations as part of the model design.

One model encompassing all cities: One approach is to create a single model that can be used across all cities. However, this approach is prone to having high bias and variance. From the data exploration, it is worth noting that the cities did not have unique (x,y) coordinates. This can potentially create difficulty for the model to recognize the patterns across each input. A strange overlap of two sequences from two different cities may cause wrong predictions in the testing phase. A single model does come with benefits. Only one model’s parameter needs to be tuned. This decreases the overall time need in parameter tuning. Furthermore, there is overall more data for the model to learn from.

Separate models based on each city: The other approach is to create a separate model for each city. By creating specific models tailored toward a single city, the complexity of model such as number of hidden layers and size can be reduced. Even with less complexity, the model produces the same or even better results as the other model. By decreasing the number of hidden layers and size, the training time is reduced significantly. The downside to having separate models is the manual labor required to tune the learning parameters for each model. Since every city is different, different parameters are required to produce accurate predictions. 6 separate models were created for each city in this project.
## Performance and Visuals

![Visual](https://i.postimg.cc/X7XCqDS0/Screen-Shot-2022-10-22-at-9-42-16-PM.png)

![Visual](https://i.postimg.cc/d3Y5wPSv/Screen-Shot-2022-10-22-at-9-43-06-PM.png)

![Visual](https://i.postimg.cc/6qtfFVyp/Screen-Shot-2022-10-22-at-9-42-55-PM.png)

![Visual](https://i.postimg.cc/dtWjSyvJ/Screen-Shot-2022-10-22-at-9-42-48-PM.png)

![Visual](https://i.postimg.cc/0QLLSBWH/Screen-Shot-2022-10-22-at-9-43-33-PM.png)