## ML-Fundamentals
Containing detailed explanations and code snippets of ML

Topics covered: Kalman Filters, Gradient Descent, and CNN architectures

- [Kalman Filter Application in Object Tracking](#kalman-filter-application-in-object-tracking)
- [Gradient Gradient Descent Back Propagation SGD Application of Gradient](#gradient-gradient-descent-back-propagation-sgd-application-of-gradient)
- [Conv1D and Conv2D](#conv1d-and-conv2d)
- [Non-linear CNN](#non-linear-cnn)
- [Sources](#sources)


# Kalman Filter Application in Object Tracking

- **Overview**

  I employed the **Kalman Filter** to refine the tracking of balls in video sequences, utilizing measurements from a CNN-based object detector. Despite the efficacy of CNN detectors, they are prone to measurement errors influenced by varying lighting conditions, obstructions, and detector limitations. The Kalman Filter enhances tracking precision by effectively merging these imperfect measurements with predicted states. This process involves continuously estimating and updating the system's states (position) and velocity (based on both predicted dynamics and new measurements). Each cycle of the filter predicts future states and corrects these predictions as new data becomes available.

- **Application of the Kalman Filter in Video Object Tracking**

  The use of the Kalman Filter in my project follows a straightforward two-step process, aligning with the typical predict-correct cycle of state estimation:
    - **Predict**: Compute the predicted next state of the ball using the current state estimates, factoring in the expected motion over time
    - **Correct**: Adjust the predicted state using fresh, albeit noisy, measurements from the CNN-based detection system to refine the position and velocity estimates

      <img src="https://github.com/aya0221/ML-Fundamentals/assets/69786640/2f370ff1-3541-4223-a7bd-29541c514e36" width="40%"> ※1
      <img src="https://github.com/aya0221/ML-Fundamentals/assets/69786640/df49494d-4ae1-4587-ade9-282ec67b5f32" width="30%"> ※2


-  **Variables used in the Kalman Filter Mathematical Model**
    - **State Vector \(x(t)\)**: Represents the estimated states of the system, including position \((x, y)\) and velocity \((vx, vy)\). Velocity is derived from changes in position over time, calculated as:
  $$vx = \frac{(x_{\text{current}} - x_{\text{previous}})}{\Delta t}$$
  $$vy = \frac{(y_{\text{current}} - y_{\text{previous}})}{\Delta t}$$
  where $\(\Delta t\)$ is the time interval between frames

    - **Measurement Vector \(z(t)\)**: The observed positions from a CNN (detection system), which can be noisy

    - **Process Covariance Matrix \(P(t|t-1)\)**: Represents the estimate of the covariance of the state vector

    - **State Transition Matrix \(F\)**: Describes how the state evolves from one point to the next without considering the new measurements. Defines the linear relationship between the current state and the next state, factoring in time dynamics

    - **Process Noise Covariance \(Q\)**: Reflects the uncertainty in the model's predictions. Quantifies the expected variability in the system dynamics, based on empirical data

    - **Measurement Noise Covariance \(R\)**: Reflects the uncertainty in the observed measurements. Represents the accuracy of the measurements, calculated from the variance of the CNN's outputs

    - **Control Matrix \(B\) and Vector \(u\)**: Generally set to zero, indicating no external forces affecting the motion of the object

    - **Measurement Matrix \(H\)**: Relates (maps) the state vector to the measurement vector, focusing primarily on the position components


-  **Operation Cycle of the Kalman Filter: Prediction & Correction**
   1. **Prediction**

       This phase uses the state transition matrix to forecast future state estimates based on current estimates:
         - **Predicted State Estimate**: $$\hat{x}^- = F \hat{x}$$
         - **Predicted Covariance Estimate**: $$P^- = FPF^T + Q$$
             , where $\(F\)$ is the state transition matrix that describes how the state variables are expected to evolve from one time step to the next, $\(P\)$ is the covariance matrix of the previous estimate indicating the uncertainty associated with that estimate, $\(Q\)$ is the process noise covariance matrix accounting for the process uncertainty, and $\(\hat{x}\)$ is the prior state estimate
   2. **Correction**

       This phase incorporates new measurement data to refine the predictions made in the previous step:
         - **Kalman Gain Calculation**: $$K = P^- H^T (H P^- H^T + R)^{-1}$$
         - **Updated State Estimate**: $$\hat{x} = \hat{x}^- + K(z - H \hat{x}^-)$$
         - **Updated Covariance Estimate**: $$P = (I - KH) P^-$$
             , where $\(K\)$ is the Kalman Gain which determines the extent to which the new measurement is incorporated into the state estimate, $\(P^-\)$ is the predicted covariance matrix from the prediction phase, $\(H\)$ is the measurement matrix that relates the state estimate to the measurement domain, $\(R\)$ is the measurement noise covariance matrix reflecting the uncertainty in the measurements, $\(I\)$ is the identity matrix, and $\(z\)$ is the new measurement
  
    here is the code snippet:
    ```
    import numpy as np
    
    class BallTracker:
        def __init__(self, process_noise_std, measurement_noise_std, dt):
            """
            Initialize the ball tracking Kalman Filter
            :param process_noise_std: Standard deviation of the process noise
            :param measurement_noise_std: Standard deviation of the measurement noise
            :param dt: Time interval between measurements
            """
            
            # Define the initial state [x, y, vx, vy] (position and velocity)
            self.x = np.zeros((4, 1))
            
            # Define the state transition model
            self.A = np.array([[1, 0, dt, 0],
                               [0, 1, 0, dt],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
            
            # Define the observation model
            self.H = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0]])
            
            # Define the process noise covariance
            q = process_noise_std**2
            self.Q = np.array([[q, 0, 0, 0],
                               [0, q, 0, 0],
                               [0, 0, q, 0],
                               [0, 0, 0, q]])
            
            # Define the measurement noise covariance
            r = measurement_noise_std**2
            self.R = np.array([[r, 0],
                               [0, r]])
            
            # Initialize the covariance of the state estimate
            self.P = np.eye(4)
        
        def predict(self):
            # Predict the next state
            self.x = np.dot(self.A, self.x)
            self.P = np.dot(self.A, np.dot(self.P, self.A.T)) + self.Q
        
        def update(self, z):
            # Update the state with a new measurement
            y = z - np.dot(self.H, self.x)
            S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
            K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
            self.x += np.dot(K, y)
            self.P = self.P - np.dot(K, np.dot(self.H, self.P))
    
        def get_estimated_position(self):
            return self.x[0, 0], self.x[1, 0]
    
    tracker = BallTracker(process_noise_std=1e-2, measurement_noise_std=1e-1, dt=0.1)
    measurements = [(5, 5), (6, 6), (7, 7), (8, 8)]  # CNN-based measurements (here, the numbers are random)
    
    for measurement in measurements:
        tracker.predict()
        tracker.update(np.array([[measurement[0]], [measurement[1]]]))
        print("estimated position is :", tracker.get_estimated_position())
    ```

# Gradient, Gradient Descent, Back Propagation, SGD, Application of Gradient

- **Gradient** is a vector of partial derivatives, represented as $$\nabla f(x)$$, which *points in the direction of the greatest increase of a function*. In machine learning, we use the gradient to update the weights of models, *moving in the direction that most reduces the loss*. This is computed as:
  $$\nabla f(x) = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n}\right]^T$$

- **Gradient Descent** is an optimization algorithm used to minimize a function by iteratively moving in the direction of the steepest descent, defined by the *negative* of the gradient. The goal is to find the model parameters that minimize a loss function. For a parameter \( p \):
  $$p = p - \alpha \frac{\partial \mathcal{L}}{\partial p}$$
  , where $\( \alpha \)$ is the learning rate and $\( \mathcal{L} \)$ is the loss function

- **Back Propagation** is used to calculate the gradient required in the gradient descent step of neural network training. This involves computing the gradient of the loss function with respect to each weight by applying the **chain rule**, working backward from the output layer to the input layer:
  $$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial w}$$
  , where $\( y \)$ is the output and $\( w \)$ are the weights
    - e.g., Mean Squared Error (MSE) is defined as: $$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
          , where $\(y_i\)$ are the true values, $\(\hat{y}_i\)$ are the predicted values, and $\(n\)$ is the number of samples

- **Difference Between SGD and Gradient Descent**:

  let the dataset:
    ```
    data = tf.constant([[2.0], [3.0], [4.0], [5.0]], dtype=tf.float32)
    targets = tf.constant([[4.0], [6.0], [8.0], [10.0]], dtype=tf.float32)
    ```

    Code Below are the same in both GD and SGD on Pytorch/Keras respectively

  ex1, Pytorch

    ```
    import torch
    
    # Define a simple dataset
    data = torch.tensor([[2.0], [3.0], [4.0], [5.0]])
    targets = torch.tensor([[4.0], [6.0], [8.0], [10.0]])
    
    # Initialize parameter with requires_grad
    w = torch.tensor([1.0], requires_grad=True)
    
    # Learning rate and epochs
    lr = 0.01
    epochs = 100
    ```

  ex2, Keras
  
    ```
    import tensorflow as tf
    # Define model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,))
    ])
    
    # Compile model with SGD optimizer
    model.compile(optimizer=tf.optimizers.SGD(learning_rate=0.01), loss='mse')
    ```
 
    - **Stochastic Gradient Descent (SGD)** updates the parameters using only a small subset of the data, which can lead to faster convergence on large datasets.
      
      ex1. Pytorch:
        ```
        for epoch in range(epochs):
            total_loss = 0
            for x, y in zip(data, targets):
                model_output = x * w
                loss = (y - model_output)**2
                loss.backward()  # Compute gradients
                with torch.no_grad():
                    w -= lr * w.grad  # Update parameters
                w.grad.zero_()  # Zero gradients
                total_loss += loss.item()
        
            print(f'Epoch {epoch+1}, Loss: {total_loss / len(data)}, w: {w.item()}')
        ```

        ex2. Keras:

        ```
        # Fit model with a batch size of 1 for true SGD behavior
        model.fit(data, targets, epochs=100, batch_size=1)
        ```
        
    - **Gradient Descent** uses the entire dataset to perform one update at a time, providing a more stable but slower convergence.
      
      ex1. Pytorch:
      ```
      for epoch in range(epochs):
        model_output = data * w
        loss = torch.mean((targets - model_output)**2)
        loss.backward()  # Compute gradients
        with torch.no_grad():
            w -= lr * w.grad  # Update parameters
        w.grad.zero_()  # Zero gradients
    
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, w: {w.item()}')
      ```
      

      ex2. Keras:
      ```
      model.fit(data, targets, epochs=100)
      ```

- **Application of Gradient**:
  During training, the parameters of the model are repeatedly adjusted using either the whole dataset (batch) or subsets of it (mini-batches), to minimize the loss function over multiple iterations or epochs. The gradient provides the necessary direction for this adjustment.
  
  (bonus ~ this is how SGD from Scratch for Gaussian Probability Density Function looks)
  
    ```
    import numpy as np
    
    # Sample data generated from a normal distribution
    data = np.random.normal(loc=0, scale=1, size=100)
    
    # Parameters to optimize (mean and standard deviation)
    mean = 5.0  # initial guess
    std_dev = 10.0  # initial guess
    
    learning_rate = 0.01
    epochs = 100
    
    for epoch in range(epochs):
        # Compute gradients for both parameters
        d_mean = np.mean((mean - data) / std_dev**2)
        d_std_dev = np.mean(((mean - data)**2 - std_dev**2) / std_dev**3)
        
        # Update parameters
        mean -= learning_rate * d_mean
        std_dev -= learning_rate * d_std_dev
    
        print(f'Epoch {epoch+1}: mean = {mean}, std_dev = {std_dev}')
    ```
  
# Conv1D and Conv2D

- **Conv1D** is used for processing 1D data, such as time-series or audio signals, where the layer will learn from patterns occurring over time
  
    ```
    model_1d = tf.keras.Sequential([
      tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(128, 1))
    ])
    ```

- **Conv2D** is used for processing 2D data, such as images. These layers learn features from the input by applying filters that capture spatial hierarchies, identifying simple edges in early layers and more complex features like textures in deeper layers.
  
    ```
    model_1d = tf.keras.Sequential([
      tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1))
    ])
    ```


# Non-linear CNN
#### ReLU (Rectified Linear Unit)
- **Formula**:
  $$f(x) = \max(0, x)$$
- **Use Cases**:
  - everything
  - helps with the vanishing gradient problem, allowing models to learn faster and perform better
    ```
    def relu(x):
        return max(0, x)
    ```

#### Sigmoid
- **Formula**:
  $$f(x) = \frac{1}{1 + e^{-x}}$$
- **Use Cases**:
  - binary classification problems at the output layer where the result is mapped between 0 and 1, making it interpretable as a probability
  - useful in models where we need to calculate probabilities that add up to one (e.g., logistic regression)
    ```
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    ```

#### Tanh (Hyperbolic Tangent)
- **Formula**:
  $$f(x) = \tanh(x) = \frac{2}{1 + e^{-2x}} - 1$$
- **Use Cases**:
  - often used in hidden layers as it centers the output between -1 and 1, which can lead to better training performance in certain cases
  - useful for feature representation in dl where data normalization might be important

#### Softmax
- **Formula**:
  $$f(x_i) = \frac{e^{x_i}}{\sum_{k} e^{x_k}}$$
- **Use Cases**:
  - typically used in the output layer to perform multi-class categorization - each output can be interpreted as a probability that the input belongs to one of the classes
  - classification problems where only one result is the correct output (e.g., classifying images of digits)

#### Leaky ReLU
- **Formula**:
  $$f(x) = \max(0.01x, x)$$
- **Use Cases**:
  - addresses the "dying ReLU" problem by allowing a small gradient when the unit is not active
  - useful in scenarios where ReLU might result in a lot of dead neurons

# Sources
※1:[Sort and Deep-SORT Based Multi-Object Tracking for Mobile Robotics: Evaluation with New Data Association Metrics](https://www.researchgate.net/publication/358134782_Sort_and_Deep-SORT_Based_Multi-Object_Tracking_for_Mobile_Robotics_Evaluation_with_New_Data_Association_Metrics)
※2: [Articles / Artificial Intelligence / Computer vision](https://www.codeproject.com/Articles/865935/Object-Tracking-Kalman-Filter-with-Ease)
