## ML-Fundamentals
Containing detailed explanations and practical code examples for key machine learning concepts, including Kalman Filters, Gradient Descent, and CNN architectures.

- [Kalman Filter Application in Object Tracking](#kalman-filter-application-in-object-tracking)
- [Gradient Gradient Descent Back Propagation SGD Application of Gradient](#gradient-gradient-descent-back-propagation-sgd-application-of-gradient)
- [Conv1D and Conv2D](#conv1d-and-conv2d)
- [Non-linear CNN](#non-linear-cnn)
- [Sources](#sources)


# Kalman Filter Application in Object Tracking

- **Overview**

    I utilized the Kalman Filter in object tracking tasks to enhance the accuracy of tracking balls in video sequences. This method leverages data from a CNN-based object detector, which, while effective, is susceptible to errors or noise due to factors like variable lighting, partial obstructions, or inherent limitations of the detector. The Kalman Filter addresses these inaccuracies by merging imperfect measurements with predicted states from prior data, resulting in more reliable tracking outcomes. The Kalman Filter excels because it continuously estimates the state of a dynamic system, refines those estimates with new measurements, and predicts future states. Essentially, it involves propagating and updating Gaussians and their covariances. Starting with the current state and estimates, we predict the next state. The subsequent correction step incorporates a noisy measurement to refine the state update.

- **My Application of the Kalman Filter in Video Object Tracking**

    After detecting objects using a CNN-based detection system, the Kalman Filter is employed to process the noisy measurements of object positions. The steps involved are:
    - **Predict**: Estimating the object's future location based on its current state
    - **Correct**: Refining this prediction with new measurement data to improve the tracking accuracy
      <img src="https://github.com/aya0221/ML-Fundamentals/assets/69786640/2f370ff1-3541-4223-a7bd-29541c514e36" width="40%"> 
      ※1


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


-  **Operation Cycle of the Kalman Filter**

   The Kalman Filter operates through two principal phases: **Prediction** and **Correction**. Each step employs mathematical models and matrices to project and adjust the state estimates of a dynamic system

     - **Prediction**

       This phase uses the state transition matrix to forecast future state estimates based on current estimates:
         - **Predicted State Estimate**: $$\hat{x}^- = F \hat{x}$$
         - **Predicted Covariance Estimate**: $$P^- = FPF^T + Q$$
             , where $\(F\)$ is the state transition matrix that describes how the state variables are expected to evolve from one time step to the next, $\(P\)$ is the covariance matrix of the previous estimate indicating the uncertainty associated with that estimate, $\(Q\)$ is the process noise covariance matrix accounting for the process uncertainty, and $\(\hat{x}\)$ is the prior state estimate
  


     - **Correction**

       This phase incorporates new measurement data to refine the predictions made in the previous step:
         - **Kalman Gain Calculation**: $$K = P^- H^T (H P^- H^T + R)^{-1}$$
         - **Updated State Estimate**: $$\hat{x} = \hat{x}^- + K(z - H \hat{x}^-)$$
         - **Updated Covariance Estimate**: $$P = (I - KH) P^-$$
             , where $\(K\)$ is the Kalman Gain which determines the extent to which the new measurement is incorporated into the state estimate, $\(P^-\)$ is the predicted covariance matrix from the prediction phase, $\(H\)$ is the measurement matrix that relates the state estimate to the measurement domain, $\(R\)$ is the measurement noise covariance matrix reflecting the uncertainty in the measurements, $\(I\)$ is the identity matrix, and $\(z\)$ is the new measurement


<img src="https://github.com/aya0221/ML-Fundamentals/assets/69786640/df49494d-4ae1-4587-ade9-282ec67b5f32" width="40%"> 
※2

# Gradient, Gradient Descent, Back Propagation, SGD, Application of Gradient

- **Gradient** is a vector of partial derivatives, represented as $$\nabla f(x)$$, which *points in the direction of the greatest increase of a function*. In machine learning, we use the gradient to update the weights of models, *moving in the direction that most reduces the loss*. This is computed as:
  $$\nabla f(x) = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n}\right]^T$$

- **Gradient Descent** is an optimization algorithm used to minimize a function by iteratively moving in the direction of the steepest descent, defined by the *negative* of the gradient. The goal is to find the model parameters that minimize a loss function. For a parameter \( p \):
  $$p = p - \alpha \frac{\partial \mathcal{L}}{\partial p}$$
  , where $\( \alpha \)$ is the learning rate and $\( \mathcal{L} \)$ is the loss function

- **Back Propagation** is used to calculate the gradient required in the gradient descent step of neural network training. This involves computing the gradient of the loss function with respect to each weight by applying the **chain rule**, working backward from the output layer to the input layer:
  $$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial w}$$
  , where \( y \) is the output and \( w \) are the weights

- **Difference Between SGD and Gradient Descent**:
    - **Stochastic Gradient Descent (SGD)** updates the parameters using only a small subset of the data, which can lead to faster convergence on large datasets.
    - **Batch Gradient Descent** uses the entire dataset to perform one update at a time, providing a more stable but slower convergence.

- **Application of Gradient**:
  During training, the parameters of the model are repeatedly adjusted using either the whole dataset (batch) or subsets of it (mini-batches), to minimize the loss function over multiple iterations or epochs. The gradient provides the necessary direction for this adjustment.

# Conv1D and Conv2D

- **Conv1D** is used for processing 1D data, such as time-series or audio signals, where the layer will learn from patterns occurring over time.
- **Conv2D** is used for processing 2D data, such as images. These layers learn features from the input by applying filters that capture spatial hierarchies, identifying simple edges in early layers and more complex features like textures in deeper layers.

# Non-linear CNN

- A Non-linear CNN incorporates non-linear activation functions like ReLU (Rectified Linear Unit) to introduce non-linear properties into the model, allowing it to learn more complex patterns. Without non-linearities, CNNs would behave just like a single linear classifier, limiting their ability to model complex relationships in the data.




# Sources
※1:[Sort and Deep-SORT Based Multi-Object Tracking for Mobile Robotics: Evaluation with New Data Association Metrics](https://www.researchgate.net/publication/358134782_Sort_and_Deep-SORT_Based_Multi-Object_Tracking_for_Mobile_Robotics_Evaluation_with_New_Data_Association_Metrics)
※2: [Articles / Artificial Intelligence / Computer vision](https://www.codeproject.com/Articles/865935/Object-Tracking-Kalman-Filter-with-Ease)
