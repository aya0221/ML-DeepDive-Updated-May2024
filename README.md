## ML-Fundamentals
Containing detailed explanations and practical code examples for key machine learning concepts, including Kalman Filters, Gradient Descent, and CNN architectures.

## Kalman Filter Application in Object Tracking

### Overview
I employed the Kalman Filter in object tracking tasks to enhance the tracking accuracy of balls in video sequences. This method improves tracking by using data from a CNN-based object detector, which, while effective, often includes some errors or 'noise.' These errors can arise due to variations in lighting, partial blockages of the object, or limitations of the detector itself. The Kalman Filter helps correct these inaccuracies by blending the imperfect measurements with predicted states from previous data, leading to more reliable tracking outcomes!

### Components
- **State Vector $x$**: Includes position $(x, y)$ and velocity $(vx, vy)$.
    Velocity is derived from changes in position over time, calculated as $$vx = (x_current - x_previous) / Δt$$  and   $$vy = (y_current - y_previous) / Δt$$, where $Δt$ is the time interval between frames
- **Measurement Vector $z$**: Consists of the detected positions from the CNN, subject to measurement noise
- **State Transition Matrix $A$**: Defines the linear relationship between the current state and the next state, factoring in time dynamics
- **Control Matrix $B$ and Vector $u$**: Generally set to zero, indicating no external forces affecting the motion of the object
- **Process Noise Covariance $Q$**: Quantifies the expected variability in the system dynamics, based on empirical data
- **Measurement Noise Covariance $R$**: Represents the accuracy of the measurements, calculated from the variance of the CNN's outputs
- **Measurement Matrix $H$**: Maps the state vector to the measurement vector, primarily focusing on the positional components and excluding velocity

### Mathematical Model
The Kalman Filter updates in two steps: Prediction and Correction. During the prediction step, the next state of the ball is estimated:


$$x_predicted = x + vx * Δt$$

$$y_predicted = $y + vy * Δt$$


The correction step adjusts this prediction based on the new measurement:

$$K = P^- H^T (H P^- H^T + R)^{-1}$$

$$x_corrected = x_predicted + K(z - H * x_predicted)$$

$$P_corrected = (I - K H) P^-$$

Where $K$ is the Kalman Gain, $P^-$ is the predicted covariance, $I$ is the identity matrix, and $z$ is the actual measurement from the object detector

### Utilization of Velocity
Velocity is critical for predicting the future position of the ball, particularly in cases of occlusion or when the ball moves out of the camera frame. By estimating velocity, the Kalman Filter can project the ball's trajectory, enhancing tracking continuity and robustness under varying conditions.

### How Kalman Filter is useful for real-time object tracking
The implementation of the Kalman Filter in this project demonstrates its efficacy in combining real-time object detection with predictive tracking, providing a robust solution for tracking objects in motion within noisy environments
