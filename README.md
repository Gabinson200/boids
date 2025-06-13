# Interactive 3D Boids Simulation with Hand Gesture Control

This project is a real-time, interactive 3D simulation of a classic boids flocking algorithm, built with Three.js. It features advanced hand gesture controls powered by Google's MediaPipe Hand Landmarker, allowing users to influence the flock's behavior directly through their webcam.

[**▶️ View the Live Demo Here**](https://github.com/Gabinson200/boids/)

![Boids Simulation Screenshot](https://github.com/user-attachments/assets/ea65bfc1-5e1d-4c70-9a54-f221df9cf92d)


---

## Features

- **Classic 3D Boids Algorithm**: Implements the three core rules of flocking behavior:
  - **Cohesion**: Boids steer towards the average position of their neighbors.
  - **Separation**: Boids steer to avoid crowding local flockmates.
  - **Alignment**: Boids steer towards the average heading of their neighbors.
- **Real-Time Hand Interaction**: Utilizes a webcam and the MediaPipe Hand Landmarker to track the user's hand position and gestures in real time.
- **Advanced Gesture Controls**:
  - **Hand Presence**: A control sphere appears in the simulation as soon as a hand is detected.
  - **Pinch to Attract**: Pinching the thumb and index finger activates an attractive force, pulling the boids towards the sphere.
  - **Extend Pinky to Repel**: Extending the pinky finger away from the other fingers triggers a powerful repulsive "explosion," scattering any nearby boids.
- **Dynamic 3D Control**: The control sphere's position is mapped to all three axes (X, Y, Z), with depth controlled by the hand's distance from the camera. The reference frame is locked on the initial gesture, allowing the user to orbit the scene with the mouse while maintaining control.
- **Interactive UI Panel**: A sidebar allows for real-time adjustments to simulation parameters, including boid count, force strengths (cohesion, separation, alignment), and visual ranges.
- **Dynamic Visual Feedback**: The control sphere changes color from cool blue (far) to hot orange (close) to provide clear visual feedback of its depth in the scene.
- **Optimized Performance**:
  - **`InstancedMesh`**: Renders all boids in a single draw call for maximum GPU efficiency.
  - **Hash Grid**: Employs a spatial partitioning grid to dramatically speed up neighbor-finding calculations, a critical optimization for flocking algorithms.
  - **Object Pooling**: Pre-allocates and reuses `Vector3` objects during calculations to minimize garbage collection pauses and ensure a smooth animation.
  - **Capped Pixel Ratio**: Limits the rendering resolution on high-DPI screens to reduce GPU load.

---

## Technologies Used

-   **HTML5 & CSS3**
-   **JavaScript (ES Modules)**
-   **Three.js**: For 3D rendering and scene management.
-   **Google MediaPipe Hand Landmarker**: For real-time hand tracking and gesture recognition.
-   **Tailwind CSS**: For styling the user interface components.

---

## Future Work

The goal of this website was to familiarize myself with boids so that I can create more robust code that utlizies boids to inform a bird flock counting computer vision algorithm. 
