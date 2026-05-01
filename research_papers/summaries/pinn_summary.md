# Research Paper Summary: PINN (Physics-Informed Neural Networks)

## Citation
Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

## Key Technical Contributions
1.  **Physics-Constrained Loss:** Instead of just minimizing error against ground truth, the loss function includes a penalty for violating physical laws (e.g., the Navier-Stokes equations for fluid flow).
2.  **Data-Efficient Learning:** By leveraging the "prior knowledge" of physics, PINNs can learn accurate models with significantly less training data than purely black-box networks.
3.  **Automatic Differentiation:** Uses the underlying neural network's graph to calculate the derivatives needed for the physical equations, making it seamless to integrate with deep learning frameworks.

## Relevance to this Project
In air quality, pollutants follow the **Advection-Diffusion Equation**. We use PINN concepts in Generation 5 to prove that our predictions are not just "guessing" but are physically consistent with how air moves through a city.
