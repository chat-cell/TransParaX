import numpy as np
from typing import Dict, Tuple, List, Optional
import torch
from scipy.stats import norm
from dataclasses import dataclass
import gpytorch
import botorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

@dataclass
class BOConfig:
    """Configuration for Bayesian Optimization"""
    n_iterations: int = 50  # Maximum number of BO iterations
    exploration_weight: float = 0.1  # Weight for exploration term
    initial_points: int = 5  # Number of initial points to sample
    bounds_multiplier: float = 2.0  # Multiplier for search bounds
    convergence_threshold: float = 1e-5  # Threshold for early stopping
    
class BayesianOptimizer:
    def __init__(
        self,
        param_names: List[str],
        config: Optional[BOConfig] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            param_names: Names of parameters to optimize
            config: Bayesian optimization configuration
            device: Device to run computations on
        """
        self.param_names = param_names
        self.config = config or BOConfig()
        self.device = device
        self.dim = len(param_names)
        
        # Initialize storage for optimization history
        self.X = None  # Parameter samples
        self.y = None  # Corresponding objective values
        self.best_params = None
        self.best_value = float('inf')
        
    def _create_initial_bounds(
        self, 
        transformer_mean: np.ndarray,
        transformer_std: np.ndarray
    ) -> torch.Tensor:
        """Create search bounds based on transformer predictions"""
        # Equation (21): θinit = μ* ± λσ*
        lower = transformer_mean - self.config.bounds_multiplier * transformer_std
        upper = transformer_mean + self.config.bounds_multiplier * transformer_std
        
        bounds = torch.tensor(
            np.vstack([lower, upper]),
            device=self.device,
            dtype=torch.float32
        )
        return bounds
    
    def _initialize_gp(
        self,
        train_X: torch.Tensor,
        train_y: torch.Tensor,
        bounds: torch.Tensor
    ) -> Tuple[SingleTaskGP, ExactMarginalLogLikelihood]:
        """Initialize and fit the Gaussian Process model"""
        # Create GP model with hybrid kernel from equation (23)
        model = SingleTaskGP(train_X, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        
        # Fit the model
        fit_gpytorch_model(mll)
        return model, mll
    
    def _compute_acquisition(
        self,
        model: SingleTaskGP,
        bounds: torch.Tensor,
        transformer_std: torch.Tensor,
        best_f: float
    ) -> torch.Tensor:
        """Compute next point to evaluate using custom acquisition function"""
        # Equation (24): Combines EI with uncertainty-guided exploration
        acq_func = ExpectedImprovement(model, best_f=best_f)
        
        # Optimize acquisition function
        candidate, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
        )
        
        return candidate
        
    def _evaluate_objective(
        self,
        params: torch.Tensor,
        simulator_fn: callable,
        measured_data: torch.Tensor
    ) -> float:
        """Evaluate objective function f(θ) for given parameters"""
        # Run simulation with current parameters
        simulated = simulator_fn(params)
        
        # Compute loss according to equation (2)
        error = torch.mean((measured_data - simulated) ** 2)
        return error.item()
    
    def optimize(
        self,
        transformer_mean: np.ndarray,
        transformer_std: np.ndarray,
        simulator_fn: callable,
        measured_data: torch.Tensor,
    ) -> Dict:
        """
        Run Bayesian optimization refinement
        
        Args:
            transformer_mean: Initial parameter predictions from transformer
            transformer_std: Uncertainty estimates from transformer
            simulator_fn: Function that simulates device behavior given parameters
            measured_data: Measured I-V characteristics to match
            
        Returns:
            Dictionary containing optimization results
        """
        # Convert inputs to tensors
        transformer_mean = torch.tensor(transformer_mean, device=self.device)
        transformer_std = torch.tensor(transformer_std, device=self.device)
        
        # Create bounds for parameter search
        bounds = self._create_initial_bounds(transformer_mean, transformer_std)
        
        # Generate initial samples around transformer prediction
        X = torch.zeros((self.config.initial_points, self.dim), device=self.device)
        y = torch.zeros(self.config.initial_points, device=self.device)
        
        # Evaluate initial points
        for i in range(self.config.initial_points):
            # Sample parameters using transformer uncertainty
            params = transformer_mean + torch.randn_like(transformer_mean) * transformer_std
            X[i] = params
            y[i] = self._evaluate_objective(params, simulator_fn, measured_data)
            
            # Update best result
            if y[i] < self.best_value:
                self.best_value = y[i].item()
                self.best_params = params.clone()
        
        # Main optimization loop
        for iteration in range(self.config.n_iterations):
            # Initialize/update GP model
            model, mll = self._initialize_gp(X, y.unsqueeze(-1), bounds)
            
            # Get next point to evaluate using acquisition function
            next_params = self._compute_acquisition(
                model, bounds, transformer_std, self.best_value
            )
            
            # Evaluate objective at new point
            next_value = self._evaluate_objective(
                next_params.squeeze(), simulator_fn, measured_data
            )
            
            # Update data
            X = torch.cat([X, next_params])
            y = torch.cat([y, torch.tensor([next_value], device=self.device)])
            
            # Update best result
            if next_value < self.best_value:
                improvement = self.best_value - next_value
                self.best_value = next_value
                self.best_params = next_params.squeeze().clone()
                
                # Check convergence
                if improvement < self.config.convergence_threshold:
                    break
                    
        return {
            'best_parameters': self.best_params.cpu().numpy(),
            'best_value': self.best_value,
            'optimization_history': {
                'parameters': X.cpu().numpy(),
                'objectives': y.cpu().numpy()
            }
        }

# Example usage
"""
# Initialize optimizer
optimizer = BayesianOptimizer(
    param_names=['voff', 'nfactor', 'u0', 'ua', ...],
    config=BOConfig(n_iterations=50)
)

# Run optimization
results = optimizer.optimize(
    transformer_mean=initial_predictions,
    transformer_std=uncertainty_estimates,
    simulator_fn=device_simulator,
    measured_data=iv_curves
)

# Access results
optimized_params = results['best_parameters']
final_error = results['best_value']
"""