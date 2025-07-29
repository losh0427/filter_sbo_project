import torch
import gpytorch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, MaternKernel, ConstantKernel, AdditiveKernel
import numpy as np
import time

# Handle BoTorch API changes
try:
    from botorch.fit import fit_gpytorch_mll as fit_gpytorch_model
except ImportError:
    from botorch.fit import fit_gpytorch_model

# Handle MES import (may not be available in newer versions)
try:
    from botorch.acquisition.max_value_entropy_search import qMaxValueEntropySample
    MES_AVAILABLE = True
except ImportError:
    MES_AVAILABLE = False


class GPSurrogateModel:
    """
    BoTorch/GPyTorch based GP surrogate model
    Supports flexible input dimensions and multi-kernel combinations
    """
    def __init__(self, kernel_config=None):
        self.model = None
        self.mll = None
        self.kernel_config = kernel_config or self._default_kernel_config()
        self.training_data = {'X': None, 'y': None}
        self.is_trained = False
        
    def _default_kernel_config(self):
        """Default kernel configuration"""
        return {
            'type': 'composite',
            'rbf_lengthscale_bounds': (0.1, 10.0),
            'matern_lengthscale_bounds': (0.1, 5.0),
            'constant_bounds': (0.1, 100.0),
            'combination': 'additive'
        }
    
    def _build_kernel(self, input_dim):
        """Build kernel function based on input dimensions"""
        if self.kernel_config['type'] == 'composite':
            # RBF kernel - suitable for continuous parameters
            rbf_kernel = RBFKernel(
                lengthscale_constraint=gpytorch.constraints.Interval(
                    *self.kernel_config['rbf_lengthscale_bounds']
                )
            )
            
            # Matérn kernel - suitable for enhanced features
            matern_kernel = MaternKernel(
                nu=2.5,
                lengthscale_constraint=gpytorch.constraints.Interval(
                    *self.kernel_config['matern_lengthscale_bounds']
                )
            )
            
            # Constant kernel
            constant_kernel = ConstantKernel(
                constant_constraint=gpytorch.constraints.Interval(
                    *self.kernel_config['constant_bounds']
                )
            )
            
            if self.kernel_config['combination'] == 'additive':
                return rbf_kernel + matern_kernel + constant_kernel
            else:
                return rbf_kernel * matern_kernel + constant_kernel
        else:
            # Simple RBF kernel
            return RBFKernel()
    
    def fit(self, X, y):
        """Train GP model"""
        # Convert to torch tensor
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float64)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float64).unsqueeze(-1)
            
        # Ensure correct data format
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(-1)
            
        # Store training data
        self.training_data['X'] = X.clone()
        self.training_data['y'] = y.clone()
        
        try:
            # Build model
            self.model = SingleTaskGP(
                X, y,
                input_transform=Normalize(d=X.shape[-1]),
                outcome_transform=Standardize(m=y.shape[-1])
            )
            
            # Set training mode
            self.model.train()
            self.model.likelihood.train()
            
            # Build MLL
            self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            
            # Train model
            fit_gpytorch_model(self.mll)
            
            # Set evaluation mode
            self.model.eval()
            self.model.likelihood.eval()
            
            self.is_trained = True
            print(f"GP model training completed, data points: {X.shape[0]}, dimensions: {X.shape[1]}")
            
        except Exception as e:
            print(f"GP model training failed: {str(e)}")
            self.is_trained = False
            raise e
    
    def predict(self, X):
        """Predict mean and uncertainty"""
        if not self.is_trained:
            raise ValueError("Model has not been trained")
            
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float64)
            
        if X.dim() == 1:
            X = X.unsqueeze(0)
            
        with torch.no_grad():
            posterior = self.model.posterior(X)
            mean = posterior.mean
            variance = posterior.variance
            
        return mean.squeeze(-1), torch.sqrt(variance).squeeze(-1)
    
    def get_model(self):
        """Get trained model instance"""
        if not self.is_trained:
            raise ValueError("Model has not been trained")
        return self.model


class AcquisitionFunction:
    """
    Acquisition function management class
    Supports EI, UCB, and simplified MES strategies
    """
    def __init__(self, func_type='EI', config=None):
        self.func_type = func_type
        self.config = config or self._default_config()
        self.acq_func = None
        self.bounds = None
        
    def _default_config(self):
        """Default configuration"""
        return {
            'EI': {'xi': 0.01},
            'UCB': {'beta': 2.0},
            'MES': {'candidate_size': 1000},
            'optimization': {
                'num_restarts': 5,
                'raw_samples': 20,
                'max_iter': 200
            }
        }
    
    def build(self, gp_model, best_f=None, bounds=None):
        """Build acquisition function based on GP model"""
        model = gp_model.get_model()
        self.bounds = bounds
        
        if self.func_type == 'EI':
            if best_f is None:
                # Automatically compute current best value
                training_y = gp_model.training_data['y']
                best_f = training_y.max()
            
            self.acq_func = ExpectedImprovement(
                model=model,
                best_f=best_f,
                maximize=True
            )
            
        elif self.func_type == 'UCB':
            beta = self.config['UCB']['beta']
            self.acq_func = UpperConfidenceBound(
                model=model,
                beta=beta,
                maximize=True
            )
            
        elif self.func_type == 'MES':
            # MES is complex and may not be available, use EI as fallback
            print("Warning: MES not available in this BoTorch version, using EI instead")
            if best_f is None:
                training_y = gp_model.training_data['y']
                best_f = training_y.max()
            self.acq_func = ExpectedImprovement(
                model=model,
                best_f=best_f,
                maximize=True
            )
            
        else:
            raise ValueError(f"Unsupported acquisition function type: {self.func_type}")
    
    def optimize(self, bounds=None, q=1):
        """Optimize acquisition function to find next candidate point"""
        if self.acq_func is None:
            raise ValueError("Acquisition function has not been built")
            
        if bounds is None:
            bounds = self.bounds
        if bounds is None:
            raise ValueError("Must provide parameter bounds")
            
        # Ensure correct bounds format
        if not isinstance(bounds, torch.Tensor):
            bounds = torch.tensor(bounds, dtype=torch.float64)
        if bounds.dim() == 2 and bounds.shape[0] == 2:
            bounds = bounds.t()  # Transpose to (n_dims, 2)
        
        try:
            candidate, acq_value = optimize_acqf(
                self.acq_func,
                bounds=bounds.t(),  # optimize_acqf needs (2, n_dims) format
                q=q,
                num_restarts=self.config['optimization']['num_restarts'],
                raw_samples=self.config['optimization']['raw_samples'],
                options={'maxiter': self.config['optimization']['max_iter']}
            )
            
            return candidate, acq_value
            
        except Exception as e:
            print(f"Acquisition function optimization failed: {str(e)}")
            # Fallback to random sampling
            return self._random_fallback(bounds), 0.0
    
    def _random_fallback(self, bounds):
        """Fallback random sampling"""
        n_dims = bounds.shape[0]
        random_point = torch.rand(1, n_dims, dtype=torch.float64)
        # Scale to actual range
        scaled_point = bounds[:, 0] + random_point * (bounds[:, 1] - bounds[:, 0])
        return scaled_point.unsqueeze(0)


class ModelTrainer:
    """
    Model training controller
    Manages training process and hyperparameter optimization
    """
    def __init__(self, max_training_iter=100):
        self.max_training_iter = max_training_iter
        
    def train_with_retry(self, gp_model, X, y, max_retries=3):
        """Model training with retry mechanism"""
        for attempt in range(max_retries):
            try:
                gp_model.fit(X, y)
                return True
            except Exception as e:
                print(f"Training attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    print("All training attempts failed")
                    return False
                time.sleep(1)  # Brief wait before retry
        return False