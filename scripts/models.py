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
import warnings

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
    BoTorch/GPyTorch based GP surrogate model with improved numerical stability
    """
    def __init__(self, kernel_config=None):
        self.model = None
        self.mll = None
        self.kernel_config = kernel_config or self._default_kernel_config()
        self.training_data = {'X': None, 'y': None}
        self.is_trained = False
        
    def _default_kernel_config(self):
        """Default kernel configuration with improved stability"""
        return {
            'type': 'matern',  # Matérn kernel is more numerically stable than RBF
            'nu': 2.5,
            'lengthscale_constraint': (0.01, 100.0),
            'outputscale_constraint': (0.01, 100.0),
            'noise_constraint': (1e-6, 1e-2)
        }
    
    def fit(self, X, y):
        """Train GP model with improved numerical stability"""
        # Convert to torch tensor with proper dtype
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float64)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float64)
            
        # Ensure correct data format
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        elif y.dim() == 2 and y.shape[1] != 1:
            y = y.unsqueeze(-1)
            
        # Data validation and preprocessing
        if X.shape[0] < 2:
            raise ValueError(f"Need at least 2 data points for GP training, got {X.shape[0]}")
            
        # Check for numerical issues
        if torch.isnan(X).any() or torch.isinf(X).any():
            raise ValueError("X contains NaN or Inf values")
        if torch.isnan(y).any() or torch.isinf(y).any():
            raise ValueError("y contains NaN or Inf values")
            
        # Check variance (avoid constant outputs)
        if y.var() < 1e-10:
            print(f"Warning: y has very low variance ({y.var().item():.2e}), adding small noise")
            y = y + torch.randn_like(y) * 1e-6
            
        # Store training data
        self.training_data['X'] = X.clone()
        self.training_data['y'] = y.clone()
        
        try:
            # Build model with numerical stability improvements
            self.model = SingleTaskGP(
                X, y,
                input_transform=Normalize(d=X.shape[-1]),
                outcome_transform=Standardize(m=y.shape[-1])
            )
            
            # Configure likelihood for better numerical stability
            self.model.likelihood.noise_covar.noise_constraint = gpytorch.constraints.Interval(
                self.kernel_config['noise_constraint'][0],
                self.kernel_config['noise_constraint'][1]
            )
            
            # Set training mode
            self.model.train()
            self.model.likelihood.train()
            
            # Build MLL
            self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            
            # Train model with improved settings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                fit_gpytorch_model(self.mll, options={
                    'maxiter': 100,
                    'lr': 0.1,
                    'line_search_fn': 'strong_wolfe'
                })
            
            # Set evaluation mode
            self.model.eval()
            self.model.likelihood.eval()
            
            self.is_trained = True
            print(f"GP model training completed, data points: {X.shape[0]}, dimensions: {X.shape[1]}")
            
            # Check model health
            self._check_model_health()
            
        except Exception as e:
            print(f"GP model training failed: {str(e)}")
            self.is_trained = False
            raise e
    
    def _check_model_health(self):
        """Check if the trained model is numerically healthy"""
        try:
            # Test prediction on training data
            with torch.no_grad():
                test_x = self.training_data['X'][:1]  # Use first training point
                posterior = self.model.posterior(test_x)
                mean = posterior.mean
                variance = posterior.variance
                
                if torch.isnan(mean).any() or torch.isinf(mean).any():
                    print("Warning: Model predictions contain NaN/Inf")
                    
                if variance.min() < 1e-10:
                    print("Warning: Model uncertainty is very low")
                    
        except Exception as e:
            print(f"Warning: Model health check failed: {e}")
    
    def predict(self, X):
        """Predict mean and uncertainty with safety checks"""
        if not self.is_trained:
            raise ValueError("Model has not been trained")
            
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float64)
            
        if X.dim() == 1:
            X = X.unsqueeze(0)
            
        try:
            with torch.no_grad():
                posterior = self.model.posterior(X)
                mean = posterior.mean
                variance = posterior.variance
                
                # Safety checks
                if torch.isnan(mean).any() or torch.isinf(mean).any():
                    print("Warning: Predictions contain NaN/Inf, using fallback")
                    mean = torch.zeros_like(mean)
                    
                if torch.isnan(variance).any() or torch.isinf(variance).any() or (variance < 0).any():
                    print("Warning: Variance contains NaN/Inf/negative, using fallback")
                    variance = torch.ones_like(variance) * 1e-3
                    
                std = torch.sqrt(torch.clamp(variance, min=1e-10))  # Ensure positive variance
                
            return mean.squeeze(-1), std.squeeze(-1)
            
        except Exception as e:
            print(f"Prediction failed: {e}")
            # Return safe fallback
            batch_size = X.shape[0]
            return torch.zeros(batch_size), torch.ones(batch_size) * 1e-3
    
    def get_model(self):
        """Get trained model instance"""
        if not self.is_trained:
            raise ValueError("Model has not been trained")
        return self.model


class AcquisitionFunction:
    """
    Acquisition function management class with improved robustness
    """
    def __init__(self, func_type='EI', config=None):
        self.func_type = func_type
        self.config = config or self._default_config()
        self.acq_func = None
        self.bounds = None
        
    def _default_config(self):
        """Default configuration with conservative settings"""
        return {
            'EI': {'xi': 0.01},  # Small improvement threshold
            'UCB': {'beta': 1.0},  # Conservative exploration
            'MES': {'candidate_size': 500},
            'optimization': {
                'num_restarts': 3,   # Reduced for stability
                'raw_samples': 10,   # Reduced for stability
                'max_iter': 100      # Reduced for stability
            }
        }
    
    def build(self, gp_model, best_f=None, bounds=None):
        """Build acquisition function based on GP model with safety checks"""
        model = gp_model.get_model()
        self.bounds = bounds
        
        try:
            if self.func_type == 'EI':
                if best_f is None:
                    # Automatically compute current best value
                    training_y = gp_model.training_data['y']
                    best_f = training_y.min()  # Minimize for cavity filter
                
                self.acq_func = ExpectedImprovement(
                    model=model,
                    best_f=best_f,
                    maximize=False  # Minimize objective
                )
                
            elif self.func_type == 'UCB':
                beta = self.config['UCB']['beta']
                self.acq_func = UpperConfidenceBound(
                    model=model,
                    beta=beta,
                    maximize=False  # Minimize objective
                )
                
            elif self.func_type == 'MES':
                # MES is complex and may not be available, use EI as fallback
                print("Warning: MES not available in this BoTorch version, using EI instead")
                if best_f is None:
                    training_y = gp_model.training_data['y']
                    best_f = training_y.min()
                self.acq_func = ExpectedImprovement(
                    model=model,
                    best_f=best_f,
                    maximize=False
                )
                
            else:
                raise ValueError(f"Unsupported acquisition function type: {self.func_type}")
                
            print(f"Acquisition function built successfully: {self.func_type}")
            
        except Exception as e:
            print(f"Failed to build acquisition function: {e}")
            # Build simpler fallback
            training_y = gp_model.training_data['y']
            best_f = training_y.min()
            self.acq_func = ExpectedImprovement(
                model=model,
                best_f=best_f,
                maximize=False
            )
            print("Built fallback EI acquisition function")
    
    def optimize(self, bounds=None, q=1, num_restarts=None, raw_samples=None):
        """Optimize acquisition function with conservative settings"""
        if self.acq_func is None:
            raise ValueError("Acquisition function has not been built")
            
        if bounds is None:
            bounds = self.bounds
        if bounds is None:
            raise ValueError("Must provide parameter bounds")
            
        # Use provided parameters or fall back to config
        num_restarts = num_restarts or self.config['optimization']['num_restarts']
        raw_samples = raw_samples or self.config['optimization']['raw_samples']
            
        # Ensure correct bounds format
        if not isinstance(bounds, torch.Tensor):
            bounds = torch.tensor(bounds, dtype=torch.float64)
        if bounds.dim() == 2 and bounds.shape[0] == 2:
            bounds = bounds.t()  # Transpose to (n_dims, 2)
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                
                candidate, acq_value = optimize_acqf(
                    self.acq_func,
                    bounds=bounds.t(),  # optimize_acqf needs (2, n_dims) format
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options={
                        'maxiter': self.config['optimization']['max_iter'],
                        'ftol': 1e-9,
                        'gtol': 1e-6
                    }
                )
            
            return candidate, acq_value
            
        except Exception as e:
            print(f"Acquisition function optimization failed: {str(e)}")
            raise e  # Let the calling function handle the fallback


class ModelTrainer:
    """
    Model training controller with improved robustness
    """
    def __init__(self, max_training_iter=100):
        self.max_training_iter = max_training_iter
        
    def train_with_retry(self, gp_model, X, y, max_retries=3):
        """Model training with retry mechanism and data preprocessing"""
        
        # Preprocess data for better numerical stability
        X_processed, y_processed = self._preprocess_data(X, y)
        
        for attempt in range(max_retries):
            try:
                print(f"Training attempt {attempt + 1}/{max_retries}")
                
                # Add small noise to avoid numerical issues if data is too regular
                if attempt > 0:
                    print(f"  Adding small noise for numerical stability")
                    y_noisy = y_processed + torch.randn_like(y_processed) * y_processed.std() * 1e-4
                else:
                    y_noisy = y_processed
                
                gp_model.fit(X_processed, y_noisy)
                print(f"  Training successful on attempt {attempt + 1}")
                return True
                
            except Exception as e:
                print(f"  Training attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    print("  All training attempts failed")
                    return False
                time.sleep(1)  # Brief wait before retry
                
        return False
    
    def _preprocess_data(self, X, y):
        """Preprocess data for better numerical stability"""
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float64)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float64)
        
        # Remove any duplicate points (can cause numerical issues)
        if X.shape[0] > 1:
            # Find unique rows
            X_unique, unique_indices = torch.unique(X, dim=0, return_inverse=True)
            if X_unique.shape[0] < X.shape[0]:
                print(f"  Removed {X.shape[0] - X_unique.shape[0]} duplicate data points")
                # Average y values for duplicate X points
                y_unique = torch.zeros(X_unique.shape[0], dtype=torch.float64)
                for i in range(X_unique.shape[0]):
                    mask = unique_indices == i
                    y_unique[i] = y[mask].mean()
                X, y = X_unique, y_unique
        
        return X, y