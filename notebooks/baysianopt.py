import os
import sys
import argparse
import random
import numpy as np
import torch
from joblib import load
from bayes_opt import BayesianOptimization
import sys 
sys.path.append('/home/hengda/material/puceng/')
# from notebooks.puceng_module import PUCENGLitModule
from src.models.puceng_module import PUCENGLitModule
# Default paths
DEFAULT_WORKDIR = '/home/hengda/material/puceng/'
DEFAULT_ENCODER_PATH = os.path.join(DEFAULT_WORKDIR, 'data/puceng/one_hot_encoder_20.pkl')
DEFAULT_CHECKPOINT_PATH = os.path.join(DEFAULT_WORKDIR, 'logs/new/train/runs/2026-01-06_10-21-43/checkpoints/epoch_087.ckpt')

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class BayesianOptRunner:
    def __init__(self, encoder_path, checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.encoder_path = encoder_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.enc = None
        self.model = None
        self.bounds = {f'x{i}': (0.6, 3.4) for i in range(1, 21)}

    def load_resources(self):
        """Load encoder and pre-trained model."""
        print(f"Loading encoder from {self.encoder_path}...")
        self.enc = load(self.encoder_path)
        
        print(f"Loading model from {self.checkpoint_path}...")
        # Load model and set to eval mode
        self.model = PUCENGLitModule.load_from_checkpoint(self.checkpoint_path).to(self.device).eval()

    def result_extraction(self, optimizer):
        target_all=[]
        params_all=[]
        for i, res in enumerate(optimizer.res):
            target=res['target']
            target_all.append(target)
            
            params=[]
            for n in range(1,21):                                 
                temp=res['params']["x%d"%(n)]
                params.append(round(temp,0))
            params=np.array(params)
            params=params.round(0)
            params_all.append(params)
            
        target_all=np.array(target_all)
        params_all=np.array(params_all, dtype=int)
        return target_all, params_all

    def objective_function(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, 
                           x11, x12, x13, x14, x15, x16, x17, x18, x19, x20):
        """
        Objective function for Bayesian Optimization.
        Expects 20 float arguments.
        """
        # Collect args into a list in order
        args = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,
                x11, x12, x13, x14, x15, x16, x17, x18, x19, x20]
        
        # Round arguments as in the original script
        params_values = [round(val, 0) for val in args]
        
        # Prepare input for model
        # Note: The original script collects params, transforms using encoder, converts to tensor
        x_test = self.enc.transform(np.array([params_values], dtype=np.float32))
        sample = torch.tensor(x_test, dtype=torch.float32).to(self.device)
        
        # Inference
        with torch.no_grad():
            pred = self.model.net.forward(sample).cpu().detach().squeeze().numpy()
        
        # Metric: number of values < -10
        # The original code: (pred < -10).sum()
        return (pred < -10).sum()

    def run(self, n_iter, init_points, random_state, probe_params=None):
        """Run the optimization process."""
        self.load_resources()
        
        optimizer = BayesianOptimization(
            f=self.objective_function,
            pbounds=self.bounds,
            verbose=1,
            random_state=random_state,
        )

        if probe_params:
            print("Probing initial parameters...")
            optimizer.probe(params=probe_params, lazy=True)

        print(f"Starting optimization with {init_points} init points and {n_iter} iterations...")
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        
        print("\nOptimization finished.")
        print("Max result:", optimizer.max)
        return optimizer

def main():
    parser = argparse.ArgumentParser(description="Bayesian Optimization for Ion Gel Structure")
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--n_iter', type=int, default=20, help='Number of optimization iterations')
    parser.add_argument('--init_points', type=int, default=5, help='Number of initial random points')
    parser.add_argument('--encoder_path', type=str, default=DEFAULT_ENCODER_PATH, help='Path to one_hot_encoder')
    parser.add_argument('--checkpoint_path', type=str, default=DEFAULT_CHECKPOINT_PATH, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, help='Optional path to save results (e.g., results.txt)')
    
    args = parser.parse_args()
    
    print(f"Setting seed to {args.seed}")
    set_seed(args.seed)
    
    runner = BayesianOptRunner(
        encoder_path=args.encoder_path,
        checkpoint_path=args.checkpoint_path
    )
    
    # Original script probed these values, preserving them as default probe
    default_probe = {
        'x1': 1.533385604316876, 'x10': 3.3547462868991396, 'x11': 2.9716419191839765, 
        'x12': 1.1522775555200895, 'x13': 2.6165787178331303, 'x14': 3.088639399314133, 
        'x15': 1.832610499194466, 'x16': 0.810378775812425, 'x17': 3.234068837754719, 
        'x18': 3.209281473056138, 'x19': 3.0025440859984136, 'x2': 1.7141329484394618, 
        'x20': 1.5720434588376628, 'x3': 2.6614033687613134, 'x4': 2.763047275313439, 
        'x5': 1.320263109178963, 'x6': 2.027592832602864, 'x7': 1.9462920339230112, 
        'x8': 1.4103735656815162, 'x9': 1.7253529174768172
    }
    
    optimizer = runner.run(
        n_iter=args.n_iter,
        init_points=args.init_points,
        random_state=args.seed,
        probe_params=default_probe
    )
    
    target_all, params_all = runner.result_extraction(optimizer)
    
    if args.output:
        print(f"Saving results to {args.output}")
        with open(args.output, 'w') as f:
            f.write(f"Max Result: {optimizer.max}\n")
            f.write("All Targets:\n")
            np.savetxt(f, target_all)
            f.write("All Params:\n")
            np.savetxt(f, params_all)

if __name__ == "__main__":
    main()