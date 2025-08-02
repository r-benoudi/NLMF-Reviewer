"""
GA-optimised Non-Local Means (OpenCV built-in) for grayscale images.
Reproduces the core pipeline of:
Benoudi et al.  "Genetic algorithm optimization of nonlocal means filter
for gaussian noise reduction"

Improvements:
- Added parameter validation and constraints
- Added reproducible random seeds
- Added error handling
- Improved mutation operator
- Added statistical tracking
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from deap import base, creator, tools, algorithms
import random
import json
from datetime import datetime

# -------------------------------------------------
# 1. Reproducible settings
# -------------------------------------------------
RANDOM_SEED    = 42                           # for reproducibility
SIGMAS         = [5, 10, 15, 20, 25, 30]      # noise levels
IMAGE_DIR      = "images"                     # put 512×512 grayscale PNGs here
RESULT_DIR     = "output_2"
os.makedirs(RESULT_DIR, exist_ok=True)

# OpenCV limits for single-channel images
H_RANGE = (0.0001, 30.0)          # h smoothing parameter
P_RANGE = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]            # templateWindowSize  (must be odd <=7)
N_RANGE = list(range(5, 22, 2))  # searchWindowSize   (odd ≤21)

# GA parameters
POP_SIZE = 25
N_GENERATIONS = 500
CXPB = 0.7
MUTPB = 0.05
TOURNSIZE = 5

# Set random seeds
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -------------------------------------------------
# 2. Parameter validation utilities
# -------------------------------------------------
def validate_parameters(h, p, n):
    """Ensure parameters are valid for OpenCV"""
    h = max(H_RANGE[0], min(H_RANGE[1], float(h)))
    
    # Ensure odd window sizes
    p = int(p)
    if p % 2 == 0:
        p = p + 1 if p < max(P_RANGE) else p - 1
    p = max(min(P_RANGE), min(max(P_RANGE), p))
    
    n = int(n)
    if n % 2 == 0:
        n = n + 1 if n < max(N_RANGE) else n - 1
    n = max(min(N_RANGE), min(max(N_RANGE), n))
    
    return h, p, n

def safe_mutate(individual, eta, low, up, indpb):
    """Custom mutation that respects parameter constraints"""
    size = len(individual)
    if not hasattr(individual, "fitness"):
        individual.fitness = creator.FitnessMax()
    
    for i in range(size):
        if random.random() < indpb:
            if i == 0:  # h parameter
                # Polynomial bounded mutation for continuous parameter
                xl, xu = low[i], up[i]
                x = individual[i]
                delta_1 = (x - xl) / (xu - xl)
                delta_2 = (xu - x) / (xu - xl)
                rand = random.random()
                mut_pow = 1.0 / (eta + 1.0)
                
                if rand <= 0.5:
                    xy = 1.0 - delta_1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * xy**(eta + 1)
                    delta_q = val**mut_pow - 1.0
                else:
                    xy = 1.0 - delta_2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy**(eta + 1)
                    delta_q = 1.0 - val**mut_pow
                
                x = x + delta_q * (xu - xl)
                individual[i] = min(max(x, xl), xu)
            else:  # window size parameters
                if i == 1:  # template window size
                    individual[i] = random.choice(P_RANGE)
                else:  # search window size
                    individual[i] = random.choice(N_RANGE)
    
    del individual.fitness.values
    return individual,

# -------------------------------------------------
# 3. GA setup
# -------------------------------------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Safe gene creators
toolbox.register("attr_h", lambda: float(np.random.uniform(*H_RANGE)))
toolbox.register("attr_p", lambda: int(np.random.choice(P_RANGE)))
toolbox.register("attr_n", lambda: int(np.random.choice(N_RANGE)))

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_h, toolbox.attr_p, toolbox.attr_n), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def safe_evaluate(individual, noisy, clean):
    """Safe evaluation with error handling and parameter validation"""
    try:
        h, p, n = validate_parameters(*individual)
        
        # Apply Non-Local Means denoising
        den = cv2.fastNlMeansDenoising(noisy, None,
                                       h=h,
                                       templateWindowSize=p,
                                       searchWindowSize=n)
        
        # Calculate PSNR as fitness
        fitness_value = psnr(clean, den)
        
        # Sanity check
        if not np.isfinite(fitness_value) or fitness_value < 0:
            return (-float('inf'),)
            
        return (fitness_value,)
        
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return (-float('inf'),)

toolbox.register("evaluate", safe_evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", safe_mutate,
                 eta=0.1,
                 low=[H_RANGE[0], min(P_RANGE), min(N_RANGE)],
                 up=[H_RANGE[1], max(P_RANGE), max(N_RANGE)],
                 indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)

# -------------------------------------------------
# 4. Results tracking
# -------------------------------------------------
def save_results(results, filename):
    """Save experiment results to JSON"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

# -------------------------------------------------
# 5. Main experiment
# -------------------------------------------------
def run_optimization(noisy, clean, img_name, sigma):
    """Run GA optimization for a single image/noise combination"""
    
    # Reset evaluation function with current images
    toolbox.register("evaluate", safe_evaluate, noisy=noisy, clean=clean)
    
    # Initialize population
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    
    # Statistics tracking
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Run GA
    pop, logbook = algorithms.eaSimple(pop, toolbox,
                                       cxpb=CXPB, mutpb=MUTPB,
                                       ngen=N_GENERATIONS, 
                                       stats=stats, 
                                       halloffame=hof,
                                       verbose=False)
    
    # Get best solution
    best_individual = hof[0]
    h_best, p_best, n_best = validate_parameters(*best_individual)
    
    # Apply best parameters
    den = cv2.fastNlMeansDenoising(noisy, None,
                                   h=h_best,
                                   templateWindowSize=p_best,
                                   searchWindowSize=n_best)
    
    # Calculate all metrics
    metrics = {
        "PSNR": float(psnr(clean, den)),
        "MSE": float(mse(clean, den)),
        "SSIM": float(ssim(clean, den)),
        "best_h": float(h_best),
        "best_p": int(p_best),
        "best_n": int(n_best),
        "generations": N_GENERATIONS,
        "final_fitness": float(best_individual.fitness.values[0])
    }
    
    # Save denoised image
    base = os.path.splitext(img_name)[0]
    out_path = os.path.join(RESULT_DIR,
                           f"{base}_sigma{sigma}_h{h_best:.2f}_P{p_best}_N{n_best}.png")
    cv2.imwrite(out_path, den)
    
    # Save noisy image for comparison
    noisy_path = os.path.join(RESULT_DIR,
                             f"{base}_sigma{sigma}_noisy.png")
    cv2.imwrite(noisy_path, noisy)
    
    return metrics, logbook

def main():
    """Main experiment loop"""
    all_results = []
    experiment_start = datetime.now()
    
    print(f"Starting experiment with seed {RANDOM_SEED}")
    print(f"Population size: {POP_SIZE}, Generations: {N_GENERATIONS}")
    print("-" * 80)
    
    for img_name in sorted(os.listdir(IMAGE_DIR)):
        if not img_name.lower().endswith((".png", ".jpg", ".bmp", ".tif")):
            continue
            
        try:
            # Load and resize image
            clean = cv2.imread(os.path.join(IMAGE_DIR, img_name), cv2.IMREAD_GRAYSCALE)
            if clean is None:
                print(f"Warning: Could not load {img_name}")
                continue
                
            clean = cv2.resize(clean, (512, 512))
            
            print(f"Processing {img_name}...")
            
            for sigma in SIGMAS:
                print(f"  Noise level σ={sigma}...")
                
                # Add Gaussian noise (reproducible with seed)
                np.random.seed(RANDOM_SEED + sigma)  # Different noise per sigma
                noise = np.random.normal(0, sigma, clean.shape)
                noisy = (clean.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)
                
                # Run optimization
                metrics, logbook = run_optimization(noisy, clean, img_name, sigma)
                
                # Store results
                result = {
                    "image": img_name,
                    "sigma": sigma,
                    "timestamp": datetime.now().isoformat(),
                    "metrics": metrics,
                    "convergence": [record for record in logbook]
                }
                all_results.append(result)
                
                # Print results
                print(f"    Best: h={metrics['best_h']:.2f}, "
                      f"P={metrics['best_p']}, N={metrics['best_n']}")
                print(f"    PSNR={metrics['PSNR']:.2f}, "
                      f"MSE={metrics['MSE']:.2f}, SSIM={metrics['SSIM']:.4f}")
                
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue
    
    # Save all results
    results_file = os.path.join(RESULT_DIR, f"experiment_results_{experiment_start.strftime('%Y%m%d_%H%M%S')}.json")
    save_results(all_results, results_file)
    
    experiment_end = datetime.now()
    duration = experiment_end - experiment_start
    
    print("-" * 80)
    print(f"Experiment completed in {duration}")
    print(f"Results saved to {results_file}")
    print(f"Processed {len(all_results)} image/noise combinations")

if __name__ == "__main__":
    main()