import numpy as np
import cv2
import random
from skimage.restoration import denoise_nl_means
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed


def add_gaussian_noise(image, sigma):
    """
    Add zero-mean Gaussian noise with standard deviation sigma to input image.
    Input image assumed in range [0,1].
    """
    noisy = image + np.random.normal(0, sigma, image.shape)
    return np.clip(noisy, 0, 1)


def evaluate_individual(params):
    """
    Evaluate PSNR for a single set of NLMF parameters.
    """
    h, patch_size, window_size, noisy, clean, sigma = params
    denoised = denoise_nl_means(
        noisy,
        h=h * sigma,
        patch_size=patch_size,
        patch_distance=window_size,
        channel_axis=None
    )
    psnr = compute_psnr(clean, denoised, data_range=1.0)
    return (h, patch_size, window_size, psnr)


class GeneticAlgorithmOptimizer:
    def __init__(self, clean_img, sigma,
                 population_size=25, generations=500,
                 crossover_prob=0.7, mutation_prob=0.05,
                 h_bounds=(0.01, 1.0),
                 patch_bounds=(3, 15), window_bounds=(3, 25)):
        self.clean = clean_img
        self.sigma = sigma
        self.noisy = add_gaussian_noise(clean_img, sigma)
        self.pop_size = population_size
        self.generations = generations
        self.cx_prob = crossover_prob
        self.mut_prob = mutation_prob
        self.h_min, self.h_max = h_bounds
        self.p_min, self.p_max = patch_bounds
        self.w_min, self.w_max = window_bounds
        self.population = self._init_population()
        self.fitness_cache = {}

    def _init_population(self):
        pop = []
        for _ in range(self.pop_size):
            h = random.uniform(self.h_min, self.h_max)
            p = random.randrange(self.p_min // 2, self.p_max // 2 + 1) * 2 + 1
            w = random.randrange(self.w_min // 2, self.w_max // 2 + 1) * 2 + 1
            pop.append((h, p, w))
        return pop

    def _evaluate_population(self, population):
        """
        Evaluate entire population in parallel, caching results.
        """
        params_list = []
        for ind in population:
            if ind not in self.fitness_cache:
                params_list.append((*ind, self.noisy, self.clean, self.sigma))
        # Parallel evaluation
        if params_list:
            with ProcessPoolExecutor() as executor:
                futures = {executor.submit(evaluate_individual, p): p for p in params_list}
                for fut in as_completed(futures):
                    h, p_sz, w_sz, psnr = fut.result()
                    self.fitness_cache[(h, p_sz, w_sz)] = psnr
        # Return fitness list
        return [self.fitness_cache[ind] for ind in population]

    def _select(self, weighted_pop):
        total = sum(weight for _, weight in weighted_pop)
        pick = random.uniform(0, total)
        current = 0
        for ind, weight in weighted_pop:
            current += weight
            if current > pick:
                return ind
        return weighted_pop[-1][0]

    def _crossover(self, p1, p2):
        if random.random() > self.cx_prob:
            return p1, p2
        h1, p1_sz, w1 = p1
        h2, p2_sz, w2 = p2
        return (h1, p2_sz, w2), (h2, p1_sz, w1)

    def _mutate(self, ind):
        h, p_sz, w_sz = ind
        if random.random() < self.mut_prob:
            h = random.uniform(self.h_min, self.h_max)
        if random.random() < self.mut_prob:
            p_sz = random.randrange(self.p_min // 2, self.p_max // 2 + 1) * 2 + 1
        if random.random() < self.mut_prob:
            w_sz = random.randrange(self.w_min // 2, self.w_max // 2 + 1) * 2 + 1
        return (h, p_sz, w_sz)

    def run(self):
        best_overall = (None, -np.inf)
        for gen in range(1, self.generations + 1):
            fitnesses = self._evaluate_population(self.population)
            weighted = list(zip(self.population, fitnesses))
            # Track best
            gen_best = max(weighted, key=lambda x: x[1])
            if gen_best[1] > best_overall[1]:
                best_overall = gen_best
            # Elitism + new generation
            new_pop = [best_overall[0]]
            while len(new_pop) < self.pop_size:
                parent1 = self._select(weighted)
                parent2 = self._select(weighted)
                c1, c2 = self._crossover(parent1, parent2)
                new_pop.append(self._mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._mutate(c2))
            self.population = new_pop

            if gen % 50 == 0 or gen == 1:
                print(f"Gen {gen}/{self.generations} - Best PSNR so far: {best_overall[1]:.2f}")

        h_opt, p_opt, w_opt = best_overall[0]
        print(f"Optimal: h={h_opt:.4f}, patch={p_opt}, window={w_opt}, PSNR={best_overall[1]:.2f}")
        return best_overall


def main():
    parser = argparse.ArgumentParser(
        description="GA-optimized NLMF for Gaussian denoising"
    )
    parser.add_argument('input', help="Path to input grayscale image (uint8)")
    parser.add_argument('--sigma', type=float, default=0.1, help="Noise std deviation")
    parser.add_argument('--pop', type=int, default=25, help="Population size")
    parser.add_argument('--gens', type=int, default=500, help="Number of generations")
    args = parser.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    ga = GeneticAlgorithmOptimizer(
        img, args.sigma,
        population_size=args.pop,
        generations=args.gens
    )
    (h_opt, p_opt, w_opt), best_psnr = ga.run()

    denoised = denoise_nl_means(
        ga.noisy,
        h=h_opt * args.sigma,
        patch_size=p_opt,
        patch_distance=w_opt,
        channel_axis=None
    )
    cv2.imwrite('noisy.png', (ga.noisy * 255).astype(np.uint8))
    cv2.imwrite('denoised.png', (denoised * 255).astype(np.uint8))
    print("Saved noisy.png and denoised.png.")


if __name__ == '__main__':
    main()
