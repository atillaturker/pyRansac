import random
import numpy as np



import random
import numpy as np

def RANSAC(data, model_func, error_func, sample_size, threshold, num_iters, random_seed, verbose=None):

    best_model = None
    best_inliers = set()   
    random.seed(random_seed)

    for i in range(num_iters):
        if isinstance(data, list):
            #print("data is List")
            sample = random.sample(data, sample_size)
        if isinstance(data, np.ndarray):
            #print("data is NumpyArray")
            indices = np.random.choice(len(data), int(sample_size), replace=False)

            sample = data[indices]
            
        model = model_func(sample)

        inliers = set() 
        for point in data:
            error = error_func(point, model)
            if error < threshold:
                inliers.add(tuple(point))
        
        if len(inliers) > len(best_inliers):
            best_model = model
            best_inliers = inliers

        if verbose:
            print(f"Iteration {i+1}/{num_iters}, Inliers: {len(inliers)}, Best Inliers: {len(best_inliers)}")

    best_model = model_func(np.array(list(best_inliers)))

    return best_model, list(best_inliers)


