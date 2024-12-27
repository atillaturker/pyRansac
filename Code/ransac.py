import random
import numpy as np



def RANSAC(data, model_func, error_func, sample_size, threshold, num_iters=1000, verbose=None):
    best_model = None
    best_inliers = []

    # random.sample cannot deal with "data" being a numpy array
    if isinstance(data, np.ndarray):
            data = data.tolist()    
            if verbose:
                print("Data is a numpy array. Converting to list.")

    
    for i in range(num_iters):
        # Random sample 
        sample = random.sample(data, int(sample_size))

        # Model fitting
        model = model_func(sample)    

        
        # Inlier calculation
        inliers = []
        for point in data:
            error = error_func(point, model)
            if error < threshold:
                inliers.append(point)
        
        if len(inliers) > len(best_inliers):
            best_model = model
            best_inliers = inliers
        if verbose:
            print(f"Iteration {i+1}/{num_iters}, Inliers: {len(inliers)}, Best Inliers: {len(best_inliers)}, Best Model: {best_model}")

    return best_model, best_inliers
