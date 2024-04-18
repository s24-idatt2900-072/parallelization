import numpy as np


def gaussian(n: np.ndarray, mu: np.ndarray, sigma: float, use_log: bool):
    N = n.shape[1] # N er 101 

    scale = ((N - 1.) / 2.) / np.log((N + 1) / 2.) if use_log else 1.

    def log(x):
        return np.log(x) if use_log else x - 1

    result = np.zeros((N, N))
    for l1 in [-N, 0, N]:
        for l2 in [-N, 0, N]:
            norm = np.linalg.norm(n + np.array([l1, l2]).reshape(-1, 1, 1), axis=0)
            result += np.exp(
                -np.square(scale * np.where(norm > 0, (n + np.array([l1, l2]).reshape(-1, 1, 1)) / norm * log(norm + 1), 0.) - mu.reshape(-1, 1, 1)).sum(axis=0)
                / sigma)
    
        spatial_domain = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(result)))
    return spatial_domain
