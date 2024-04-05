import numpy as np

# n er størrelsen på bilde/filteret som er n*n fylt med verdier fra (n-1)/2 til -(n-1)/2 
# mu er koordinatene en velger å starte filteret på
# sigma bestemmer hvor store frekvenser som går gjennom"
# use_log er om den skal bruke logaritme til utergning eller ikke

#n: np.ndarray, mu: np.ndarray, sigma: float, use_log: bool

def gaussian(size, mu: np.ndarray, sigma, use_log):

    mu = np.array(mu)
    n1 = np.arange(size).reshape((-1, 1)).repeat(size, axis=1) - (size - 1) / 2
    n2 = n1.T
    n = np.array([n1, n2])  

    N = n.shape[1]
    scale = ((N - 1.) / 2.) / np.log((N + 1) / 2.) if use_log else 1.

    def log(x):
        return np.log(x) if use_log else x - 1

    result = np.zeros((size, size))
    for l1 in [-size, 0, size]:
        for l2 in [-size, 0, size]:
            norm = np.linalg.norm(n + np.array([l1, l2]).reshape(-1, 1, 1), axis=0)
            result += np.exp(
                -np.square(scale * np.where(norm > 0, (n + np.array([l1, l2]).reshape(-1, 1, 1)) / norm * log(norm + 1), 0.) - mu.reshape(-1, 1, 1)).sum(axis=0)
                / sigma)    

    spatial_domain = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(result)))
    return (spatial_domain.real, spatial_domain.imag) 