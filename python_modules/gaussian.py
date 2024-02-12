import numpy as np

# n er størrelsen på bilde/filteret som er n*n fylt med verdier fra (n-1)/2 til -(n-1)/2 
# mu er koordinatene en velger å starte filteret på
# sigma bestemmer hvor store frekvenser som går gjennom"
# use_log er om den skal bruke logaritme til utergning eller ikke

#n: np.ndarray, mu: np.ndarray, sigma: float, use_log: bool

def gaussian(n, mu: np.ndarray, sigma, use_log):

    mu = np.array(mu)

    N = 9
    
    print("Hello world")
    

    n1 = np.arange(N).reshape((-1, 1)).repeat(N, axis=1) - (N - 1) / 2

    n2 = n1.T
    n = np.array([n1, n2])

    print(n)
        
    N = n.shape[1]

    print(f"dette er mu {mu}")


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

     
    print(f"dette er result {result}")
    print(f"dette er typen til result {type(result)}")
    print(f"Dette er shapen til result {result.shape}")
    return result
    
    # if use_log : 
    #     return sigma * 10
    # else:
    #     return sigma/10