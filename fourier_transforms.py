from typing import List
import numpy as np


def dft(x: List[float]) -> List[complex]:
    '''
           N-1
    X[k] = Σ (x[n] * exp(-j2πkn/N))
           n=0
    '''
    N = len(x)
    X = [0 + 0j for i in range(N)]
    for k in range(N):
        summation = 0 + 0j
        for n in range(N):
            summation += x[n] * np.exp(-2j * np.pi * k * n / N)
        X[k] = summation

    return X


def fft(x: List[float]) -> List[complex]:
    '''
    FFT on buffer `x` of length N, 
    where N = 2**p, for p∈Z+

    ```
           (N/2)-1                               (N/2)-1
    X[k] = Σ (x(2r) * w_N ** 2rk) + (w_N ** k) * Σ (x(2r+1) * w_N ** 2rk)
           r=0                                   r=0
    ```

    where w_N = exp(-j2πkn/N)
    '''
    # base case - only 1 sample left
    # bin k = 0. This simplifies to:
    # => X[0] = x[0] + w_N**0 * 0
    # => X[0] = x[0]
    if len(x) <= 1:
        return np.asarray(x, dtype=float)

    # recursively do X_even + w_N**k * X_odd
    N = len(x)
    X_even = fft(x[::2])
    X_odd = fft(x[1::2])
    X = np.array([0 + 0j for _ in range(N)])
    N_half = N // 2

    for r in range(N_half):
        # w_coeff refers to the w_N ** k term
        w_coeff = np.exp(-2j * np.pi * r / N)
        X[r] = X_even[r] + w_coeff * X_odd[r]

    for r in range(N_half):
        k = N_half + r  # bin index
        w_coeff = np.exp(-2j * np.pi * k / N)
        X[k] = X_even[r] + w_coeff * X_odd[r]

    # With numpy
    # w_coeff = np.exp(-2j * np.pi * np.arange(N) / N)
    # return np.concatenate([
    #     X_even + w_coeff[:N // 2] * X_odd,
    #     X_even + w_coeff[N // 2:] * X_odd,
    # ])
    return X
