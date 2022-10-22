'''
Testing code to generate an audio signal and plot the signal, DFT and FFT of the signal
'''
from typing import List, Union
import numpy as np
from fourier_transforms import dft, fft
import matplotlib.pyplot as plt
from timeit import default_timer
import json


def add_sin_wave(buffer: List[float], freq: float, sample_rate: Union[float, int], amplitude: float = 1.0):
    '''
    Superpose sin wave with `amplitude` and `freq` to existing audio `buffer`
    '''
    sample_no = 0

    for i in range(len(buffer)):
        t = sample_no / sample_rate
        buffer[i] += amplitude * np.sin(2 * np.pi * freq * t)
        sample_no += 1


def log_results(buffer: List[float], sample_rate: Union[float, int], dft_timing: float, fft_timing: float):
    log_obj = {
        'buffer': buffer,  # buffer size 1024 is small enough to dump to json
        'sample_rate': sample_rate,
        'timings': {
            'DFT': dft_timing,
            'FFT': fft_timing,
        },
    }
    with open('log.json', 'w') as f:
        json.dump(log_obj, f, indent=4)


if __name__ == '__main__':
    buffer_size = 512
    buffer = [0 for _ in range(buffer_size)]
    fs = buffer_size * 4

    # Amaj7
    add_sin_wave(buffer, 440, fs, amplitude=1/3)
    add_sin_wave(buffer, 554.37, fs, amplitude=1/3)
    add_sin_wave(buffer, 659.25, fs, amplitude=1/3)
    add_sin_wave(buffer, 830.61, fs, amplitude=1/3)

    N = len(buffer)
    freq_resolution = fs / N

    start = default_timer()
    freq_response_dft = dft(buffer)
    dur_dft = default_timer() - start
    print(f'DFT on buffer of {buffer_size} took {dur_dft}s')

    start = default_timer()
    freq_response_fft = fft(buffer)
    dur_fft = default_timer() - start
    print(f'FFT on buffer of {buffer_size} took {dur_fft}s')

    # add audio buffer, sample rate and timings to json
    log_results(buffer, fs, dur_dft, dur_fft)

    # plot audio signal generated
    x1 = np.array([1000 * s / fs for s in range(len(buffer))])  # time in ms
    y1 = np.array(buffer)
    plt.subplot(3, 1, 1)
    plt.title('Audio Signal Generated')
    plt.xlabel('t / ms')
    plt.ylabel('Amplitude')
    plt.xlim(left=0, right=1000 * buffer_size / fs)
    plt.plot(x1, y1)

    # range of frequencies of the frequency response to plot
    freq_lower = 400
    freq_upper = 900
    bins_lower = int(freq_lower / freq_resolution)
    bins_upper = int(freq_upper / freq_resolution)
    freq_domain_to_plot = range(bins_lower, bins_upper+1)

    # plot DFT
    x2 = np.array([k * freq_resolution for k in freq_domain_to_plot])
    y2 = np.array(
        [abs(freq_response_dft[i]) for i in freq_domain_to_plot])
    plt.subplot(3, 1, 2)
    plt.stem(x2, y2, 'b', markerfmt=' ')
    plt.title('DFT frequency response')
    plt.xlabel('f / Hz')
    plt.ylabel('Amplitude')
    plt.ylim(bottom=0)
    plt.xlim(left=x2[0], right=x2[-1])

    # plot FFT
    x3 = x2
    y3 = np.array(
        [abs(freq_response_fft[i]) * 2 / N for i in freq_domain_to_plot])
    plt.subplot(3, 1, 3)
    plt.stem(x3, y3, 'b', markerfmt=' ')
    plt.title('FFT frequency response')
    plt.xlabel('f / Hz')
    plt.ylabel('Amplitude')
    plt.ylim(bottom=0)
    plt.xlim(left=x3[0], right=x3[-1])

    plt.tight_layout()
    plt.savefig('result.png', dpi=300)
    plt.show()
