import matplotlib.pyplot as plt
import numpy as np

def read_ber_data(filename):
    snr = []
    ber = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.split()
            snr.append(float(parts[0]))
            ber.append(float(parts[1]))
    return snr, ber

snr_layered, ber_layered = read_ber_data("ber_snr_results_layered.txt")
snr, ber = read_ber_data("ber_snr_results_flooding.txt")
# snr_flodding_with_numba, ber_flodding_with_numba = read_ber_data("ber_snr_results_flooding_withnumba.txt")
# snr_layered_with_numba, ber_layered_with_numba = read_ber_data("ber_snr_results_layered_withnumba.txt")
snr_hard, ber_hard = read_ber_data("ber_snr_results_hard.txt")

plt.figure(figsize=(10, 6))

plt.semilogy(snr_layered, ber_layered, 'o-', label='Layered')
plt.semilogy(snr, ber, 's-', label='Flooding')
# plt.semilogy(snr, ber, 'b-', label='Flooding with Numba')
# plt.semilogy(snr, ber, 'b-', label='Layered with Numba')
plt.semilogy(snr_hard, ber_hard, 'r-', label='Hard')

plt.title('BER vs SNR Comparison')
plt.xlabel('SNR (Eb/N0) dB')
plt.ylabel('BER (Bit Error Rate)')

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()

plt.show()
