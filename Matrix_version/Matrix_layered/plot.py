import matplotlib.pyplot as plt
import numpy as np

def read_ber_data(filename):
    snr = []
    ber = []
    bler = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 3:
                    snr.append(float(parts[0]))  # SNR
                    ber.append(float(parts[1]))  # BER
                    # bler.append(float(parts[2]))  # BLER
    return snr, ber

snr_layered, ber_layered = read_ber_data("ber_snr_results_layered_with_multi.txt")
snr, ber = read_ber_data("ber_snr_results_flooding_with_multi.txt")
# snr_flodding_with_numba, ber_flodding_with_numba = read_ber_data("ber_snr_results_flooding_withnumba.txt")
# snr_layered_with_numba, ber_layered_with_numba = read_ber_data("ber_snr_results_layered_withnumba.txt")
# snr_hard, ber_hard = read_ber_data("ber_snr_results_hard.txt")

plt.figure(figsize=(10, 6))

plt.semilogy(snr_layered, ber_layered, 'o-', label='Layered')
plt.semilogy(snr, ber, 's-', label='Flooding')
# plt.semilogy(snr, ber, 'b-', label='Flooding with Numba')
# plt.semilogy(snr, ber, 'b-', label='Layered with Numba')
# plt.semilogy(snr_hard, ber_hard, 'r-', label='Hard')

plt.xlabel('SNR (Eb/N0) dB')
plt.title('BER vs SNR')
plt.ylabel('BER (Bit Error Rate)')
# plt.title('Block Error Rate vs SNR ')
# plt.ylabel('BLER (Block Error Rate)')

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()

plt.show()
