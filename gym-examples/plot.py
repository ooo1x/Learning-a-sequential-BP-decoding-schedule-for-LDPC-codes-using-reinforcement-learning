import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

def plot_q_table(q_table_filepath):
    # Load the Q-table from a file
    q_table = np.load(q_table_filepath)
    plt.figure(figsize=(10, 8))
    sns.heatmap(q_table, annot=True, fmt=".8f",cmap="coolwarm")
    plt.title("Q-table Heatmap")
    plt.xlabel("Actions")
    plt.ylabel("States")
    plt.show()

plot_q_table("q_table.npy")
data = np.load('q_table.npy')

print(data)

# def read_ber_data(filename):
#     snr = []
#     ber = []
#     bler = []
#     with open(filename, 'r') as file:
#         for line in file:
#             line = line.strip()
#             if line:
#                 parts = line.split()
#                 if len(parts) >= 3:
#                     snr.append(float(parts[1]))  # SNR
#                     ber.append(float(parts[4]))  # BER
#                     # bler.append(float(parts[2]))  # BLER
#     return snr, ber

# snr_layered, ber_layered = read_ber_data("ber_snr_results_layered_with_multi.txt")
# snr, ber = read_ber_data("ber_results.txt")
# snr_flodding_with_numba, ber_flodding_with_numba = read_ber_data("ber_snr_results_flooding_withnumba.txt")
# snr_layered_with_numba, ber_layered_with_numba = read_ber_data("ber_snr_results_layered_withnumba.txt")
# snr_hard, ber_hard = read_ber_data("ber_snr_results_hard.txt")
#
# plt.figure(figsize=(10, 6))

# plt.semilogy(snr_layered, ber_layered, 'o-', label='Layered')
# plt.semilogy(snr, ber, 's-', label='RL')
# plt.semilogy(snr, ber, 'b-', label='Flooding with Numba')
# plt.semilogy(snr, ber, 'b-', label='Layered with Numba')
# plt.semilogy(snr_hard, ber_hard, 'r-', label='Hard')


# plt.xlabel('SNR (Eb/N0) dB')
# plt.title('BER vs SNR')
# plt.ylabel('BER (Bit Error Rate)')
# plt.title('Block Error Rate vs SNR ')
# plt.ylabel('BLER (Block Error Rate)')

# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.legend()
#
# plt.show()

import matplotlib.pyplot as plt

# 数据点
episodes = [5000,11000,18000,26000,35000]
rewards = [11013.800287617438,12448.434221993506, 12433.098139925267, 13373.708733930791,18657.5765365411]
ber_values = [0.017879270667374755, 0.016118836915297093, 0.015615337043908472, 0.014723032069970846, 0.010695753468177485]

fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:blue'
ax1.set_xlabel('Episodes')
ax1.set_ylabel('Reward', color=color)
ax1.plot(episodes, rewards, color=color, marker='o', label='Average Reward')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('BER', color=color)
ax2.plot(episodes, ber_values, color=color, marker='o', label='BER')
ax2.tick_params(axis='y', labelcolor=color)

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.title('Reward and BER vs. Episodes')
plt.show()

