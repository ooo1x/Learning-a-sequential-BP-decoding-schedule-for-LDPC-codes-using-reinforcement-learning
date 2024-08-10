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

# import matplotlib.pyplot as plt
#
# # 数据点
# episodes = [5000 * i for i in range(1, 17)]  # 每5000个Episodes记录一次
# average_rewards = [
#     32.178140813213226, 31.92035151254548, 32.104332193450816, 31.583970738310008,
#     32.2000245895698, 31.826606656274038, 31.93927107779231, 32.30925745669669,
#     31.881597298569154, 32.517368376063544, 32.52590741995922, 32.32404720958816,
#     32.24000137212179, 32.57902043119239, 31.83405055223952, 32.41194628168952
# ]
# ber_values = [
#     0.02658301876761125, 0.02669902912621359, 0.02545112112188542, 0.024985661920554573,
#     0.02667733760170735, 0.024495394865765237, 0.024478605698619406, 0.023071958696353662,
#     0.024151325789562573, 0.026026078130286547, 0.025705554556893762, 0.025417703519374335,
#     0.024591573516766982, 0.026106667361454887, 0.0246381805651275, 0.02530684550170821
# ]
#
# # 创建图表和轴
# fig, ax1 = plt.subplots(figsize=(12, 6))
#
# # 绘制平均奖励
# color = 'tab:blue'
# ax1.set_xlabel('Episodes')
# ax1.set_ylabel('Average Reward', color=color)
# ax1.plot(episodes, average_rewards, color=color, marker='o', label='Average Reward')
# ax1.tick_params(axis='y', labelcolor=color)
#
# # 创建共享x轴的第二个y轴
# ax2 = ax1.twinx()
# color = 'tab:red'
# ax2.set_ylabel('BER', color=color)
# ax2.plot(episodes, ber_values, color=color, marker='o', label='BER')
# ax2.tick_params(axis='y', labelcolor=color)
#
# # 添加图例
# ax1.legend(loc='upper left')
# ax2.legend(loc='upper right')
#
# # 显示图表
# plt.title('Average Reward and BER vs. Episodes')
# plt.show()

