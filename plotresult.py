import matplotlib.pyplot as plt

eb_n0 = [0, 0.5, 1.0, 1.5, 2.0]


ber_optimal = [0.024282560706401765, 0.017812611328820806,0.012507191635190235,0.008533747090768037,0.0056548560582094275]
#ber_qlearning  = [0.02,0.015714285714285715,0.011428571428571429,0.007142857142857142,0.0014285714285714286]
ber_fixed = [0.019152318388140886, 0.012029822434058084, 0.009058527853617084, 0.005993253104289781, 0.003496173438171921]
ber_flooding =[0.035937142857142834,0.024078571428571408,0.019992857142857128,0.014189999999999998,0.00949714285714286]

plt.semilogy(eb_n0, ber_flooding, label='Flooding', marker='*')
plt.semilogy(eb_n0, ber_optimal, label='Optimal Fixed Schedule', marker='s')
#plt.semilogy(eb_n0, ber_qlearning, label='Q-Learning', marker='o')
plt.semilogy(eb_n0, ber_fixed, label='Noise Dependent Schedule', marker='^')



plt.xticks(eb_n0, [f"{x} dB" for x in eb_n0])
plt.legend()


plt.title('BER vs. Eb/N0')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('BER')


plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.show()