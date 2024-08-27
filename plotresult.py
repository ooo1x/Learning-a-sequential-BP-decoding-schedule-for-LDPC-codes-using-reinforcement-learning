import matplotlib.pyplot as plt

eb_n0 = [0, 0.5, 1.0, 1.5, 2.0]


ber_optimal = [0.02558788157928405, 0.01808066759388039,0.01312528682882056,0.00957359220326651,0.005456765626192475]
ber_qlearning  = [0.024583917201366864,
0.016013952750911685,
0.010695753468177485,
0.008732099196646874,
0.00529677519862907,
]
ber_fixed = [0.019152318388140886, 0.012029822434058084, 0.009058527853617084, 0.005993253104289781, 0.003496173438171921]
ber_flooding =[0.035937142857142834,0.024078571428571408,0.019992857142857128,0.014189999999999998,0.00949714285714286]

plt.semilogy(eb_n0, ber_flooding, label='Flooding', marker='*')
plt.semilogy(eb_n0, ber_optimal, label='Optimal Fixed Schedule', marker='s')
plt.semilogy(eb_n0, ber_qlearning, label='Q-Learning', marker='o')
plt.semilogy(eb_n0, ber_fixed, label='Noise Dependent Schedule', marker='^')



plt.xticks(eb_n0, [f"{x} dB" for x in eb_n0])
plt.legend()


plt.title('BER vs. Eb/N0')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('BER')


plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.show()