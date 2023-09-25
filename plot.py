import matplotlib.pyplot as plt
import pickle
import numpy as np

DATASET = "cifar10"

# ATTACK_METHOD = 'corner'
# ATTACK_METHOD = 'low_xai_pattern'
ATTACK_METHOD = 'high_xai_pattern'


lr_list = ['0', '0.001', '0.005', '0.01', '0.02', '0.03', '0.04', '0.05', '0.1', '0.2']

# report_cln = pickle.load(open(f'./report/{ATTACK_METHOD}_cln','rb'))
# report_adv = pickle.load(open(f'./report/{ATTACK_METHOD}_adv','rb'))
report_cln = pickle.load(open(f'./report/All_normal_cln','rb'))


cln = [0.9555000066757202]
adv = [0.9416000247001648]

cln = cln + report_cln
# adv = adv + report_adv


# print(report_cln)
# print(report_adv)

plt.subplots(figsize=(7, 4), dpi=160)

plt.plot(lr_list, cln, 'rs--', label='normal')
# plt.plot(lr_list, adv, 'b^--', label='backdoor')
plt.legend(loc='upper right')          # ncol = 1
plt.xlabel('Learning Rate')
plt.ylabel('Acc')

# plt.plot(lr_list, cln, 'rs--',  lr_list, adv, 'b^--')
# plt.legend(loc='upper right')
plt.savefig(f"./All_cln.png")