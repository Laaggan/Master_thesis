from return_values_from_tensorboard import *
scalar_name = 'accuracy'

x1,y1 = return_tensorborad_data('/Users/carlrosengren/Desktop/20191202_all_modalities_lee2_dropout/20191202-221022-all-lr-0.0001-n-15539/events.out.tfevents.1575324633.ml-server-2711-60gbram-p4gpu', scalar_name)
#x2,y2 = return_tensorborad_data('/Users/carlrosengren/Desktop/20191202_all_modalities_lee2/20191203-053819-all-lr-0.01-n-15539/events.out.tfevents.1575351504.ml-60gb-16vcpu-t4gpu-200g', scalar_name)
#x3,y3 = return_tensorborad_data('/Users/carlrosengren/Desktop/20191202_all_modalities_lee2/20191203-041405-all-lr-0.001-n-15539/events.out.tfevents.1575346450.ml-60gb-16vcpu-t4gpu-200g', scalar_name)
#x4,y4 = return_tensorborad_data('/Users/carlrosengren/Desktop/20191202_all_modalities_lee2/20191203-023112-all-lr-0.0001-n-15539/events.out.tfevents.1575340276.ml-60gb-16vcpu-t4gpu-200g', scalar_name)
#x5,y5 = return_tensorborad_data('/Users/carlrosengren/Desktop/20191202_all_modalities_lee2/20191203-005323-all-lr-1e-05-n-15539/events.out.tfevents.1575334407.ml-60gb-16vcpu-t4gpu-200g', scalar_name)

#plt.plot(x1, abs(y1[:,0]-y1[:,1]), label='Loss Error $\eta=10^{-4}$')
plt.plot(x1, y1[:,1], label=r' val acc. $\eta=10^{-4}$')
#plt.plot(x2, abs(y2[:,0] - y2[:,1]), label='Loss Error $\eta=10^{-2}$')
#plt.plot(x2, y2[:,1], label='val acc. $\eta=10^{-2}$')
#plt.plot(x3, abs(y3[:,0] - y3[:,1]), label='Loss Error $\eta=10^{-3}$')
#plt.plot(x3, y3[:,1], label='val acc $\eta=10^{-3}$')
#plt.plot(x4, abs(y4[:,0] - y4[:,1]),  label='Loss Error $\eta=10^{-4}$')
#plt.plot(x4, y4[:,1], label='val acc $\eta=10^{-4}$')
#plt.plot(x5, abs(y5[:,0] - y5[:,1]), label='Loss Error $\eta=10^{-5}$')
#plt.plot(x5, y5[:,1], label='val acc. $\eta=10^{-5}$')

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.axis([0, 18, 0.92, 0.98])
plt.title("Train Progress")
plt.legend(loc='upper right', frameon=True, bbox_to_anchor=(1.1, 1.15))
plt.show()
