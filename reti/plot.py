print(f'Summary loss train: {summary_loss_train}')
print(f'Summary loss train shape: {np.shape(summary_loss_train)}')
print(f'Summary loss val: {summary_loss_train}')
print(f'Summary loss val shape: {np.shape(summary_loss_val)}')
# summary_acc_train_cpu = summary_acc_train.cpu()
#print(f'Summary acc train shape: {np.shape(summary_acc_train.cpu())}')

sommario_acc_train_array = []
for idx in range(len(summary_acc_train)):
    sommario_acc_train_array.append(summary_acc_train[idx].cpu().clone().detach().numpy())
    #print(sommario_acc_train_array[idx])
print(f'Sommario acc train array shape: {np.shape(sommario_acc_train_array)}')


sommario_acc_val_array = []
for idx in range(len(summary_acc_val)):
    sommario_acc_val_array.append(summary_acc_val[idx].cpu().clone().detach().numpy())
    print(sommario_acc_val_array[idx])
print(f'Sommario acc val array shape: {np.shape(sommario_acc_val_array)}')

x = [i for i in range(num_epochs)]
print(x)

#Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,6))

x = [i for i in range(num_epochs)]
print(x)

ax1.plot(x, summary_loss_train,  label = 'Training Loss')
ax1.plot(x, summary_loss_val, label = 'Validation Loss')
ax1.legend()

sommario_acc_train_array = []
for idx in range(len(summary_acc_train)):
    sommario_acc_train_array.append(summary_acc_train[idx].cpu().clone().detach().numpy())
print(f'Sommario acc train array: {np.shape(sommario_acc_train_array)}')

sommario_acc_val_array = []
for idx in range(len(summary_acc_val)):
    sommario_acc_val_array.append(summary_acc_val[idx].cpu().clone().detach().numpy())
print(f'Sommario acc val array: {np.shape(sommario_acc_val_array)}')

# ax2.plot(x, sommario_acc_train_array, label='Training Accuracy')
# ax2.plot(x, sommario_acc_val_array, label='Validation Accuracy')
# ax2.legend()

plt.show()