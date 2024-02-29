#for cycle for training
for epoch in range(num_epochs):
    
    total_batch = len(train_data)//batch_size

    acc_current_train = 0.0
    cost_current_train = 0.0


    for i, data in enumerate(train_loader):
        
        #pass data to cuda device
        batch_images, batch_labels = data
        if torch.cuda.is_available():
          batch_images, batch_labels = batch_images.cuda(), batch_labels.cuda()

        X = batch_images
        # print(X)
        Y = batch_labels
        # print(Y)
        # print(i)
      
        #IF YOU ARE *NOT* USING aux_logits
        if (model.aux_logits == False):
          pre = model(X)
          cost = loss(pre, Y)
          _, preds = torch.max(pre, 1)

        
        #ELSE IF YOU ARE USING aux_logits: in these lines of code pre are the outputs of the model applied to the batch, then pre is converted into a tensor called "output"
        #with wich is calculated the loss function
        else:
          output, pre = model(X)
          cost = loss(output,Y)
          _, preds = torch.max(output, 1)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        #the next three lines are useful to print accuracy during training 
        acc_current_train += torch.sum(preds == Y.data)
        epoch_acc = acc_current_train.float() / len(train_data)
        cost_current_train += cost.item() * batch_images.size(0)
        epoch_loss = cost_current_train / len(train_loader)
        summary_acc_train.append(epoch_acc)

        if (i+1) % 5 == 0:
            print('Epoch [%d/%d], lter [%d/%d] Loss: %.4f, Accuracy:%.4f %%'
                 %(epoch+1, num_epochs, i+1, total_batch, cost.item(), (100 * epoch_acc.item())))
          
    scheduler.step()
#evaluating the model
    model.eval()

    correct = 0
    total = 0
    
    loss_current_val = 0.0
    acc_current_val = 0.0
    
    total_batch_val = len(test_data)//batch_size

    for j, data in enumerate(test_loader):

        #pass data to cuda device
        images, labels = data
        if torch.cuda.is_available():
          images, labels = images.cuda(), labels.cuda()
        
    #   images = images.cuda()
        outputs = model(images)
        
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()
        
        val_cost = loss(outputs, labels)
        
        loss_current_val += val_cost.item() * images.size(0)
        val_epoch_loss = loss_current_val / len(test_loader)
        acc_current_val += torch.sum(predicted == labels.data)
        val_epoch_acc = acc_current_val.float() / len(test_loader)

        if (j+1) % 5 == 0:
          # print('Accuracy of test images with %f epochs and %f: %f %%' % (100 * float(correct) / total))
          print(f'Epoch: {epoch+1}, batch {batch_size}, learning rate {lr}, iter: [{j+1}/{total_batch_val}]: %f %%' %(100 * float(correct) / total))

    #Plot

    summary_loss_train.append(epoch_loss)
    summary_acc_train.append(epoch_acc)  
    summary_loss_val.append(val_epoch_loss)
    summary_acc_val.append(val_epoch_acc)


    if val_epoch_acc > best_acc:

         best_acc = val_epoch_acc
         best_model_wts = copy.deepcopy(model.state_dict())
         summary_acc_train.append(epoch_acc)

        
    # print('Accuracy of test images with %f epochs and %f: %f %%' % (100 * float(correct) / total))
    #print(f'{num_epochs} epochs, batch {batch_size}, learning rate 0.001: %f %%' %(100 * float(correct) / total))

print(f'Summary: Neural Networl:{model_down}, epochs:{epoch+1},batch:{batch_size}, learning rate: {lr}, \n Training: Total batch: [{total_batch}], Loss:{epoch_loss}, Accuracy: {100 * epoch_acc}%, \n Validation: batch size:[{total_batch_val}], Loss: {val_epoch_loss}, Accuracy: {100 * val_epoch_acc}%')
