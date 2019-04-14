import time
import copy
import torch
import numpy as np

# helper function
def train_model(device, model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
  # Send the model to GPU
  model= model.to(device)

  since = time.time()

  train_loss_history = []
  train_acc_history = []
  val_loss_history = []
  val_acc_history = []

  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # confusion matrix initialization
    cfm = np.array([[0 for col in range(65)] for row in range(65)])

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()  # Set model to training mode. MR: train_loader
      else:
        model.eval()   # Set model to evaluate mode. MR: valid_loader

      running_loss = 0.0
      running_corrects = 0

      # Iterate over data.
      for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)
        '''
          if train_on_gpu: data, target = data.cuda(), target.cuda()
        '''

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'): # MR: 如果是train就会需要backward & gradient calculation
          # Get model outputs and calculate loss
          # Special case for inception because in training it has an auxiliary output. In train
          #   mode we calculate the loss by summing the final output and the auxiliary output
          #   but in testing we only consider the final output.
          if is_inception and phase == 'train':
            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
            outputs, aux_outputs = model(inputs)
            loss1 = criterion(outputs, labels)
            loss2 = criterion(aux_outputs, labels)
            loss = loss1 + 0.4*loss2
          else:
            outputs = model(inputs) # to(device)？
            loss = criterion(outputs, labels)

          _, preds = torch.max(outputs, 1)

          if phase == 'train': # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()
          else: # update confusion matrix
            for i in range(len(preds)):
              cfm[labels.cpu().numpy()[i]][preds.cpu().numpy()[i]] += 1 # results of the current batch

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

      epoch_loss = running_loss / len(dataloaders[phase].dataset)
      epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

      print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

      if phase == 'train':
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc.item())
      if phase == 'val':
        val_loss_history.append(epoch_loss)
        val_acc_history.append(epoch_acc.item())
        # deep copy the model
        if epoch_acc > best_acc:
          best_acc = epoch_acc
          best_model_wts = copy.deepcopy(model.state_dict())
          best_cfm = copy.deepcopy(cfm)
          torch.save(model.state_dict(), 'epoch_backup_model.pkl')

    print() # Each epoch has a training and validation phase

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:4f}'.format(best_acc))
  # np.set_printoptions(threshold=np.inf)
  # print('Confusion matrix under best val acc:\n', best_cfm)

  # load best model weights
  model.load_state_dict(best_model_wts)
  return model, train_loss_history, train_acc_history, \
         val_loss_history, val_acc_history, best_cfm

