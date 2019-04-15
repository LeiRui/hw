# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.

import torch.optim as optim

def createOptimizer(model_ft, feature_extract):
  params_to_update = model_ft.parameters()
  print("Params to learn:")
  if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
      if param.requires_grad == True:
        params_to_update.append(param)
        print("\t",name)
  else:
    for name,param in model_ft.named_parameters():
      if param.requires_grad == True:
        print("\t",name)

  # Observe that all parameters are being optimized
  #optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
  optimizer_ft = optim.Adam(params_to_update, lr=0.001)
  return optimizer_ft

