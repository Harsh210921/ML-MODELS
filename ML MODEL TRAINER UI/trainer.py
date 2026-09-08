def training_code(num_epoch,model,train_loader,optimizer,criterion,test_loader):
    import copy
    import torch
    num=num_epoch
    best_loss=float("inf")
    best_model=None
    for epoch in range(num):
     train_loss=0
     model.train()
     for inputs, labels in train_loader:
        optimizer.zero_grad()
        output=model(inputs)
        loss=criterion(output,labels)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
     train_epoch_loss=train_loss/len(train_loader)
     print("-----------------")
     print(f"EPOCH :{epoch}")
     print(f"EPOCH TRAIN LOSS : {train_epoch_loss}")
     model.eval()
     with torch.no_grad():
      val_loss=0
      for inputs, labels in test_loader:
        output=model(inputs)
        loss=criterion(output,labels)
        val_loss+=loss.item()
      val_epoch_loss=val_loss/len(test_loader)
      if val_epoch_loss < best_loss:
        best_loss=val_epoch_loss
        best_model=copy.deepcopy(model.state_dict())
      print(f"VALIDATION LOSS : {val_epoch_loss} ,LOWEST VAL LOSS: {best_loss} ")
    return model, best_loss