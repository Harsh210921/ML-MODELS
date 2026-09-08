from flask import Flask, request, jsonify
from flask import Flask
from flask_cors import CORS
def model_training(model_name, num_output, num_epoch):
    import torch
    import torchvision
    from pathlib import Path
    from torchvision.datasets import ImageFolder
    from torchvision.transforms import transforms
    from torch.utils.data import DataLoader
    dataset=Path("dataset")
    train_path=dataset/"train"
    test_path=dataset/"test"
    val_path=dataset/"valid"
    print("DATASET PATH LOADED")
    #-------------------------------#
    transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    train_transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    print("DATASET TRANSFORMATION")
    #---------------------------------#
    
    train_dataset=ImageFolder(train_path,transform=train_transform)
    test_dataset=ImageFolder(test_path,transform=transform)
    # val_dataset=ImageFolder(val_path,transform=transform)
    print("DATASET LOADED TO IMAGEFOLDER")
    train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
    test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)
    # val_loader=DataLoader(val_dataset,batch_size=32,shuffle=False)
    print("DATASET LOADED TO DATALOADER")

    #----------------------------------#
    weight=None
    optimizer=None

    import torch.nn as nn
    criterion=nn.CrossEntropyLoss()
    if model_name=="RESNET 50":
       from torchvision.models import resnet50, ResNet50_Weights
       weight=ResNet50_Weights
       model=resnet50(weights=weight.DEFAULT)
       model.fc=nn.Linear(2048,num_output)
       for parameter in model.parameters():
        parameter.requires_grad=False
       for parameter in model.fc.parameters():
        parameter.requires_grad=True
       optimizer=torch.optim.Adam(
        model.fc.parameters(),
         lr=0.001
         )

    elif model_name=="EFFICIENT NET B0":
       from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
       weight=EfficientNet_B0_Weights.DEFAULT
       model=efficientnet_b0(weights=weight)
       model.classifier[1] = nn.Linear(1280,num_output)
       for parameter in model.parameters():
        parameter.requires_grad=False
       for parameter in model.classifier[1].parameters():
        parameter.requires_grad=True
       optimizer=torch.optim.Adam(
          model.classifier[1].parameters(),
         lr=0.001)
    #----------------------------------#
    from trainer import training_code
    # NEW
    return training_code(num_epoch, model, train_loader, optimizer, criterion, test_loader)
    

       
       




   















app=Flask(__name__)
CORS(app)
@app.route("/test")
def test():
 return "THIS IS BACKEND : TEST SUCCESSFULL"

@app.route("/send", methods=["POST"])
def collect():
    data = request.get_json()
    print("Received parameters:", data)
    model_name=data.get("model")
    num_output=data.get("output", 5)
    num_epoch=data.get("epoch", 5)
    trained_model, lowest_loss = model_training(model_name, num_output, num_epoch)
    return jsonify({
        "status": "SUCCESS",
        "message": f"Finished training {model_name}",
        "lowest_val_loss": lowest_loss
    })



app.run()
