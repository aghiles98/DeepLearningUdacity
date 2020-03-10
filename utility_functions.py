# Imports here
from torchvision import transforms, datasets,models
from torch.utils.data import DataLoader
import torch
from torch import optim, cuda
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict 
import numpy as np
import os 
from PIL import Image
import json

# defining the directories of our data.
def data_dir(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data = {'train': train_dir, 'valid': valid_dir, 'test': test_dir}
    return data

# Done: Define your transforms for the training, validation, and testing sets
def loaders(data):
    # Done: Define your transforms for the training, validation, and testing sets
# Image transformations
    data_transforms = {
        # Train uses data augmentation
        'train':
        transforms.Compose([
            transforms.RandomRotation(40),
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # Imagenet standards
        ]),
        # Validation does not use augmentation
        'valid':
        transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # Test does not use augmentation
        'test':
        transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Done: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data['train'], data_transforms['train'])
    valid_data = datasets.ImageFolder(data['valid'], data_transforms['valid'])
    test_data = datasets.ImageFolder(data['test'], data_transforms['test'])

    # Done: Using the image datasets and the trainforms, define the dataloaders
    # Data iterators
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    validationloader = DataLoader(valid_data, batch_size=64, shuffle=True)
    testloader = DataLoader(test_data, batch_size=64, shuffle=True)
    
    return {'train': trainloader, 'valid': validationloader, 'test': testloader, 'trainset': train_data}

#Loading in a pre-trained model
def load_pretrained_network(arch):
    resnet18 = models.resnet18(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)
    densenet161 = models.densenet161(pretrained=True)
    _models = {'resnet':resnet18,'alexnet': alexnet,'vgg': vgg16, 'densenet': densenet161, }
    network = _models['vgg'] 
    if arch in _models:
        network = _models[arch]
    # Freeze model weights
    for param in network.parameters():
        param.requires_grad = False
    return network

# Build the classifier
def build_classifier(model, arch,hidden_units, train_loader):
    # freeze parameters of the model
    for param in model.parameters():
        param.requires_grad = False
    if (arch=='densenet'):
         n_inputs = model.classifier.in_features # number of inputs
    else: 
         n_inputs = model.classifier[-1].in_features 
         
    n_classes = len(train_loader.dataset.classes) # the number of classes in the last layer

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(n_inputs, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(int(hidden_units), n_classes)),
        ('out', nn.LogSoftmax(dim=1))
    ]))
    if (arch=='densenet'):
         model.classifier = classifier 
    else:
        model.classifier[-1] = classifier 
    return model

# Implement a function for the validation pass
def validation(model, device,testloader, criterion):
    model.eval()
    loss = 0
    accuracy = 0
    for images, labels in testloader:  
        if device!='cpu':
            images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1]).to(device)
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return loss, accuracy 

def train(model, 
          device,
          train_loader,
          valid_loader,
          learning_rate,
          epochs=2,
          print_every = 2):
    print('Start training ..')
    steps = 0
    criterion = nn.CrossEntropyLoss() # define the criterion
    optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=learning_rate) # define the optimizer 
    model.to(device)
    running_loss = 0
    for e in range(epochs):
        model.train()
        for images, labels in train_loader:
            steps += 1
            if device!='cpu':
                images, labels = images.to(device), labels.to(device)     
            optimizer.zero_grad()       
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()  
                    
            running_loss += loss.item()
            
            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()
                
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model,device ,valid_loader, criterion)
                    
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                    "Test Loss: {:.3f}.. ".format(test_loss/len(valid_loader)),
                    "Test Accuracy: {:.3f}".format(accuracy/len(valid_loader)))
                
                running_loss = 0
                
                # Make sure training is back on
                model.train()
                
# Function to test the model 
def test_model(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    if model!='cpu':
        model.to(device)
    model.eval()
    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        test_loss, test_acc = validation(model,device ,test_loader, criterion)
        test_loss,test_acc = test_loss/len(test_loader), test_acc/len(test_loader)
    #Print the result 
    print("Test Loss: {:.3f}.. ".format(test_loss),"Test Accuracy: {:.2f}%".format( 100 * test_acc))
    
    
# TODO: Save the checkpoint 
def save_model(model, path, arch, trainset,hidden_units,lr,epochs):
    print('Saving the model to ./{}/checkpoint.pth'.format(path))
    model.class_to_idx = trainset.class_to_idx
    checkpoint = {
        'arch': arch,
        'model': model,
        'input_size':(3,244,244),
        'output_size': 102, 
        'hidden_units': hidden_units,
        'learning_rate': lr,
        'features': model.features,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'epochs': epochs
    }
    if not os.path.exists(path):
        print('save directories...', flush = True)
        os.makedirs(path)
    torch.save(checkpoint, path + '/checkpoint.pth')
    
    
# Load function 
def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)  # Load the saved file  
    model = checkpoint['model']  
    model.classifier = checkpoint['classifier']
    model.features = checkpoint['features']
    #model.cat_to_name = checkpoint['cat_to_name']
    model.class_to_idx = checkpoint['class_to_idx']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    return model



# Process Image function 
def process_image(image_path):
    image = Image.open(image_path)
    image = image.resize((256, 256)) # resize the image 
    cropped_image = image.crop((0, 0, 249, 249)) # crop the image 
    image_to_nparray= np.array(cropped_image) # image to numpy array
    # Normalize the image 
    image_to_nparray = (image_to_nparray / 45) - np.array([0.485, 0.456, 0.406]) / np.array([0.229, 0.224, 0.225])
    image_to_nparray = image_to_nparray.transpose(2, 0, 1) #  Reorder the dimension of the color
    return torch.from_numpy(image_to_nparray)


def predict(image, device, model, topk):
    if len(model.class_to_idx) < topk: # if the given top k classes is much more the number of model outputs 
        topk = len(model.class_to_idx)
    model.to(device)
    with torch.no_grad():
        model.eval()
        image = image.float().to(device)
        output = model.forward(image) # predict the result
        prediction = torch.exp(output).data[0].topk(topk)
        return prediction
    

