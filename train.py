import argparse 
from utility_functions import * 
from workspace_utils import active_session 
import sys

# Get input args 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type = str, default = 'flowers')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--epochs',  type=int, default=20)  
    parser.add_argument('--learning_rate',  type=float, default=0.01)  
    parser.add_argument('--arch',type=str ,default = 'vgg')
    parser.add_argument('--hidden_units', type=int, default=256)
    return parser.parse_args() 

# min function 
def main():
    args = get_args() # get input args
    device = 'cpu' # chose the device cpu default 
    if args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'  
    data = data_dir(args.data_dir) # set the dataset
    data_loaders = loaders(data) # load dataset
    model = load_pretrained_network(args.arch) # load pretrained model
    model= build_classifier(model, args.arch,args.hidden_units, data_loaders['train']) # build the classifier
    print("The classifier => {} ".format(model.classifier))
    # train the model 
    train(model,
        device,
        data_loaders['train'],
        data_loaders['valid'],
        args.learning_rate,
        epochs=args.epochs,
        print_every = 40)
    test_model(model, data_loaders['test'], device)   # test the model 
    # Save the model 
    save_model(model, args.save_dir,args.arch,data_loaders['trainset'],args.hidden_units,args.learning_rate,args.epochs)
    print(" Model Training Completed Successfully !")
    
    
if __name__ == "__main__":
    sys.stdout.flush()
    with active_session():
         main()