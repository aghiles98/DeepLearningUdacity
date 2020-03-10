import argparse
from utility_functions import *

# Get input args 
def get_args():    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Create command line argumentsusing add_argument() from ArguementParser method
    parser.add_argument('image', type = str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--category_names', type=str, default=' ')
    parser.add_argument('--gpu', action='store_true')
    return parser.parse_args()


def main():
    
    args = get_args() # get inpyut args
    device = 'cpu' # define the device to use cpu default
    if args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'
    model = load_checkpoint(args.checkpoint) # load the model
    image = process_image(args.image).unsqueeze_(0) # process the image 
    top_probs, top_classes = predict(image, device, model, args.top_k)
    top_probs, top_classes = top_probs.cpu().numpy(), top_classes.cpu().numpy()
    # Log the result
    if args.category_names != ' ':
        names = []
        # loading the json file of the category names
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        for category in top_classes:
            name = model.class_to_idx[str(category + 1)]
            names.append(cat_to_name[str(name + 1)])
        for result in range(len(top_probs)):
            print('Rank: {:<4}, ClassName: {}, Class: {:<4}, Probability: {:.4f}\n'.format(result + 1, names[result], top_classes[result], top_probs[result]))     
    else:
        for result in range(len(top_probs)):
            print('Rank: {:<4}, Class: {:<4}, Probability: {:.4f}\n'.format(result + 1, top_classes[result], top_probs[result]))

# calling the main function
if __name__ == "__main__":
    main()
    