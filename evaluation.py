import torch
import torchvision.transforms as transforms
from PIL import Image
from model import CNNtoRNN
from Data_Loader import FlickrDataset

# Hyperparameters
embed_size = 256
hidden_size = 256
vocab_size = 2994
num_layers = 1
device = 'cuda'
#learning_rate = 0.001
#num_epochs = 40
root_folder=r"flickr8k\images",
annotation_file=r"flickr8k\captions.txt"

model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to('cuda')

def print_examples(model, device, root_folder, annotation_file):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = FlickrDataset(root_folder, annotation_file, transform = transform)
    model.eval()
    test_img1 = transform(Image.open(r"flickr8k\images\44129946_9eeb385d77.jpg").convert("RGB")).unsqueeze(
        0
    )
    print(
        "OUTPUT: "
        + " ".join(model.caption_image(test_img1.to(device), dataset.vocab))
    )

print_examples(model, device, root_folder, annotation_file)