import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from model import CNNtoRNN
from Data_Loader import FlickrDataset

# Hyperparameters
embed_size = 256
hidden_size = 256
vocab_size = 2994
num_layers = 1
device = 'cuda'
root_folder = r"flickr8k\images"
annotation_file = r"flickr8k\captions.txt"

model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
# Load model weights
model.load_state_dict(torch.load(r'flickr8k/my_checkpoint.pth.tar', weights_only=True)['state_dict'], strict=False)
model.eval()

def denormalize(image_tensor):
    """Helper function to denormalize the image for display."""
    image_tensor = image_tensor.clone().squeeze(0)
    # Move mean and std to the same device as image_tensor
    mean = torch.tensor([0.5, 0.5, 0.5], device=image_tensor.device)
    std = torch.tensor([0.5, 0.5, 0.5], device=image_tensor.device)
    image_tensor = image_tensor * std[:, None, None] + mean[:, None, None]  # Undo normalization
    return image_tensor

def clean_caption(caption):
    """Remove <SOS> and <EOS> tokens from the caption."""
    return caption.replace("<SOS>", "").replace("<EOS>", "").strip()

def get_dynamic_font_size(text, max_width, max_fontsize=12, min_fontsize=6):
    """Calculate dynamic font size based on text length and available space."""
    fig, ax = plt.subplots()
    renderer = fig.canvas.get_renderer()
    
    font_size = max_fontsize
    text_obj = ax.text(0, 0, text, fontsize=font_size)
    while text_obj.get_window_extent(renderer=renderer).width > max_width and font_size > min_fontsize:
        font_size -= 1
        text_obj.set_fontsize(font_size)
    
    plt.close(fig)
    return font_size

def display_images_with_text(images, captions):
    """Display images with their predicted captions in a 3x3 grid."""
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    axs = axs.flatten()  # Flatten the 3x3 grid to easily access each subplot
    
    for i, (image, caption) in enumerate(zip(images, captions)):
        # Convert tensor image to numpy and denormalize
        image = denormalize(image)
        
        # Clean the predicted text by removing <SOS> and <EOS>
        caption = clean_caption(caption)
        
        # Convert to numpy and clip image values to ensure valid range [0, 1] for display
        image = image.permute(1, 2, 0).cpu().numpy()
        image = np.clip(image, 0, 1)
        
        # Display the image
        axs[i].imshow(image)

        # Create a rectangle for the text at the bottom
        rect_width = 300
        rect_height = 80
        rect = patches.Rectangle((0, 0), rect_width, rect_height, linewidth=1, edgecolor='black', facecolor='white', alpha=0.7)
        axs[i].add_patch(rect)

        # Calculate dynamic font size for predicted text
        predicted_font_size = get_dynamic_font_size(caption, rect_width - 20)

        # Display predicted text with dynamic font size
        axs[i].text(10, 40, f"Predicted: {caption}", fontsize=predicted_font_size, color='red')

        axs[i].axis('off')  # Hide axes

    plt.tight_layout()
    plt.show()

def print_examples(model, device, root_folder, annotation_file):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = FlickrDataset(root_folder, annotation_file, transform=transform)
    model.eval()
    
    # Load 9 images and their paths
    image_paths = [
        r"flickr8k\images\33108590_d685bfe51c.jpg",
        r"C:\Users\pc\Downloads\dog.jpg",  # Replace with actual image paths
        r"C:\Users\pc\Downloads\child.jpg",
        r"C:\Users\pc\Downloads\bus.png",
        r"C:\Users\pc\Downloads\boat.png",
        r"C:\Users\pc\Downloads\horse.png",
        r"flickr8k\images\47870024_73a4481f7d.jpg",
        r"flickr8k\images\3726025663_e7d35d23f6.jpg",
        r"flickr8k\images\3725177385_62d5e13634.jpg",
    ]
    
    images = []
    captions = []

    for img_path in image_paths:
        test_img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0)
        predicted_caption = " ".join(model.caption_image(test_img.to(device), dataset.vocab))
        images.append(test_img)
        captions.append(predicted_caption)
    
    # Display the images with their predicted captions
    display_images_with_text(images, captions)

print_examples(model, device, root_folder, annotation_file)
