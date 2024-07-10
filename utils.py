import os
import torch
import torchvision
from torchvision import transforms, datasets
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def plot_images(images):
    plt.figure(figsize=(32, 32))
    # Concatenate images along the last dimension for display
    concatenated_images = torch.cat([i for i in images.cpu()], dim=-1)
    concatenated_images = concatenated_images.view(16, -1).cpu().numpy()
    plt.imshow(concatenated_images, cmap='gray')
    plt.axis('off')
    plt.show()

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


# def get_data(args):
#     print("Entering get data")
    
#     # Define transformations
#     transform = transforms.Compose([
#         transforms.Resize(80),  # args.image_size + 1/4 * args.image_size
#         transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
#         transforms.Grayscale(num_output_channels=1),  # Ensure the image is single-channel grayscale
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))  # Normalize for grayscale images
#     ])
    
#     # Create the dataset
#     dataset = datasets.ImageFolder(args.dataset_path, transform=transform)
    
#     # Custom label mapping: map class_n to label n
#     custom_class_to_idx = {cls_name: int(cls_name.split('_')[1]) for cls_name in dataset.classes}
    
#     # Apply the custom label mapping
#     dataset.targets = [custom_class_to_idx[dataset.classes[i]] for i in dataset.targets]
    
#     # Update the dataset classes to reflect the new order
#     dataset.class_to_idx = custom_class_to_idx
#     dataset.classes = list(custom_class_to_idx.keys())
    
#     print("This is dataset class", dataset.classes)
#     print("Custom class to idx mapping:", custom_class_to_idx)
    
#     # Create the DataLoader
#     dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
#     return dataloader



def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
