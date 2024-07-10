import streamlit as st
import sys
import os
from ddpm_conditional import Diffusion
from ddpm_conditional_64 import Diffusion as DF64
from modules import UNet_conditional
from modules_64 import UNet_conditional as UN64
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np 
import torchvision.transforms as T


transform = T.ToPILImage()




if torch.cuda.is_available():
    st.write("CUDA is available.")
    device = "cuda"
else:
    st.write("CUDA is not available.")
    device = "cpu"

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.title('Diffusion Model for Image Generation')
st.write('Choose a model')

# Create a select box for model selection
model_choice = st.selectbox('Select a model', ('Diffusion for Number', 'DF64 for RGB'))

# Based on the selection, load the appropriate model
if model_choice == 'Diffusion for Number':
    diffusion = Diffusion(img_size=16, device=device)
    model = UNet_conditional(num_classes=10).to(device)
    ckpt = torch.load("./completed_models/ddpm_con_alld/ckpt.pt")
    model.load_state_dict(ckpt)
elif model_choice == 'DF64 for RGB':
    diffusion = DF64(img_size=64, device=device)
    model = UN64(num_classes=10).to(device)
    ckpt = torch.load("./completed_models/ddpm_con_64/ckpt.pt")
    model.load_state_dict(ckpt)

st.write(f'You selected the {model_choice} model.')

# Create checkboxes for class selection
st.write('Select classes to generate:')
selected_classes = []
for i in range(10):
    if st.checkbox(f'Class {i}'):
        selected_classes.append(i)

label_tensor = torch.tensor(selected_classes).to(device)
show_last_10_steps = st.checkbox('Show last 10 steps')

st.write(f'You selected the following classes: {selected_classes}')

if st.button('Generate Images'):
    label_tensor = torch.tensor(selected_classes).to(device)

    if model_choice == "Diffusion for Number":
        def plot_sampled_images(sampled_imgs):

            num_images = sampled_imgs.size(0)
            num_cols = 5
            num_rows = num_images // num_cols + 1  # Adjust the number of rows based on the number of images
            images = []
            for i in range(num_images):
                img = sampled_imgs[i].squeeze().cpu().numpy()
                img = Image.fromarray(img, mode='L')
                images.append(img)
            return images

        def sample(diffusion, model, n, labels, cfg_scale=1):
            model.eval()
            with torch.no_grad():
                x = torch.randn((n, 1, diffusion.img_size, diffusion.img_size)).to(diffusion.device)
                last_10_imgs = []
                for i in tqdm(reversed(range(1, diffusion.noise_steps)), position=3):
                    if i > 200:
                        continue
                    t = (torch.ones(n) * i).long().to(diffusion.device)

                    predicted_noise = model(x, t, labels)
                    if cfg_scale > 0:
                        uncond_predicted_noise = model(x, t, None)
                        predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                    alpha = diffusion.alpha[t][:, None, None, None]
                    alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
                    beta = diffusion.beta[t][:, None, None, None]
                    if i > 1:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)
                    x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                    x = x.clamp(-1, 1)
                    
                    if show_last_10_steps and i <= 10:
                        tmp_x = (x.clamp(-1, 1) + 1) / 2
                        tmp_x = (tmp_x * 255).type(torch.uint8)
                        last_10_imgs.append(tmp_x.cpu())

                model.train()
                x = (x.clamp(-1, 1) + 1) / 2
                x = (x * 255).type(torch.uint8)
                return x, last_10_imgs

        sigmas = diffusion.beta
        sampled_imgs, last_10_imgs = sample(diffusion, model, n=len(label_tensor), labels=label_tensor)
        sampled_images = plot_sampled_images(sampled_imgs)
        print(type(sampled_images[0]))
        print(type(last_10_imgs[0]))

        # last_10_imgs_pil = []
        # for img in last_10_imgs:
        #     img_ten = transform(img)
        #     last_10_imgs_pil.append(img_ten)

        images_to_display = sampled_images 
        num_images = len(images_to_display)
        num_cols = 5
        num_rows = (num_images + num_cols - 1) // num_cols  # Ceiling division    
        st.write('Generated Images:')
        print(type(sampled_images))
        for row in range(num_rows):
            st.write(f'Row {row + 1}:')
            start_idx = row * num_cols
            end_idx = min((row + 1) * num_cols, num_images)
            row_images = images_to_display[start_idx:end_idx]
            
            row_col1, row_col2, row_col3, row_col4, row_col5 = st.columns(5)
            
            if len(row_images) > 0:
                row_col1.image(row_images[0], caption='Generated Image', width=128)
            if len(row_images) > 1:
                row_col2.image(row_images[1], caption='Generated Image', width=128)
            if len(row_images) > 2:
                row_col3.image(row_images[2], caption='Generated Image', width=128)
            if len(row_images) > 3:
                row_col4.image(row_images[3], caption='Generated Image', width=128)
            if len(row_images) > 4:
                row_col5.image(row_images[4], caption='Generated Image', width=128)
            
        if show_last_10_steps:
            st.write('Images from the last 10 steps:')
            for step, imgs in enumerate(last_10_imgs, 1):
                st.write(f'Step {step}')
                images = plot_sampled_images(imgs)
                for img in images:
                    st.image(img, caption=f'Step {step}', width=128)
        

    

    elif model_choice =="DF64 for RGB":
        def plot_sampled_images(sampled_imgs):

            num_images = sampled_imgs.size(0)
            images = []
            for i in range(num_images):
                img = sampled_imgs[i].cpu().numpy()
                if img.shape[0] == 1:  # Grayscale image
                    img = img.squeeze()  # Remove channel dimension
                    img = Image.fromarray(img, mode='L')
                elif img.shape[0] == 3:  # RGB image
                    img = np.transpose(img, (1, 2, 0))  # Change from CHW to HWC format
                    img = Image.fromarray(img.astype(np.uint8))
                images.append(img)
            return images

        def sample(diffusion, model, n, labels, cfg_scale=3):
            model.eval()
            with torch.no_grad():
                x = torch.randn((n, 3, diffusion.img_size, diffusion.img_size)).to(diffusion.device)
                last_10_imgs = []
                for i in tqdm(reversed(range(1, diffusion.noise_steps)), position=3):
                    if i > 1000:
                        continue
                    t = (torch.ones(n) * i).long().to(diffusion.device)

                    predicted_noise = model(x, t, labels)
                    if cfg_scale > 0:
                        uncond_predicted_noise = model(x, t, None)
                        predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                    alpha = diffusion.alpha[t][:, None, None, None]
                    alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
                    beta = diffusion.beta[t][:, None, None, None]
                    if i > 1:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)
                    x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                    x = x.clamp(-1, 1)
                    
                    if show_last_10_steps and i <= 10:
                        tmp_x = (x.clamp(-1, 1) + 1) / 2
                        tmp_x = (tmp_x * 255).type(torch.uint8)
                        last_10_imgs.append(tmp_x.cpu())

                model.train()
                x = (x.clamp(-1, 1) + 1) / 2
                x = (x * 255).type(torch.uint8)
                return x, last_10_imgs

        sigmas = diffusion.beta
        sampled_imgs, last_10_imgs = sample(diffusion, model, n=len(label_tensor), labels=label_tensor)
        sampled_images = plot_sampled_images(sampled_imgs)
        
        st.write('Generated Images:')
        for img in sampled_images:
            st.image(img, caption='Generated Image', width=128)
        
        if show_last_10_steps:
            st.write('Images from the last 10 steps:')
            for step, imgs in enumerate(last_10_imgs, 1):
                st.write(f'Step {step}')
                images = plot_sampled_images(imgs)
                for img in images:
                    st.image(img, caption=f'Step {step}', width=128)
        