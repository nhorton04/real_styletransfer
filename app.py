import streamlit as st
from models import TransformerNet
from utils import *
import torch
import numpy as np
from torch.autograd import Variable
import argparse
import tkinter as tk
import os
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tqdm
from torchvision.utils import save_image
from PIL import Image

def main():

    uploaded_file = st.file_uploader("Choose a picc", type=['jpg', 'png', 'webm', 'mp4', 'gif', 'jpeg'])
    if uploaded_file is not None:
        st.image(uploaded_file, width=200)

    folder = os.path.abspath(os.getcwd())
    folder = folder + '/models'

    fnames = []

    for basename in os.listdir(folder):
        print(basename)
        fname = os.path.join(folder, basename)

        if fname.endswith('.pth'):
            fnames.append(fname)
    checkpoint = st.selectbox('Select a pretrained model', fnames)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--image_path", type=str, required=True, help="Path to image")
    # parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
    # args = parser.parse_args()
    # print(args)

    os.makedirs("images/outputs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    transform = style_transform()
    try:
        # Define model and load model checkpoint
        transformer = TransformerNet().to(device)
        transformer.load_state_dict(torch.load(checkpoint))
        transformer.eval()

        # Prepare input
        image_tensor = Variable(transform(Image.open(uploaded_file).convert('RGB'))).to(device)
        image_tensor = image_tensor.unsqueeze(0)

        # Stylize image
        with torch.no_grad():
            stylized_image = denormalize(transformer(image_tensor)).cpu()

        # colormaps = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']

    # colormap = st.selectbox('Select a colormap', colormaps)

    # plt.imshow(stylized_image.numpy()[0][0], cmap=colormap)
    # plt.imshow(stylized_image.numpy()[0][0], cmap='gist_rainbow')
    # img = np.squeeze(stylized_image)
    # plt.imshow(img[0])
    # plt.show()
    # st.image(img)
    # # Save image

        fn = str(np.random.randint(0, 100)) + 'image.jpg'
        save_image(stylized_image, f"images/outputs/stylized-{fn}")

        st.image(f"images/outputs/stylized-{fn}")
    except:
        st.write('Choose an image')
    # imagee = cv2.imread(f"images/outputs/stylized-{fn}")
    # cv2.imshow('Image', imagee)

if __name__ == "__main__":
    main()
