# DCGAN Lung X-Ray Generator

This repository contains a Jupyter notebook implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) for generating lung X-ray images. The project is inspired by several works on GANs and applies state-of-the-art deep learning techniques to medical imaging, specifically chest X-rays for pneumonia detection.


## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [References](#references)
- [License](#license)

---

## Overview

This notebook demonstrates how to train a DCGAN model with Keras, PyTorch, and TensorFlow to generate realistic lung X-ray images. The primary goal is to augment medical datasets and explore generative modeling in healthcare.

### Key Objectives

- Load and preprocess the Chest X-Ray dataset (including pneumonia and normal cases).
- Visualize and analyze dataset statistics.
- Implement DCGAN architecture for image generation.
- Train, evaluate, and visualize the results of the generator.

---

## Features

- **Hybrid Frameworks**: Uses Keras, PyTorch, and TensorFlow for flexibility and performance.
- **Visualization**: Extensive use of matplotlib for dataset and result visualization.
- **Custom Utilities**: Functions for timing, grid image creation, and data preprocessing.
- **GPU Acceleration**: Designed to utilize GPU for faster training.

---

## Dataset

The primary dataset used is the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) available on Kaggle.

- **Normal:** 1583 images
- **Pneumonia:** 4273 images

**Directory Structure:**
```
chest_xray/
├─ train/
│  ├─ NORMAL/
│  └─ PNEUMONIA/
├─ test/
│  ├─ NORMAL/
│  └─ PNEUMONIA/
└─ val/
   ├─ NORMAL/
   └─ PNEUMONIA/
```

---

## Prerequisites

Make sure you have the following installed:

- Python 3.7+
- [Jupyter Notebook](https://jupyter.org/)
- [Keras](https://keras.io/) (Tested on version 2.x)
- [TensorFlow](https://www.tensorflow.org/) (Tested on version 2.x)
- [PyTorch](https://pytorch.org/) (Tested on version 1.x)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Pillow](https://python-pillow.org/)
- [imageio](https://imageio.readthedocs.io/)
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)

Install dependencies using pip:

```bash
pip install numpy pandas matplotlib pillow imageio tensorflow keras torch scikit-learn
```

---

## Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/DarkLord-13/DCGAN_Lung_XRay_Generator.git
   cd DCGAN_Lung_XRay_Generator
   ```

2. **Download the dataset:**

   Download [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) from Kaggle and place it in the correct directory.

3. **Open the notebook:**

   ```bash
   jupyter notebook lung_xray_dcgan.ipynb
   ```

4. **Run the notebook step-by-step:**
   - Inspect the dataset.
   - Preprocess images.
   - Visualize samples and grids.
   - Train the DCGAN.
   - Generate synthetic lung X-ray images.

---

## Project Structure

```
DCGAN_Lung_XRay_Generator/
├─ lung_xray_dcgan.ipynb      # Main notebook
├─ assets/
│  └─ generated_samples.png   # Example generated images (add your own)
├─ README.md                  # This file
└─ LICENSE                    # License information
```

---

## References

- [DCGAN with Keras (Kaggle)](https://www.kaggle.com/waltermaffy/dcgan-with-keras)
- [Chest X-Ray Images (Pneumonia) Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- [Keras-GANs](https://github.com/eriklindernoren/Keras-GAN)
- [Dog Memorizer GAN (Kaggle)](https://www.kaggle.com/cdeotte/dog-memorizer-gan)
- [MIT 6.S191: Introduction to Deep Learning](http://introtodeeplearning.com)
- [YouTube: Generative Adversarial Networks](https://www.youtube.com/watch?v=dCKbRCUyop8)
- [towardsdatascience.com: GANs vs. Autoencoders](https://towardsdatascience.com/gans-vs-autoencoders-comparison-of-deep-generative-models-985cf15936ea)

For a full list of references, see the "References" section in the notebook.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Special thanks to all the contributors and Kaggle authors whose work inspired this notebook.
