from skimage.metrics import structural_similarity as ssim

# computes Structural Similarity Index of two images
def compute_ssim(img1, img2):
    # Assumes shape (C, H, W); convert to (H, W) if grayscale
    img1 = img1.squeeze().cpu().detach().numpy()
    img2 = img2.squeeze().cpu().detach().numpy()
    return ssim(img1, img2, data_range=1.0)