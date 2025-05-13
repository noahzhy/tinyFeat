import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
from transformers import AutoImageProcessor, AutoModel


path = 'test_0.jpg'
image = Image.open(path)

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
model = AutoModel.from_pretrained('facebook/dinov2-small')

inputs = processor(images=image, return_tensors="pt", use_fast=True)
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
# 1x257x384 channel 1 class token + 256 patches (16x16)
class_token, patch_token = last_hidden_states[0][0], last_hidden_states[0][1:]
patch_tokens_reshaped = patch_token.permute(1, 0).reshape(384, 16, 16)

fg_pca = PCA(n_components=3)

reduced_patches = fg_pca.fit_transform(patch_token.detach().numpy())
norm_patches = minmax_scale(reduced_patches) # Shape (256, 3)
fg_mask_flat = (norm_patches > 0.5) # Shape (256, 3), user's original fg_mask logic

# Reshape norm_patches and the mask to image dimensions (16, 16, 3)
norm_patches_img = norm_patches.reshape(16, 16, 3)
fg_mask_img = fg_mask_flat.reshape(16, 16, 3)
only_object = np.zeros((16, 16, 3))

only_object[fg_mask_img] = norm_patches_img[fg_mask_img]

plt.figure(figsize=(5,5))
plt.imshow(only_object)
plt.axis('off')
plt.colorbar()
plt.savefig('dinov2.png', bbox_inches='tight', pad_inches=0)
# plt.show()
