import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import monai
from segment_anything import sam_model_registry
from pathlib import Path
import tifffile
import imageio
import cv2
import glob

# set seed
torch.manual_seed(2023)


def resize_expand_sequence(img_array, size):
  resized_img_list = []
  expanded_img_list = []
  for jj in range(len(img_array)):
    img = img_array[jj]
    resized_img = cv2.resize(img, (size, size), interpolation = cv2.INTER_NEAREST)
    resized_img_list.append(resized_img)
    expand_img = np.expand_dims(resized_img, axis=2)
    expand_img = np.repeat(expand_img, 3, axis=2)
    expanded_img_list.append(expand_img)
  return np.array(expanded_img_list)


def resize_sequence(img_array, size):
  resized_img_list = []
  for jj in range(len(img_array)):
    img = img_array[jj]
    resized_img = cv2.resize(img, (size, size), interpolation = cv2.INTER_NEAREST)
    resized_img_list.append(resized_img)
  return np.array(resized_img_list)

results_path = Path('/PROVIDE_PATH_TO_SAVE_RESULTS')

"""load input images and corresponding masks"""
all_dataset_paths = sorted(glob.glob("/PATH_TO_DATASET_EXAMPLE/Example*"+".tif"))
all_mask_paths = sorted(glob.glob("/PATH_TO_DATASET_EXAMPLE/mask_Example*"+".tif"))

all_resized_images = []
all_resized_masks = []

new_size = 1024
for jj in range(len(all_dataset_paths)):
    img_seq = tifffile.imread(all_dataset_paths[jj])
    img_seq = img_seq.astype(np.uint16)
    img_seq_r = resize_expand_sequence(img_seq,new_size)
    all_resized_images.append(img_seq_r)
    mask_seq = np.array(imageio.mimread(all_mask_paths[jj],memtest=False))
    mask_seq = mask_seq.astype(np.uint8)
    mask_seq_r = resize_sequence(mask_seq,new_size)
    all_resized_masks.append(mask_seq_r)

shuffler = np.random.permutation(len(all_resized_images))
all_resized_images_shuffled = [all_resized_images[ss] for ss in shuffler]
all_resized_masks_shuffled = [all_resized_masks[ss] for ss in shuffler]

images = np.vstack(all_resized_images_shuffled)
masks = np.vstack(all_resized_masks_shuffled)

# Filter the image and mask arrays to keep only the non-empty pairs
valid_indices = [i for i, mask in enumerate(masks) if mask.max() != 0]
filtered_images = images[valid_indices]
filtered_masks = masks[valid_indices]

# sanity check 
rng_c = np.random.default_rng(42)
fig, axes = plt.subplots(1, 2, figsize=(10, 10))

rand_idx = rng_c.integers(0, len(filtered_images))

# Plot the image
axes[0].imshow(filtered_images[rand_idx][:,:,0], cmap='gray')
axes[0].set_title("Image %i"%(rand_idx))

# Plot the ground truth mask
axes[1].imshow(filtered_masks[rand_idx], cmap='gray')
axes[1].set_title("GT Mask %i"%(rand_idx))

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])      
plt.savefig(str(results_path)+'/sanity_check_test.png', dpi=300)
plt.close()

class SAMDataset(Dataset):
    def __init__(self, input_images, input_masks):
        self.input_images = input_images
        self.input_masks = input_masks
    def __len__(self):
        return len(self.input_images)
    def __getitem__(self, idx):
        # Normalize image
        pixel_mean = np.mean(self.input_images[idx], axis=(0,1))
        pixel_std = np.std(self.input_images[idx], axis=(0,1))
        img = (self.input_images[idx] - pixel_mean) / pixel_std
        img = img.astype('float')
        
        img = np.transpose( img, (2, 0, 1))
        mask = self.input_masks[idx]
        mask_2D = mask.astype('uint8')
        assert np.max(mask_2D) == 1 and np.min(mask_2D) == 0.0, "ground truth should be 0, 1"
        return torch.tensor(img).float(), torch.tensor(mask_2D[None, :,:]).long()
    

model_type = 'vit_b'
sam_checkpoint = '/PATH_TO_CHECKPOINT/sam_vit_b_01ec64.pth'
gpu = 0
device = torch.device(gpu if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(gpu)
sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
sam_model.train()
# Set up the optimizer, hyperparameter tuning will improve performance here
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')


#%% train
num_epochs = 250
losses = []
best_loss = 1e10

val_split = 0.2
train_split = 1-val_split

train_dataset = SAMDataset(filtered_images[0:int(train_split*images.shape[0])],filtered_masks[0:int(train_split*masks.shape[0])])
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Validation dataset
val_dataset = SAMDataset(filtered_images[int(train_split*images.shape[0]):],filtered_masks[int(train_split*images.shape[0]):])
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
val_losses = []

for epoch in range(num_epochs):
    epoch_loss = 0
    # train
    for step, (image, gt2D) in enumerate(tqdm(train_dataloader)):
        # do not compute gradients for image encoder and prompt encoder
        with torch.no_grad():
            image = image.to(device)
            gt2D = gt2D.to(device)
            # resize image to 1024 by 1024
            image = TF.resize(image, (1024, 1024), antialias=True)

            B,_, H, W = gt2D.shape
            # set the bbox as the image size for fully automatic segmentation
            box_torch = torch.from_numpy(np.array([[0,0,W,H]]*B)).float().to(device)
            # perform random rotation for both input image and mask tl diversify the dataset
            rot_trans = T.RandomRotation(degrees=360)
            state = torch.get_rng_state()
            rot_image = rot_trans(image)
            torch.set_rng_state(state)
            rot_gt2D = rot_trans(gt2D)

            image_embedding = sam_model.image_encoder(rot_image)
    
            # get prompt embeddings
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes = box_torch[:, None, :],
                masks=None,
            )
            
        # predicted masks
        mask_predictions, _ = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          )

        mask_predictions = nn.functional.interpolate(mask_predictions,
                size=(new_size, new_size),
                mode='bilinear',
                align_corners=False)    

        loss = seg_loss(mask_predictions, rot_gt2D.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    pred_mask_prob = torch.sigmoid(mask_predictions)
    # convert soft mask to hard mask
    pred_mask_prob = pred_mask_prob.cpu().detach().numpy().squeeze()

    pred_mask = (pred_mask_prob > 0.5).astype(np.uint8)

    orig_img = np.transpose(rot_image.cpu(), (2, 3, 1, 0))
    gt_mask = np.transpose(rot_gt2D.cpu(), (2, 3, 1, 0))

    if epoch%5 == 0:
        # plot ground truth vs predicted mask
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # Plot the image
        axes[0].imshow(np.array(orig_img[:,:,0,0]), cmap='gray') 
        axes[0].set_title("Image")

        # Plot the ground truth mask
        axes[1].imshow(gt_mask[:,:,0,0], cmap='gray') 
        axes[1].set_title("GT Mask")

        # Plot the predicted mask
        axes[2].imshow(pred_mask)  
        axes[2].set_title("Predicted Mask")

        # Hide axis ticks and labels
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])      
        plt.savefig(str(results_path)+'/gt_vs_pred_epoch%i_test.png'%(epoch), dpi=300)
        plt.close()

    epoch_loss /= step
    losses.append(epoch_loss)

    # save the latest model checkpoint
    torch.save(sam_model.state_dict(), str(results_path)+'/microbundle_sam_model_latest_test.pth')
    # save the best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(sam_model.state_dict(),str(results_path)+'/microbundle_sam_model_best_test.pth')


    # validate
    epoch_val_loss = 0
    for step, (val_image, val_gt2D) in enumerate(tqdm(val_dataloader)):
        # do not compute gradients for image encoder and prompt encoder
        with torch.no_grad():
            val_image = val_image.to(device)
            val_gt2D = val_gt2D.to(device)
            # resize image to 1024 by 1024
            val_image = TF.resize(val_image, (1024, 1024), antialias=True)

            B,_, H, W = val_gt2D.shape
            # set the bbox as the image size for fully automatic segmentation
            val_box_torch = torch.from_numpy(np.array([[0,0,W,H]]*B)).float().to(device)

            rot_trans = T.RandomRotation(degrees=360)
            state = torch.get_rng_state()
            rot_val_image = rot_trans(val_image)
            torch.set_rng_state(state)
            rot_val_gt2D = rot_trans(val_gt2D)

            val_image_embedding = sam_model.image_encoder(rot_val_image)
            
            # get prompt embeddings
            val_sparse_embeddings, val_dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes = val_box_torch[:, None, :],
                masks=None,
            )
            
        # predicted masks
        val_mask_predictions, _ = sam_model.mask_decoder(
            image_embeddings=val_image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=val_sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=val_dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          )

        val_mask_predictions = nn.functional.interpolate(val_mask_predictions,
                size=(new_size, new_size),
                mode='bilinear',
                align_corners=False)    

        val_loss = seg_loss(val_mask_predictions, rot_val_gt2D.to(device))
        epoch_val_loss += val_loss.item()
        
    val_pred_mask_prob = torch.sigmoid(val_mask_predictions)
    # convert soft mask to hard mask
    val_pred_mask_prob = val_pred_mask_prob.cpu().detach().numpy().squeeze()

    val_pred_mask = (val_pred_mask_prob > 0.5).astype(np.uint8)

    val_orig_img = np.transpose(rot_val_image.cpu(), (2, 3, 1, 0))
    val_gt_mask = np.transpose(rot_val_gt2D.cpu(), (2, 3, 1, 0))

    if epoch%5 == 0:
        # plot ground truth vs predicted mask
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # Plot the image
        axes[0].imshow(np.array(val_orig_img[:,:,0,0]), cmap='gray') 
        axes[0].set_title("Val Image")

        # Plot the ground truth mask
        axes[1].imshow(val_gt_mask[:,:,0,0], cmap='gray') 
        axes[1].set_title("Val GT Mask")

        # Plot the predicted mask
        axes[2].imshow(val_pred_mask)
        axes[2].set_title("Val Predicted Mask")

        # Hide axis ticks and labels
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])      
        plt.savefig(str(results_path)+'/validation_gt_vs_pred_epoch%i_Type2_pillars_extended.png'%(epoch), dpi=300)
        plt.close()        
    
    epoch_val_loss /= step
    val_losses.append(epoch_val_loss)
   
    print(f'EPOCH: {epoch}, Train_Loss: {epoch_loss}, Val_Loss: {epoch_val_loss}')

# save losses into a text file 
np.savetxt(str(results_path)+'/train_loss_test.txt',np.asarray(losses))
np.savetxt(str(results_path)+'/validate_loss_test.txt',np.asarray(val_losses))

# plot loss
plt.figure()
plt.plot(losses, label='train')
plt.plot(val_losses, label='validate')
plt.title('Dice + Cross Entropy Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(str(results_path)+'/train_validate_loss_test.png', dpi=300)
plt.close()

