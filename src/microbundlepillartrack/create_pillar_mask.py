from microbundlepillartrack import image_analysis as ia
from segment_anything import sam_model_registry
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage import morphology, exposure
from skimage.morphology import disk
from skimage.filters import rank
import matplotlib.pyplot as plt
from typing import List, Tuple
from pathlib import Path
import numpy as np
import warnings
import torch
import cv2

###########################################################################
# Threshold-based pillar segmentation
###########################################################################

def check_frame_low_contrast(img: np.ndarray) -> bool:
    """Given an image. Will check if it is a low contrast image."""
    low_contrast = exposure.is_low_contrast(img)
    return low_contrast


def check_movie_low_contrast(img_list: List) -> bool:
    """Given an image sequence as a list. Will check if the sequence is overall low contrast."""
    all_contrast = []
    num_imgs = len(img_list)
    for ii in range(len(img_list)):
        img = img_list[ii]
        contrast = check_frame_low_contrast(img)
        all_contrast.append(contrast)
    num_low_contrast = sum(all_contrast)
    if num_low_contrast/num_imgs > 0.1:
         warnings.warn('Input video is a low contrast video.'' Pillar mask segmentation is suboptimal and might fail.',category = UserWarning, stacklevel=2)


def thresh_img_local(img: np.ndarray) -> np.ndarray:
    """Given an uint16 image. Will return a binary image based on local otsu thresholding."""
    img = img.astype('uint16')
    radius = 100
    selem = disk(radius)
    img = ia.uint16_to_uint8(img)
    local_otsu = rank.otsu(img, selem)
    binary_img = img > local_otsu
    return binary_img


def label_regions(binary_img: np.ndarray) -> List:
    """Given a binary image. Will return a list of properties for each labelled region."""
    label_image = label(binary_img)
    region_props = regionprops(label_image)
    return region_props


def remove_small_large_regions(region_props, thresh_size_small: int = 1500, thresh_size_large: int = 12000) -> List:
    """Given a list of region properties. Will keep the labelled regions whose areas fall within the specified lower and upper boundaries and return them in a list."""
    region_props_new = []
    region_areas = get_regions_area(region_props)
    for ii in range(len(region_areas)):
        reg_area = region_areas[ii]
        if reg_area > thresh_size_small and reg_area < thresh_size_large:
            region_props_new.append(region_props[ii])
    return region_props_new


def get_roundest_regions(region_props: List, num_regions: int = 2) -> List:
    """Given a list of region properties. Will return a list of roundest regions of length 'num_regions'."""
    eccentricity_list = []
    for region in region_props:
        eccentricity_list.append(region.eccentricity)
    ecc_rank = np.argsort(eccentricity_list)
    num_regions = np.min([num_regions, len(eccentricity_list)])
    regions_keep = []
    for kk in range(0, num_regions):
        xx = ecc_rank[kk]
        if eccentricity_list[xx] < 0.9:
            regions_keep.append(region_props[xx])
    return regions_keep


def remove_border_regions(binary_img: np.ndarray) -> np.ndarray:
    """Given a binary image. Will remove masks touching the border."""
    cleared_binary_img = clear_border(binary_img)
    return cleared_binary_img


def get_largest_regions(region_props: List, num_regions: int = 2) -> List:
    """Given a list of region properties. Will return a list of largest regions of length 'num_regions'."""
    area_list = []
    for region in region_props:
        area_list.append(region.area_convex)
    area_rank = np.argsort(area_list)[::-1]
    num_regions = np.min([num_regions, len(area_list)])
    regions_keep = []
    for kk in range(0, num_regions):
        xx = area_rank[kk]
        regions_keep.append(region_props[xx])
    return regions_keep


def get_regions_area(regions_list: List) -> List:
    """Given a regions list. Will compute the convex area of the masks."""
    area_list = []
    for region in regions_list:
        reg_area = region.area_convex
        area_list.append(reg_area)
    return area_list


def region_to_coords(regions_list: List) -> List:
    """Given a regions list. Will return the coordinates of all regions in the list."""
    coords_list = []
    for region in regions_list:
        coords = region.coords
        coords_list.append(coords)
    return coords_list


def coords_to_mask(coords: np.ndarray, array: np.ndarray) -> np.ndarray:
    """Given coordinates and template array. Will turn coordinates into a binary mask."""
    mask = np.zeros(array.shape)
    # for coords in coords_list:
    for kk in range(0, coords.shape[0]):
        mask[coords[kk, 0], coords[kk, 1]] = 1
    return mask


def close_region(array: np.ndarray, radius: int = 5) -> np.ndarray:
    """Given an array with a small hole. Will return a closed array."""
    footprint = morphology.disk(radius, dtype=bool)
    closed_array = morphology.binary_closing(array, footprint)
    return closed_array


def dilate_mask(mask: np.ndarray, kernel_size: int = 5 , iter: int = 1) -> np.ndarray:
    """Given a closed mask. Will apply OpenCV's dilate morphological operator to increase 
    the area of the mask according to an elliptical kernel."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
    dilated_mask = cv2.dilate(mask.astype('uint8'), kernel, iterations=iter)
    return dilated_mask


def check_regions_to_dilate(regions_list: List, min_area: int = 2000) -> List:
    """Given a regions list. Will check if the area of each region is less than the set 'min_area' and return the region index when true."""
    reg_areas = get_regions_area(regions_list)
    reg_areas = np.array(reg_areas)
    true_idx = np.where(reg_areas < min_area) [0]
    return true_idx.astype('int')


def run_dilate_mask(mask_idx: List, mask_1: np.ndarray, mask_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Given the indices of small kept regions and the two pillar masks. Will dilate the small pillar masks if necessary."""
    for idx in mask_idx: 
        if idx == 0:
            mask_1 = dilate_mask(mask_1)
            print("Segmented `pillar_mask_1` has been dilated to increase its area. Examine results for pillar_1 with caution.")
        elif idx==1:
            mask_2 = dilate_mask(mask_2)
            print("Segmented `pillar_mask_2` has been dilated to increase its area. Examine results for pillar_1 with caution.")     
    return mask_1, mask_2


def get_mask_centroid(mask_region) -> np.ndarray:
    """Given a mask region. Will return the coordinates of the centroid."""
    r0,c0 = mask_region.centroid
    return np.array([r0,c0])


def order_pillar_masks(regions_list: List) -> List:
    """Given a list of region masks. Will order the masks from closest to farthest with respect to the x=0 axis (left edge)."""
    c_centroid_list = []
    for region in regions_list:
        _,c0 = get_mask_centroid(region)
        c_centroid_list.append(c0)
    centroid_rank = np.argsort(c_centroid_list)
    ordered_masks= []
    num_regions = len(regions_list)
    for kk in range(0, num_regions):
        xx = centroid_rank[kk]
        ordered_masks.append(regions_list[xx])
    return ordered_masks


###########################################################################
# SAM-based pillar segmentation
###########################################################################

def resize_image(image: np.ndarray, new_size:int = 1024) -> np.ndarray:
    """Given an image. Will resize image to dimensions indicated by 'new_size'."""
    resized_img = cv2.resize(image, (new_size, new_size), interpolation = cv2.INTER_NEAREST)
    return resized_img


def expand_image(image: np.ndarray) -> np.ndarray:
    """Given a grayscale image. Will expand image to a 3D image."""
    expanded_img = np.expand_dims(image, axis=2)
    expanded_img = np.repeat(expanded_img, 3, axis=2)
    return expanded_img


def find_image_mean(image: np.ndarray) -> np.ndarray:
    """Given a 3D image. Will find the pixel mean per channel."""
    pixel_mean = np.mean(image, axis=(0,1))
    return pixel_mean


def find_image_sd(image: np.ndarray) -> np.ndarray:
    """Given a 3D image. Will find the pixel standard deviation per channel."""
    pixel_std = np.std(image, axis=(0,1))
    return pixel_std


def normalize_image(image: np.ndarray, pixel_mean: np.ndarray, pixel_std: np.ndarray) -> np.ndarray:
    """Given an image, pixel mean and pixel standard deviation. Will return the normalized image."""
    norm_image = (image - pixel_mean) / pixel_std
    return norm_image.astype('float')


def ndarray_image_to_tensor(image: np.ndarray) -> torch.tensor:
    """Given a 3D image. Will return the image as a troch tensor in the permuted order (Channel,Height,Width)."""
    img_tr = np.transpose(image, (2, 0, 1))
    img_torch = torch.tensor(img_tr).float()
    return img_torch[None,:,:,:] # expand to (B,C,H,W) with B=1


def load_ft_SAM_model(checkpoint_path: Path, device: torch.device, model_type: str = "vit_b"):
    """Given path to SAM checkpoint, torch device, and SAM model type. Will load pretrained SAM."""
    sam_checkpoint = str(checkpoint_path)
    microbundle_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)  
    return microbundle_sam


def get_embeddings(mb_sam, img_torch: torch.tensor, device: torch.device) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """Given image as torch tensor. Will return image and prompt embeddings required to run SAM for pillar segmentation."""
    with torch.no_grad():   
        input_img = img_torch.to(device)
        B,_, H, W = input_img.shape
        # set the bbox as the image size for fully automatic segmentation
        box_torch = torch.from_numpy(np.array([[0,0,W,H]]*B)).float().to(device)
        # get image embeddings
        image_embedding = mb_sam.image_encoder(input_img)
        # get prompt embeddings
        sparse_embeddings, dense_embeddings = mb_sam.prompt_encoder(points=None, boxes = box_torch, masks=None)
        return image_embedding, sparse_embeddings, dense_embeddings


def get_pred_mask_prob(mb_sam, image_embedding: torch.tensor, sparse_embeddings: torch.tensor, dense_embeddings: torch.tensor) -> torch.tensor:
    """Given a pretrained SAM, and image, sparse and dense embeddings. Will return predicted mask probability tensor. """
    mask_prediction_prob, _ = mb_sam.mask_decoder(
    image_embeddings=image_embedding, # (B, 256, 64, 64)
    image_pe=mb_sam.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
    sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
    dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
    multimask_output=False)
    return mask_prediction_prob


def resize_pred_mask_prob(mask_prob: torch.tensor, orig_img_size: np.ndarray) -> torch.tensor:
    """Given a predicted mask probability tensor. Will resize to the original input image size."""
    res_mask_prob = torch.nn.functional.interpolate(mask_prob, size=(orig_img_size[0], orig_img_size[1]), mode='bilinear', align_corners=False)  
    return res_mask_prob


def mask_prob_to_binary(mask_prob: torch.tensor) -> np.ndarray:
    """Given a predicted mask probability tensor. Will return a binary mask."""
    pred_mask_prob = torch.sigmoid(mask_prob)
    # convert soft mask to hard mask
    pred_mask_prob = pred_mask_prob.cpu().detach().numpy().squeeze()
    binary_pred_mask = (pred_mask_prob > 0.5).astype(np.uint8)
    return binary_pred_mask


def segment_microbundle_pillars_SAM(img: np.ndarray, checkpoint_path: Path, device: torch.device):
    """Given a grayscale image, SAM checkpoint, and device. Will return a binary mask of the microbundle pillars."""
    orig_img_shape = img.shape
    resized_img = resize_image(img)
    expanded_img = expand_image(resized_img)
    img_pix_avg = find_image_mean(expanded_img)
    img_pix_sd = find_image_sd(expanded_img)
    norm_img = normalize_image(expanded_img,img_pix_avg, img_pix_sd)
    img_torch = ndarray_image_to_tensor(norm_img)
    microbundle_SAM = load_ft_SAM_model(checkpoint_path, device)
    assert img_torch.shape == (1, 3, microbundle_SAM.image_encoder.img_size, microbundle_SAM.image_encoder.img_size), 'input image should be resized to 1024*1024'
    img_embd, sparse_embd, dense_embd = get_embeddings(microbundle_SAM, img_torch, device)
    pred_mask_p = get_pred_mask_prob(microbundle_SAM, img_embd, sparse_embd, dense_embd)    
    resized_pred_mask_p = resize_pred_mask_prob(pred_mask_p,orig_img_shape)
    binary_mask = mask_prob_to_binary(resized_pred_mask_p)
    return binary_mask


def run_microbundle_SAM(img: np.ndarray, checkpoint_path: Path, gpu: int = 0) -> np.ndarray:
    """Given a grayscale image, SAM checkpoint, device and gpu. Will return a binary mask of the microbundle pillars."""
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    pillar_masks = segment_microbundle_pillars_SAM(img, checkpoint_path, device)
    return pillar_masks

###########################################################################
# Run pillar segmentation
###########################################################################

def create_pillar_masks_type1(img: np.ndarray, checkpoint_path_type_1: Path, gpu: int = 0) -> Tuple[np.ndarray,np.ndarray]:
    """Given a grayscale image of type1. Will segment and save the 2 pillar masks."""
    binary_img = thresh_img_local(img)
    cleared_binary_img = remove_border_regions(binary_img)
    region_props = label_regions(cleared_binary_img)
    region_props = remove_small_large_regions(region_props)
    regions_keep = get_roundest_regions(region_props)
    if len(regions_keep) < 2:
        warnings.warn('Masks for both pillars could not be automatically segmented via thresholding.'' Implementing microbundle_SAM for pillar segmentation instead.',category = UserWarning, stacklevel=2)
        SAM_binary_img = run_microbundle_SAM(img, checkpoint_path_type_1, gpu)
        SAM_region_props = label_regions(SAM_binary_img)
        SAM_regions_keep = remove_small_large_regions(SAM_region_props)
        if len(SAM_regions_keep) < 2:
            raise IndexError("Could not segment masks for both pillars automatically."" Aborting pillar tracking.")
        else:
            regions_keep = SAM_regions_keep
    ordered_regions_keep = order_pillar_masks(regions_keep)
    small_region_idx = check_regions_to_dilate(ordered_regions_keep)
    regions_keep_coords = region_to_coords(ordered_regions_keep)
    # make two masks, one for each pillar
    mask_1 = coords_to_mask(regions_keep_coords[0], img)
    closed_mask_1 = close_region(mask_1)
    mask_2 = coords_to_mask(regions_keep_coords[1], img)
    closed_mask_2 = close_region(mask_2)
    closed_mask_1, closed_mask_2 = run_dilate_mask(small_region_idx, closed_mask_1, closed_mask_2)
    return closed_mask_1, closed_mask_2


def create_pillar_masks_type2(img: np.ndarray, checkpoint_path_type_2: Path, gpu: int = 0) -> Tuple[np.ndarray,np.ndarray]:
    """Given a grayscale image of type2. Will segment and save the 2 pillar masks."""
    SAM_binary_img = run_microbundle_SAM(img, checkpoint_path_type_2, gpu)
    SAM_region_props = label_regions(SAM_binary_img)
    SAM_regions_keep = remove_small_large_regions(SAM_region_props, thresh_size_small = 7000, thresh_size_large = 20000)
    if len(SAM_regions_keep) < 2:
        raise IndexError("Could not segment masks for both pillars automatically."" Aborting pillar tracking.")
    else:
        regions_keep = SAM_regions_keep
    ordered_regions_keep = order_pillar_masks(regions_keep)
    regions_keep_coords = region_to_coords(ordered_regions_keep)
    # make two masks, one for each pillar
    mask_1 = coords_to_mask(regions_keep_coords[0], img)
    closed_mask_1 = close_region(mask_1, radius= 10)
    mask_2 = coords_to_mask(regions_keep_coords[1], img)
    closed_mask_2 = close_region(mask_2, radius= 10)
    return closed_mask_1, closed_mask_2


def save_mask(folder_path: Path, mask_1: np.ndarray, mask_2: np.ndarray, fname: str = "pillar_mask") -> Tuple[Path,Path,Path,Path]:
    """Given a folder path and 2 pillar masks. Will save the files."""
    new_path = ia.create_folder(folder_path, "masks")
    file_path_1 = new_path.joinpath(fname + "_1.txt").resolve()
    np.savetxt(str(file_path_1), mask_1, fmt="%i")
    img_path_1 = new_path.joinpath(fname + "_1.png").resolve()
    plt.imsave(img_path_1, mask_1)
    file_path_2 = new_path.joinpath(fname + "_2.txt").resolve()
    np.savetxt(str(file_path_2), mask_2, fmt="%i")
    img_path_2 = new_path.joinpath(fname + "_2.png").resolve()
    plt.imsave(img_path_2, mask_2)
    return file_path_1, img_path_1, file_path_2, img_path_2


def run_create_pillar_mask(folder_path: Path, checkpoint_path: Path = Path("../src/microbundlepillartrack"), microbundle_type: str = "type1", fname: str = "pillar_mask", frame_num: int = 0) -> Tuple[Path,Path,Path,Path,Path]:
    """Given a folder and mask segmentation settings. Will segment and save the pillar masks."""
    # load the first image
    movie_folder_path = folder_path.joinpath("movie").resolve()
    name_list_path = ia.image_folder_to_path_list(movie_folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    # check if low contrast video 
    check_movie_low_contrast(tiff_list)
    img = tiff_list[frame_num]
    # create the pillar masks
    if microbundle_type == "type1":
        checkpoint_path_type_1 = checkpoint_path.joinpath('microbundle_SAM_Type1_pillars.pth')
        mask_1, mask_2 = create_pillar_masks_type1(img, checkpoint_path_type_1, gpu = 0)
        # save the pillar masks
        file_path_m1, img_path_m1, file_path_m2, img_path_m2 = save_mask(folder_path, mask_1, mask_2, fname)
        # plot and save a figure overlay of frame and masks
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.imshow(mask_1+mask_2, alpha=0.3)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(str(folder_path.joinpath('masks').resolve()) + '/pillar_masks_overlay.png', dpi=300)
    elif microbundle_type == "type2":
        checkpoint_path_type_2 = checkpoint_path.joinpath('microbundle_SAM_Type2_pillars.pth')
        mask_1, mask_2 = create_pillar_masks_type2(img, checkpoint_path_type_2, gpu = 0)
        # save the pillar masks
        file_path_m1, img_path_m1, file_path_m2, img_path_m2 = save_mask(folder_path, mask_1, mask_2, fname)
        # plot and save a figure overlay of frame and masks
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.imshow(mask_1+mask_2, alpha=0.3)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(str(folder_path.joinpath('masks').resolve()) + '/pillar_masks_overlay.png', dpi=300)
    else:
        warnings.warn("Input 'microbundle_type' should be specified as 'type1' or 'type2'.", category = UserWarning, stacklevel=2)
        file_path_m1, img_path_m1, file_path_m2, img_path_m2 = None, None, None, None
    return file_path_m1, img_path_m1, file_path_m2, img_path_m2