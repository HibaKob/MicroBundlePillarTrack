import glob
from microbundlepillartrack import create_pillar_mask as cpm
import numpy as np
from pathlib import Path
from scipy import ndimage
from skimage import io
import warnings
import torch
import pytest
from skimage import morphology
from skimage.morphology import footprints



def files_path():
    self_path_file = Path(__file__)
    self_path = self_path_file.resolve().parent
    data_path = self_path.joinpath("files").resolve()
    return data_path


def example_path(example_name):
    data_path = files_path()
    example_path = data_path.joinpath(example_name).resolve()
    return example_path


def glob_movie(example_name):
    folder_path = example_path(example_name)
    movie_path = folder_path.joinpath("movie").resolve()
    name_list = glob.glob(str(movie_path) + '/*.TIF')
    name_list.sort()
    name_list_path = []
    for name in name_list:
        name_list_path.append(Path(name))
    return name_list


def test_check_frame_low_contrast():
    img_low = np.random.rand(250,250)*0.1
    img_normal = np.random.rand(250,250)*250
    low_contrast = cpm.check_frame_low_contrast(img_low)
    nromal_contrast = cpm.check_frame_low_contrast(img_normal)
    assert low_contrast == True
    assert nromal_contrast == False


def test_check_movie_low_contrast():
    img_sequence = []
    for ii in range(250):
        img_low = np.random.rand(250,250)*0.1
        img_sequence.append(img_low)
    with warnings.catch_warnings(record=True) as record:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        cpm.check_movie_low_contrast(img_sequence)
    assert len(record) == 1
                       
                       
def test_thresh_img_local():
    rad_1 = 25
    disk_1 = morphology.disk(rad_1, dtype=bool)
    dim = 100
    array = np.zeros((dim, dim))
    array[0:disk_1.shape[0], 0:disk_1.shape[1]] = disk_1
    thresh_array = cpm.thresh_img_local(array)
    assert np.sum(thresh_array) == np.sum(array)


def test_label_regions():
    rad_1 = 25
    disk_1 = morphology.disk(rad_1, dtype=bool)
    rad_2 = 10
    disk_2 = morphology.disk(rad_2, dtype=bool)
    dim = 100
    array = np.zeros((dim, dim))
    array[0:disk_1.shape[0], 0:disk_1.shape[1]] = disk_1
    array[-disk_2.shape[0]:, -disk_2.shape[1]:] = disk_2
    thresh_array = cpm.thresh_img_local(array)
    region_props = cpm.label_regions(thresh_array)
    assert len(region_props) == 2


def test_remove_small_large_regions():
    rad_1 = 25
    disk_1 = morphology.disk(rad_1, dtype=bool)
    rad_2 = 2
    disk_2 = morphology.disk(rad_2, dtype=bool)
    rad_3 = 15
    disk_3 = morphology.disk(rad_3, dtype=bool)
    dim = 100
    array = np.zeros((dim, dim))
    array[0:disk_1.shape[0], 0:disk_1.shape[1]] = disk_1
    array[-disk_2.shape[0]:, -disk_2.shape[1]:] = disk_2
    array[0:disk_3.shape[0], -disk_3.shape[1]:] = disk_3
    thresh_array = cpm.thresh_img_local(array)
    region_props = cpm.label_regions(thresh_array)
    kept_region_props = cpm.remove_small_large_regions(region_props, thresh_size_small = 500, thresh_size_large = 15000)
    assert len(kept_region_props) == 2


def test_get_roundest_regions():
    rad_1 = 25
    disk_1 = morphology.disk(rad_1, dtype=bool)
    width_1 = 30
    height_1 = 17
    ellipse_1 = footprints.ellipse(width_1, height_1, dtype=bool)
    width_2 = 40
    height_2 = 35
    ellipse_2 = footprints.ellipse(width_2, height_2, dtype=bool)
    width_3 = 30
    height_3 = 40
    ellipse_3 = footprints.ellipse(width_3, height_3, dtype=bool)
    dim = 200
    array = np.zeros((dim, dim))
    array[0:disk_1.shape[0], 0:disk_1.shape[1]] = disk_1
    array[0:ellipse_1.shape[0], -ellipse_1.shape[1]:] = ellipse_1
    array[-ellipse_2.shape[0]:, -ellipse_2.shape[1]:] = ellipse_2
    array[int(dim/2 - ellipse_3.shape[0]/2):int(dim/2 + ellipse_3.shape[0]/2), int(dim/2 - ellipse_3.shape[1]/2):int(dim/2 + ellipse_3.shape[1]/2)] = ellipse_3
    thresh_array = cpm.thresh_img_local(array)
    region_props = cpm.label_regions(thresh_array)
    kept_roundest_regions = cpm.get_roundest_regions(region_props, num_regions = 2)
    assert len(kept_roundest_regions) == 2


def test_remove_border_regions():
    dim = 100
    array = np.zeros((dim, dim))
    array [10:60,0:6] = 1
    array [90:,20:30] = 1
    array [30:70,30:60] = 1
    cleared_array = cpm.remove_border_regions(array)
    assert np.sum(cleared_array[:,0:10]) == 0
    assert np.sum(cleared_array[:,90:]) == 0
    assert np.sum(cleared_array[0:10,:]) == 0
    assert np.sum(cleared_array[90:,:]) == 0


def test_get_largest_regions():
    rad_1 = 25
    disk_1 = morphology.disk(rad_1, dtype=bool)
    rad_2 = 10
    disk_2 = morphology.disk(rad_2, dtype=bool)
    rad_3 = 5
    disk_3 = morphology.disk(rad_3, dtype=bool)
    dim = 100
    array = np.zeros((dim, dim))
    array[0:disk_1.shape[0], 0:disk_1.shape[1]] = disk_1
    array[-disk_2.shape[0]:, -disk_2.shape[1]:] = disk_2
    array[80:80+disk_3.shape[0]:, 0:disk_3.shape[1]] = disk_3
    thresh_array = cpm.thresh_img_local(array)
    region_props = cpm.label_regions(thresh_array)
    regions_keep = cpm.get_largest_regions(region_props, num_regions = 2)
    assert len(regions_keep) == 2


def test_get_regions_area():
    rad_1 = 25
    disk_1 = morphology.disk(rad_1, dtype=bool)
    rad_2 = 10
    disk_2 = morphology.disk(rad_2, dtype=bool)
    dim = 100
    array = np.zeros((dim, dim))
    array[0:disk_1.shape[0], 0:disk_1.shape[1]] = disk_1
    array[-disk_2.shape[0]:, -disk_2.shape[1]:] = disk_2
    thresh_array = cpm.thresh_img_local(array)
    region_props = cpm.label_regions(thresh_array)
    area_list = []
    for region in region_props:
        reg_area = region.area_convex
        area_list.append(reg_area)
    area_list_cpm = cpm.get_regions_area(region_props)
    diff_area = np.subtract(area_list,area_list_cpm)
    assert np.all(diff_area) == 0


def test_region_to_coords():
    rad_1 = 25
    disk_1 = morphology.disk(rad_1, dtype=bool)
    rad_2 = 10
    disk_2 = morphology.disk(rad_2, dtype=bool)
    dim = 100
    array = np.zeros((dim, dim))
    array[0:disk_1.shape[0], 0:disk_1.shape[1]] = disk_1
    array[-disk_2.shape[0]:, -disk_2.shape[1]:] = disk_2
    thresh_array = cpm.thresh_img_local(array)
    region_props = cpm.label_regions(thresh_array)
    coords_list = cpm.region_to_coords(region_props)
    assert len(coords_list) == 2


def test_coords_to_mask():
    rad_1 = 25
    disk_1 = morphology.disk(rad_1, dtype=bool)
    dim = 100
    array = np.zeros((dim, dim))
    array[0:disk_1.shape[0], 0:disk_1.shape[1]] = disk_1
    thresh_array = cpm.thresh_img_local(array)
    region_props = cpm.label_regions(thresh_array)
    coords_list = cpm.region_to_coords(region_props)
    mask = cpm.coords_to_mask(coords_list[0],array)
    assert np.sum(mask) == np.sum(thresh_array)


def test_close_region():
    val = 10
    array = np.zeros((val, val))
    array[3:7, 3:7] = 1
    array_missing = np.copy(array)
    array_missing[5, 5] = 0
    array_closed = cpm.close_region(array_missing, radius=2)
    assert np.allclose(array_closed, array)


def test_dilate_mask():
    rad_1 = 25
    disk_1 = morphology.disk(rad_1, dtype=bool)
    dim = 100
    array = np.zeros((dim, dim))
    array[0:disk_1.shape[0], 0:disk_1.shape[1]] = disk_1
    thresh_array = cpm.thresh_img_local(array)
    region_props = cpm.label_regions(thresh_array)
    coords_list = cpm.region_to_coords(region_props)
    mask = cpm.coords_to_mask(coords_list[0],array)
    dilated_mask = cpm.dilate_mask(mask)
    assert np.sum(dilated_mask) > np.sum(mask)


def test_check_regions_to_dilate():
    rad_1 = 10
    disk_1 = morphology.disk(rad_1, dtype=bool)
    rad_2 = 30
    disk_2 = morphology.disk(rad_2, dtype=bool)
    dim = 100
    array = np.zeros((dim, dim))
    array[0:disk_1.shape[0], 0:disk_1.shape[1]] = disk_1
    array[-disk_2.shape[0]:, -disk_2.shape[1]:] = disk_2
    thresh_array = cpm.thresh_img_local(array)
    region_props = cpm.label_regions(thresh_array)
    idx_small = cpm.check_regions_to_dilate(region_props, min_area = 2000)
    assert idx_small == 0


def test_run_dilate_mask_1():
    rad_1 = 10
    disk_1 = morphology.disk(rad_1, dtype=bool)
    rad_2 = 5
    disk_2 = morphology.disk(rad_2, dtype=bool)
    dim = 100
    array = np.zeros((dim, dim))
    array[0:disk_1.shape[0], 0:disk_1.shape[1]] = disk_1
    array[-disk_2.shape[0]:, -disk_2.shape[1]:] = disk_2
    thresh_array = cpm.thresh_img_local(array)
    region_props = cpm.label_regions(thresh_array)
    idx_small = cpm.check_regions_to_dilate(region_props, min_area = 2000)
    coords_list = cpm.region_to_coords(region_props)
    masks_o_1 = cpm.coords_to_mask(coords_list[0],array)
    masks_o_2 = cpm.coords_to_mask(coords_list[1],array)
    masks_o = [masks_o_1,masks_o_2]
    mask_1, mask_2 = cpm.run_dilate_mask(idx_small,masks_o_1,masks_o_2)
    masks = [mask_1,mask_2]
    for idx in idx_small:
        if len(idx_small) == 1:
            assert np.sum(masks[idx]) > np.sum(masks_o[idx])
            assert np.sum(masks[1-idx]) == np.sum(masks_o[1-idx])
        else:
            assert np.sum(masks[idx]) > np.sum(masks_o[idx])


def test_run_dilate_mask_2():
    rad_1 = 10
    disk_1 = morphology.disk(rad_1, dtype=bool)
    rad_2 = 30
    disk_2 = morphology.disk(rad_2, dtype=bool)
    dim = 100
    array = np.zeros((dim, dim))
    array[0:disk_1.shape[0], 0:disk_1.shape[1]] = disk_1
    array[-disk_2.shape[0]:, -disk_2.shape[1]:] = disk_2
    thresh_array = cpm.thresh_img_local(array)
    region_props = cpm.label_regions(thresh_array)
    idx_small = cpm.check_regions_to_dilate(region_props, min_area = 2000)
    coords_list = cpm.region_to_coords(region_props)
    masks_o_1 = cpm.coords_to_mask(coords_list[0],array)
    masks_o_2 = cpm.coords_to_mask(coords_list[1],array)
    masks_o = [masks_o_1,masks_o_2]
    mask_1, mask_2 = cpm.run_dilate_mask(idx_small,masks_o[0],masks_o[1])
    masks = [mask_1,mask_2]
    for idx in idx_small:
        if len(idx_small) == 1:
            assert np.sum(masks[idx]) > np.sum(masks_o[idx])
            assert np.sum(masks[1-idx]) == np.sum(masks_o[1-idx])
        else:
            assert np.sum(masks[idx]) > np.sum(masks_o[idx])


def test_get_mask_centroid():
    rad_1 = 20
    disk_1 = morphology.disk(rad_1, dtype=bool)
    dim = 100
    array = np.zeros((dim, dim))
    array[int(dim/2 - disk_1.shape[0]/2):int(dim/2 + disk_1.shape[0]/2), int(dim/2 - disk_1.shape[1]/2):int(dim/2 + disk_1.shape[1]/2)] = disk_1
    thresh_array = cpm.thresh_img_local(array)
    thresh_array = cpm.thresh_img_local(array)
    region_props = cpm.label_regions(thresh_array)
    mask = region_props[0]
    r0,c0 = cpm.get_mask_centroid(mask)
    assert r0 == dim/2-1
    assert c0 == dim/2-1


def test_order_pillar_masks():
    rad_1 = 25
    disk_1 = morphology.disk(rad_1, dtype=bool)
    rad_2 = 5
    disk_2 = morphology.disk(rad_2, dtype=bool)
    rad_3 = 15
    disk_3 = morphology.disk(rad_3, dtype=bool)
    dim = 100
    array = np.zeros((dim, dim))
    array[0:disk_1.shape[0], 0:disk_1.shape[1]] = disk_1
    array[-disk_2.shape[0]:, -disk_2.shape[1]:] = disk_2
    array[0:disk_3.shape[0], -disk_3.shape[1]:] = disk_3
    thresh_array = cpm.thresh_img_local(array)
    masks = cpm.label_regions(thresh_array)
    ordered_masks = cpm.order_pillar_masks(masks)
    assert ordered_masks[0].area == np.sum(disk_1)
    assert ordered_masks[1].area == np.sum(disk_3)
    assert ordered_masks[2].area == np.sum(disk_2)


def test_resize_image():
    img = np.zeros((250,250))
    img_shape = img.shape
    new_size = 1024
    resized_img = cpm.resize_image(img, new_size)
    resized_img_shape = resized_img.shape
    assert resized_img_shape > img_shape
    assert resized_img_shape == (new_size,new_size)


def test_expand_image():
    img = np.zeros((250,250))
    expanded_img = cpm.expand_image(img)
    assert expanded_img.shape[2] == 3


def test_find_image_mean():
    mean = 100
    stdv = 5 
    size = 250
    dist = np.random.normal(mean, stdv, size**2)
    img = dist.reshape(size,-1)
    expanded_img = cpm.expand_image(img)
    pixel_mean = cpm.find_image_mean(expanded_img)
    assert np.allclose(pixel_mean, mean, atol=0.5)


def test_find_image_sd():
    mean = 100
    stdv = 5 
    size = 250
    dist = np.random.normal(mean, stdv, size**2)
    img = dist.reshape(size,-1)
    expanded_img = cpm.expand_image(img)
    pixel_std = cpm.find_image_sd(expanded_img)
    assert np.allclose(pixel_std, stdv, atol=0.1)


def test_normalize_image():
    mean = 100
    stdv = 5 
    size = 250
    dist = np.random.normal(mean, stdv, size**2)
    img = dist.reshape(size,-1)
    expanded_img = cpm.expand_image(img)
    norm_image = cpm.normalize_image(expanded_img, mean, stdv)
    pixel_mean = cpm.find_image_mean(norm_image)
    pixel_std = cpm.find_image_sd(norm_image)
    assert np.allclose(pixel_mean, 0, atol=0.02)
    assert np.allclose(pixel_std, 1, atol=0.02)


def test_ndarray_image_to_tensor():
    img = np.zeros((250,250))
    expanded_img = cpm.expand_image(img)
    img_torch = cpm.ndarray_image_to_tensor(expanded_img)
    assert torch.is_tensor(img_torch)


def test_load_ft_SAM_model():
    src_path = Path('./src/microbundlepillartrack')
    checkpoint_path = src_path.joinpath('microbundle_SAM_Type2_pillars.pth').resolve()
    gpu = 0
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    model_type = "vit_b"
    microbundle_sam = cpm.load_ft_SAM_model(checkpoint_path, device, model_type=model_type)
    assert microbundle_sam is not None


def test_get_embeddings():
    src_path = Path('./src/microbundlepillartrack')
    checkpoint_path = src_path.joinpath('microbundle_SAM_Type2_pillars.pth').resolve()
    gpu = 0
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    model_type = "vit_b"
    mean = 100
    stdv = 5 
    size = 250
    new_size = 1024
    dist = np.random.normal(mean, stdv, size**2)
    img = dist.reshape(size,-1)
    resized_img = cpm.resize_image(img, new_size)
    expanded_img = cpm.expand_image(resized_img)
    norm_image = cpm.normalize_image(expanded_img, mean, stdv)
    img_torch = cpm.ndarray_image_to_tensor(norm_image)
    microbundle_sam = cpm.load_ft_SAM_model(checkpoint_path, device, model_type=model_type)
    image_embedding, sparse_embeddings, dense_embeddings = cpm.get_embeddings(microbundle_sam, img_torch, device)
    B,_, H, W = img_torch.shape
    box_torch = torch.from_numpy(np.array([[0,0,W,H]]*B)).float().to(device)
    assert torch.is_tensor(image_embedding)
    assert torch.is_tensor(sparse_embeddings)
    assert torch.is_tensor(dense_embeddings)

    assert img_torch.shape == (1, 3, microbundle_sam.image_encoder.img_size, microbundle_sam.image_encoder.img_size)
    assert image_embedding.shape == microbundle_sam.image_encoder(img_torch).shape
    assert sparse_embeddings.shape == microbundle_sam.prompt_encoder(points=None, boxes = box_torch, masks=None)[0].shape
    assert dense_embeddings.shape == microbundle_sam.prompt_encoder(points=None, boxes = box_torch, masks=None)[1].shape


def test_get_pred_mask_prob():
    src_path = Path('./src/microbundlepillartrack')
    checkpoint_path = src_path.joinpath('microbundle_SAM_Type2_pillars.pth').resolve()
    gpu = 0
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    model_type = "vit_b"
    mean = 100
    stdv = 5 
    size = 250
    new_size = 1024
    dist = np.random.normal(mean, stdv, size**2)
    img = dist.reshape(size,-1)
    resized_img = cpm.resize_image(img, new_size)
    expanded_img = cpm.expand_image(resized_img)
    norm_image = cpm.normalize_image(expanded_img, mean, stdv)
    img_torch = cpm.ndarray_image_to_tensor(norm_image)
    microbundle_sam = cpm.load_ft_SAM_model(checkpoint_path, device, model_type=model_type)
    image_embedding, sparse_embeddings, dense_embeddings = cpm.get_embeddings(microbundle_sam, img_torch, device)
    mask_pred_prob = cpm.get_pred_mask_prob(microbundle_sam, image_embedding, sparse_embeddings, dense_embeddings)
    assert torch.is_tensor(mask_pred_prob)
    assert torch.all(torch.sigmoid(mask_pred_prob) < 1)


def test_resize_pred_mask_prob():
    mean = 100
    stdv = 5 
    size = 256
    orig_array = np.zeros([size*2,size*2])
    orig_size = orig_array.shape
    dist = np.random.normal(mean, stdv, size**2)
    array = dist.reshape(size,-1)
    array = array[:,:,np.newaxis]
    array_torch = cpm.ndarray_image_to_tensor(array)
    resized_array_torch = cpm.resize_pred_mask_prob(array_torch, orig_size)
    assert resized_array_torch.shape == (1,1,orig_size[0],orig_size[1])


def test_mask_prob_to_binary():
    mean = 100
    stdv = 5 
    size = 256
    dist = np.random.normal(mean, stdv, size**2)
    array = dist.reshape(size,-1)
    array = array[:,:,np.newaxis]
    array_torch = cpm.ndarray_image_to_tensor(array)
    binary_array = cpm.mask_prob_to_binary(array_torch)
    nonzero_elem = np.count_nonzero(binary_array)
    sum_elem = np.sum(binary_array)
    assert nonzero_elem == sum_elem


def test_segment_microbundle_pillars_SAM():
    src_path = Path('./src/microbundlepillartrack')
    checkpoint_path = src_path.joinpath('microbundle_SAM_Type2_pillars.pth').resolve()
    gpu = 0
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    mean = 100
    stdv = 5 
    size = 512
    dist = np.random.normal(mean, stdv, size**2)
    img = dist.reshape(size,-1)
    binary_mask = cpm.segment_microbundle_pillars_SAM(img, checkpoint_path, device)
    nonzero_elem = np.count_nonzero(binary_mask)
    sum_elem = np.sum(binary_mask)
    assert binary_mask.shape == (size,size)
    assert nonzero_elem == sum_elem


def test_run_microbundle_SAM():
    src_path = Path('./src/microbundlepillartrack')
    checkpoint_path = src_path.joinpath('microbundle_SAM_Type2_pillars.pth').resolve()
    gpu = 0
    mean = 100
    stdv = 5 
    size = 512
    dist = np.random.normal(mean, stdv, size**2)
    img = dist.reshape(size,-1)
    binary_mask = cpm.run_microbundle_SAM(img, checkpoint_path, gpu)
    nonzero_elem = np.count_nonzero(binary_mask)
    sum_elem = np.sum(binary_mask)
    assert binary_mask.shape == (size,size)
    assert nonzero_elem == sum_elem


def test_create_pillar_masks_type1_thresh():
    img_path = glob_movie("real_example_pillar_masks")[0]
    src_path = Path('./src/microbundlepillartrack')
    checkpoint_path = src_path.joinpath('microbundle_SAM_Type1_pillars.pth').resolve()
    gpu = 0
    img = io.imread(img_path)
    closed_mask_1, closed_mask_2 = cpm.create_pillar_masks_type1(img, checkpoint_path, gpu)
    assert closed_mask_1.shape == img.shape
    assert closed_mask_2.shape == img.shape


def test_create_pillar_masks_type1_SAM():
    img_path = glob_movie("real_example_SAM_type1")[0]
    src_path = Path('./src/microbundlepillartrack')
    checkpoint_path = src_path.joinpath('microbundle_SAM_Type1_pillars.pth').resolve()
    gpu = 0
    img = io.imread(img_path)
    closed_mask_1, closed_mask_2 = cpm.create_pillar_masks_type1(img, checkpoint_path, gpu)
    assert closed_mask_1.shape == img.shape
    assert closed_mask_2.shape == img.shape


def test_create_pillar_masks_type1_SAM_fail():
    img_path = glob_movie("real_example_SAM_type1_fail")[0]
    src_path = Path('./src/microbundlepillartrack')
    checkpoint_path = src_path.joinpath('microbundle_SAM_Type1_pillars.pth').resolve()
    gpu = 0
    img = io.imread(img_path)
    with pytest.raises(IndexError) as error:
        _, _ = cpm.create_pillar_masks_type1(img, checkpoint_path, gpu)
    assert error.typename == "IndexError"


def test_create_pillar_masks_type2_SAM():
    img_path = glob_movie("real_example_SAM_type2")[0]
    src_path = Path('./src/microbundlepillartrack')
    checkpoint_path = src_path.joinpath('microbundle_SAM_Type2_pillars.pth').resolve()
    gpu = 0
    img = io.imread(img_path)
    closed_mask_1, closed_mask_2 = cpm.create_pillar_masks_type2(img, checkpoint_path, gpu)
    assert closed_mask_1.shape == img.shape
    assert closed_mask_2.shape == img.shape


def test_create_pillar_masks_type2_SAM_fail():
    img_path = glob_movie("real_example_SAM_type2_fail")[0]
    src_path = Path('./src/microbundlepillartrack')
    checkpoint_path = src_path.joinpath('microbundle_SAM_Type2_pillars.pth').resolve()
    gpu = 0
    img = io.imread(img_path)
    with pytest.raises(IndexError) as error:
        _, _ = cpm.create_pillar_masks_type2(img, checkpoint_path, gpu)
    assert error.typename == "IndexError"


def test_save_mask():
    folder_path = example_path("real_example_pillar_masks")
    img_path = glob_movie("real_example_pillar_masks")[0]
    src_path = Path('./src/microbundlepillartrack')
    checkpoint_path = src_path.joinpath('microbundle_SAM_Type1_pillars.pth').resolve()
    gpu = 0
    img = io.imread(img_path)
    closed_mask_1, closed_mask_2 = cpm.create_pillar_masks_type1(img, checkpoint_path, gpu)
    file_path_1, img_path_1, file_path_2, img_path_2 = cpm.save_mask(folder_path, closed_mask_1, closed_mask_2, fname = "pillar_mask")
    assert file_path_1.is_file()
    assert img_path_1.is_file()
    assert file_path_2.is_file()
    assert img_path_2.is_file()


def test_run_create_pillar_mask_type1():
    folder_path = example_path("real_example_SAM_type1")
    src_path = Path('./src/microbundlepillartrack')
    microbundle_type = "type1"
    fname = "pillar_mask"
    frame_num = 0
    file_path_m1, img_path_m1, file_path_m2, img_path_m2 = cpm.run_create_pillar_mask(folder_path,src_path,microbundle_type,fname,frame_num)
    assert file_path_m1.is_file()
    assert img_path_m1.is_file()
    assert file_path_m2.is_file()
    assert img_path_m2.is_file()


def test_run_create_pillar_mask_type2():
    folder_path = example_path("real_example_SAM_type2")
    src_path = Path('./src/microbundlepillartrack')
    microbundle_type = "type2"
    fname = "pillar_mask"
    frame_num = 0
    file_path_m1, img_path_m1, file_path_m2, img_path_m2 = cpm.run_create_pillar_mask(folder_path,src_path,microbundle_type,fname,frame_num)
    assert file_path_m1.is_file()
    assert img_path_m1.is_file()
    assert file_path_m2.is_file()
    assert img_path_m2.is_file()


def test_run_create_pillar_mask_wrong_type():
    folder_path = example_path("real_example_SAM_type2")
    src_path = Path('./src/microbundlepillartrack')
    microbundle_type = "type3"
    fname = "pillar_mask"
    frame_num = 0
    with warnings.catch_warnings(record=True) as record:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        cpm.run_create_pillar_mask(folder_path,src_path,microbundle_type,fname,frame_num)
    assert len(record) == 1



