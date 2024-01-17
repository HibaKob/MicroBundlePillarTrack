from microbundlepillartrack import pillar_analysis as pa
from microbundlepillartrack import create_pillar_mask as cpm
from microbundlepillartrack import image_analysis as ia
import numpy as np
import glob
from pathlib import Path
import warnings
import shutil
import pytest


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


def pillar_masks_path(example_name):
    ex_path = example_path(example_name)
    mask_path = ex_path.joinpath("masks").resolve()
    p_m_paths = sorted(glob.glob(str(mask_path) + "/*pillar_mask*.txt"))
    return p_m_paths


def test_rename_folder():
    folder_path = example_path("test_rename_folder")
    folder_name = 'movie'
    ia.create_folder(folder_path, folder_name)
    new_folder_name = 'unadjusted_movie'
    new_path = pa.rename_folder(folder_path, folder_name, new_folder_name)
    assert new_path.is_dir()
    assert new_path.name == new_folder_name
    if new_path.exists():
        _ = pa.rename_folder(folder_path,'unadjusted_movie','movie')


def test_rename_folder_false():
    folder_path = example_path("test_rename_folder")
    folder_name = 'movies'
    new_folder_name = 'unadjusted_movie'
    with warnings.catch_warnings(record=True) as record:
        warnings.filterwarnings("ignore", category=DeprecationWarning) 
        pa.rename_folder(folder_path, folder_name, new_folder_name)
    assert len(record) == 1


def test_rename_folder_exists():
    folder_path = example_path("test_rename_folder")
    folder_name = 'movie'
    new_folder_name = 'new_movie'
    with warnings.catch_warnings(record=True) as record:
        warnings.filterwarnings("ignore", category=DeprecationWarning) 
        pa.rename_folder(folder_path, folder_name, new_folder_name)
    assert len(record) == 1



def test_compute_otsu_thresh():
    dim = 10
    known_lower = 10
    known_upper = 100
    std_lower = 2
    std_upper = 10
    select = 0.8
    x1 = np.random.normal(known_lower, std_lower, dim * dim * dim)
    x1 = np.reshape(x1, (dim, dim, dim))
    x2 = np.random.normal(known_upper, std_upper, dim * dim * dim)
    x2 = np.reshape(x2, (dim, dim, dim))
    choose = np.random.random((dim, dim, dim)) > select
    x1[choose] = x1[choose] + x2[choose]
    found = pa.compute_otsu_thresh(x1)
    assert found > known_lower and found < (known_upper + known_lower)


def test_apply_otsu_thresh():
    dim = 10
    known_lower = 10
    known_upper = 10000
    std_lower = 0.1
    std_upper = 10
    select = 0.8
    x1 = np.random.normal(known_lower, std_lower, dim * dim * dim)
    x1 = np.reshape(x1, (dim, dim, dim))
    x2 = np.random.normal(known_upper, std_upper, dim * dim * dim)
    x2 = np.reshape(x2, (dim, dim, dim))
    choose = np.random.random((dim, dim, dim)) > select
    x1[choose] = x1[choose] + x2[choose]
    known = x1 > np.mean(x1)
    found = pa.apply_otsu_thresh(x1)
    assert np.allclose(known, found)


def test_invert_mask():
    array_half = np.zeros((10, 10))
    array_half[0:5, :] = 1
    array_invert = pa.invert_mask(array_half)
    assert np.allclose(array_invert + array_half, np.ones((10, 10)))


def test_close_region():
    val = 10
    array = np.zeros((val, val))
    array[3:7, 3:7] = 1
    array_missing = np.copy(array)
    array_missing[5, 5] = 0
    array_closed = pa.close_region(array_missing, radius=1)
    assert np.allclose(array_closed, array)


def test_moving_mean():
    a = np.array([4, 8, 6, -1, -2, -3, -1, 3, 4, 5])
    window = 3
    true_mov_mean = []
    for ii in range(len(a)-window+1):
        inter_mov_mean = np.sum(a[ii:ii+window])/window
        true_mov_mean.append(inter_mov_mean)
    mov_mean = pa.moving_mean(a,window)
    assert np.allclose(mov_mean,np.asarray(true_mov_mean))


def test_project_vector_timeseries():
    size = 50
    array_1 = np.random.randint(5, size=size) + np.random.rand(size)
    array_2 = np.random.randint(2, size=size) + np.random.rand(size)
    dir_vector = np.array([1,1])
    manual_proj = array_1-array_2 #dir_vector is transformed to [1,-1] in 'project_vector_timeseries' due to flipped image vertical coordinates
    proj_arrays = pa.project_vector_timeseries(array_1,array_2,dir_vector)
    assert np.allclose(manual_proj,proj_arrays)


def test_find_valleys():
    x = np.linspace(0, np.pi * 2.0, 250)
    timeseries = np.sin(4*x - np.pi / 2.0)
    valleys = pa.find_valleys(timeseries)
    assert valleys.shape[0] == 4
    assert np.isclose(timeseries[valleys[0]], -1, atol=.01)
    assert np.isclose(timeseries[valleys[1]], -1, atol=.01)
    assert np.isclose(timeseries[valleys[2]], -1, atol=.01)
    assert np.isclose(timeseries[valleys[3]], -1, atol=.01)
    li = 10 * [-0.99] + list(timeseries) + 10 * [-0.99]
    timeseries = np.asarray(li)
    valleys = pa.find_valleys(timeseries)
    assert np.isclose(timeseries[valleys[0]], -1, atol=.01)
    assert np.isclose(timeseries[valleys[1]], -1, atol=.01)
    assert np.isclose(timeseries[valleys[2]], -1, atol=.01)
    assert np.isclose(timeseries[valleys[3]], -1, atol=.01)


def test_compute_peaks():
    x = np.linspace(0, np.pi * 2.0, 250)
    timeseries = np.sin(4*x - np.pi / 2.0)
    peaks = pa.find_beat_peaks(timeseries)
    assert peaks.shape[0] == 4
    assert np.isclose(timeseries[peaks[0]], 1, atol=.01)
    assert np.isclose(timeseries[peaks[1]], 1, atol=.01)
    assert np.isclose(timeseries[peaks[2]], 1, atol=.01)
    assert np.isclose(timeseries[peaks[3]], 1, atol=.01)
    li = 10 * [-0.99] + list(timeseries) + 10 * [-0.99]
    timeseries = np.asarray(li)
    peaks = pa.find_beat_peaks(timeseries)
    assert np.isclose(timeseries[peaks[0]], 1, atol=.01)
    assert np.isclose(timeseries[peaks[1]], 1, atol=.01)
    assert np.isclose(timeseries[peaks[2]], 1, atol=.01)
    assert np.isclose(timeseries[peaks[3]], 1, atol=.01)


def test_find_peak_widths():
    x = np.linspace(0, np.pi * 2.0, 250)
    timeseries = np.sin(4*x - np.pi / 2.0) + 1
    peaks = pa.find_beat_peaks(timeseries)
    num_beats = len(peaks)
    t_100 = 250/num_beats
    relative_height = 0.5
    true_width_t50 = (1-relative_height)*t_100
    true_amp_t50 = (1-relative_height)*timeseries[peaks]
    width_info_t50 = pa.find_peak_widths(timeseries, peaks, relative_height)
    assert np.allclose(width_info_t50[0],true_width_t50,atol=0.5)
    assert np.allclose(width_info_t50[1],true_amp_t50,atol=0.01)
    relative_height = 0.8
    true_amp_t80 = (1-relative_height)*timeseries[peaks]
    width_info_t80 = pa.find_peak_widths(timeseries, peaks, relative_height)
    assert np.allclose(width_info_t80[1],true_amp_t80,atol=0.01)


def test_adjust_first_valley():
    folder_path = example_path('example_pillar_frame0_not_valley')
    movie_path = folder_path.joinpath("movie").resolve()
    new_movie_path = folder_path.joinpath("unadjusted_movie").resolve()
    movie_lst = glob_movie(folder_path)
    original_frames = len(movie_lst)
    first_valley_idx = 3
    new_img_lst = pa.adjust_first_valley(folder_path,first_valley_idx)
    assert len(new_img_lst) == original_frames - first_valley_idx
    if new_movie_path.exists() and new_movie_path.is_dir():
        shutil.rmtree(movie_path)
        _ = pa.rename_folder(folder_path,'unadjusted_movie','movie')


def test_check_frame0_valley():
    folder_path = example_path('example_pillar_frame0_not_valley')
    movie_path_lst = glob_movie(folder_path)
    tiff_list = ia.read_all_tiff(movie_path_lst)
    img_list_uint8 = ia.uint16_to_uint8_all(tiff_list)
    file_path_m1, _, file_path_m2, _ = cpm.run_create_pillar_mask(folder_path)
    closed_mask_1 = ia.read_txt_as_mask(file_path_m1)
    closed_mask_2 = ia.read_txt_as_mask(file_path_m2)
    l_mask, r_mask = pa.order_pillar_masks(closed_mask_1,closed_mask_2)
    center_mask_l = pa.get_mask_centroid(l_mask)
    center_mask_r = pa.get_mask_centroid(r_mask)
    clip_fraction = 0.1
    tracker_0, tracker_1 = ia.track_all_steps_with_adjust_param_dicts(img_list_uint8, l_mask, clip_fraction)
    _,mean_disp_all_0, mean_disp_all_1 = pa.compute_pillar_position_timeseries(tracker_0,tracker_1)
    vector = pa.find_vector_between_centroids(center_mask_l,center_mask_r)
    proj_mean_disp_all = pa.project_vector_timeseries(mean_disp_all_0, mean_disp_all_1, vector)
    valleys = pa.find_valleys(proj_mean_disp_all)
    with warnings.catch_warnings(record=True) as record:
        warnings.filterwarnings("ignore", category=DeprecationWarning) 
        adjusted_img_list = pa.check_frame0_valley(folder_path, tiff_list, proj_mean_disp_all, valleys)
        assert len(record) == 1
    assert len(adjusted_img_list) < len(img_list_uint8)
    movie_path = folder_path.joinpath("movie").resolve()
    new_movie_path = folder_path.joinpath("unadjusted_movie").resolve()
    if new_movie_path.exists() and new_movie_path.is_dir():
        shutil.rmtree(movie_path)
        _ = pa.rename_folder(folder_path,'unadjusted_movie','movie')


def test_check_frame0_valley_true():
    folder_path = example_path('real_example_pillar_short')
    movie_path_lst = glob_movie(folder_path)
    tiff_list = ia.read_all_tiff(movie_path_lst)
    img_list_uint8 = ia.uint16_to_uint8_all(tiff_list)
    file_path_m1, _, file_path_m2, _ = cpm.run_create_pillar_mask(folder_path)
    closed_mask_1 = ia.read_txt_as_mask(file_path_m1)
    closed_mask_2 = ia.read_txt_as_mask(file_path_m2)
    l_mask, r_mask = pa.order_pillar_masks(closed_mask_1,closed_mask_2)
    center_mask_l = pa.get_mask_centroid(l_mask)
    center_mask_r = pa.get_mask_centroid(r_mask)
    clip_fraction = 0.1
    tracker_0, tracker_1 = ia.track_all_steps_with_adjust_param_dicts(img_list_uint8, l_mask, clip_fraction)
    _,mean_disp_all_0, mean_disp_all_1 = pa.compute_pillar_position_timeseries(tracker_0,tracker_1)
    vector = pa.find_vector_between_centroids(center_mask_l,center_mask_r)
    proj_mean_disp_all = pa.project_vector_timeseries(mean_disp_all_0, mean_disp_all_1, vector)
    valleys = pa.find_valleys(proj_mean_disp_all)
    with warnings.catch_warnings(record=True) as record:
        warnings.filterwarnings("ignore", category=DeprecationWarning) 
        adjusted_img_list = pa.check_frame0_valley(folder_path, tiff_list, proj_mean_disp_all, valleys)
        assert len(record) == 0
    assert len(adjusted_img_list) == len(img_list_uint8)


def test_prepare_valley_info():
    valleys = np.array([2,15,27,38,51])
    info = pa.prepare_valley_info(valleys)
    assert info.shape[0] == 4
    assert info.shape[1] == 3


def test_compute_pillar_secnd_moment():
    pillar_width = 163
    pillar_thickness = 33.2
    pillar_diameter = 40
    pillar_secnd_moment_area = pa.compute_pillar_secnd_moment_rectangular(pillar_width, pillar_thickness)
    assert np.isclose(pillar_secnd_moment_area, (pillar_width*(pillar_thickness)**3)/12, atol=1)
    pillar_secnd_moment_area = pa.compute_pillar_secnd_moment_circular(pillar_diameter)
    assert np.isclose(pillar_secnd_moment_area, (np.pi*(pillar_diameter)**4)/64, atol=1)


def test_compute_pillar_stiffnes():
    pillar_modulus = 1.61
    pillar_width = 163
    pillar_thickness = 33.2 
    pillar_diameter = 40   
    pillar_length = 199
    force_location = 163
    pillar_profile_rect = 'rectangular'
    pillar_profile_circ = 'circular'
    pillar_profile_wrong = 'circ'
    I_rect = pa.compute_pillar_secnd_moment_rectangular(pillar_width, pillar_thickness)
    I_circ = pa.compute_pillar_secnd_moment_circular(pillar_diameter)
    pillar_stiffness_gt_rect = (6*pillar_modulus*I_rect)/((force_location**2)*(3*pillar_length-force_location))
    pillar_stiffness_gt_circ = (6*pillar_modulus*I_circ)/((force_location**2)*(3*pillar_length-force_location))
    pillar_stiffness_rect = pa.compute_pillar_stiffnes(pillar_profile_rect, pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location)
    pillar_stiffness_circ = pa.compute_pillar_stiffnes(pillar_profile_circ, pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location)
    pillar_stiffness_wrong = pa.compute_pillar_stiffnes(pillar_profile_wrong, pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location)
    assert np.isclose(pillar_stiffness_rect, pillar_stiffness_gt_rect, atol=0.01)
    assert np.isclose(pillar_stiffness_circ, pillar_stiffness_gt_circ, atol=0.01)
    assert np.isclose(pillar_stiffness_wrong, 0, atol=0.01)
     

def test_compute_pillar_force():
    pillar_stiffness = 0.42
    pillar_avg_deflection = 2
    length_scale = 1
    pillar_force_gt = pillar_stiffness*pillar_avg_deflection*length_scale
    pillar_force = pa.compute_pillar_force(pillar_stiffness, pillar_avg_deflection, length_scale)
    assert np.isclose(pillar_force, pillar_force_gt, atol=0.01)


def test_compute_pillar_position_timeseries():
    num_pts = 3
    num_frames = 100
    tracker_0 = 100 * np.ones((num_pts, num_frames)) + np.random.random((num_pts, num_frames))
    tracker_1 = 50 * np.ones((num_pts, num_frames)) + np.random.random((num_pts, num_frames))
    disp_abs_mean, mean_disp_0_all, mean_disp_1_all = pa.compute_pillar_position_timeseries(tracker_0, tracker_1)
    assert disp_abs_mean.shape[0] == num_frames
    assert np.max(disp_abs_mean) < np.sqrt(2.0)
    assert mean_disp_0_all.shape[0] == num_frames
    assert np.max(mean_disp_0_all) < 1
    assert mean_disp_1_all.shape[0] == num_frames
    assert np.max(mean_disp_1_all) < 1


def test_pillar_force_all_steps():
    pillar_stiffnes = None
    pillar_profile = 'rectangular'
    pillar_modulus = 1.61
    pillar_width = 163
    pillar_thickness = 33.2   
    pillar_diameter = 40  
    pillar_length = 199
    force_location = 163
    num_frames = 100
    pillar_mean_abs_disp = np.sqrt(2)*np.ones(num_frames)
    pillar_mean_disp_row = np.ones(num_frames)
    pillar_mean_disp_col = np.ones(num_frames)
    length_scale = 1

    pillar_k = pa.compute_pillar_stiffnes(pillar_profile, pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location)
    pillar_F_abs, pillar_F_row, pillar_F_col = pa.pillar_force_all_steps(pillar_mean_abs_disp,pillar_mean_disp_row, pillar_mean_disp_col, pillar_stiffnes, pillar_profile, 
                                                                         pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location, length_scale)
    assert len(pillar_F_abs) == num_frames
    assert np.allclose(pillar_F_abs/pillar_k, pillar_mean_abs_disp, atol=0.01)
    assert len(pillar_F_row) == num_frames
    assert np.allclose(pillar_F_row/pillar_k, pillar_mean_disp_row, atol=0.01)
    assert len(pillar_F_col) == num_frames
    assert np.allclose(pillar_F_col/pillar_k, pillar_mean_disp_col, atol=0.01)


def test_pillar_force_all_steps_given_K():
    pillar_stiffnes = 0.42
    pillar_profile = 'rectangular'
    pillar_modulus = 1.61
    pillar_width = 163
    pillar_thickness = 33.2 
    pillar_diameter = 40    
    pillar_length = 199
    force_location = 163
    num_frames = 100
    pillar_mean_abs_disp = np.sqrt(2)*np.ones(num_frames)
    pillar_mean_disp_row = np.ones(num_frames)
    pillar_mean_disp_col = np.ones(num_frames)
    length_scale = 1

    pillar_F_abs, pillar_F_row, pillar_F_col = pa.pillar_force_all_steps(pillar_mean_abs_disp,pillar_mean_disp_row, pillar_mean_disp_col, pillar_stiffnes, pillar_profile, 
                                                                         pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location, length_scale)
    pillar_k = pa.compute_pillar_stiffnes(pillar_profile, pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location)
    assert len(pillar_F_abs) == num_frames
    assert np.allclose(pillar_F_abs/pillar_stiffnes, pillar_mean_abs_disp, atol=0.01)
    assert len(pillar_F_row) == num_frames
    assert np.allclose(pillar_F_row/pillar_stiffnes, pillar_mean_disp_row, atol=0.01)
    assert len(pillar_F_col) == num_frames
    assert np.allclose(pillar_F_col/pillar_stiffnes, pillar_mean_disp_col, atol=0.01)
    assert pillar_stiffnes != pillar_k


def test_pillar_force_all_steps_given_K_2():
    pillar_stiffnes = 0.42
    num_frames = 100
    pillar_mean_abs_disp = np.sqrt(2)*np.ones(num_frames)
    pillar_mean_disp_row = np.ones(num_frames)
    pillar_mean_disp_col = np.ones(num_frames)
    length_scale = 1

    pillar_F_abs, pillar_F_row, pillar_F_col = pa.pillar_force_all_steps(pillar_mean_abs_disp,pillar_mean_disp_row, pillar_mean_disp_col, pillar_stiffnes, length_scale=length_scale)
    assert len(pillar_F_abs) == num_frames
    assert np.allclose(pillar_F_abs/pillar_stiffnes, pillar_mean_abs_disp, atol=0.01)
    assert len(pillar_F_row) == num_frames
    assert np.allclose(pillar_F_row/pillar_stiffnes, pillar_mean_disp_row, atol=0.01)
    assert len(pillar_F_col) == num_frames
    assert np.allclose(pillar_F_col/pillar_stiffnes, pillar_mean_disp_col, atol=0.01)


def test_compute_pillar_velocity():
    y = np.linspace(0, np.pi * 2.0, 250)
    length_scale = 4 
    fps = 30
    true_velocity = (np.diff(y))*length_scale*fps
    computed_velocity = pa.compute_pillar_velocity(y,length_scale,fps)
    assert np.allclose(true_velocity[:-2],computed_velocity)


def test_compute_contraction_relaxation_peaks():
    x = np.linspace(0, np.pi * 2.0, 250)
    velocity = np.sin(6*x)
    contraction_peaks_idx, relaxation_peaks_idx = pa.compute_contraction_relaxation_peaks(velocity)
    assert len(contraction_peaks_idx) == len(relaxation_peaks_idx) == 6
    assert np.allclose(np.diff(contraction_peaks_idx),250/6,atol=1)
    assert np.allclose(np.diff(relaxation_peaks_idx),250/6,atol=1)


def test_save_pillar_velocity_results():
    folder_path = example_path("real_example_pillar_short")
    pillar_velocity = np.zeros((10,100))
    contraction_info = np.array([3,33,63,93])
    relaxation_info = np.array([15,45,75])
    saved_paths = pa.save_pillar_velocity_results(folder_path=folder_path,pillar_velocity=pillar_velocity,contraction_info=contraction_info,relaxation_info=relaxation_info,fname=None)
    for paths in saved_paths:
        assert paths.is_file()
    assert len(saved_paths) == 3
    saved_paths = pa.save_pillar_velocity_results(folder_path=folder_path,pillar_velocity=pillar_velocity,contraction_info=contraction_info,relaxation_info=relaxation_info,fname="pillar_1")
    for paths in saved_paths:
        assert paths.is_file()
    assert len(saved_paths) == 3


def test_get_mask_centroid():
    array = np.zeros((100,100))
    array[20:61,40:61] = 1
    true_centroid = np.array([(60+20)/2,(40+60)/2])
    centroid = pa.get_mask_centroid(array)
    assert np.allclose(centroid, true_centroid)


def test_order_pillar_masks():
    array_1 = np.zeros((512,512))
    array_2 = np.zeros((512,512))
    array_1[250:350,400:490] = 1
    array_2[250:350,90:150] = 1
    ordered_masks = pa.order_pillar_masks(array_1,array_2)
    assert np.sum(ordered_masks[0]) == np.sum(array_2)
    assert np.sum(ordered_masks[1]) == np.sum(array_1)


def test_find_vector_between_centroids():
    centroid1 = np.array([8,5])
    centroid2 = np.array([40,25])
    vector = centroid2 - centroid1
    magnitude = np.sqrt((vector[0])**2 + (vector[1])**2)
    true_unit_vector = vector/magnitude
    unit_vector = pa.find_vector_between_centroids(centroid1,centroid2)
    assert np.allclose(unit_vector,true_unit_vector)


def test_save_orientation_info():
    folder_path = example_path("real_example_pillar_short")
    vector = np.array([1,0]) # given as (row,col)
    angle = np.pi
    saved_path = pa.save_orientation_info(folder_path,vector,angle)
    assert saved_path.is_file


def test_compute_midpt():
    pt1 = np.array([10,15])
    pt2 = np.array([30,45])
    true_midpt = np.array((pt1[0]+pt2[0])/2,(pt1[1]+pt2[1])/2)
    midpt = pa.compute_midpt(pt1,pt2)
    np.allclose(midpt,true_midpt)


def test_extract_ROI_for_width():
    array = np.zeros((250,250))
    midpt = np.array([250/2,250/2])
    pillar_col_dist = 100
    ratio_r = 0.4
    ratio_c = 0.2
    img_ROI = pa.extract_ROI_for_width(array,midpt,pillar_col_dist,ratio_r,ratio_c)
    assert img_ROI.shape[0] == 2*ratio_r*pillar_col_dist
    assert img_ROI.shape[1] == 2*ratio_c*pillar_col_dist


def test_extract_ROI_for_width_out_of_bound():
    array = np.zeros((90,90))
    midpt = np.array([90/2,90/2])
    pillar_col_dist = 100
    ratio_r = 0.5
    ratio_c = 0.2
    img_ROI = pa.extract_ROI_for_width(array,midpt,pillar_col_dist,ratio_r,ratio_c)
    assert img_ROI.shape[0] == array.shape[0]
    assert img_ROI.shape[1] == 2*ratio_c*pillar_col_dist


def test_find_tissue_width():
    folder_path = example_path("real_example_pillar_short")
    movie_path_list = glob_movie(folder_path)
    tiff_list = ia.read_all_tiff(movie_path_list)
    masks_path = pillar_masks_path(folder_path)
    mask_1 = ia.read_txt_as_mask(masks_path[0])
    mask_2 = ia.read_txt_as_mask(masks_path[1])
    l_mask, r_mask = pa.order_pillar_masks(mask_1,mask_2)
    center_mask_l = pa.get_mask_centroid(l_mask)
    center_mask_r = pa.get_mask_centroid(r_mask)
    manual_tissue_width = 111
    with warnings.catch_warnings(record=True) as record:
        warnings.filterwarnings("ignore", category=DeprecationWarning) 
        tissue_width = pa.find_tissue_width(folder_path,tiff_list,center_mask_l,center_mask_r)
    assert np.isclose(tissue_width,manual_tissue_width, atol=3)
    assert len(record) == 0

def test_find_tissue_width_one_peak():
    folder_path = example_path("real_example_pillar_super_short_cropped_tissue")
    movie_path_list = glob_movie(folder_path)
    tiff_list = ia.read_all_tiff(movie_path_list)
    masks_path = pillar_masks_path(folder_path)
    mask_1 = ia.read_txt_as_mask(masks_path[0])
    mask_2 = ia.read_txt_as_mask(masks_path[1])
    l_mask, r_mask = pa.order_pillar_masks(mask_1,mask_2)
    center_mask_l = pa.get_mask_centroid(l_mask)
    center_mask_r = pa.get_mask_centroid(r_mask)
    manual_tissue_width = 111
    with warnings.catch_warnings(record=True) as record:
        warnings.filterwarnings("ignore", category=DeprecationWarning) 
        tissue_width = pa.find_tissue_width(folder_path,tiff_list,center_mask_l,center_mask_r)
    assert np.isclose(tissue_width,manual_tissue_width, atol=3)
    assert len(record) == 1


def test_find_tissue_width_dark():
    folder_path = example_path("real_example_pillar_super_short_inverted_colors")
    movie_path_list = glob_movie(folder_path)
    tiff_list = ia.read_all_tiff(movie_path_list)
    masks_path = pillar_masks_path(folder_path)
    mask_1 = ia.read_txt_as_mask(masks_path[0])
    mask_2 = ia.read_txt_as_mask(masks_path[1])
    l_mask, r_mask = pa.order_pillar_masks(mask_1,mask_2)
    center_mask_l = pa.get_mask_centroid(l_mask)
    center_mask_r = pa.get_mask_centroid(r_mask)
    manual_tissue_width = 108
    with warnings.catch_warnings(record=True) as record:
        warnings.filterwarnings("ignore", category=DeprecationWarning) 
        tissue_width = pa.find_tissue_width(folder_path,tiff_list,center_mask_l,center_mask_r)
    assert np.isclose(tissue_width,manual_tissue_width, rtol=1)
    assert len(record) == 1


def test_save_tissue_width_info():
    folder_path = example_path("real_example_pillar_short")
    tissue_width = 111 
    length_scale = 4
    saved_path = pa.save_tissue_width_info(folder_path,tissue_width,length_scale)
    assert saved_path.is_file


def test_compute_tissue_stress_all_steps():
    pillar_abs_force = np.ones((10,100))
    tissue_width = 111
    tissue_depth = 350
    length_scale = 4
    true_tissue_stress = pillar_abs_force/(tissue_width*tissue_depth*length_scale)
    tissue_stress = pa.compute_tissue_stress_all_steps(pillar_abs_force,tissue_width,tissue_depth,length_scale)
    assert np.allclose(tissue_stress, true_tissue_stress)


def test_detect_irregular_beats():
    peaks_regular = np.array([5,25,46,65,83])
    peaks_irregular = np.array([5,25,46,60,85])
    with warnings.catch_warnings(record=True) as record:
        warnings.filterwarnings("ignore", category=DeprecationWarning) 
        pa.detect_irregular_beats(peaks_regular)
    assert len(record) == 0   
    with warnings.catch_warnings(record=True) as record:
        warnings.filterwarnings("ignore", category=DeprecationWarning) 
        pa.detect_irregular_beats(peaks_irregular)
    assert len(record) == 1


def test_find_slope():
    x_pts = np.arange(1,100,5)
    true_slope = 3.5
    true_intercept = 2.1
    y_pts = true_slope*x_pts+true_intercept
    slope, y_interc = pa.find_slope(x_pts,y_pts)
    assert np.isclose(slope, true_slope, atol=0.01)
    assert np.isclose(y_interc, true_intercept, atol=0.01)


def test_find_slope_0_intercept():
    x_pts = np.arange(1,100,5)
    true_slope = 3.5
    true_intercept = 2.1
    y_pts = true_slope*x_pts+true_intercept
    slope, y_interc = pa.find_slope_0_intercept(x_pts,y_pts)
    assert slope > true_slope
    assert np.isclose(y_interc, 0, atol=0.01)


def test_find_slope_0_intercept_true():
    x_pts = np.arange(1,100,5)
    true_slope = 3.5
    true_intercept = 0
    y_pts = true_slope*x_pts+true_intercept
    slope, y_interc = pa.find_slope_0_intercept(x_pts,y_pts)
    assert np.isclose(slope, true_slope, atol=0.01)
    assert np.isclose(y_interc, true_intercept, atol=0.01)


def test_save_pillar_position():
    folder_path = example_path("real_example_pillar_short")
    tracker_row_all = np.zeros((10, 100))
    tracker_col_all = np.zeros((10, 100))
    info = [[0, 10, 30], [1, 30, 35], [2, 35, 85]]
    info = np.asarray(info)
    saved_paths = pa.save_pillar_position(folder_path=folder_path, tracker_row_all=tracker_row_all, tracker_col_all = tracker_col_all, info = info, split_track = False, fname = None)
    for paths in saved_paths:
        assert paths.is_file()
    assert len(saved_paths) == 3


def test_detect_drift_no_split():
    x = np.linspace(0, np.pi * 2.0, 250)
    timeseries = np.sin(4*x - np.pi / 2.0) + 1
    drift = 0.02*x
    timeseries_drift = timeseries + drift 
    peaks = pa.find_beat_peaks(timeseries_drift)
    valleys = pa.find_valleys(timeseries_drift)
    split = False
    with warnings.catch_warnings(record=True) as record:
        warnings.filterwarnings("ignore", category=DeprecationWarning) 
        pa.detect_drift(timeseries_drift, peaks, valleys, split)
    assert len(record) == 0
    drift = 0.2*x
    timeseries_drift = timeseries + drift 
    peaks = pa.find_beat_peaks(timeseries_drift)
    valleys = pa.find_valleys(timeseries_drift)
    split = False
    with warnings.catch_warnings(record=True) as record:
        warnings.filterwarnings("ignore", category=DeprecationWarning) 
        pa.detect_drift(timeseries_drift, peaks, valleys, split)
    assert len(record) == 1


def test_detect_drift_no_split_subpixel():
    x = np.linspace(0, np.pi * 2.0, 250)
    timeseries = 0.4*(np.sin(4*x - np.pi / 2.0) + 1)
    drift = 0.02*x
    timeseries_drift = timeseries + drift 
    peaks = pa.find_beat_peaks(timeseries_drift)
    valleys = pa.find_valleys(timeseries_drift)
    split = False
    with warnings.catch_warnings(record=True) as record:
        warnings.filterwarnings("ignore", category=DeprecationWarning) 
        pa.detect_drift(timeseries_drift, peaks, valleys, split)
    assert len(record) == 0
    drift = 0.2*x
    timeseries_drift = timeseries + drift 
    peaks = pa.find_beat_peaks(timeseries_drift)
    valleys = pa.find_valleys(timeseries_drift)
    split = False
    with warnings.catch_warnings(record=True) as record:
        warnings.filterwarnings("ignore", category=DeprecationWarning) 
        pa.detect_drift(timeseries_drift, peaks, valleys, split)
    assert len(record) == 1


def test_detect_drift_split():
    x = np.linspace(0, np.pi * 2.0, 250)
    timeseries = np.sin(4*x - np.pi / 2.0) + 1
    drift = 0.02*x
    timeseries_drift = timeseries + drift
    peaks = pa.find_beat_peaks(timeseries_drift)
    valleys = pa.find_valleys(timeseries_drift)
    split = True
    with warnings.catch_warnings(record=True) as record:
        warnings.filterwarnings("ignore", category=DeprecationWarning) 
        pa.detect_drift(timeseries_drift, peaks, valleys, split)
    assert len(record) == 0
    drift = 0.2*x
    timeseries_drift = timeseries + drift
    peaks = pa.find_beat_peaks(timeseries_drift)
    valleys = pa.find_valleys(timeseries_drift)
    split = True
    with warnings.catch_warnings(record=True) as record:
        warnings.filterwarnings("ignore", category=DeprecationWarning) 
        pa.detect_drift(timeseries_drift, peaks, valleys, split)
    assert len(record) == 1


def test_save_pillar_position_split():
    folder_path = example_path("real_example_pillar_short_split")
    tracker_row_all = np.zeros((10, 100))
    tracker_col_all = np.zeros((10, 100))
    info = [[0, 10, 30], [1, 30, 35], [2, 35, 85]]
    info = np.asarray(info)
    saved_paths = pa.save_pillar_position(folder_path=folder_path, tracker_row_all=tracker_row_all, tracker_col_all = tracker_col_all, info = info, split_track = True, fname = None)
    for paths in saved_paths:
        assert paths.is_file()
    assert len(saved_paths) == 7


def test_save_pillar_position_fname():
    folder_path = example_path("real_example_pillar_short")
    tracker_row_all = np.zeros((10, 100))
    tracker_col_all = np.zeros((10, 100))
    info = [[0, 10, 30], [1, 30, 35], [2, 35, 85]]
    info = np.asarray(info)
    saved_paths = pa.save_pillar_position(folder_path=folder_path, tracker_row_all=tracker_row_all, tracker_col_all = tracker_col_all, info = info, split_track = False, fname = 'Pillar_1_')
    for paths in saved_paths:
        assert paths.is_file()
    assert len(saved_paths) == 3


def test_save_pillar_force():
    folder_path = example_path("real_example_pillar_short")
    pillar_force_abs = np.sqrt(2)*np.ones(100)
    pillar_force_row = np.ones(100)
    pillar_force_col = np.ones(100)
    saved_paths = pa.save_pillar_force(folder_path=folder_path, pillar_force_abs = pillar_force_abs, pillar_force_row = pillar_force_row, pillar_force_col = pillar_force_col, fname=None)
    for paths in saved_paths:
        assert paths.is_file()
    assert len(saved_paths) == 3


def test_save_pillar_force_fname():
    folder_path = example_path("real_example_pillar_short")
    pillar_force_abs = np.sqrt(2)*np.ones(100)
    pillar_force_row = np.ones(100)
    pillar_force_col = np.ones(100)
    saved_paths = pa.save_pillar_force(folder_path=folder_path, pillar_force_abs = pillar_force_abs, pillar_force_row = pillar_force_row, pillar_force_col = pillar_force_col, fname="Pillar_1_")
    for paths in saved_paths:
        assert paths.is_file()
    assert len(saved_paths) == 3


def test_save_peaks():
    folder_path = example_path("real_example_pillar_short")
    peaks = np.array([15,30,45,60])
    fname = "pillar_1"
    saved_path = pa.save_peaks(folder_path=folder_path, peaks=peaks, fname=None)
    assert saved_path.is_file()
    saved_path = pa.save_peaks(folder_path=folder_path, peaks=peaks, fname=fname)
    assert saved_path.is_file()


def test_save_beat_width_info():
    folder_path = example_path("real_example_pillar_short")
    pillar_width_info = [np.array([10,11,11,10]),np.array([1.9,2,2.1,2]),np.array([19,66,114,160]),np.array([29,77,125,170])]
    fname = "pillar_1_t50_"
    saved_path = pa.save_beat_width_info(folder_path=folder_path, pillar_width_info=pillar_width_info, fname=None)
    assert saved_path.is_file()
    saved_path = pa.save_beat_width_info(folder_path=folder_path, pillar_width_info=pillar_width_info, fname=fname)
    assert saved_path.is_file()


def test_save_tissue_stress():
    folder_path = example_path("real_example_pillar_short")
    tissue_stress = np.ones((10,100))
    saved_path = pa.save_tissue_stress(folder_path=folder_path, tissue_stress=tissue_stress)
    assert saved_path.is_file


def test_run_pillar_tracking():
    folder_path = example_path("real_example_pillar_short")
    movie_path_lst = glob_movie(folder_path)
    num_frames = len(movie_path_lst)
    tissue_depth = 1
    pillar_orientation_vector = np.array([1,0])
    pillar_stiffnes = None
    pillar_profile = 'rectangular'
    pillar_modulus = 1.61
    pillar_width = 163
    pillar_thickness = 33.2 
    pillar_diameter = 40    
    pillar_length = 199
    force_location = 163
    length_scale = 1
    fps = 1
    split_track = False
    mean_disp_all_0, mean_disp_all_1, saved_paths_pos = pa.run_pillar_tracking(folder_path, tissue_depth, pillar_orientation_vector, pillar_stiffnes, pillar_profile, pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location, length_scale, fps, split_track)
    assert len(mean_disp_all_0) == num_frames
    assert len(mean_disp_all_1) == num_frames
    assert len(saved_paths_pos) == 3
    for pa_p in saved_paths_pos:
        assert pa_p.is_file()


def test_run_pillar_tracking_split():
    folder_path = example_path("real_example_pillar_short_split")
    tissue_depth = 1
    pillar_orientation_vector = np.array([1,0])
    pillar_stiffnes = None
    pillar_profile = 'rectangular'
    pillar_modulus = 1.61
    pillar_width = 163
    pillar_thickness = 33.2 
    pillar_diameter = 40    
    pillar_length = 199
    force_location = 163
    length_scale = 1
    fps = 1
    split_track = True
    _, _, saved_paths_pos = pa.run_pillar_tracking(folder_path, tissue_depth, pillar_orientation_vector, pillar_stiffnes, pillar_profile, pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location, length_scale, fps, split_track)
    assert len(saved_paths_pos) == 7
    for pa_p in saved_paths_pos:
        assert pa_p.is_file()


def test_run_pillar_tracking_vec():
    folder_path = example_path("real_example_pillar_short")
    tissue_depth = 1
    pillar_orientation_vector = None
    pillar_stiffnes = None
    pillar_profile = 'rectangular'
    pillar_modulus = 1.61
    pillar_width = 163
    pillar_thickness = 33.2 
    pillar_diameter = 40    
    pillar_length = 199
    force_location = 163
    length_scale = 1
    fps = 1
    split_track = False
    _, _,saved_paths_pos = pa.run_pillar_tracking(folder_path, tissue_depth, pillar_orientation_vector, pillar_stiffnes, pillar_profile, pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location, length_scale, fps, split_track)
    for pa_p in saved_paths_pos:
        assert pa_p.is_file()


def test_load_pillar_tracking_results():
    folder_path = example_path("real_example_pillar_short")
    folder_path_split = example_path("real_example_pillar_short_split")
    tissue_depth = 1
    pillar_orientation_vector = np.array([1,0])
    pillar_stiffnes = None
    pillar_profile = 'rectangular'
    pillar_modulus = 1.61
    pillar_width = 163
    pillar_thickness = 33.2 
    pillar_diameter = 40    
    pillar_length = 199
    force_location = 163
    length_scale = 1
    fps = 1
    fname = 'pillar1_'
    _ = pa.run_pillar_tracking(folder_path, tissue_depth, pillar_orientation_vector, pillar_stiffnes, pillar_profile, pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location, length_scale, fps, split_track = False)
    _ = pa.load_pillar_tracking_results(folder_path=folder_path, split_track = False, fname = fname)
    
    _ = pa.run_pillar_tracking(folder_path_split, tissue_depth, pillar_orientation_vector, pillar_stiffnes, pillar_profile, pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location, length_scale, fps, split_track = True)
    _ = pa.load_pillar_tracking_results(folder_path=folder_path_split, split_track = True, fname = fname)

    folder_path = example_path("io_testing_examples")
    folder_path_0 = folder_path.joinpath("fake_example_0").resolve()
    with pytest.raises(FileNotFoundError) as error:
        pa.load_pillar_tracking_results(folder_path=folder_path_0)
    assert error.typename == "FileNotFoundError"
    folder_path_1 = folder_path.joinpath("fake_example_3").resolve()
    with pytest.raises(FileNotFoundError) as error:
        pa.load_pillar_tracking_results(folder_path=folder_path_1)
    assert error.typename == "FileNotFoundError"


def test_load_pillar_force_results():
    folder_path = example_path("real_example_pillar_short")
    tissue_depth = 1
    pillar_orientation_vector = np.array([1,0])
    pillar_stiffnes = None
    pillar_profile = 'rectangular'
    pillar_modulus = 1.61
    pillar_width = 163
    pillar_thickness = 33.2 
    pillar_diameter = 40    
    pillar_length = 199
    force_location = 163
    length_scale = 1
    fps = 1
    fname = 'pillar1_'
    _ = pa.run_pillar_tracking(folder_path, tissue_depth, pillar_orientation_vector, pillar_stiffnes, pillar_profile, pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location, length_scale, fps, split_track = False)
    _ = pa.load_pillar_force_results(folder_path=folder_path, fname=fname)  
    
    folder_path = example_path("io_testing_examples")
    folder_path_0 = folder_path.joinpath("fake_example_0").resolve()
    with pytest.raises(FileNotFoundError) as error:
        pa.load_pillar_force_results(folder_path=folder_path_0)
    assert error.typename == "FileNotFoundError"
    folder_path_1 = folder_path.joinpath("fake_example_3").resolve()
    with pytest.raises(FileNotFoundError) as error:
        pa.load_pillar_force_results(folder_path=folder_path_1)
    assert error.typename == "FileNotFoundError"


def test_load_pillar_velocity_results():
    folder_path = example_path("real_example_pillar_short")
    tissue_depth = 1
    pillar_orientation_vector = np.array([1,0])
    pillar_stiffnes = None
    pillar_profile = 'rectangular'
    pillar_modulus = 1.61
    pillar_width = 163
    pillar_thickness = 33.2 
    pillar_diameter = 40    
    pillar_length = 199
    force_location = 163
    length_scale = 1
    fps = 1
    fname = 'pillar1_'
    _ = pa.run_pillar_tracking(folder_path, tissue_depth, pillar_orientation_vector, pillar_stiffnes, pillar_profile, pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location, length_scale, fps, split_track = False)
    _ = pa.load_pillar_velocity_results(folder_path=folder_path, fname=fname) 
    _ = pa.load_pillar_velocity_info(folder_path=folder_path, fname=fname)
    folder_path_1 = folder_path.joinpath("fake_example_3").resolve()
    with pytest.raises(FileNotFoundError) as error:
        pa.load_pillar_velocity_results(folder_path=folder_path_1)
    assert error.typename == "FileNotFoundError"
    with pytest.raises(FileNotFoundError) as error:
        pa.load_pillar_velocity_info(folder_path=folder_path_1)
    assert error.typename == "FileNotFoundError"
    

def test_load_tissue_stress_results():
    folder_path = example_path("real_example_pillar_short")
    tissue_depth = 1
    pillar_orientation_vector = np.array([1,0])
    pillar_stiffnes = None
    pillar_profile = 'rectangular'
    pillar_modulus = 1.61
    pillar_width = 163
    pillar_thickness = 33.2 
    pillar_diameter = 40    
    pillar_length = 199
    force_location = 163
    length_scale = 1
    fps = 1
    _ = pa.run_pillar_tracking(folder_path, tissue_depth, pillar_orientation_vector, pillar_stiffnes, pillar_profile, pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location, length_scale, fps, split_track = False)
    _ = pa.load_tissue_stress_results(folder_path=folder_path) 
    folder_path_1 = folder_path.joinpath("fake_example_3").resolve()
    with pytest.raises(FileNotFoundError) as error:
        pa.load_tissue_stress_results(folder_path=folder_path_1)
    assert error.typename == "FileNotFoundError"


def test_load_beat_info():
    folder_path = example_path("real_example_pillar_short")
    tissue_depth = 1
    pillar_orientation_vector = np.array([1,0])
    pillar_stiffnes = None
    pillar_profile = 'rectangular'
    pillar_modulus = 1.61
    pillar_width = 163
    pillar_thickness = 33.2 
    pillar_diameter = 40    
    pillar_length = 199
    force_location = 163
    length_scale = 1
    fps = 1
    fname = 'pillar1_'
    fname_t50 = 'pillar1_t50_'
    _ = pa.run_pillar_tracking(folder_path, tissue_depth, pillar_orientation_vector, pillar_stiffnes, pillar_profile, pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location, length_scale, fps, split_track = False)
    _ = pa.load_beat_peaks(folder_path=folder_path, fname=fname) 
    _ = pa.load_beat_width_info(folder_path=folder_path, fname=fname_t50) 
    folder_path_1 = folder_path.joinpath("fake_example_3").resolve()
    with pytest.raises(FileNotFoundError) as error:
        pa.load_beat_peaks(folder_path=folder_path_1)
    assert error.typename == "FileNotFoundError"
    with pytest.raises(FileNotFoundError) as error:
        pa.load_beat_width_info(folder_path=folder_path_1)
    assert error.typename == "FileNotFoundError"


def test_visualize_pillar_tracking():
    folder_path = example_path("real_example_pillar_short")
    folder_path_split = example_path("real_example_pillar_short_split")
    tissue_depth = 1
    pillar_orientation_vector = np.array([1,0])
    pillar_stiffnes = None
    pillar_profile = 'rectangular'
    pillar_modulus = 1.61
    pillar_width = 163
    pillar_thickness = 33.2 
    pillar_diameter = 40    
    pillar_length = 199
    force_location = 163
    length_scale = 1
    fps = 1
    _ = pa.run_pillar_tracking(folder_path, tissue_depth, pillar_orientation_vector, pillar_stiffnes, pillar_profile, pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location, length_scale, fps, split_track = False)
    saved_path = pa.visualize_pillar_tracking(folder_path, split_track = False)
    assert saved_path.is_file
    
    folder_path = example_path("real_example_pillar_short")
    _ = pa.run_pillar_tracking(folder_path_split, tissue_depth, pillar_orientation_vector, pillar_stiffnes, pillar_profile, pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location, length_scale, fps, split_track = True)
    saved_path = pa.visualize_pillar_tracking(folder_path_split, split_track = True)
    assert saved_path.is_file