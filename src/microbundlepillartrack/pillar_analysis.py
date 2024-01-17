from microbundlepillartrack import image_analysis as ia
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import center_of_mass
from skimage.filters import threshold_otsu
from skimage import morphology
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import List, Tuple, Union
from pathlib import Path
from skimage import io
import numpy as np
import warnings
import glob
import os



def rename_folder(folder_path: Path, folder_name: str, new_folder_name: str) -> Path:
    """Given a path to a directory, a folder in the given directory, and a new folder name. 
    Will rename the folder."""
    original_folder_path = folder_path.joinpath(folder_name).resolve()
    new_folder_path = folder_path.joinpath(new_folder_name).resolve()
    if os.path.exists(original_folder_path) is False:
        warnings.warn('Original folder path does not exist! Cannot rename folder.', category = UserWarning, stacklevel=2)
        new_folder_path = folder_path
    elif os.path.exists(new_folder_path) == True:
        warnings.warn('%s folder already exists.'%(new_folder_name), category = UserWarning, stacklevel=2)
    else:
        os.rename(original_folder_path,new_folder_path)
    return new_folder_path


def compute_otsu_thresh(array: np.ndarray) -> Union[float, int]:
    """Given an image array. Will return the otsu threshold applied by skimage."""
    thresh = threshold_otsu(array)
    return thresh


def apply_otsu_thresh(array: np.ndarray) -> np.ndarray:
    """Given an image array. Will return a boolean numpy array with an otsu threshold applied."""
    thresh = compute_otsu_thresh(array)
    thresh_img = array > thresh
    return thresh_img


def invert_mask(mask: np.ndarray) -> np.ndarray:
    """Given a mask. Will return an inverted mask."""
    invert_mask = mask == 0
    return invert_mask


def close_region(array: np.ndarray, radius: int = 1) -> np.ndarray:
    """Given an array with a small hole. Will return a closed array."""
    footprint = morphology.disk(radius, dtype=bool)
    closed_array = morphology.binary_closing(array, footprint)
    return closed_array


def moving_mean(x: np.ndarray, N: Union[int,float]) -> np.ndarray:
    """Given a timeseries. Will return the moving mean with window 'N'."""
    m_mean = np.convolve(x, np.ones((N,))/N, mode='valid')
    return m_mean


def project_vector_timeseries(col_all: np.ndarray, row_all: np.ndarray, dir_vector: np.ndarray) -> np.ndarray:
    """Given tracked results and a unit vector. Will compute the projected results."""
    vector_transp = np.array([col_all,row_all]).T
    dir_vector_col_row = np.array([dir_vector[1],-1*dir_vector[0]])
    proj_vector_timeseries = np.dot(vector_transp,dir_vector_col_row)
    proj_vector_timeseries = proj_vector_timeseries.astype('float')
    return proj_vector_timeseries


def find_valleys(input_array: np.ndarray, init_prom: float = 0.001) -> np.ndarray:
    """Given a timeseries. Will find the valleys."""
    _, valley_props = find_peaks(-1*input_array, prominence=init_prom, width=0.5)
    valley_prominence = 0.5*np.mean(valley_props['prominences'])
    valleys, _ = find_peaks(np.concatenate(([min(-1*input_array)],-1*input_array)), prominence=valley_prominence)
    valleys -=1
    return valleys


def find_beat_peaks(input_array: np.ndarray, init_prom: float = 0.001) -> np.ndarray:
    """Given a timeseries. Will find the peaks."""
    _, peak_props = find_peaks(input_array, prominence=init_prom, width=0.5)
    peak_prominence = 0.5*np.mean(peak_props['prominences'])
    peaks, _ = find_peaks(np.concatenate(([min(input_array)],input_array)), prominence=peak_prominence)
    peaks -=1
    return peaks


def find_peak_widths(input_array: np.ndarray, peaks: np.ndarray, relative_height: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Given a timeseries. Will find the peak widths at the specified relative height."""
    width_info = peak_widths(input_array, peaks, relative_height)
    return width_info


def adjust_first_valley(folder_path: Path, valley_frame_num: int) -> List:
    """ Given a folder path of images. Will remove images prior to the specified "valley_image".
    The adjusted movie frames are saved in 'movie' folder. The original movie frames are saved in 'unadjusted_movie' folder."""
    # rename folder to retain original images
    unadjusted_imgs_folder = rename_folder(folder_path,"movie","unadjusted_movie")
    unadjusted_list_path = ia.image_folder_to_path_list(unadjusted_imgs_folder)
    # create a new "movie" folder to save adjusted frames
    adjusted_movie_folder = ia.create_folder(folder_path, "movie")
    number_unadjusted_images = len(unadjusted_list_path)
    # save adjusted images in new folder
    adjusted_img_list = []
    for ff in range(valley_frame_num, number_unadjusted_images):
        img = ia.read_tiff(unadjusted_list_path[ff])
        img = img.astype('uint16')
        fn = adjusted_movie_folder.joinpath("%04d.TIF"%(ff-valley_frame_num)).resolve()
        io.imsave(fn,img,check_contrast=False)
        adjusted_img_list.append(img)
    return adjusted_img_list


def check_frame0_valley(example_path: Path, tiff_list: List, proj_mean_disp_all: np.ndarray, valleys: np.ndarray) -> List:
    """Given folder path, of images, a list of tiff images, the tracked mean directional pillar displacement arrays, and the unit vector between both pillars. 
    Will check if frame 0 is a valley frame and will adjust the tiff list to start at a valley frame otherwise."""
    lower_idx = np.argmin(proj_mean_disp_all[valleys[0:2]])
    valleys_mean_proj_disp = np.mean(proj_mean_disp_all[valleys[0:2]])
    approx_peaks = (valleys[:-1] + valleys[1:])/2
    approx_peaks = approx_peaks.astype('int')

    if valleys_mean_proj_disp < -0.05 and example_path.joinpath("unadjusted_movie").resolve().exists() is False:
        warnings.warn('Input video does not start from a valley position.' 'It has been adjusted using the preprocessing function "adjust_first_valley" to start from frame %i.'%(valleys[lower_idx]-1),category = UserWarning, stacklevel=2)
        adjusted_img_list = adjust_first_valley(example_path,valleys[lower_idx]-1)
    else: 
        adjusted_img_list = tiff_list
    return adjusted_img_list


def prepare_valley_info(valleys: np.ndarray) -> np.ndarray:
    """Given a list of valley frame indices. Will reformat to indicate start and end of each beat."""
    info = []
    for kk in range(0, len(valleys) - 1):
        # beat number, start index wrt movie, end index wrt movie
        info.append([kk, valleys[kk], valleys[kk + 1]])
    return np.asarray(info)


def compute_pillar_secnd_moment_rectangular(pillar_width: float, pillar_thickness: float)-> float: 
    """Given pillar width and thickness in micrometers (um).
    Will compute the pillar (taken as a rectangular beam) second moment of area in (um)^4."""
    secnd_moment_area = (pillar_width*pillar_thickness**3)/12
    return secnd_moment_area


def compute_pillar_secnd_moment_circular(pillar_diameter: float)-> float: 
    """Given pillar diameter in micrometers (um).
    Will compute the pillar (taken as a circular beam) second moment of area in (um)^4."""
    secnd_moment_area = (np.pi*pillar_diameter**4)/64
    return secnd_moment_area


def compute_pillar_stiffnes(pillar_profile: str, pillar_modulus: float, pillar_width: float, 
                            pillar_thickness: float, pillar_diameter: float, pillar_length: float, 
                            force_location: float) -> float:
    """Given pillar material Elastic modulus (in MPa), width, thickness, length and force application location 
    in micrometers (um). Will compute the pillar stiffness in (uN/um)."""
    if pillar_profile == 'rectangular':
        I = compute_pillar_secnd_moment_rectangular(pillar_width, pillar_thickness)
    elif pillar_profile == 'circular':
        I = compute_pillar_secnd_moment_circular(pillar_diameter)
    else:
        print("Pillar_profile should be either 'rectangular' or 'circular'")
        I = 0
    pillar_stiffness = (6*pillar_modulus*I)/((force_location**2)*(3*pillar_length-force_location))
    return pillar_stiffness


def compute_pillar_force(pillar_stiffness: float, pillar_avg_deflection: np.ndarray, length_scale: float) -> np.ndarray:
    """Given pillar stiffness in (uN/um), pillar average deflection in pixels and a length scale 
    conversion from pixels to micrometers (um). Will compute pillar force in microNewtons (uN)."""
    pillar_force = pillar_stiffness*pillar_avg_deflection*length_scale
    return pillar_force


def compute_pillar_position_timeseries(tracker_0: np.ndarray, tracker_1: np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Given tracker arrays. Will return single timeseries of mean absolute displacement, 
    mean row displacement and mean column displacement."""
    mean_tracker_0 = np.mean(tracker_0,axis=0)
    mean_tracker_1 = np.mean(tracker_1,axis=0)
    
    mean_tracker_0_0 = np.ones(np.shape(mean_tracker_0))*mean_tracker_0[0]
    mean_disp_0_all = mean_tracker_0 - mean_tracker_0_0
    
    mean_tracker_1_0 = np.ones(np.shape(mean_tracker_1))*mean_tracker_1[0]
    mean_disp_1_all = mean_tracker_1 - mean_tracker_1_0
    
    disp_abs_mean = (mean_disp_0_all ** 2.0 + mean_disp_1_all ** 2.0) ** 0.5
    return disp_abs_mean, mean_disp_0_all, -1*mean_disp_1_all


def pillar_force_all_steps(pillar_mean_abs_disp: np.ndarray, pillar_mean_disp_row: np.ndarray, 
                           pillar_mean_disp_col: np.ndarray, pillar_stiffnes: float = None, pillar_profile: str = 'rectangular',
                           pillar_modulus: float = 1.61, pillar_width: float = 163, pillar_thickness: float = 33.2, 
                           pillar_diameter: float = 400, pillar_length: float = 199.3, force_location: float = 163,
                           length_scale: float = 1) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Given pillar material Elastic modulus (in MPa), width, thickness, length, force application location 
    in micrometers (um), pillar tracking results in pixels, and a length scale conversion from pixels to 
    micrometers (um). Will compute pillar force in microNewtons (uN) for all steps."""
    
    if pillar_stiffnes is not None:
        pillar_k = pillar_stiffnes
    else:
        pillar_k = compute_pillar_stiffnes(pillar_profile, pillar_modulus, pillar_width, pillar_thickness, pillar_diameter, pillar_length, force_location)

    pillar_F_row = compute_pillar_force(pillar_k,pillar_mean_disp_row,length_scale)
    pillar_F_col = compute_pillar_force(pillar_k,pillar_mean_disp_col,length_scale)
    pillar_F_abs = compute_pillar_force(pillar_k,pillar_mean_abs_disp,length_scale)

    return pillar_F_abs, pillar_F_row, pillar_F_col


def compute_pillar_velocity(pillar_mean_abs_disp: np.ndarray, length_scale: Union[int,float], fps: Union[int,float]) -> np.ndarray:
    """Given pillar mean absolute displacement, length scale (um/pixel) and fps (frames/seconds) . Will compute the pillar beating velocity in um/s."""
    pillar_mean_abs_disp_um = pillar_mean_abs_disp*length_scale
    time_step = 1/fps
    pillar_velocity = moving_mean(np.diff(pillar_mean_abs_disp_um)/time_step,3)
    return pillar_velocity


def compute_contraction_relaxation_peaks(pillar_velocity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Given pillar mean velocity. Will return the indices for peak contraction and relaxation velocities."""
    contraction_peaks_idx = find_beat_peaks(pillar_velocity,init_prom = np.max(abs(pillar_velocity))/10)
    relaxation_peaks_idx = find_valleys(pillar_velocity,init_prom = np.max(abs(pillar_velocity))/10)#[1:]
    if relaxation_peaks_idx[0] < contraction_peaks_idx[0]:
        relaxation_peaks_idx = relaxation_peaks_idx[1:]
    return contraction_peaks_idx, relaxation_peaks_idx


def save_pillar_velocity_results(*, folder_path: Path, pillar_velocity: np.ndarray, contraction_info: np.ndarray, relaxation_info: np.ndarray, fname: str = None) -> Tuple[Path, Path, Path]:
    """Given pillar velocity results in um/s. Will save as text file."""
    new_path = ia.create_folder(folder_path, "pillar_results")
    if fname is not None:
        file_path_res = new_path.joinpath(fname + "pillar_velocity.txt").resolve()
        np.savetxt(str(file_path_res), pillar_velocity)
        file_path_info_cont = new_path.joinpath(fname + "pillar_contraction_vel_info.txt").resolve()
        np.savetxt(str(file_path_info_cont), contraction_info)
        file_path_info_relax = new_path.joinpath(fname + "pillar_relaxation_vel_info.txt").resolve()
        np.savetxt(str(file_path_info_relax), relaxation_info)
    else:
        file_path_res = new_path.joinpath("pillar_velocity.txt").resolve()
        np.savetxt(str(file_path_res), pillar_velocity)
        file_path_info_cont = new_path.joinpath("pillar_contraction_vel_info.txt").resolve()
        np.savetxt(str(file_path_info_cont), contraction_info)
        file_path_info_relax = new_path.joinpath("pillar_relaxation_vel_info.txt").resolve()
        np.savetxt(str(file_path_info_relax), relaxation_info)
    return file_path_res, file_path_info_cont, file_path_info_relax


def get_mask_centroid(mask:np.ndarray) -> np.ndarray:
    centroid_coord = center_of_mass(mask)
    return np.array(centroid_coord)


def order_pillar_masks(pillar_1:np.ndarray, pillar_2:np.ndarray) -> List:
    """Given a list of region masks. Will order the masks from closest to farthest with respect to the x=0 axis (left edge)."""
    p1_center_col = get_mask_centroid(pillar_1)[1]
    p2_center_col = get_mask_centroid(pillar_2)[1]
    if p1_center_col < p2_center_col:
        ordered_masks = [pillar_1, pillar_2]
    else: 
        ordered_masks = [pillar_2, pillar_1]
    return ordered_masks


def find_vector_between_centroids(centroid1:np.ndarray, centroid2: np.ndarray) ->  np.ndarray:
    """Given the coordinates of two centroids. Will find the unit vector (row,column) between the first and the second centroid."""
    vector = np.asarray([centroid2[0]-centroid1[0],centroid2[1]-centroid1[1]],dtype=float)
    unit_vector = vector/np.linalg.norm(vector)
    return unit_vector


def save_orientation_info(folder_path: Path, vector: np.ndarray, angle: float) -> Path:
    """Given pillar masks orientation info. Will save in a text file."""
    res_folder_path = ia.create_folder(folder_path,"pillar_results")
    file_path = res_folder_path.joinpath("orientation_info.txt").resolve()
    orientation_info = np.asarray([vector[1],-1*vector[0],-1*angle])
    np.savetxt(str(file_path), orientation_info.reshape(1,-1))
    return file_path


def compute_midpt(pt_1: np.ndarray,pt_2: np.ndarray) -> np.ndarray:
    """Given the coordinates of two points. Will return the coordinates of the midpoint."""
    midpt = (pt_1 + pt_2)/2
    return midpt


def extract_ROI_for_width(img:np.ndarray, midpt: np.ndarray, pillar_col_dist: float, ratio_r: float, ratio_c: float) -> np.ndarray:
    """Given an image, midpoint between pillar centroids, the horizontal (column) distance between the pillar centroids, 
    and clipping ratios in the row and column directions. Will return a cropped image."""
    img_r, _ = img.shape
    strt_row = np.max([0,int(midpt[0]-ratio_r*pillar_col_dist)])
    end_row = np.min([img_r,int(midpt[0]+ratio_r*pillar_col_dist)])
    strt_col = int(midpt[1]-ratio_c*pillar_col_dist)
    end_col = int(midpt[1]+ratio_c*pillar_col_dist)
    img_ROI = img[strt_row:end_row,strt_col:end_col]
    return img_ROI


def find_tissue_width(folder_path, tiff_list: List, pillar_1_centroid: np.ndarray, pillar_2_centroid: np.ndarray) -> float:
    """Given an image sequence and pillar centroids. Will return the tissue width at the middle region."""
    # find midpoint of segment between the 2 pillar centroids
    pillar_midpt = compute_midpt(pillar_1_centroid,pillar_2_centroid)
    # find distance between the 2 pillar centroids 
    pillar_dist = np.sqrt((pillar_2_centroid[0] - pillar_1_centroid[0])**2 + (pillar_2_centroid[1] - pillar_1_centroid[1])**2)
    # find pillar orientation
    vector = find_vector_between_centroids(pillar_1_centroid,pillar_2_centroid)
    pillar_rot_mat, pillar_ang = ia.rot_vec_to_rot_mat_and_angle(vector)
    # rotate first valley frame to get tissue width (if necessary)
    _, rot_frame_0, trans_row, trans_col = ia.rotate_test_img(folder_path,tiff_list,pillar_ang,pillar_midpt[0],pillar_midpt[1],pillar_rot_mat)
    trans_pillar_midpt_row, trans_pillar_midpt_col  = ia.translate_points(pillar_midpt[0],pillar_midpt[1],trans_row,trans_col)
    trans_pillar_midpt = np.array([trans_pillar_midpt_row,trans_pillar_midpt_col], dtype=float)
    # isolate a middle region of the tissue to find the width
    tissue_mid_ROI = extract_ROI_for_width(rot_frame_0, trans_pillar_midpt, pillar_dist, 0.4, 1/6)
    mask_mid_ROI = apply_otsu_thresh(tissue_mid_ROI)
    mask_mid_ROI = invert_mask(mask_mid_ROI)
    mask_mid_ROI = close_region(mask_mid_ROI,10)
    # plot and save the tissue mask in the middle region on which tissue width calculations are based 
    plt.figure()
    plt.imshow(mask_mid_ROI, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(str(folder_path.joinpath('masks').resolve())+'/mid_tissue_mask.png',dpi=300)
    plt.close()

    mean_mid_ROI = np.mean(mask_mid_ROI, axis=1)
    diff_mean_mid_ROI = moving_mean(np.diff(mean_mid_ROI),6) 
    peaks = find_beat_peaks(abs(diff_mean_mid_ROI), init_prom = 0.01)
    peak_slope = diff_mean_mid_ROI[peaks]
    max_idx = np.argmax(peak_slope)
    min_idx = np.argmin(peak_slope)
    sorted_idx = np.sort([max_idx,min_idx])
    if len(peaks) < 2:
        warnings.warn('An approximate value of the microbundle width is outputted due to imaging noise or the microbundle being too close to the image border.', category = UserWarning, stacklevel=2)
        tissue_width = 2*abs(peaks[0] - tissue_mid_ROI.shape[0]/2)
    else: 
        tissue_width = abs(peaks[min_idx] - peaks[max_idx])
        if diff_mean_mid_ROI[peaks[sorted_idx[0]]] <= 0 or  diff_mean_mid_ROI[peaks[sorted_idx[1]]] >= 0:
            warnings.warn('The microbundle width measurement might not be accurate due to imaging noise or the microbundle being too close to the image border.', category = UserWarning, stacklevel=2)
    return tissue_width


def save_tissue_width_info(folder_path: Path, tissue_width: float, length_scale: float) -> Path:
    """Given tissue width info. Will save in a text file."""
    res_folder_path = ia.create_folder(folder_path,"pillar_results")
    file_path = res_folder_path.joinpath("tissue_width_info.txt").resolve()
    tissue_width_um = tissue_width*length_scale
    tissue_width_info = np.asarray([tissue_width,tissue_width_um])
    np.savetxt(str(file_path), tissue_width_info.reshape(1,-1))
    return file_path


def compute_tissue_stress_all_steps(pillar_abs_force: np.ndarray,tissue_width: float, tissue_depth: float, length_scale: float) -> float:
    """Given pillar absolute force, tissue width and depth, and length scale. Will compute tissue stress."""
    tissue_width_um = tissue_width*length_scale
    tissue_cross_sectional_area = tissue_width_um*tissue_depth
    tissue_stress = pillar_abs_force / tissue_cross_sectional_area
    return tissue_stress


def detect_irregular_beats(peaks:np.ndarray):
    """Given peaks. Will check for irregular beats."""
    diff_peaks = np.diff(peaks)
    mean_freq = np.mean(diff_peaks)
    if np.allclose(diff_peaks,mean_freq,atol=np.ceil(0.2*mean_freq)):
        pass
    else:
        warnings.warn('Irregular beats were detected!',category = UserWarning, stacklevel=2)


def find_slope(col_pts:np.ndarray, row_pts:np.ndarray) -> Tuple[float,float]: 
    """Given the column and row positions of a set of points. Will perform linear fitting and 
    return the slope and y-intercept."""   
    fit = np.polyfit(col_pts, row_pts, 1)
    slope = fit[0]  # Gradient
    y_inter = fit[1]  # y-intercept
    return slope, y_inter


def find_slope_0_intercept(col_pts:np.ndarray, row_pts:np.ndarray) -> Tuple[float,float]:
    """Given the column and row positions of a set of points. Will perform linear fitting with 
     y-intercept forced to 0 and return the slope and y-intercept (outputted for sanity check)."""   
    xx = np.vstack([col_pts, np.zeros(len(col_pts))]).T
    fit = np.linalg.lstsq(xx, row_pts, rcond=None)[0]
    slope = fit[0]  # Gradient
    y_inter = fit[1]  # y-intercept
    return slope, y_inter


def normalize_timeseries(timeseries:np.ndarray) -> np.ndarray:
    """Given a timeseries. Will scale it into the range [0,1]."""
    min_ts = np.min(timeseries)
    max_ts = np.max(timeseries)
    normalized_timeseries = (timeseries - min_ts)/(max_ts-min_ts)
    return normalized_timeseries


def detect_drift(mean_abs_disp:np.ndarray, peaks:np.ndarray, valleys:np.ndarray, split_track:bool):
    """Given tracked mean absolute displacement, """
    if np.max(mean_abs_disp) < 1:
        mean_abs_disp = normalize_timeseries(mean_abs_disp)
    if split_track:
        slope_peaks, _ = find_slope(peaks, mean_abs_disp[peaks])
        if abs(slope_peaks) > 0.0015:
            warnings.warn('High drift could not be eliminated by splitting due to large imgaging noise! Consider analyzing half of the movie only.',category = UserWarning, stacklevel=2)
    else:
        slope_valleys, _ = find_slope(valleys, mean_abs_disp[valleys])
        if abs(slope_valleys) > 0.0015:
            warnings.warn('High drift was detected during tracking! Consider setting "split" to "True".',category = UserWarning, stacklevel=2)


def save_pillar_position(*, folder_path: Path, tracker_col_all: List, tracker_row_all: List,
                         info: np.ndarray = None, split_track: bool = False, fname: str = None) -> List:
    """Given pillar tracking results. Will save as text files."""
    new_path = ia.create_folder(folder_path, "pillar_results")
    saved_paths = []
    if split_track:
        num_beats = info.shape[0]
        for kk in range(0, num_beats):
            tracker_row = tracker_row_all[kk]
            tracker_col = tracker_col_all[kk]
            if fname is not None:
                file_path = new_path.joinpath(fname + "beat%i_row.txt"%(kk)).resolve()
                saved_paths.append(file_path)
                np.savetxt(str(file_path), tracker_row)
                file_path = new_path.joinpath(fname + "beat%i_col.txt"%(kk)).resolve()
                saved_paths.append(file_path)
                np.savetxt(str(file_path), tracker_col)
            else:
                file_path = new_path.joinpath("beat%i_row.txt"%(kk)).resolve()
                saved_paths.append(file_path)
                np.savetxt(str(file_path), tracker_row)
                file_path = new_path.joinpath("beat%i_col.txt"%(kk)).resolve()
                saved_paths.append(file_path)
                np.savetxt(str(file_path), tracker_col)
    else:
        if fname is not None:
            file_path = new_path.joinpath(fname + "row.txt").resolve()
            saved_paths.append(file_path)
            np.savetxt(str(file_path), tracker_row_all)
            file_path = new_path.joinpath(fname + "col.txt").resolve()
            saved_paths.append(file_path)
            np.savetxt(str(file_path), tracker_col_all)
        else:
            file_path = new_path.joinpath("row.txt").resolve()
            saved_paths.append(file_path)
            np.savetxt(str(file_path), tracker_row_all)
            file_path = new_path.joinpath("col.txt").resolve()
            saved_paths.append(file_path)
            np.savetxt(str(file_path), tracker_col_all)
    if info is not None:
        if fname is not None:
            file_path = new_path.joinpath(fname + "info.txt").resolve()
            np.savetxt(str(file_path), info)
            saved_paths.append(file_path)       
        else:
            file_path = new_path.joinpath("info.txt").resolve()
            np.savetxt(str(file_path), info)
            saved_paths.append(file_path)
    return saved_paths


def save_pillar_force(*, folder_path: Path, pillar_force_abs: np.ndarray, pillar_force_row: np.ndarray, 
                         pillar_force_col: np.ndarray, fname: str = None) -> List:
    """Given pillar force results. Will save as text files."""
    new_path = ia.create_folder(folder_path, "pillar_results")
    saved_paths = [] 
    if fname is not None:
        file_path = new_path.joinpath(fname + "pillar_force_abs.txt").resolve()
        saved_paths.append(file_path)
        np.savetxt(str(file_path), pillar_force_abs)   
        file_path = new_path.joinpath(fname + "pillar_force_row.txt").resolve()
        saved_paths.append(file_path)
        np.savetxt(str(file_path), pillar_force_row)
        file_path = new_path.joinpath(fname + "pillar_force_col.txt").resolve()
        saved_paths.append(file_path)
        np.savetxt(str(file_path), pillar_force_col)
    else:
        file_path = new_path.joinpath("pillar_force_abs.txt").resolve()
        saved_paths.append(file_path)
        np.savetxt(str(file_path), pillar_force_abs)
        file_path = new_path.joinpath("pillar_force_row.txt").resolve()
        saved_paths.append(file_path)
        np.savetxt(str(file_path), pillar_force_row)
        file_path = new_path.joinpath("pillar_force_col.txt").resolve()
        saved_paths.append(file_path)
        np.savetxt(str(file_path), pillar_force_col)
    return saved_paths


def save_peaks(*, folder_path: Path, peaks: np.ndarray, fname: str = None) -> Path: 
    """Given computed peaks. Will save as text files."""
    new_path = ia.create_folder(folder_path, "pillar_results")
    if fname is not None:
        file_path = new_path.joinpath(fname + "beat_peaks.txt").resolve()
    else:
        file_path = new_path.joinpath("beat_peaks.txt").resolve()
    np.savetxt(str(file_path), peaks)
    return file_path


def save_beat_width_info(*, folder_path: Path, pillar_width_info: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] , fname: str = None) -> Path:
    """Given pillar contraction beat width results. Will save as text files."""
    new_path = ia.create_folder(folder_path, "pillar_results")
    if fname is not None:
        file_path = new_path.joinpath(fname + "beat_width_info.txt").resolve()
    else:
        file_path = new_path.joinpath("beat_width_info.txt").resolve()
    np.savetxt(str(file_path), pillar_width_info)
    return file_path


def save_tissue_stress(*, folder_path: Path, tissue_stress: np.ndarray) -> Path:
    """Given tissue stress results. Will save as text file."""
    new_path = ia.create_folder(folder_path, "pillar_results")
    file_path = new_path.joinpath("tissue_stress.txt").resolve()
    np.savetxt(str(file_path), tissue_stress)
    return file_path


def run_pillar_tracking(folder_path: Path, tissue_depth: float = 350, pillar_orientation_vector: np.ndarray = None, pillar_stiffnes: float = None, pillar_profile: str = 'rectangular', 
                        pillar_modulus: float = 1.61, pillar_width: float = 163, pillar_thickness: float = 33.2, 
                        pillar_diameter: float = 400, pillar_length: float = 199.3, force_location: float = 163, 
                        length_scale: float = 1, fps: Union[int, float] = 1, split_track: bool = False) -> Tuple[np.ndarray,np.ndarray,List]:
    """Given a folder path, tissue depth in micrometers (um), unit vector between pillars (optional), pillar material Elastic modulus (in MPa), width, thickness, length, force application 
    location in micrometers (um), and a length scale conversion from pixels to micrometers (um). Will perform tracking, 
    compute pillar force in microNewtons (uN) and save results as text files."""
    # read images and mask file
    movie_folder_path = folder_path.joinpath("movie").resolve()
    name_list_path = ia.image_folder_to_path_list(movie_folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    img_list_uint8 = ia.uint16_to_uint8_all(tiff_list)
    mask_folder_path = folder_path.joinpath("masks").resolve()
    mask_file_list = glob.glob(str(mask_folder_path) + "/*pillar_mask*.txt")
    mask_file_path_1 = mask_folder_path.joinpath("pillar_mask_1.txt").resolve()
    mask_file_path_2 = mask_folder_path.joinpath("pillar_mask_2.txt").resolve()
    mask_1 = ia.read_txt_as_mask(mask_file_path_1)
    mask_2 = ia.read_txt_as_mask(mask_file_path_2)
    l_mask, r_mask = order_pillar_masks(mask_1,mask_2)
    clip_fraction = 0.1
    # find centroids, midpoint, and the column distance of the pillar masks
    center_mask_l = get_mask_centroid(l_mask)
    center_mask_r = get_mask_centroid(r_mask)
    # check if first frame is a valley frame using left mask
    tracker_0, tracker_1 = ia.track_all_steps_with_adjust_param_dicts(img_list_uint8, l_mask, clip_fraction)
    _,mean_disp_all_0, mean_disp_all_1 = compute_pillar_position_timeseries(tracker_0,tracker_1)
    if pillar_orientation_vector is not None:
        vector = pillar_orientation_vector
    else:
    # find pillar orientation
        vector = find_vector_between_centroids(center_mask_l,center_mask_r)
    _, pillar_ang = ia.rot_vec_to_rot_mat_and_angle(vector)
    save_orientation_info(folder_path,vector,pillar_ang)
    # find projected mean displacement vector
    proj_mean_disp_all = project_vector_timeseries(mean_disp_all_0, mean_disp_all_1, vector)
    # find valleys
    valleys_adj = find_valleys(proj_mean_disp_all, init_prom=np.max(abs(proj_mean_disp_all))/10)
    # adjust image list to start from a valley frame
    adjusted_img_list = check_frame0_valley(folder_path, tiff_list, proj_mean_disp_all, valleys_adj)
    adjusted_img_list_uint8 = ia.uint16_to_uint8_all(adjusted_img_list)
    # find tissue width
    tissue_width = find_tissue_width(folder_path, adjusted_img_list_uint8, center_mask_l, center_mask_r)
    # save tissue width info
    save_tissue_width_info(folder_path, tissue_width, length_scale)
    # track each pillar
    abs_pillar_force_all = []
    for ml in range(len(mask_file_list)): 
        # load pillar masks
        mask_file_path = mask_folder_path.joinpath("pillar_mask_%i.txt"%(ml+1)).resolve()
        mask = ia.read_txt_as_mask(mask_file_path)
        # perform tracking using updated image list
        tracker_0, tracker_1 = ia.track_all_steps_with_adjust_param_dicts(adjusted_img_list_uint8, mask, clip_fraction)
        # perform timeseries analysis
        mean_abs_disp, mean_disp_all_0, mean_disp_all_1 = compute_pillar_position_timeseries(tracker_0,tracker_1)
        # updated valley positions based on adjusted frames
        valleys = find_valleys(mean_abs_disp, init_prom=np.max(mean_abs_disp)/10)
        # restructure valleys info   
        info = prepare_valley_info(valleys)
        if split_track:
            mean_abs_disp,_,_,_ = ia.compute_abs_position_timeseries(tracker_0,tracker_1)
            valleys = find_valleys(mean_abs_disp,  init_prom=np.max(mean_abs_disp)/10)
            info = prepare_valley_info(valleys)
            mean_abs_disp_all = []
            mean_disp_all_0_all = []
            mean_disp_all_1_all = []
            tracker_0_all, tracker_1_all = ia.split_tracking(tracker_0, tracker_1, info)
            num_beats = len(tracker_0_all)
            for nb in range(num_beats):
                tracker_0_beat = tracker_0_all[nb]
                tracker_1_beat = tracker_1_all[nb]
                mean_abs_disp_beat, mean_disp_all_0_beat, mean_disp_all_1_beat = compute_pillar_position_timeseries(tracker_0_beat,tracker_1_beat)
                mean_abs_disp_all.append(mean_abs_disp_beat)
                mean_disp_all_0_all.append(mean_disp_all_0_beat)
                mean_disp_all_1_all.append(mean_disp_all_1_beat)
            
            mean_abs_disp = [disp for disp_lst in mean_abs_disp_all for disp in disp_lst]
            mean_abs_disp = np.asarray(mean_abs_disp)
            mean_disp_all_0 = [disp_0 for disp_0_lst in mean_disp_all_0_all for disp_0 in disp_0_lst]
            mean_disp_all_0 = np.asarray(mean_disp_all_0)
            mean_disp_all_1 = [disp_1 for disp_1_lst in mean_disp_all_1_all for disp_1 in disp_1_lst]
            mean_disp_all_1 = np.asarray(mean_disp_all_1)
  
            saved_paths_pos = save_pillar_position(folder_path=folder_path, tracker_col_all=tracker_0_all, tracker_row_all=tracker_1_all, info=info, split_track = True, fname='pillar%i_'%(ml+1))
        else:  
            # save pillar tracking results
            saved_paths_pos = save_pillar_position(folder_path=folder_path, tracker_col_all=tracker_0, tracker_row_all=tracker_1, info=info, split_track = False, fname='pillar%i_'%(ml+1))
        # compute pillar force 
        pillar_force_all, pillar_row_force_all, pillar_col_force_all = pillar_force_all_steps(mean_abs_disp, mean_disp_all_0, mean_disp_all_1, pillar_stiffnes = pillar_stiffnes, pillar_profile = pillar_profile, pillar_modulus = pillar_modulus, pillar_width = pillar_width, pillar_thickness = pillar_thickness, pillar_diameter = pillar_diameter, pillar_length = pillar_length, force_location = force_location, length_scale = length_scale)
        # save pillar force results
        saved_paths_force = save_pillar_force(folder_path=folder_path, pillar_force_abs=pillar_force_all, pillar_force_row=pillar_row_force_all, pillar_force_col=pillar_col_force_all, fname='pillar%i_'%(ml+1))
        abs_pillar_force_all.append(pillar_force_all)
        # compute pillar velocity
        mean_pillar_velocity = compute_pillar_velocity(mean_abs_disp, length_scale, fps)
        contraction_peaks_idx, relaxation_peaks_idx = compute_contraction_relaxation_peaks(mean_pillar_velocity)
        # save pillar velocity results
        saved_paths_velocity = save_pillar_velocity_results(folder_path=folder_path, pillar_velocity=mean_pillar_velocity, contraction_info=contraction_peaks_idx, relaxation_info=relaxation_peaks_idx, fname='pillar%i_'%(ml+1))
        # Compute time intervals 
        pillar_peaks = find_beat_peaks(mean_abs_disp, init_prom=np.max(mean_abs_disp)/10)
        wdith_info_t50 = find_peak_widths(mean_abs_disp, pillar_peaks, 0.5)
        wdith_info_t80 = find_peak_widths(mean_abs_disp, pillar_peaks, 0.8)
        # save time intervals 
        saved_paths_peaks = save_peaks(folder_path=folder_path, peaks=pillar_peaks, fname='pillar%i_'%(ml+1))
        saved_paths_t50 = save_beat_width_info(folder_path=folder_path, pillar_width_info=wdith_info_t50, fname='pillar%i_t50_'%(ml+1))
        saved_paths_t80 = save_beat_width_info(folder_path=folder_path, pillar_width_info=wdith_info_t80, fname='pillar%i_t80_'%(ml+1))
        # detect high drift
        detect_drift(mean_abs_disp,pillar_peaks,valleys,split_track)
        # detect irregular beats
        detect_irregular_beats(pillar_peaks) 
    # compute pillar stress results
    dim_0 = np.max([len(abs_pillar_force_all[0]),len(abs_pillar_force_all[1])])
    comb_abs_pillar_force = np.ma.empty((dim_0,1,2))
    comb_abs_pillar_force.mask = True
    comb_abs_pillar_force[:abs_pillar_force_all[0].shape[0],0,0] = abs_pillar_force_all[0]
    comb_abs_pillar_force[:abs_pillar_force_all[1].shape[0],0,1] = abs_pillar_force_all[1]
    masked_avg_abs_pillar_force = comb_abs_pillar_force.mean(axis=2)
    avg_abs_pillar_force = np.ma.getdata(masked_avg_abs_pillar_force).ravel()
    tissue_stress = compute_tissue_stress_all_steps(avg_abs_pillar_force, tissue_width, tissue_depth, length_scale)
    # save pillar stress results
    saved_path_stress = save_tissue_stress(folder_path=folder_path, tissue_stress=tissue_stress)
    return mean_disp_all_0, mean_disp_all_1, saved_paths_pos


def load_pillar_tracking_results(folder_path: Path, split_track: bool = False, fname: str = "") -> Tuple[np.ndarray,np.ndarray]:
    """Given folder path. Will load pillar tracking results. If there are none, will return an error."""
    res_folder_path = folder_path.joinpath("pillar_results").resolve()
    if res_folder_path.exists() is False:
        raise FileNotFoundError("tracking results are not present.")
    if split_track:  
        num_files = len(glob.glob(str(res_folder_path) + "/" + fname + "beat*.txt"))
        num_beats = int((num_files) / 2)
        pillar_row_all = []
        pillar_col_all = []
        for kk in range(0, num_beats):
            pillar_row = np.loadtxt(str(res_folder_path) + "/" + fname + "beat%i_row.txt" % (kk))
            pillar_col = np.loadtxt(str(res_folder_path) + "/" + fname + "beat%i_col.txt" % (kk))
            pillar_row_all.append(pillar_row)
            pillar_col_all.append(pillar_col)
    else:
        pillar_row_all = np.loadtxt(str(res_folder_path) + "/" + fname + "row.txt")
        pillar_col_all = np.loadtxt(str(res_folder_path) + "/" + fname + "col.txt")   
    return pillar_row_all, pillar_col_all


def load_pillar_force_results(folder_path: Path, fname: str = "") -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Given folder path. Will load pillar tracking results. If there are none, will return an error."""
    res_folder_path = folder_path.joinpath("pillar_results").resolve()
    if res_folder_path.exists() is False:
        raise FileNotFoundError("tracking results are not present -- therefore pillar force results must not be present either")
    force_file_list = glob.glob(str(res_folder_path) + "/*pillar_force*")
    if len(force_file_list) == 0:
        raise FileNotFoundError("pillar force results are not present")
    pillar_abs_force_all = np.loadtxt(str(res_folder_path) + "/" + fname + "pillar_force_abs.txt")
    pillar_row_force_all = np.loadtxt(str(res_folder_path) + "/" + fname + "pillar_force_row.txt")
    pillar_col_force_all = np.loadtxt(str(res_folder_path) + "/" + fname + "pillar_force_col.txt")
    return pillar_abs_force_all, pillar_row_force_all, pillar_col_force_all


def load_pillar_velocity_results(folder_path: Path,  fname: str = "") -> np.ndarray:
    """Given folder path. Will load pillar velocity results. If there are none, will return an error."""
    res_folder_path = folder_path.joinpath("pillar_results").resolve()
    velocity_file_list = glob.glob(str(res_folder_path) + "/*pillar_velocity*")
    if len(velocity_file_list) == 0:
        raise FileNotFoundError("pillar velocity results are not present")
    pillar_mean_velocity_all = np.loadtxt(str(res_folder_path) + "/" + fname + "pillar_velocity.txt")
    return pillar_mean_velocity_all


def load_pillar_velocity_info(folder_path: Path,  fname: str = "") -> np.ndarray:
    """Given folder path. Will load pillar contraction and relaxation velocity info. If there are none, will return an error."""
    res_folder_path = folder_path.joinpath("pillar_results").resolve()
    velocity_info_file_list = glob.glob(str(res_folder_path) + "/*vel_info*")
    if len(velocity_info_file_list) == 0:
        raise FileNotFoundError("pillar velocity results are not present")
    pillar_cont_info = np.loadtxt(str(res_folder_path) + "/" + fname + "pillar_contraction_vel_info.txt")
    pillar_relax_info = np.loadtxt(str(res_folder_path) + "/" + fname + "pillar_relaxation_vel_info.txt")
    return pillar_cont_info.astype('int'), pillar_relax_info.astype('int')


def load_tissue_stress_results(folder_path: Path) -> np.ndarray:
    """Given folder path. Will load tissue stress results. If there are none, will return an error."""
    res_folder_path = folder_path.joinpath("pillar_results").resolve()
    stress_file_list = glob.glob(str(res_folder_path) + "/*tissue_stress*")
    if len(stress_file_list) == 0:
        raise FileNotFoundError("tissue stress results are not present")
    tissue_stress_all = np.loadtxt(str(res_folder_path) + "/tissue_stress.txt")
    return tissue_stress_all


def load_beat_peaks(folder_path: Path,  fname: str = "") -> np.ndarray:
    """Given folder path. Will load pillar peaks. If there are none, will return an error."""
    res_folder_path = folder_path.joinpath("pillar_results").resolve()
    width_info_file_list = glob.glob(str(res_folder_path) + "/*beat_peaks*")
    if len(width_info_file_list) == 0:
        raise FileNotFoundError("pillar contraction results are not present")
    pillar_beat_peaks = np.loadtxt(str(res_folder_path) + "/" + fname + "beat_peaks.txt")
    return pillar_beat_peaks.astype('int')


def load_beat_width_info(folder_path: Path,  fname: str = "") -> np.ndarray:
    """Given folder path. Will load pillar contraction and relaxation beat width info. If there are none, will return an error."""
    res_folder_path = folder_path.joinpath("pillar_results").resolve()
    width_info_file_list = glob.glob(str(res_folder_path) + "/*beat_width_info*")
    if len(width_info_file_list) == 0:
        raise FileNotFoundError("pillar contraction results are not present")
    pillar_beat_width_info = np.loadtxt(str(res_folder_path) + "/" + fname + "beat_width_info.txt")
    return pillar_beat_width_info


def visualize_pillar_tracking(folder_path: Path, length_scale: float = 1, fps: Union[int, float] = 1, split_track: bool = False) -> Path:
    """Given a folder path where tracking has already been run. Will save visualizations."""
    vis_folder_path = ia.create_folder(folder_path, "pillar_visualizations")
    mask_folder_path = folder_path.joinpath("masks").resolve()
    mask_file_list = glob.glob(str(mask_folder_path) + "/*pillar_mask*.txt")
    # Visualize pillar displacement results
    color_lst = ['dodgerblue','firebrick','lightcoral','lightskyblue']
    plt.figure()
    for ml in range(len(mask_file_list)): 
        # load pillar tracking results
        pillar_tracker_row, pillar_tracker_col = load_pillar_tracking_results(folder_path,split_track,fname='pillar%i_'%(ml+1))
        if split_track:
            num_beats = len(pillar_tracker_row)
            mean_abs_disp_all = []
            for nb in range(num_beats):
                tracker_0_beat = pillar_tracker_col[nb]
                tracker_1_beat = pillar_tracker_row[nb]
                mean_abs_disp_beat, mean_disp_all_0_beat, mean_disp_all_1_beat = compute_pillar_position_timeseries(tracker_0_beat,tracker_1_beat)
                mean_abs_disp_all.append(mean_abs_disp_beat)
            mean_abs_disp = [disp for disp_lst in mean_abs_disp_all for disp in disp_lst]
            mean_abs_disp = np.asarray(mean_abs_disp)
        else:
            mean_abs_disp, _, _ = compute_pillar_position_timeseries(pillar_tracker_col, pillar_tracker_row)
    
        plt.plot(mean_abs_disp, color = color_lst[ml], label='pillar %i'%(ml+1))
    plt.ylabel(r'pillar mean absolute displacement (pixels)')
    plt.xlabel('frame')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.savefig(str(vis_folder_path)+'/pillar_mean_absolute_displacement.pdf', format='pdf')
    # plt.savefig(str(vis_folder_path)+'/pillar_mean_absolute_displacement.svg', format='svg')
    plt.close()
    plt.figure()
    count = 1
    for ml in range(len(mask_file_list)): 
        pillar_tracker_row, pillar_tracker_col = load_pillar_tracking_results(folder_path,split_track,fname='pillar%i_'%(ml+1))
        
        if split_track:
            num_beats = len(pillar_tracker_row)
            mean_disp_all_0_all = []
            mean_disp_all_1_all = []
            for nb in range(num_beats):
                tracker_0_beat = pillar_tracker_col[nb]
                tracker_1_beat = pillar_tracker_row[nb]
                _, mean_disp_all_0_beat, mean_disp_all_1_beat = compute_pillar_position_timeseries(tracker_0_beat,tracker_1_beat)
                mean_disp_all_0_all.append(mean_disp_all_0_beat)
                mean_disp_all_1_all.append(mean_disp_all_1_beat)
            
            mean_disp_all_col = [disp_0 for disp_0_lst in mean_disp_all_0_all for disp_0 in disp_0_lst]
            mean_disp_all_col = np.asarray(mean_disp_all_col)
            mean_disp_all_row = [disp_1 for disp_1_lst in mean_disp_all_1_all for disp_1 in disp_1_lst]
            mean_disp_all_row = np.asarray(mean_disp_all_row)
        else:
            _, mean_disp_all_col, mean_disp_all_row = compute_pillar_position_timeseries(pillar_tracker_col,pillar_tracker_row)
        
        plt.plot(mean_disp_all_col, color = color_lst[ml], label='pillar %i column (horizontal)'%(ml+1))
        plt.plot(mean_disp_all_row, color = color_lst[-count], label='pillar %i row (vertical)'%(ml+1))
        count +=1
        
    plt.ylabel(r'pillar mean displacement (pixels)')
    plt.xlabel('frame')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.savefig(str(vis_folder_path)+'/pillar_directional_displacement.pdf', format='pdf')
    plt.close()

    # Visualize pillar force results
    plt.figure()
    for ml in range(len(mask_file_list)): 
        # load pillar force results
        all_pillar_force, _, _ = load_pillar_force_results(folder_path,fname='pillar%i_'%(ml+1))
        plt.plot(all_pillar_force, color = color_lst[ml], label='pillar %i'%(ml+1))

    plt.ylabel(r'pillar absolute force ($\mu$N)')
    plt.xlabel('frame')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.savefig(str(vis_folder_path)+'/pillar_force_absolute.pdf', format='pdf')
    # plt.savefig(str(vis_folder_path)+'/pillar_force_absolute.svg', format='svg')
    plt.close()

    # Visualize pillar temporal results
    time_scale = 1/fps
    bar_width = 0.4
    bar_width_incr = 0
    count_bar = 0
    color_lst_bar = ['navy','slategrey','darkred','rosybrown']
    y_label = ('Contraction', 'Relaxation')
    y_pos = np.arange(len(y_label))
    plt.figure()
    fig, axs = plt.subplots(2, 1, figsize=(5,8), gridspec_kw={'height_ratios': [2.5, 1], 'hspace': .36})
    for ml in range(len(mask_file_list)): 
        pillar_velocity = load_pillar_velocity_results(folder_path,fname='pillar%i_'%(ml+1))
        pillar_cont_peaks, pillar_relax_peaks = load_pillar_velocity_info(folder_path,fname='pillar%i_'%(ml+1))
        pillar_cont_velocity = pillar_velocity[pillar_cont_peaks]
        pillar_rel_velocity = pillar_velocity[pillar_relax_peaks]
        avg_cont_vel = np.mean(pillar_cont_velocity)
        avg_rel_vel = np.mean(pillar_rel_velocity)
        avg_peak_vel = np.array((avg_cont_vel,avg_rel_vel))

        axs[0].plot(time_scale*np.arange(len(pillar_velocity)),pillar_velocity,color=color_lst[ml],zorder=-1, label='pillar%i'%(ml+1))
        axs[0].scatter(pillar_cont_peaks*time_scale,pillar_cont_velocity, color=color_lst_bar[count_bar], marker='o')
        axs[0].scatter(pillar_relax_peaks*time_scale,pillar_rel_velocity,  color=color_lst_bar[count_bar+1],  marker='o')

        axs[1].barh(y_pos+bar_width_incr, abs(avg_peak_vel), align='center', color =color_lst_bar[count_bar:count_bar+2], height=bar_width)
        axs[1].scatter(abs(pillar_cont_velocity), (y_pos[0]+bar_width_incr)*np.ones(len(pillar_cont_velocity)), color ='dimgrey', marker='x')
        axs[1].scatter(abs(pillar_rel_velocity), (y_pos[1]+bar_width_incr)*np.ones(len(pillar_rel_velocity)), color ='dimgrey', marker='x')
        
        bar_width_incr += bar_width
        count_bar += 2

    axs[0].set_xlabel('time (s)')
    axs[0].set_ylabel(r'pillar velocity ($\mu$m/s)')
    handles, labels = axs[0].get_legend_handles_labels()

    axs[1].set_yticks(y_pos+bar_width/2, labels=y_label)
    axs[1].invert_yaxis()  # labels read top-to-bottom
    axs[1].set_xlabel(r'absolute velocity ($\mu$m/s)')
    peaks_handles = Line2D([0], [0], marker='o', markeredgecolor='k', markerfacecolor='k', linestyle='')
    plt.legend([handles[0], handles[1], peaks_handles], [labels[0], labels[1], 'peaks'] ,loc='center', bbox_to_anchor=(0.52, 0.33), bbox_transform=fig.transFigure, ncol=3)    
    plt.savefig(str(vis_folder_path)+'/pillar_velocity_results.pdf', bbox_inches='tight', format='pdf')
    # plt.savefig(str(vis_folder_path)+'/pillar_velocity_results.svg', format='svg')
    plt.close()

    # Visualize time intervals 
    time_scale = 1/fps
    bar_width = 0.4
    bar_width_incr = 0
    color_lst_disp = ['dodgerblue','firebrick']
    color_lst_time = ['navy','darkred']
    y_label = (r't$_{80}$', r't$_{50}$')
    y_pos = np.arange(len(y_label))
    plt.figure()
    fig, axs = plt.subplots(2, 1, figsize=(5,9), gridspec_kw={'height_ratios': [2.5, 1], 'hspace': .4})
    for ml in range(len(mask_file_list)): 
        # load pillar tracking results
        pillar_tracker_row, pillar_tracker_col = load_pillar_tracking_results(folder_path,split_track,fname='pillar%i_'%(ml+1))
        if split_track:
            num_beats = len(pillar_tracker_row)
            mean_abs_disp_all = []
            for nb in range(num_beats):
                tracker_0_beat = pillar_tracker_col[nb]
                tracker_1_beat = pillar_tracker_row[nb]
                mean_abs_disp_beat, mean_disp_all_0_beat, mean_disp_all_1_beat = compute_pillar_position_timeseries(tracker_0_beat,tracker_1_beat)
                mean_abs_disp_all.append(mean_abs_disp_beat)

            mean_abs_disp = [disp for disp_lst in mean_abs_disp_all for disp in disp_lst]
            mean_abs_disp = np.asarray(mean_abs_disp)
        else:
            mean_abs_disp, _, _ = compute_pillar_position_timeseries(pillar_tracker_row,pillar_tracker_col)
        
        mean_abs_disp_metric = mean_abs_disp*length_scale
        peaks = load_beat_peaks(folder_path, fname='pillar%i_'%(ml+1))
        width_info_t50 = load_beat_width_info(folder_path, fname='pillar%i_t50_'%(ml+1))
        width_info_t80 = load_beat_width_info(folder_path, fname='pillar%i_t80_'%(ml+1))

        mean_t80 = np.mean(width_info_t80[0])
        mean_t50 = np.mean(width_info_t50[0])

        mean_t80_t50 = np.array([mean_t80,mean_t50])

        axs[0].plot(time_scale*np.arange(len(mean_abs_disp_metric)),mean_abs_disp_metric, c=color_lst_disp[ml], label='pillar %i'%(ml+1),zorder=-1)
        axs[0].scatter(time_scale*peaks, mean_abs_disp_metric[peaks], marker="o", c=color_lst_time[ml])
        axs[0].scatter(width_info_t80[2]*time_scale,width_info_t80[1]*length_scale,marker='>', c='white', edgecolors=color_lst_time[ml])
        axs[0].scatter(width_info_t80[3]*time_scale,width_info_t80[1]*length_scale,marker='>', c='white', edgecolors=color_lst_time[ml])
        axs[0].scatter(width_info_t50[2]*time_scale,width_info_t50[1]*length_scale,marker='<', c=color_lst_time[ml])
        axs[0].scatter(width_info_t50[3]*time_scale,width_info_t50[1]*length_scale,marker='<', c=color_lst_time[ml])

        axs[1].barh(y_pos+bar_width_incr, mean_t80_t50*time_scale, align='center', color = [color_lst_time[ml],'w'], edgecolor=color_lst_time[ml], height=bar_width, zorder=-1)
        axs[1].scatter(width_info_t80[0]*time_scale, (y_pos[0]+bar_width_incr)*np.ones(len(width_info_t80[0])), color ='dimgrey', marker='x')
        axs[1].scatter(width_info_t50[0]*time_scale, (y_pos[1]+bar_width_incr)*np.ones(len(width_info_t50[0])), color ='dimgrey', marker='x')

        bar_width_incr += bar_width
    
    axs[0].set_xlabel('time(s)')
    axs[0].set_ylabel(r'mean absolute displacement ($\mu$m)')

    axs[1].set_yticks(y_pos+bar_width/2, labels=y_label)
    axs[1].invert_yaxis()  # labels read top-to-bottom
    axs[1].set_xlabel(r'time interval (s)')

    handles, labels = axs[0].get_legend_handles_labels()
    peaks = Line2D([0], [0], marker='o', markeredgecolor='k', markerfacecolor='k', linestyle='')
    l_t80 = Line2D([0], [0], marker='>', markeredgecolor='k', markerfacecolor='w', linestyle='')
    l_t50 = Line2D([0], [0], marker='<', markeredgecolor='k', markerfacecolor='k', linestyle='')
    plt.legend([handles[0],l_t50,handles[1],peaks,l_t80], [labels[0],r't$_{50}$',labels[1],'peaks',r't$_{80}$'],loc='center', bbox_to_anchor=(0.52, 0.335), bbox_transform=fig.transFigure, ncol=3)
    plt.savefig(str(vis_folder_path)+'/pillar_time_intervals.pdf', bbox_inches='tight', format='pdf')
    # plt.savefig(str(vis_folder_path)+'/pillar_time_intervals.svg', format='svg')
    plt.close()

    # Visualize tissue stress results
    tissue_stress_all = load_tissue_stress_results(folder_path)
    tissue_stress_all_kpa = tissue_stress_all*(10**3)
    plt.figure()
    plt.plot(tissue_stress_all_kpa, color = 'k')
    plt.ylabel(r'tissue stress (kPa)')
    plt.xlabel('frame')
    plt.tight_layout()
    plt.savefig(str(vis_folder_path)+'/tissue_stress.pdf', format='pdf')
    # plt.savefig(str(vis_folder_path)+'/tissue_stress.svg', format='svg')
    plt.close()
    return vis_folder_path