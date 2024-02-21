from pathlib import Path
import numpy as np
import csv


def txt_results_to_mean_max_csv(folder_path: Path, fps: float) -> Path:
    """Given the path of the 'pillar_results' folder and frames/sec (fps). Will output the 
    maximum or the mean output at the beat peaks and save them in a '.csv' file."""

    output_path = str(folder_path) + '/summary_results.csv'
    header = ['P1_max_peak_abs_force_(uN)', 'P1_mean_peak_abs_force_(uN)', 'P1_max_cont_vel_(um/s)', 'P1_mean_cont_vel_(um/s)', 'P1_max_rel_vel_(um/s)', 'P1_mean_rel_vel_(um/s)',
            'P1_mean_t50_width_(s)', 'P1_mean_t80_width_(s)', 'P2_max_peak_abs_force_(uN)', 'P2_mean_peak_abs_force_(uN)', 'P2_max_cont_vel_(um/s)', 'P2_mean_cont_vel_(um/s)', 
            'P2_max_rel_vel_(um/s)', 'P2_mean_rel_vel_(um/s)', 'P2_mean_t50_width_(s)', 'P2_mean_t80_width_(s)', 'max_peak_stress_(MPa)', 'mean_peak_stress_(MPa)']
    all_results = []
    for ii in range(2):
        absolute_force_results = np.loadtxt(str(folder_path)+'/pillar%i_pillar_force_abs.txt'%(ii+1))
        velocity_results = np.loadtxt(str(folder_path)+'/pillar%i_pillar_velocity.txt'%(ii+1))
        t50_results = np.loadtxt(str(folder_path)+'/pillar%i_t50_beat_width_info.txt'%(ii+1))
        t80_results = np.loadtxt(str(folder_path)+'/pillar%i_t80_beat_width_info.txt'%(ii+1))
        beat_peaks = np.loadtxt(str(folder_path)+'/pillar%i_beat_peaks.txt'%(ii+1)).astype(np.int64)
        contraction_peaks = np.loadtxt(str(folder_path)+'/pillar%i_pillar_contraction_vel_info.txt'%(ii+1)).astype(np.int64)
        relaxation_peaks = np.loadtxt(str(folder_path)+'/pillar%i_pillar_relaxation_vel_info.txt'%(ii+1)).astype(np.int64)

        absolute_force_peaks = absolute_force_results[beat_peaks]
        contraction_velocity_peaks = velocity_results[contraction_peaks]
        relaxation_velocity_peaks = abs(velocity_results[relaxation_peaks])
        t50_width = t50_results[0]/fps
        t80_width = t80_results[0]/fps
        
        max_absolute_force_peaks = np.max(absolute_force_peaks)
        mean_absolute_force_peaks = np.mean(absolute_force_peaks)

        max_contraction_velocity_peaks = np.max(contraction_velocity_peaks)
        max_relaxation_velocity_peaks = np.max(relaxation_velocity_peaks)

        mean_contraction_velocity_peaks = np.mean(contraction_velocity_peaks)
        mean_relaxation_velocity_peaks = np.mean(relaxation_velocity_peaks)

        mean_t50_width = np.mean(t50_width)
        mean_t80_width = np.mean(t80_width)

        results_list = [max_absolute_force_peaks, mean_absolute_force_peaks, max_contraction_velocity_peaks, 
                        mean_contraction_velocity_peaks, max_relaxation_velocity_peaks, mean_relaxation_velocity_peaks,
                        mean_t50_width, mean_t80_width]
        all_results.extend(results_list)

    stress_results = np.loadtxt(str(folder_path)+'/tissue_stress.txt')

    stress_peaks = stress_results[beat_peaks]
    max_stress_peaks = np.max(stress_peaks)
    mean_stress_peaks = np.mean(stress_peaks)
    stress_results = [max_stress_peaks,mean_stress_peaks]
    all_results.extend(stress_results)

    with open(output_path, 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(hh for hh in header)
        writer.writerow(all_results)
        return Path(output_path)