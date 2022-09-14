import torch
import numpy as np
from print_plot import print_spectrogram,print_hist
from data_function import get_prior_phoneme_sepctrogram_info,get_prior_matrix_sepctrogram_info
from utils import process_text
import os
np_paths = ["v9_vv_mm_ground_truth_0.npy",
               "v9_vv_mm_predict_0.npy",
              "v9_vv_mm_matrix_B_abs_0.npy",
              "v9_vv_mm_matrix_A_abs_0.npy",
              "v9_vv_mm_g_velocity_abs_0.npy",
              "v9_vv_mm_g_pressure_abs_0.npy"] + \
            [r"v9_no_predict_0.npy",
              "v9_no_matrix_B_abs_0.npy",
              "v9_no_matrix_A_abs_0.npy",
              "v9_no_g_velocity_abs_0.npy",
              "v9_no_g_pressure_abs_0.npy"]

for p in np_paths:
    pic_name = p.split(".")[0]
    p = "./data/np_data/" + p
    print("Np data",p)
    is_mel = False
    spec = np.load(p)
    if "predict_0" in p:
        clim = [-12,0]
        is_mel=True
    if "ground_truth" in p:
        clim = [-12,0]
        is_mel=True
    if "matrix_A" in p:
        clim = [0,8]
        print_spectrogram(torch.tensor(spec.T),clim=[0,2], pic_name=pic_name + "_lowmagn", version="",
                          is_mel=is_mel)
    if "matrix_B" in p:
        clim = [0,2]
    if "pressure" in p:
        clim = [0,8]
    if "velocity" in p:
        clim = [0,0.1]
        print_spectrogram(torch.tensor(spec.T),ylim=[0,800], pic_name=pic_name+"_lowfreq", clim=clim, version="", is_mel=is_mel)
    if "rosen" in p:
        clim=[0,0.1]
    print_spectrogram(torch.tensor(spec.T), pic_name=pic_name,clim=clim,version="",is_mel=is_mel)



idx_to_i = {}
all_text = process_text(os.path.join("data", "train_12000.txt"))
top_n_idx = set()
for i in range(len(all_text)):
    idx = all_text[i].split("|")[0]
    idx_to_i[idx] = i


vv = get_prior_phoneme_sepctrogram_info(("LJ001-0014"))
#print(vv["LJ001-0001"])
print_spectrogram(torch.tensor(vv["LJ001-0014"].T),clim=[0,0.1],ylim=[0,10000],pic_name="rosen_prior_spec")
print_spectrogram(torch.tensor(vv["LJ001-0014"].T),clim=[0,0.1],ylim=[0,800],pic_name="rosen_prior_spec_lowfreq")
ma,mb = get_prior_matrix_sepctrogram_info(("LJ001-0014"),idx_to_i)
#print(ma["LJ001-0001"].size())
from print_plot import print_spectrogram
print_spectrogram(ma["LJ001-0014"],pic_name="combine_matrix_A_target_spec",clim=[0,8])
print_spectrogram(ma["LJ001-0014"],pic_name="combine_matrix_A_target_spec_lowmagn",clim=[0,2])
print_spectrogram(mb["LJ001-0014"],pic_name="combine_matrix_B_target_spec",clim=[0,2])



ground_truth = "./data/np_data/" + "v9_vv_mm_ground_truth_0.npy"
vvmm_pre = "./data/np_data/" + "v9_vv_mm_predict_0.npy"
no_pre = "./data/np_data/" + "v9_no_predict_0.npy"

ground_truth_spec = np.load(ground_truth)
vvmm_pre_spec = np.load(vvmm_pre)
no_pre_spec = np.load(no_pre)

mse_spec_no = np.power(no_pre_spec - ground_truth_spec,2)
mse_spec_vvmm = np.power(vvmm_pre_spec - ground_truth_spec,2)

print_spectrogram(torch.tensor(mse_spec_no.T),clim = [0,5],ylim=[0,10000],pic_name="squared_no_prior")
print_spectrogram(torch.tensor(mse_spec_vvmm.T),clim = [0,5], ylim=[0,10000],pic_name="suqared_vvmm_prior")


mse_no = np.mean(mse_spec_no)
mse_vvmm = np.mean(mse_spec_vvmm)

print(mse_no)
print(mse_vvmm)
