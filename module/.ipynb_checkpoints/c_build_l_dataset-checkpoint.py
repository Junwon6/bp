import build_l_dataset.d00_s01 as d00_s01
import build_l_dataset.d02_s01 as d02_s01

def build_l_dataset(d_id, s_id, s_len, n_fold):
    if (d_id == 'd00') & (s_id == 's00'):
        return

    elif (d_id == 'd00') & (s_id == 's01'):
        l_path = '../_dataset/open_dataset'              # load dataset path
        s_path = f'./l-dataset/{d_id}/{s_id}'            # save dataset path
        v_fname = 'valid_uni_bvp_boundary.csv'           # validation boundary file name
        f_name = ['rise_time', 'pwa', 'pwd']             # feature name

        t_dataset, t_idx = d00_s01.build_case_l_dataset(s_len, n_fold, l_path, s_path, v_fname, f_name)
        d00_s01.build_total_l_dataset(t_dataset, t_idx, s_path, f_name, s_len)

    if (d_id == 'd02') & (s_id == 's01'):
        l_path = '../_dataset/collected_dataset'         # load dataset path
        s_path = f'./l-dataset/{d_id}/{s_id}'            # save dataset path
        f_name = ['rise_time', 'pwa', 'pwd']             # feature name
        time = 30

        t_dataset, t_idx = d02_s01.build_case_l_dataset(s_len, n_fold, l_path, s_path, time, f_name)
        d02_s01.build_total_l_dataset(t_dataset, t_idx, n_fold, s_path, f_name, s_len)
        return

        
if __name__ == "__main__":
    d_id = 'd00'
    s_id = 's01'
    s_len = 100
    n_fold = 5
    print(d00_s01.build_total_l_dataset)
    # build_l_dataset(d_id, s_id, s_len, n_fold)