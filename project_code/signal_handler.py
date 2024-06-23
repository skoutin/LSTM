## μόλις πάρεις τα τελικά δεδομένα μη χαλάσεις το tranpy, αλλά φτιάξε ένα νέο module το signal_handler μόλις πάρεις τα τελικά δεδομένα

def sdirs (name):
    ### WT1_1in6= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1376.7 channel 1 (B2R)/satb1_1376.7_4_channel1.mat'
    WT1_1in6= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1376.7 channel 1 (B2R)/satb1_1376.7_control_channel1.mat' 
    WT1_2in6= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1376.7 channel 1 (B2R)/satb1_1376.7_first 20 min in 0 Mg_channel1.mat'
    WT1_3in6= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1376.7 channel 1 (B2R)/satb1_1376.7_second 20 min in 0 Mg_channel1.mat'
    WT1_4in6 ='D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1376.7 channel 1 (B2R)/satb1_1376.7_third 20 min in 0 Mg_channel1.mat'
    WT1_5in6= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1376.7 channel 1 (B2R)/satb1_1376.7_fourth_channel1.mat'
    WT1_6in6= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1376.7 channel 1 (B2R)/satb1__1376.7_fifth 20 min in 0 Mg_channel1.mat'

    WT2_1in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1376.7 channel 2 (B2R)/satb1_1376.7_control_channel2.mat'
    WT2_2in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1376.7 channel 2 (B2R)/satb1_1376.7_first 20 min in 0 Mg_channel2.mat'
    WT2_3in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1376.7 channel 2 (B2R)/satb1_1376.7_second 20 min in 0 Mg_channel2.mat'
    WT2_4in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1376.7 channel 2 (B2R)/satb1_1376.7_third 20 min in 0 Mg_channel2.mat'
    WT2_5in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1376.7 channel 2 (B2R)/satb1_1376.7_fourth 20 min in 0 Mg_channel2.mat'

    WT3_1in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1451.4 channel2/satb1_1451.4_control_channel2.mat'
    WT3_2in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1451.4 channel2/satb1_1451.4_first 20 min in 0 Mg_channel2.mat'
    WT3_3in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1451.4 channel2/satb1_1451.4_second 20 min in 0 Mg_channel2_Nikos.mat'
    WT3_4in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1451.4 channel2/satb1_1451.4_third 20 min in 0 Mg_channel2.mat'
    WT3_5in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1451.4 channel2/satb1_1451.4_fourth 20 min in 0 Mg_channel2.mat'

    WT4_1in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1451.4 channel4/satb1_1451.4_control_channel4.mat'
    WT4_2in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1451.4 channel4/satb1_1451.4_first 20 min in 0 Mg_channel4.mat'
    WT4_3in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1451.4 channel4/satb1_1451.4_second 20 min in 0 Mg_channel4_Nikos.mat'
    WT4_4in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1451.4 channel4/satb1_1451.4_third 20 min in 0 Mg_channel4.mat'
    WT4_5in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1451.4 channel4/satb1_1451.4_fourth 20 min in 0 Mg_channel4.mat'

    WT5_1in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1494.2 channel1/satb1_1494.2_control_channel1.mat'
    WT5_2in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1494.2 channel1/satb1_1494.2_first 20 min in 0 Mg_channel1.mat'
    WT5_3in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1494.2 channel1/satb1_1494.2_second 20 min in 0 Mg_channel1.mat'
    WT5_4in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1494.2 channel1/satb1_1494.2_third 20 min in 0 Mg_channel1.mat'
    WT5_5in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1494.2 channel1/satb1_1494.2_fourth 20 min in 0 Mg_channel1.mat'

    WT6_1in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1494.2 channel2/satb1_1494.2_control_channel2.mat'
    WT6_2in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1494.2 channel2/satb1_1494.2_first 20 min in 0 Mg_channel2.mat'
    WT6_3in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1494.2 channel2/satb1_1494.2_second 20 min in 0 Mg_channel2.mat'
    WT6_4in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1494.2 channel2/satb1_1494.2_third 20 min in 0 Mg_channel2.mat'
    WT6_5in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1494.2 channel2/satb1_1494.2_fourth 20 min in 0 Mg_channel2.mat'

    Mu1_1in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/mutant_0Mg/1375.5 channel 1 (F2R)/satb1_1375.5_control_channel1.mat'
    Mu1_2in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/mutant_0Mg/1375.5 channel 1 (F2R)/satb1_1375.5_first 20 min in 0 Mg_channel1.mat'
    Mu1_3in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/mutant_0Mg/1375.5 channel 1 (F2R)/satb1_1375.5_second 20 min in 0 Mg_channel1.mat'
    Mu1_4in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/mutant_0Mg/1375.5 channel 1 (F2R)/satb1_1375.5_third 20 min in 0 Mg_channel1.mat'
    Mu1_5in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/mutant_0Mg/1375.5 channel 1 (F2R)/satb1_1375.5_fourth 20 min in 0 Mg_channel1.mat'

    Mu2_1in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/mutant_0Mg/1473.1 channel4/satb1_1473.1_control_channel4.mat'
    Mu2_2in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/mutant_0Mg/1473.1 channel4/satb1_1473.1_first 20 min in 0 Mg_channel4.mat'
    Mu2_3in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/mutant_0Mg/1473.1 channel4/satb1_1473.1_second 20 min in 0 Mg_channel4.mat'
    Mu2_4in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/mutant_0Mg/1473.1 channel4/satb1_1473.1_third 20 min in 0 Mg_channel4.mat'
    Mu2_5in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/mutant_0Mg/1473.1 channel4/satb1_1473.1_fourth 20 min in 0 Mg_channel4.mat'

    Mu3_1in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/mutant_0Mg/1476.7 channel4/satb1_1476.7_control_channel4.mat'
    Mu3_2in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/mutant_0Mg/1476.7 channel4/satb1_1476.7_first 20 min in 0 Mg_channel4.mat'
    Mu3_3in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/mutant_0Mg/1476.7 channel4/satb1_1476.7_second 20 min in 0 Mg_channel4.mat'
    Mu3_4in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/mutant_0Mg/1476.7 channel4/satb1_1476.7_third 20 min in 0 Mg_channel4.mat'
    Mu3_5in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/mutant_0Mg/1476.7 channel4/satb1_1476.7_fourth 20 min in 0 Mg_channel4.mat'

    WT1_4AP_1in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_4ΑΡ/1580.4 channel 2/4AP_1580.4_control_channel2.mat'
    WT1_4AP_2in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_4ΑΡ/1580.4 channel 2/4AP_1580.4_1_channel2.mat'
    WT1_4AP_3in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_4ΑΡ/1580.4 channel 2/4AP_1580.4_2_channel2.mat'
    WT1_4AP_4in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_4ΑΡ/1580.4 channel 2/4AP_1580.4_3_channel2.mat'
    WT1_4AP_5in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_4ΑΡ/1580.4 channel 2/4AP_1580.4_3_channel2.mat'

    WT2_4AP_1in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_4ΑΡ/1677.3 channel4/4AP_1677.3_control_channel4.mat'
    WT2_4AP_2in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_4ΑΡ/1677.3 channel4/4AP_1677.3_1_channel4.mat'
    WT2_4AP_3in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_4ΑΡ/1677.3 channel4/4AP_1677.3_2_channel4.mat'
    WT2_4AP_4in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_4ΑΡ/1677.3 channel4/4AP_1677.3_3_channel4.mat'
    WT2_4AP_5in5= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_4ΑΡ/1677.3 channel4/per.mat'  # αυτό είναι όντως το 5ο αρχείο; κάντο plot για να δεις

    dirs={'WT1_1in6':WT1_1in6, 'WT1_2in6':WT1_2in6, 'WT1_3in6':WT1_3in6, 'WT1_4in6':WT1_4in6, 'WT1_5in6':WT1_5in6, 'WT1_6in6':WT1_6in6, 
          'WT2_1in5':WT2_1in5, 'WT2_2in5':WT2_2in5, 'WT2_3in5':WT2_3in5, 'WT2_4in5':WT2_4in5, 'WT2_5in5':WT2_5in5,
          'WT3_1in5':WT3_1in5, 'WT3_2in5':WT3_2in5, 'WT3_3in5':WT3_3in5,'WT3_4in5':WT3_4in5, 'WT3_5in5':WT3_5in5,
          'WT4_1in5':WT4_1in5, 'WT4_2in5':WT4_2in5, 'WT4_3in5':WT4_3in5, 'WT4_4in5':WT4_4in5, 'WT4_5in5':WT4_5in5,
          'WT5_1in5':WT5_1in5, 'WT5_2in5':WT5_2in5, 'WT5_3in5':WT5_3in5, 'WT5_4in5':WT5_4in5, 'WT5_5in5':WT5_5in5,
          'WT6_1in5':WT6_1in5, 'WT6_2in5':WT6_2in5, 'WT6_3in5':WT6_3in5, 'WT6_4in5':WT6_4in5, 'WT6_5in5':WT6_5in5,
          'Mu1_1in5':Mu1_1in5, 'Mu1_2in5':Mu1_2in5, 'Mu1_3in5':Mu1_3in5, 'Mu1_4in5':Mu1_4in5, 'Mu1_5in5':Mu1_5in5,
          'Mu2_1in5':Mu2_1in5, 'Mu2_2in5':Mu2_2in5, 'Mu2_3in5':Mu2_3in5, 'Mu2_4in5':Mu2_4in5, 'Mu2_5in5':Mu2_5in5,
          'Mu3_1in5':Mu3_1in5, 'Mu3_2in5':Mu3_2in5, 'Mu3_3in5':Mu3_3in5, 'Mu3_4in5':Mu3_4in5, 'Mu3_5in5':Mu3_5in5,
          'WT1_4AP_1in5':WT1_4AP_1in5, 'WT1_4AP_2in5':WT1_4AP_2in5, 'WT1_4AP_3in5':WT1_4AP_3in5, 'WT1_4AP_4in5':WT1_4AP_4in5, 'WT1_4AP_5in5':WT1_4AP_5in5,
          'WT2_4AP_1in5':WT2_4AP_1in5, 'WT2_4AP_2in5':WT2_4AP_2in5, 'WT2_4AP_3in5':WT2_4AP_3in5, 'WT2_4AP_4in5':WT2_4AP_4in5,'WT2_4AP_5in5':WT2_4AP_5in5}

    return dirs[name]


def list_of_names(selector):
    WT1 = ['WT1_1in6', 'WT1_2in6', 'WT1_3in6', 'WT1_4in6', 'WT1_5in6', 'WT1_6in6']
    WT2 = ['WT2_1in5', 'WT2_2in5', 'WT2_3in5', 'WT2_4in5', 'WT2_5in5']
    WT3=['WT3_1in5', 'WT3_2in5', 'WT3_3in5', 'WT3_4in5', 'WT3_5in5']
    WT4 = ['WT4_1in5', 'WT4_2in5', 'WT4_3in5', 'WT4_4in5', 'WT4_5in5']
    WT5 = ['WT5_1in5', 'WT5_2in5', 'WT5_3in5', 'WT5_4in5', 'WT5_5in5']
    WT6 = ['WT6_1in5', 'WT6_2in5', 'WT6_3in5', 'WT6_4in5', 'WT6_5in5']
    Mu1= ['Mu1_1in5', 'Mu1_2in5', 'Mu1_3in5', 'Mu1_4in5', 'Mu1_5in5']
    Mu2= ['Mu2_1in5', 'Mu2_2in5', 'Mu2_3in5', 'Mu2_4in5', 'Mu2_5in5']
    Mu3= ['Mu3_1in5', 'Mu3_2in5', 'Mu3_3in5', 'Mu3_4in5', 'Mu3_5in5']
    WT1_4AP = ['WT1_4AP_1in5', 'WT1_4AP_2in5', 'WT1_4AP_3in5', 'WT1_4AP_4in5', 'WT1_4AP_5in5']
    WT2_4AP = ['WT2_4AP_1in5', 'WT2_4AP_2in5', 'WT2_4AP_3in5', 'WT2_4AP_4in5', 'WT2_4AP_5in5']
    All_WT_0Mg = WT1 + WT2 + WT3 + WT4 + WT5 + WT6
    All_Mu_0Mg = Mu1 + Mu2 + Mu3
    All_EA = WT1[0] + WT2[0] + WT3[0] + WT4[0] + WT5[0] + WT6[0] + Mu1[0] + Mu2[0] + Mu3[0] + WT1_4AP[0] + WT2_4AP[0]
    All_after_EA = WT1[1:] + WT2[1:] + WT3[1:] + WT4[1:] + WT5[1:] + WT6[1:] + Mu1[1:] + Mu2[1:] + Mu3[1:] + WT1_4AP[1:] + WT2_4AP[1:]
    All = All_WT_0Mg + All_Mu_0Mg + WT1_4AP + WT2_4AP
    dict= {'WT1':WT1, 'WT2':WT2, 'WT3':WT3, 'WT4':WT4, 'WT5':WT5, 'WT6':WT6, 'Mu1':Mu1, 'Mu2':Mu2,'Mu3':Mu3, 'WT1_4AP':WT1_4AP, 'WT2_4AP':WT2_4AP,
           'All_WT_0Mg':All_WT_0Mg, 'All_Mu_0Mg':All_Mu_0Mg, 'All_EA':All_EA, 'All_after_EA':All_after_EA, 'All':All}
    return dict[selector]

def time_series(name, downsampling):
    import h5py
    import numpy as np
    import scipy.signal as sn
    dr=sdirs(name)
    f=h5py.File(dr, 'r') # h5py.File can open any '.mat' file. Here it opens a struct as a dictionary
    s=list(f.keys())[1]
    struct=f['s']
    signal=np.array(struct['signal'])
    time=np.array(struct['time'])
    if downsampling=='None':
        time_ser=np.vstack((signal.T, time.T))
    else:
        signal=sn.decimate(signal, q=downsampling, axis=0)
        # signal=signal[0::downsample] # alternative way to simply take the 10th element without sn.decimate which also performs an anti-aliasing filter.
        time=time[0::downsampling]
        time_ser=np.vstack((signal.T, time.T))
    return time_ser

def extract_all_data(downsampling, save_path):
    import numpy as np
    name_list = list_of_names('All')
    n=18461538 # make all time_series samples the same length, by zero-paddding, in order to take same length results
    data_list=[]
    for name in name_list:
        ts=time_series(name, downsampling)
        signal=ts[0,:].copy()
        time=ts[1,:]
        signal.resize((int(n/downsampling),), refcheck=False) #gia na einai ola ta simata isou mikous
        # print(signal.shape)
        data_list.append(signal)
    data_list = np.array(data_list)
    print('The shape of extracted data is', data_list.shape)
    np.save(save_path,data_list)
    return data_list

def windowing(signal, window_length, overlap): # this function chooses windows by percentage of overlapping between windows
    import numpy as np
    windows=[]
    window_movement=int(window_length * (1-overlap))
    sp=np.arange(0,len(signal),window_movement)
    for i in sp:
        segm=signal[i:i+window_length]
        if len(segm) < window_length:
            break
        windows.append(segm)
    windowed_signal=np.asarray(windows)
    #windowed_signal=windows
    return windowed_signal

def windowing2(signal, window_length, window_movement): # this function chooses windows by number of points sliding for the next window
    import numpy as np
    windows=[]
    sp=np.arange(0,len(signal),window_movement)
    for i in sp:
        segm=signal[i:i+window_length]
        if len(segm) < window_length:
            break
        windows.append(segm)
    windowed_signal=np.asarray(windows)
    #windowed_signal=windows
    return windowed_signal


def combine (list_of_names, downsampling):
    import numpy as np
    signal = time_series(list_of_names[0], downsampling)
    lfp=signal[0,:]
    time= signal[1,:]
    fs=time[1]-time[0]
    for name in list_of_names[1:]:
        signal = time_series(name, downsampling)
        lfp_new=signal[0,:]
        time_new= signal[1,:]
        lfp = np.hstack((lfp, lfp_new))
        time = np.hstack((time, time_new + len(time)*fs))
    signal = np.vstack((lfp,time))
    #print(lfp.shape)
    #print(time.shape)
    return signal


 