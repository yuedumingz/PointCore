import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
from scipy.stats import rankdata
import os
real_3d_classes = ['airplane','car','candybar','chicken',
                   'diamond','duck','fish','gemstone',
                   'seahorse','shell','starfish','toffees']
def evaluation_pmae_fpfh_xyz(real_3d_classes_index=0,a_a=1,b_b=0.26,index_print=0):
    
    fpfh_data=[]
    xyz_data=[]
    pmae_data=[]
    select_all=[["scores","labels_gt"],["segmentations","masks_gt"]]
    print_what=["image_level","point_level"]
    select=select_all[index_print]
    if os.path.exists("data_save_scores_myself/"+real_3d_classes[real_3d_classes_index]+select[0]+"_xyz.pkl")\
        and os.path.exists("data_save_scores_myself/"+real_3d_classes[real_3d_classes_index]+select[0]+"_pmae.pkl")\
            and os.path.exists("data_save_scores_myself/"+real_3d_classes[real_3d_classes_index]+select[1]+".pkl"):
        #for scale_index in scale_indexs:
            #fpfh_data.append(pickle.load(open(real_3d_classes[real_3d_classes_index]+"_fpfh.pkl", 'rb'))[0][0])
        xyz_data.append(pickle.load(open("data_save_scores_myself/"+real_3d_classes[real_3d_classes_index]+select[0]+"_xyz.pkl", 'rb')))
        pmae_data.append(pickle.load(open("data_save_scores_myself/"+real_3d_classes[real_3d_classes_index]+select[0]+"_pmae.pkl", 'rb')))
        #fpfh_data=np.array(fpfh_data)
        xyz_data=np.array(xyz_data)
        pmae_data=np.array(pmae_data)
        # fpfh_data=rankdata(fpfh_data,axis=1)/fpfh_data.shape[0]
        xyz_data=rankdata(xyz_data,axis=1)/xyz_data.shape[0]
        pmae_data=rankdata(pmae_data,axis=1)/pmae_data.shape[0]
        scores=(a_a*xyz_data+b_b*pmae_data)/3#缺陷值排名-分层取平均-pmae+xyz 最好成绩,1.26
        #scores=(0.*fpfh_data+1*xyz_data+0.*pmae_data)/1
        scores=np.mean(scores,axis=0)
        labels_gt=np.array(pickle.load(open("data_save_scores_myself/"+real_3d_classes[real_3d_classes_index]+select[1]+".pkl", 'rb')))
        labels_gt=labels_gt.flatten()
        image_ap = average_precision_score(labels_gt,scores)
        full_image_auroc = roc_auc_score(labels_gt,scores)
        #print(real_3d_classes[real_3d_classes_index]+"_"+print_what[index_print]+"_ap: ",image_ap)
        #print(real_3d_classes[real_3d_classes_index]+"_"+print_what[index_print]+"_auc: ",full_image_auroc)
    else:
        image_ap=0
        full_image_auroc=0
    return image_ap,full_image_auroc