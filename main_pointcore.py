import contextlib
import logging
import os
import sys
import pickle
import click
import numpy as np
import torch
import tqdm
import patchcore.backbones
import patchcore.common
import patchcore.patchcore
import patchcore.utils
import patchcore.sampler
import patchcore.metrics
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import label
from bisect import bisect
import time
from dataset_pc import Dataset3dad_train,Dataset3dad_test
from torch.utils.data import DataLoader
import open3d as o3d
#from utils.visualization import save_anomalymap
from scipy.stats import rankdata
import argparse
import pandas as pd
LOGGER = logging.getLogger(__name__)
import debugpy
#保证host和端口一致，listen可以只设置端口。则为localhost,否则设置成(host,port)
debugpy.listen(12361)
print('wait debugger')
debugpy.wait_for_client()
print("Debugger Attached")

@click.group(chain=True)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--memory_size", type=int, default=10000, show_default=True)
@click.option("--anomaly_scorer_num_nn", type=int, default=5)
@click.option("--faiss_on_gpu", is_flag=True, default=True)
@click.option("--faiss_num_workers", type=int, default=8)
def main(**kwargs):
    pass


@main.result_callback()
def run(
    methods,
    gpu,
    seed,
    memory_size,
    anomaly_scorer_num_nn,
    faiss_on_gpu,
    faiss_num_workers
):
    data_save_pd = pd.DataFrame(columns=['Task','image_auc','pixel_auc','image_ap','pixel_ap','time_cost'])
    methods = {key: item for (key, item) in methods}

    device = patchcore.utils.set_torch_device(gpu)
    # Device context here is specifically set and used later
    # because there was GPU memory-bleeding which I could only fix with
    # context managers.
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []
    root_dir = '/data02/zbz/point_anomaly/PointCore/dataset/pcd'
    save_root_dir = './benchmark/reg3dad/'
    print('Task start: Reg3DAD')
    
    # real_3d_classes = ['airplane','car','candybar','chicken',
    #                 'diamond','duck','fish','gemstone',
    #                 'seahorse','shell','starfish','toffees']#'airplane','car','candybar','chicken',
                    #'diamond','duck','fish','gemstone',
                    #'seahorse','shell','starfish','toffees'
    #根据root_dir下所有文件夹的名字组成一个real_3d_classes
    real_3d_classes = os.listdir(root_dir)
    
    #dict_name_index={"airplane":0,"car":1,"candybar":2,"chicken":3,"diamond":4,"duck":5,"fish":6,"gemstone":7,"seahorse":8,"shell":9,"starfish":10,"toffees":11}
    dict_name_index = {class_name: index for index, class_name in enumerate(real_3d_classes)}
    data_save={}
    downsample_list = ["_01","_008","_006"]
    for item in real_3d_classes:
        data_save[item+"_labels_gt"]=[]
        data_save[item+"_masks_gt"]=[]
        for downsample in downsample_list:
            for dis in ["_xyz","_fpfh","_pmae"]:
                data_save[item+downsample+dis]=[]
    for dataset_count, dataset_name in enumerate(real_3d_classes):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataset_name,
                dataset_count + 1,
                len(real_3d_classes),
            )
        )
        if( not os.path.exists(save_root_dir+dataset_name)):
            os.makedirs(save_root_dir+dataset_name)
        patchcore.utils.fix_seeds(seed, device)
        train_loader = DataLoader(Dataset3dad_train(root_dir, dataset_name, 1024, if_norm=False), num_workers=1,
                                batch_size=1, shuffle=False, drop_last=False)
        test_loader = DataLoader(Dataset3dad_test(root_dir, dataset_name, 1024, if_norm=False), num_workers=1,
                                batch_size=1, shuffle=False, drop_last=False)

        for data, mask, label, path in train_loader:
            basic_template = data.squeeze(0).cpu().numpy()
            break
        
        with device_context:
            torch.cuda.empty_cache()
            ap_seg=[]
            scores=[]
            for sampler_num in range(1):
                if sampler_num==0:
                    sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(0.1, device)
                elif sampler_num==1:
                    sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(0.08, device)
                elif sampler_num==2:
                    sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(0.06, device)
            # sampler = methods["get_sampler"](
            #     device,
            # )
            # PatchCore_list = methods["get_patchcore"](imagesize, sampler, device)
                nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)

                PatchCore = patchcore.patchcore.PatchCore(device)
                PatchCore.load(
                    backbone=None,
                    layers_to_extract_from=None,
                    device=device,
                    input_shape=None,
                    pretrain_embed_dimension=1024,
                    target_embed_dimension=1024,
                    patchsize=16,
                    featuresampler=sampler,
                    anomaly_scorer_num_nn=anomaly_scorer_num_nn,
                    nn_method=nn_method,
                    basic_template=basic_template,
                )
                PatchCore.set_deep_feature_extractor()
                start_time = time.time()
                torch.cuda.empty_cache()
                memory_feature_xyz,sample_2dim = PatchCore.fit_with_limit_size_myself(train_loader, memory_size)
                memory_feature_pmae = PatchCore.fit_with_limit_size_pmae_myself(train_loader, memory_size,sample_2dim)
                scores_xyz, segmentations_xyz,scores_pmae, segmentations_pmae, labels_gt, masks_gt = PatchCore.predict_myself(
                    test_loader,memory_feature_xyz,memory_feature_pmae
                )
                #全部归一化
                scores_xyz=(scores_xyz-np.min(scores_xyz))/(np.max(scores_xyz)-np.min(scores_xyz))
                scores_pmae=(scores_pmae-np.min(scores_pmae))/(np.max(scores_pmae)-np.min(scores_pmae))
                segmentations_xyz=(segmentations_xyz-np.min(segmentations_xyz))/(np.max(segmentations_xyz)-np.min(segmentations_xyz))
                segmentations_pmae=(segmentations_pmae-np.min(segmentations_pmae))/(np.max(segmentations_pmae)-np.min(segmentations_pmae))
                scores_xyz = rankdata(scores_xyz)/len(scores_xyz)
                scores_pmae = rankdata(scores_pmae)/len(scores_pmae)
                #segmentations_xyz = rankdata(segmentations_xyz)/len(segmentations_xyz)
                #segmentations_pmae = rankdata(segmentations_pmae)/len(segmentations_pmae)
                end_time = time.time()
                time_cost = (end_time - start_time)/len(test_loader)
                LOGGER.info("Computing evaluation metrics.")
                scores.append((scores_xyz+0.1*scores_pmae)/1.1)
                ap_seg.append((0.1*segmentations_pmae+segmentations_xyz)/1.1)
            scores = np.array(scores)
            scores=np.mean(scores,axis=0)
            ap_seg = np.array(ap_seg)
            ap_seg=np.mean(ap_seg,axis=0)
            auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores, labels_gt
            )["auroc"]

            img_ap = average_precision_score(labels_gt,scores)
            ap_mask = np.asarray(masks_gt)
            ap_mask = ap_mask.flatten().astype(np.int32)
            pixel_ap = average_precision_score(ap_mask,ap_seg)
            full_pixel_auroc = roc_auc_score(ap_mask,ap_seg)
            print('Task:{}, image_auc:{}, pixel_auc:{}, image_ap:{}, pixel_ap:{}, time_cost:{}'.format
                    (dataset_name,auroc,full_pixel_auroc,img_ap,pixel_ap,time_cost))
            from evaluation import evaluation_pmae_fpfh_xyz
            img_ap_old,full_image_auroc_old = evaluation_pmae_fpfh_xyz(real_3d_classes=real_3d_classes,real_3d_classes_index=dict_name_index[dataset_name],a_a=1,b_b=0.26,index_print=0)
            if(img_ap_old<img_ap and full_image_auroc_old<auroc):
                with open("data_save_scores_myself/"+dataset_name+"scores_xyz"+'.pkl','wb') as f:
                    pickle.dump(scores_xyz,f)
                with open("data_save_scores_myself/"+dataset_name+"scores_pmae"+'.pkl','wb') as f:
                    pickle.dump(scores_pmae,f)
                with open("data_save_scores_myself/"+dataset_name+"segmentations_xyz"+'.pkl','wb') as f:
                    pickle.dump(segmentations_xyz,f)
                with open("data_save_scores_myself/"+dataset_name+"segmentations_pmae"+'.pkl','wb') as f:
                    pickle.dump(segmentations_pmae,f)
                with open("data_save_scores_myself/"+dataset_name+"labels_gt"+'.pkl','wb') as f:
                    pickle.dump(labels_gt,f)
                with open("data_save_scores_myself/"+dataset_name+"masks_gt"+'.pkl','wb') as f:
                    pickle.dump(masks_gt,f)
            data_save_pd.loc[dataset_count] = [dataset_name,auroc,full_pixel_auroc,img_ap,pixel_ap,time_cost]
            #data_save_pd.loc[2*dataset_count+1] = [dataset_name+"_01",auroc_01,full_pixel_auroc_01,img_ap_01,pixel_ap_01]
    data_save_pd.to_csv('GR+LR_PointCore_RB.csv',index=False)



@main.command("sampler")
@click.argument("name", type=str, default="approx_greedy_coreset")
@click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
def sampler(name, percentage):
    def get_sampler(device):
        if name == "identity":
            return patchcore.sampler.IdentitySampler()
        elif name == "greedy_coreset":
            return patchcore.sampler.GreedyCoresetSampler(percentage, device)
        elif name == "approx_greedy_coreset":
            return patchcore.sampler.ApproximateGreedyCoresetSampler(percentage, device)

    return ("get_sampler", get_sampler)

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
