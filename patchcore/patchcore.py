"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle
from sklearn.neighbors import KNeighborsRegressor,NearestNeighbors
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from pointnet2_ops import pointnet2_utils
import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler
import cupy as cp
from timm.models import create_model
import argparse
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from sklearn.cluster import KMeans
import open3d as o3d
from M3DM.cpu_knn import fill_missing_values
from feature_extractors.ransac_position import get_registration_np,get_registration_refine_np
#from utils.utils import get_args_point_mae
from M3DM.models import Model1
from sklearn.decomposition import PCA


LOGGER = logging.getLogger(__name__)



class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=patchcore.sampler.IdentitySampler(),
        nn_method=patchcore.common.FaissNN(False, 4),
        basic_template=None,
        **kwargs,
    ):
        # self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})
        self.voxel_size = 0.5 #0.1

        # feature_aggregator = patchcore.common.NetworkFeatureAggregator(
        #     self.backbone, self.layers_to_extract_from, self.device
        # )
        # feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        # self.forward_modules["feature_aggregator"] = feature_aggregator

        # preprocessing = patchcore.common.Preprocessing(
        #     feature_dimensions, pretrain_embed_dimension
        # )
        # self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        # self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
        #     device=self.device, target_size=input_shape[-2:]
        # )

        self.featuresampler = featuresampler
        self.dataloader_count = 0
        self.basic_template = basic_template
        self.deep_feature_extractor = None
        # self.pca = PCA(n_components=10)
        
    def set_deep_feature_extractor(self):
        # args = get_args_point_mae()
        self.deep_feature_extractor = Model1(device='cuda', 
                        rgb_backbone_name='vit_base_patch8_224_dino', 
                        xyz_backbone_name='Point_MAE', 
                        group_size = 128, 
                        num_group = 16384)
        self.deep_feature_extractor = self.deep_feature_extractor.cuda()
    
    def set_dataloadercount(self, dataloader_count):
        self.dataloader_count = dataloader_count

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)
    
    def embed_xyz(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed_xyz(data)
    
    def _embed_xyz(self, point_cloud, detach=True):
        reg_data = get_registration_refine_np(point_cloud.squeeze(0).cpu().numpy(),self.basic_template)
        reg_data = reg_data.astype(np.float32)
        #reg_data=point_cloud.squeeze(0).cpu().numpy().astype(np.float32)
        return reg_data
    
    def _embed_fpfh(self, point_cloud, detach=True):
        reg_data = get_registration_np(point_cloud.squeeze(0).cpu().numpy(),self.basic_template)
        #reg_data=point_cloud.squeeze(0).cpu().numpy()
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(reg_data))
        radius_normal = self.voxel_size * 2
        o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = self.voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(o3d_pc, o3d.geometry.KDTreeSearchParamHybrid
        (radius=radius_feature, max_nn=100))
        fpfh = pcd_fpfh.data.T
        # fpfh = torch.from_numpy(pcd_fpfh.data.T)
        # print(fpfh.shape)
        fpfh = fpfh.astype(np.float32)
        return fpfh
    
    def _embed_pointmae(self, point_cloud, sample_idx,detach=True):
        reg_data = get_registration_refine_np(point_cloud.squeeze(0).cpu().numpy(),self.basic_template)
        #reg_data=point_cloud.squeeze(0).cpu().numpy()
        pointcloud_data = torch.from_numpy(reg_data).permute(1,0).unsqueeze(0).cuda().float()
        pmae_features, center, ori_idx, center_idx = self.deep_feature_extractor(pointcloud_data,sample_idx)
        pmae_features = pmae_features.squeeze(0).permute(1,0).cpu().numpy()
        pmae_features = pmae_features.astype(np.float32)
        return pmae_features,center_idx

    def _embed_downpointmae_xyz(self, point_cloud, detach=True):
        reg_data = get_registration_np(point_cloud.squeeze(0).cpu().numpy(),self.basic_template)
        pointcloud_data = torch.from_numpy(reg_data).permute(1,0).unsqueeze(0).cuda().float()
        pmae_features, center, ori_idx, center_idx = self.deep_feature_extractor(pointcloud_data)
        pmae_features = pmae_features.squeeze(0).permute(1,0).cpu().numpy()
        pmae_features = pmae_features.astype(np.float32)
        # pmae_features = self.pca.fit_transform(pmae_features)
        mask_idx = center_idx.squeeze().long()
        xyz = reg_data[mask_idx.cpu().numpy(),:]
        xyz = xyz.repeat(333,1)
        features = np.concatenate([pmae_features,xyz],axis=1)
        return features.astype(np.float32),center_idx
    
    def _embed_upfpfh_xyz(self, point_cloud, detach=True):
        reg_data = get_registration_np(point_cloud.squeeze(0).cpu().numpy(),self.basic_template)
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(reg_data))
        radius_normal = self.voxel_size * 2
        o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = self.voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(o3d_pc, o3d.geometry.KDTreeSearchParamHybrid
        (radius=radius_feature, max_nn=100))
        fpfh = pcd_fpfh.data.T
        # fpfh = torch.from_numpy(pcd_fpfh.data.T)
        # print(fpfh.shape)
        fpfh = fpfh.astype(np.float32)
        xyz = reg_data.repeat(11,1)
        features = np.concatenate([fpfh,xyz],axis=1)
        return features.astype(np.float32)
    
    # def _embed(self, images, detach=True, provide_patch_shapes=False):
    #     """Returns feature embeddings for images."""

    #     def _detach(features):
    #         if detach:
    #             return [x.detach().cpu().numpy() for x in features]
    #         return features

    #     _ = self.model.eval()
    #     with torch.no_grad():
    #         features = self.model(images)['seg_feat']
    #         features = features.reshape(-1,768)
    #         # print(features.shape)

    #     patch_shapes = [14,14]
    #     if provide_patch_shapes:
    #         return _detach(features), patch_shapes
    #     return _detach(features)
    

    def fit(self, training_data):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run(features)

        self.anomaly_scorer.fit(detection_features=[features])
    
    def fit_with_return_feature(self, training_data):
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            training_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run(features)

        self.anomaly_scorer.fit(detection_features=[features])
        return features
    
    def get_all_features(self, training_data):
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            training_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        return features
    
    def fit_with_limit_size(self, training_data, limit_size):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._fill_memory_bank_with_limit_size(training_data, limit_size)
        
    def _fill_memory_bank_with_limit_size(self, input_data, limit_size):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_pointcloud):
            with torch.no_grad():
                
                return self._embed_xyz(input_pointcloud)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                features.append(_image_to_features(input_pointcloud))
        features_len=[]
        for feature in features:
            features_len.append(len(feature))#记录下每个点云的点数，方便后面计算获得的索引在哪个点云中
        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        self.anomaly_scorer.fit(detection_features=[features])
        print(features.shape)
        return features
    def fit_with_limit_size_myself(self, training_data, limit_size):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._fill_memory_bank_with_limit_size_myself(training_data, limit_size)
    def _fill_memory_bank_with_limit_size_myself(self, input_data, limit_size):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_pointcloud):
            with torch.no_grad():
                
                return self._embed_xyz(input_pointcloud)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                features.append(_image_to_features(input_pointcloud))
        features_len=[0]
        for feature in features:
            features_len.append(features_len[-1]+len(feature))#记录下每个点云的点数，方便后面计算获得的索引在哪个点云中
        features = np.concatenate(features, axis=0)
        features,sample_indices = self.featuresampler.run_with_limit_memory_myself(features, limit_size)
        #self.anomaly_scorer.fit(detection_features=[features])
        print(features.shape)
        sample_2dim=[]
        # for i in range(len(sample_indices)):
        #     for j in range(len(features_len)):
        #         if sample_indices[i]<features_len[j]:
        #             sample_2dim.append([j-1,sample_indices[i]-features_len[j-1]])
        #             break
        #将sample_indices进行排序，从小到大
        arrIndex = np.array(sample_indices).argsort()
        sample_indices = sample_indices[arrIndex]
        features=features[arrIndex]
        sample_indices_point=0
        for i in range(len(features_len)-1):
            temp=[]
            while sample_indices_point<len(sample_indices) and sample_indices[sample_indices_point]<features_len[i+1]:
                temp.append(sample_indices[sample_indices_point]-features_len[i])
                sample_indices_point+=1
            sample_2dim.append(temp)
        return features,sample_2dim
    
    def fit_with_limit_size_fpfh(self, training_data, limit_size):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._fill_memory_bank_with_limit_size_fpfh(training_data, limit_size)
        
    def _fill_memory_bank_with_limit_size_fpfh(self, input_data, limit_size):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_pointcloud):
            with torch.no_grad():
                
                return self._embed_fpfh(input_pointcloud)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                features.append(_image_to_features(input_pointcloud))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        self.anomaly_scorer.fit(detection_features=[features])
        print(features.shape)
        return features

    def fit_with_limit_size_fpfh_upxyz(self, training_data, limit_size):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._fill_memory_bank_with_limit_size_fpfh_upxyz(training_data, limit_size)
        
    def _fill_memory_bank_with_limit_size_fpfh_upxyz(self, input_data, limit_size):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_pointcloud):
            with torch.no_grad():
                
                return self._embed_upfpfh_xyz(input_pointcloud)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                features.append(_image_to_features(input_pointcloud))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        self.anomaly_scorer.fit(detection_features=[features])
        print(features.shape)
        return features

    def fit_with_limit_size_pmae(self, training_data, limit_size):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._fill_memory_bank_with_limit_size_pmae(training_data, limit_size)
        
    def _fill_memory_bank_with_limit_size_pmae(self, input_data, limit_size):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_pointcloud):
            with torch.no_grad():
                pmae_features, sample_idx =self._embed_pointmae(input_pointcloud)
                return pmae_features

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                features.append(_image_to_features(input_pointcloud))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        self.anomaly_scorer.fit(detection_features=[features])
        print(features.shape)
        return features
    def fit_with_limit_size_pmae_myself(self, training_data, limit_size,sample_2dim):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._fill_memory_bank_with_limit_size_pmae_myself(training_data, limit_size,sample_2dim)
        
    def _fill_memory_bank_with_limit_size_pmae_myself(self, input_data, limit_size,sample_2dim):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_pointcloud,sample_idx):
            with torch.no_grad():
                pmae_features, sample_idx_out =self._embed_pointmae(input_pointcloud,sample_idx)
                return pmae_features

        features = []
        sample_2dim_point=0
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                if len(sample_2dim[sample_2dim_point])==0:
                    continue
                features.append(_image_to_features(input_pointcloud,sample_2dim[sample_2dim_point]))
                sample_2dim_point+=1

        features = np.concatenate(features, axis=0)
        #features = self.featuresampler.run_with_limit_memory(features, limit_size)
        #self.anomaly_scorer.fit(detection_features=[features])
        print(features.shape)
        return features

    def fit_with_limit_size_downpmae_xyz(self, training_data, limit_size):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._fill_memory_bank_with_limit_size_downpmae_xyz(training_data, limit_size)
        
    def _fill_memory_bank_with_limit_size_downpmae_xyz(self, input_data, limit_size):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_pointcloud):
            with torch.no_grad():
                pmae_features, sample_idx =self._embed_downpointmae_xyz(input_pointcloud)
                return pmae_features

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                features.append(_image_to_features(input_pointcloud))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        self.anomaly_scorer.fit(detection_features=[features])
        print(features.shape)
        return features

    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                labels_gt.extend(label.numpy().tolist())
                masks_gt.extend(mask[0].numpy().tolist())
                _scores, _masks = self._predict(input_pointcloud)
                # for score, mask in zip(_scores, _masks):
                scores.extend(_scores)
                masks.extend(_masks)
        return scores, masks, labels_gt, masks_gt
    def interpolation_pmae(self,x1,x2,x3,a1,a2,a3):
        pmae_feature_interpolation=[]
        for (a1_temp,a2_temp,a3_temp) in zip(a1,a2,a3):
            pmae_feature_interpolation.append(cp.asnumpy((x2*x3*cp.asarray(a1_temp)+x1*x3*cp.asarray(a2_temp)+x1*x2*cp.asarray(a3_temp))/(x1*x2+x1*x3+x2*x3)))
        pmae_feature_interpolation=np.array(pmae_feature_interpolation).swapaxes(0,1)
        return pmae_feature_interpolation
    def compute_masks_pmae(self,memory_feature_pmae,indices,pmae_features,indices_pmae):
        # x1=distance_pmae[:,0]#.reshape((len(distance_pmae[:,0]),1))
        # x2=distance_pmae[:,1]#.reshape((len(distance_pmae[:,0]),1))
        # x3=distance_pmae[:,2]#.reshape((len(distance_pmae[:,0]),1))
        # a1=memory_feature_pmae[indices_pmae[:,0],:]#np.swapaxes(,0,1)
        # a2=memory_feature_pmae[indices_pmae[:,1],:]#np.swapaxes(,0,1)
        # a3=memory_feature_pmae[indices_pmae[:,2],:]#np.swapaxes(,0,1)
        distance_pmae_des=[]
        for index in np.array_split(np.arange(len(indices)),3):
            pmae_feature_interpolation=pmae_features[indices_pmae[index,0],:]
        #     #pmae_feature_interpolation=self.interpolation_pmae(x1[index],x2[index],x3[index],a1[:,index],a2[:,index],a3[:,index])
        # #pmae_feature_interpolation=self.interpolation_pmae(x1,x2,x3,a1,a2,a3)#(x2*x3*a1+x1*x3*a2+x1*x2*a3)/(x1*x2+x1*x3+x2*x3)
        # #pmae_feature_interpolation=pmae_feature_interpolation
        #     x1_temp=x1[index].reshape((len(index),1))
        #     x2_temp=x2[index].reshape((len(index),1))
        #     x3_temp=x3[index].reshape((len(index),1))
        #     a1_temp=memory_feature_pmae[indices_pmae[index,0],:]
        #     a2_temp=memory_feature_pmae[indices_pmae[index,1],:]
        #     a3_temp=memory_feature_pmae[indices_pmae[index,2],:]
        #     # a2_temp=cp.asarray(a2[index,:])
        #     # a3_temp=cp.asarray(a3[index,:])
        #     pmae_feature_interpolation=(x2_temp*x3_temp*a1_temp+x1_temp*x3_temp*a2_temp+x1_temp*x2_temp*a3_temp)/(x1_temp*x2_temp+x1_temp*x3_temp+x2_temp*x3_temp)
        #     distance_pmae_des_temp=[]
        #     for i in range(indices.shape[1]):
        #         distance_pmae_des_temp.append(np.sqrt(np.sum((pmae_feature_interpolation-memory_feature_pmae[indices[index,i],:])**2,axis=1)))
        #     distance_pmae_des_temp=np.array(distance_pmae_des_temp)
        #     distance_pmae_des_temp=np.min(distance_pmae_des_temp,axis=0)
        #     distance_pmae_des.extend(distance_pmae_des_temp)
            distance_pmae_des_temp=[]
            for i in range(indices.shape[1]):
                distance_pmae_des_temp.append(np.sqrt(np.sum((pmae_feature_interpolation-memory_feature_pmae[indices[index,i],:])**2,axis=1)))
            distance_pmae_des_temp=np.array(distance_pmae_des_temp)
            distance_pmae_des_temp=np.min(distance_pmae_des_temp,axis=0)
            distance_pmae_des.extend(distance_pmae_des_temp)
        return distance_pmae_des
    def _predict_xyz_pmae_myself(self, input_pointcloud,memory_feature_xyz,memory_feature_pmae):
        input_pointcloud_input = torch.from_numpy(input_pointcloud).permute(1,0).unsqueeze(0).cuda().float()
        fps_idx = pointnet2_utils.furthest_point_sample(input_pointcloud_input.transpose(-1, -2), 1024)
        fps_idx=np.array(fps_idx.to("cpu"))[0]
        pmae_features, center, ori_idx, center_idx = self.deep_feature_extractor(input_pointcloud_input,fps_idx)
        with torch.no_grad():
            pmae_features=np.array(pmae_features.transpose(-1,-2).to("cpu")[0])
        center_idx=np.array(center_idx.to("cpu")[0])
        nn_pmae = NearestNeighbors(n_neighbors=3)
        nn_pmae.fit(input_pointcloud[center_idx,:])
        distances_pmae,indices_pmae = nn_pmae.kneighbors(input_pointcloud)
        #indices_pmae=indices_pmae.reshape((len(indices_pmae)))
        # 创建最近邻居模型
        # arrIndex = np.array(center_idx.cpu()).argsort()
        # center_idx = center_idx[arrIndex]
        # pmae_features = pmae_features[arrIndex]
        nn = NearestNeighbors(n_neighbors=3)
        nn.fit(memory_feature_xyz)
        # 找到每个点的最近邻居
        distances, indices = nn.kneighbors(input_pointcloud)
        #_masks_xyz = np.sqrt(np.sum((input_pointcloud - memory_feature_xyz[indices[:,0]])**2,axis=1))
        #_masks_xyz=np.mean(distances,axis=1)
        _masks_xyz=distances[:,0]
        _scores_xyz=np.max(_masks_xyz)
        #_masks_pmae=np.sqrt(np.sum((pmae_features[indices_pmae,:] - memory_feature_pmae[indices[:,0]])**2,axis=1))
        _masks_pmae=self.compute_masks_pmae(memory_feature_pmae,indices,pmae_features,indices_pmae)
        _scores_pmae=np.max(_masks_pmae)
        return _scores_xyz, _masks_xyz,_scores_pmae, _masks_pmae
    def predict_myself(self,test_loader,memory_feature_xyz,memory_feature_pmae):
        _ = self.forward_modules.eval()
        scores_xyz=[]
        scores_pmae=[]
        masks_xyz=[]
        masks_pmae=[]
        labels_gt=[]
        masks_gt=[]
        with tqdm.tqdm(test_loader, desc="Inferring...", leave=False) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                input_pointcloud=self._embed_xyz(input_pointcloud)
                labels_gt.extend(label.numpy().tolist())
                masks_gt.extend(mask[0].numpy().tolist())
                _scores_xyz, _masks_xyz,_scores_pmae, _masks_pmae = self._predict_xyz_pmae_myself(input_pointcloud,memory_feature_xyz,memory_feature_pmae)
                #_scores_pmae, _masks_pmae = self._predict_pmae(input_pointcloud,memory_feature_pmae)
                #print(_scores_xyz)
                scores_xyz.append(_scores_xyz)
                masks_xyz.extend(_masks_xyz)
                scores_pmae.append(_scores_pmae)
                masks_pmae.extend(_masks_pmae)
        return scores_xyz,masks_xyz,scores_pmae,masks_pmae,labels_gt,masks_gt
    def save_predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        paths = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                labels_gt.extend(label.numpy().tolist())
                masks_gt.extend(mask.numpy().tolist())
                _scores, _masks = self._predict(input_pointcloud)
                # for score, mask in zip(_scores, _masks):
                scores.extend(_scores)
                masks.append(_masks)
                paths.append(path[0])
        data_save={
            'scores':scores,
            'masks':masks,
            'labels_gt':labels_gt,
            'masks_gt':masks_gt,
            'paths':paths
        }
        with open('airplane_xyz.pkl','wb') as f:
            pickle.dump(data_save,f)

    def _predict(self, input_pointcloud):
        """Infer score and mask for a batch of images."""
        # images = images.to(torch.float).to(self.device)
        # _ = self.forward_modules.eval()

        # batchsize = images.shape[0]
        with torch.no_grad():
            features = self._embed_xyz(input_pointcloud)
            # print(patch_shapes) [32,32]
            features = np.asarray(features)
            # print(features.shape)
            # features = np.repeat(features,2,axis=1)

            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = np.max(image_scores)
            # print(patch_scores.shape)
            # print(image_scores)
            # image_scores = self.patch_maker.unpatch_scores(
            #     image_scores, batchsize=batchsize
            # )
            # image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            # print(image_scores.shape)
            # image_scores = self.patch_maker.score(image_scores)
            # print(image_scores.shape)

            # patch_scores = self.patch_maker.unpatch_scores(
            #     patch_scores, batchsize=batchsize
            # )
            # patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            # masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [image_scores], [mask for mask in patch_scores]
        # return [score for score in image_scores], [mask for mask in image_scores]
    
    def predict_fpfh(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader_fpfh(data)
        return self._predict_fpfh(data)

    def _predict_dataloader_fpfh(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                labels_gt.extend(label.numpy().tolist())
                masks_gt.extend(mask.numpy().tolist())
                _scores, _masks = self._predict_fpfh(input_pointcloud)
                scores.extend(_scores)
                masks.extend(_masks)
        return scores, masks, labels_gt, masks_gt
    def save_predict_dataloader_fpfh(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        paths = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                labels_gt.extend(label.numpy().tolist())
                masks_gt.extend(mask.numpy().tolist())
                _scores, _masks = self._predict_fpfh(input_pointcloud)
                scores.extend(_scores)
                masks.extend(_masks)
                paths.append(path[0])
        data_save={
            'scores':scores,
            'masks':masks,
            'labels_gt':labels_gt,
            'masks_gt':masks_gt,
            'paths':paths
        }
        with open('airplane_fpfh.pkl','wb') as f:
            pickle.dump(data_save,f)
    def _predict_fpfh(self, input_pointcloud):
        """Infer score and mask for a batch of images."""
        with torch.no_grad():
            features = self._embed_fpfh(input_pointcloud)
            features = np.asarray(features)
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = np.max(image_scores)
        return [image_scores], [mask for mask in patch_scores]

    def predict_fpfh_upxyz(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader_fpfh_upxyz(data)
        return self._predict_fpfh_upxyz(data)

    def _predict_dataloader_fpfh_upxyz(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                labels_gt.extend(label.numpy().tolist())
                masks_gt.extend(mask.numpy().tolist())
                _scores, _masks = self._predict_fpfh_upxyz(input_pointcloud)
                scores.extend(_scores)
                masks.extend(_masks)
        return scores, masks, labels_gt, masks_gt

    def _predict_fpfh_upxyz(self, input_pointcloud):
        """Infer score and mask for a batch of images."""
        with torch.no_grad():
            features = self._embed_upfpfh_xyz(input_pointcloud)
            features = np.asarray(features)
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = np.max(image_scores)
        return [image_scores], [mask for mask in patch_scores]
    
    def predict_pmae(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader_pmae(data)
        return self._predict_pmae(data)

    def _predict_dataloader_pmae(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        #paths = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                labels_gt.extend(label.numpy().tolist())
                masks_gt.extend(mask.numpy().tolist())
                _scores, _masks = self._predict_pmae(input_pointcloud)
                scores.extend(_scores)
                masks.extend(_masks)
                #paths.append(path[0])
        return scores, masks, labels_gt, masks_gt
    def save_predict_dataloader_pmae(self, dataloader):
        _ = self.forward_modules.eval()
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        paths = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                labels_gt.extend(label.numpy().tolist())
                masks_gt.extend(mask.numpy().tolist())
                _scores, _masks = self._predict_pmae(input_pointcloud)
                scores.extend(_scores)
                masks.append(_masks)
                paths.append(path[0])
        data_save={
            'scores':scores,
            'masks':masks,
            'labels_gt':labels_gt,
            'masks_gt':masks_gt,
            'paths':paths
        }
        with open('airplane_pmae.pkl','wb') as f:
            pickle.dump(data_save,f)
    def _predict_pmae(self, input_pointcloud):
        """Infer score and mask for a batch of images."""
        with torch.no_grad():
            features, sample_dix = self._embed_pointmae(input_pointcloud)
            features = np.asarray(features,order='C').astype('float32')
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = np.max(image_scores)
            mask_idx = sample_dix.squeeze().long()
            xyz_sampled = input_pointcloud[0][mask_idx.cpu(),:]
            # print(patch_scores.shape)
            # print(input_pointcloud.shape)
            # print(xyz_sampled.shape)
            full_scores = fill_missing_values(xyz_sampled,patch_scores,input_pointcloud[0], k=1)
        return [image_scores], [mask for mask in full_scores]

    
    def predict_downpmae_xyz(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader_downpmae_xyz(data)
        return self._predict_downpmae_xyz(data)

    def _predict_dataloader_downpmae_xyz(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                labels_gt.extend(label.numpy().tolist())
                masks_gt.extend(mask.numpy().tolist())
                _scores, _masks = self._predict_downpmae_xyz(input_pointcloud)
                scores.extend(_scores)
                masks.extend(_masks)
        return scores, masks, labels_gt, masks_gt

    def _predict_downpmae_xyz(self, input_pointcloud):
        """Infer score and mask for a batch of images."""
        with torch.no_grad():
            features, sample_dix = self._embed_downpointmae_xyz(input_pointcloud)
            features = np.asarray(features,order='C').astype('float32')
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = np.max(image_scores)
            mask_idx = sample_dix.squeeze().long()
            xyz_sampled = input_pointcloud[0][mask_idx.cpu(),:]
            # print(patch_scores.shape)
            # print(input_pointcloud.shape)
            # print(xyz_sampled.shape)
            full_scores = fill_missing_values(xyz_sampled,patch_scores,input_pointcloud[0], k=1)
        return [image_scores], [mask for mask in full_scores]
    
    def _predict_past_tasks(self, features, data):
        pass
            
    def _fit_past_tasks(self, features, data):
        pass
        

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: patchcore.common.FaissNN(False, 4),
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
