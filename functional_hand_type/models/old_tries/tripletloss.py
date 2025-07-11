import torch
import torch.nn.functional as torch_f

from einops import repeat

from models import resnet
from models.transformer import Transformer_Encoder, PositionalEncoding
from models.actionbranch import ActionClassificationBranch
from models.objclassbranch import ObjClassBranch
from models.handtypebranch import HandTypeClassificationBranch
from models.utils import  To25DBranch,compute_hand_loss,loss_str2func
from models.mlp import MultiLayerPerceptron
from datasets.queries import BaseQueries, TransQueries 
import clip


class ResNet_(torch.nn.Module):
    def __init__(self,resnet_version=18):
        super().__init__()
        if int(resnet_version) == 18:
            img_feature_size = 512
            self.base_net = resnet.resnet18(pretrained=True)
        elif int(resnet_version) == 50:
            img_feature_size = 2048
            self.base_net = resnet.resnet50(pretrained=True)
        else:
            self.base_net=None
    
    
    def forward(self, image):
        features, res_layer5 = self.base_net(image)
        return features, res_layer5
 

class TemporalNet(torch.nn.Module):
    def __init__(self,  is_single_hand,
                        transformer_d_model,
                        transformer_dropout,
                        transformer_nhead,
                        transformer_dim_feedforward,
                        transformer_num_encoder_layers_action,
                        transformer_num_encoder_layers_pose,
                        transformer_normalize_before=True,

                        lambda_action_loss=None,
                        lambda_hand_2d=None,
                        lambda_hand_z=None,
                        ntokens_pose=1,
                        ntokens_action=1,
                        
                        dataset_info=None,
                        trans_factor=100,
                        scale_factor=0.0001,
                        pose_loss='l2',
                        dim_grasping_feature=128,):

        super().__init__()
        
        self.ntokens_pose= ntokens_pose
        self.ntokens_action=ntokens_action

        self.pose_loss=loss_str2func()[pose_loss]
        
        self.lambda_hand_z=lambda_hand_z
        self.lambda_hand_2d=lambda_hand_2d        
        self.lambda_action_loss=lambda_action_loss
        self.lambda_handtype_loss=1.0

        # Hyperparameters
        self.lambda_hand_z=100 # 100
        self.lambda_hand_2d=1 # 1    
        self.lambda_action_loss=1 # 1

        self.is_single_hand=is_single_hand
        self.num_joints=21 if self.is_single_hand else 42
        
        
        # 1-1. 회전 특징을 위한 인코더 추가
        # 입력: 500 (npy 원본 길이), 출력: 128 (임베딩 차원)
        self.rotation_encoder = MultiLayerPerceptron(
            base_neurons=[500, 256, 128],
            out_dim=128,
            act_hidden='leakyrelu',
            act_final='none' # 정규화는 Triplet Loss 전에 하는 것이 일반적
        )
        self.final_feature_fusion_layer = torch.nn.Linear(
            transformer_d_model + 128, transformer_d_model
        )

        # 1-2. Triplet Loss 함수와 가중치(lambda) 정의
        self.triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
        self.lambda_contrastive_loss = 0.3 # 튜닝이 필요한 하이퍼파라미터
        
        # Feature Extraction
        self.meshregnet = ResNet_(resnet_version=18)

        # Egocentric Knowledge Module (Local Transformer)
        self.transformer_pe=PositionalEncoding(d_model=transformer_d_model) 

        self.transformer_pose=Transformer_Encoder(d_model=transformer_d_model, 
                                nhead=transformer_nhead, 
                                num_encoder_layers=transformer_num_encoder_layers_pose,
                                dim_feedforward=transformer_dim_feedforward,
                                dropout=0.0, 
                                activation="relu", 
                                normalize_before=transformer_normalize_before)
                                    
       
        # Hand Pose Estimation
        # self.scale_factor = scale_factor 
        # self.trans_factor = trans_factor
        # self.image_to_hand_pose=MultiLayerPerceptron(base_neurons=[transformer_d_model, transformer_d_model,transformer_d_model], out_dim=self.num_joints*3,
        #                         act_hidden='leakyrelu',act_final='none')        
        # self.postprocess_hand_pose=To25DBranch(trans_factor=self.trans_factor,scale_factor=self.scale_factor)
        
        # Object classification
        self.num_objects=dataset_info.num_objects
        self.image_to_olabel_embed=torch.nn.Linear(transformer_d_model,transformer_d_model)
        self.obj_classification=ObjClassBranch(num_obj=self.num_objects, feature_dim=transformer_d_model)
        
        # Feature to Action
        self.hand_pose3d_to_action_input=torch.nn.Linear(self.num_joints*2,transformer_d_model)
        self.olabel_to_action_input=torch.nn.Linear(self.num_objects,transformer_d_model)

        # Egocentric Action Module (Global Transformer)
        self.concat_to_action_input=torch.nn.Linear(transformer_d_model*2,transformer_d_model)
        self.num_actions=dataset_info.num_actions
        self.action_token=torch.nn.Parameter(torch.randn(1,1,transformer_d_model))
        
        self.transformer_action=Transformer_Encoder(d_model=transformer_d_model, 
                            nhead=transformer_nhead, 
                            num_encoder_layers=transformer_num_encoder_layers_action,
                            dim_feedforward=transformer_dim_feedforward,
                            dropout=0.0,
                            activation="relu", 
                            normalize_before=transformer_normalize_before) 
        
        self.action_classification= ActionClassificationBranch(num_actions=self.num_actions, action_feature_dim=transformer_d_model)

        # Hand Type Prior
        self.num_handtypes=dataset_info.num_handtypes
        self.hand_type_classification = HandTypeClassificationBranch(num_types=self.num_handtypes, hand_feature_dim=transformer_d_model)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.hand_pred_label_txt = 'None,Quadpod,small Diameter,Medium Diameter,Thumb up,Thumb-Middle Grip,Tip Pinch,Disk Grip,Dynamic Tripod,Fixed Hook,Fist,Large Diameter,parallel Extension,Thumb-2 Finger,Writing Tripod,Tripod,Hand Clench,Pincer Grip,Open Hand,Stirring,Spray-Trigger Grip,Index Finger Flexion,Thumb Tucked,Extended Index Curl,Relaxing Hand,Dynamic Flatten,Dynamic Pinch,Index Finger,Full Rotation,Dynamic Parallel Extension,Dynamic Lateral Pinch,Poking,Middle Rotation,Index Rotation,Adduction Grip,Dynamic Diameter,Palmar,Hammering,Extension Type'.split(',')
        self.tokenized_label = clip.tokenize(self.hand_pred_label_txt).to(self.device)
        self.hlabel_features = self.clip_model.encode_text(self.tokenized_label).detach()
        self.hlabel_concat_to_action_input=torch.nn.Linear(transformer_d_model*2,transformer_d_model)

        self.dataset_name = dataset_info.name

    def forward(self, batch_flatten, epoch=0, train=True, verbose=False):
        flatten_images=batch_flatten[TransQueries.IMAGE].cuda()
        rotation_features = batch_flatten[TransQueries.ROTATION_FEATURE].cuda()
        
                
        #Loss
        total_loss = torch.Tensor([0]).cuda()
        losses = {}
        results = {}

        # ======== Feature Extraction ========
        flatten_in_feature, _ =self.meshregnet(flatten_images) 
        
        # ======== Egocentric Knowledge Module ========
        batch_seq_pin_feature=flatten_in_feature.contiguous().view(-1,self.ntokens_pose,flatten_in_feature.shape[-1])
        batch_seq_pin_pe=self.transformer_pe(batch_seq_pin_feature)
         
        batch_seq_pweights=batch_flatten['not_padding'].cuda().float().view(-1,self.ntokens_pose)
        batch_seq_pweights[:,0]=1.
        batch_seq_pmasks=(1-batch_seq_pweights).bool()
         
        batch_seq_pout_feature,_=self.transformer_pose(src=batch_seq_pin_feature, src_pos=batch_seq_pin_pe,
                            key_padding_mask=batch_seq_pmasks, verbose=False)
 
 
        flatten_pout_feature=torch.flatten(batch_seq_pout_feature,start_dim=0,end_dim=1)
        
        # Hand Pose Estimation
        # flatten_hpose=self.image_to_hand_pose(flatten_pout_feature)
        # flatten_hpose=flatten_hpose.view(-1,self.num_joints,3)
        # flatten_hpose_25d_3d=self.postprocess_hand_pose(sample=batch_flatten,scaletrans=flatten_hpose,verbose=verbose) 

        # weights_hand_loss=batch_flatten['not_padding'].cuda().float()
        # hand_results,total_loss,hand_losses=self.recover_hand(flatten_sample=batch_flatten,flatten_hpose_25d_3d=flatten_hpose_25d_3d,weights=weights_hand_loss,
        #                 total_loss=total_loss,verbose=verbose)        
        # results.update(hand_results)
        # losses.update(hand_losses)

        # Object Classification
        flatten_olabel_feature=self.image_to_olabel_embed(flatten_pout_feature)
        
        weights_olabel_loss=batch_flatten['not_padding'].cuda().float()
        olabel_results,total_loss,olabel_losses=self.predict_object(sample=batch_flatten,features=flatten_olabel_feature,
                        weights=weights_olabel_loss,total_loss=total_loss,verbose=verbose)
        results.update(olabel_results)
        losses.update(olabel_losses)

        # Hand Type Classification
        weights_hlabel_loss=batch_flatten['not_padding'].cuda().float()
        if self.dataset_name == 'h2o':
            hlabel_results, total_loss, hlabel_losses = self.predict_handtype_h2o(
                sample=batch_flatten, 
                features=flatten_in_feature,
                weights=weights_hlabel_loss,
                total_loss=total_loss,
                verbose=verbose
            )
        else:
            hlabel_results, total_loss, hlabel_losses = self.predict_handtype(
                sample=batch_flatten, 
                features=flatten_in_feature,
                weights=weights_hlabel_loss,
                total_loss=total_loss,
                verbose=verbose
            )
        results.update(hlabel_results)
        losses.update(hlabel_losses)
        
        #Rotation contrastive
        rotation_embedding = self.rotation_encoder(rotation_features)

        # ======== Egocentric Action Module ========
        # flatten_hpose2d=torch.flatten(flatten_hpose[:,:,:2],1,2)
        # flatten_ain_feature_hpose=self.hand_pose3d_to_action_input(flatten_hpose2d) # flatten_ain_feature_hpose shape : (B * 128, 512)
        flatten_ain_feature_olabel=self.olabel_to_action_input(olabel_results["obj_reg_possibilities"]) # flatten_ain_feature_olabel shape : (B * 128, 512)
        
        hand_pred_label_features = torch.stack([self.hlabel_features[int(value)] for value in hlabel_results['hand_pred_labels']])
        flatten_ain_feature_hlabel_txt=torch.nn.functional.normalize(hand_pred_label_features).to(torch.cuda.current_device())
        
        flatten_ain_feature=torch.cat((flatten_pout_feature,flatten_ain_feature_olabel),dim=1)
        flatten_ain_feature=self.concat_to_action_input(flatten_ain_feature) # (B * 128, 512)
        

        # if train:
        flatten_ain_feature=torch.cat((flatten_ain_feature, flatten_ain_feature_hlabel_txt), dim=1) # (B * 128, 1024)
        flatten_ain_feature=self.hlabel_concat_to_action_input(flatten_ain_feature)
        
        #ROTATION CONCAT
        expanded_rotation_embedding = rotation_embedding.unsqueeze(1).expand(-1, self.ntokens_action, -1)
        flatten_rotation_embedding = expanded_rotation_embedding.reshape(-1, 128)
        flatten_ain_feature = torch.cat((flatten_ain_feature, flatten_rotation_embedding), dim=1)
        flatten_ain_feature = self.final_feature_fusion_layer(flatten_ain_feature)

        batch_seq_ain_feature=flatten_ain_feature.contiguous().view(-1,self.ntokens_action,flatten_ain_feature.shape[-1])
        
        # Concat trainable token
        batch_aglobal_tokens = repeat(self.action_token,'() n d -> b n d',b=batch_seq_ain_feature.shape[0])
        batch_seq_ain_feature=torch.cat((batch_aglobal_tokens,batch_seq_ain_feature),dim=1)
        batch_seq_ain_pe=self.transformer_pe(batch_seq_ain_feature)
 
        batch_seq_weights_action=batch_flatten['not_padding'].cuda().float().view(-1,self.ntokens_action)
        batch_seq_amasks_frames=(1-batch_seq_weights_action).bool()
        batch_seq_amasks_global=torch.zeros_like(batch_seq_amasks_frames[:,:1]).bool() 
        batch_seq_amasks=torch.cat((batch_seq_amasks_global,batch_seq_amasks_frames),dim=1)        
         
        batch_seq_aout_feature,_=self.transformer_action(src=batch_seq_ain_feature, src_pos=batch_seq_ain_pe,
                                key_padding_mask=batch_seq_amasks, verbose=False)
        
        # Action Classification
        batch_out_action_feature=torch.flatten(batch_seq_aout_feature[:,0],1,-1)     
        weights_action_loss=torch.ones_like(batch_flatten['not_padding'].cuda().float()[0::self.ntokens_action]) 

        action_results, total_loss, action_losses=self.predict_action(sample=batch_flatten,features=batch_out_action_feature, weights=weights_action_loss,
                        total_loss=total_loss,verbose=verbose)
        
        results.update(action_results)
        losses.update(action_losses)
        
        # --- ✅ 2. Contrastive Loss 계산 (이 위치로 이동) ---
        
        if train:
            anchor, positive, negative = self._get_triplets_from_batch(
                embeddings=torch.nn.functional.normalize(rotation_embedding, p=2, dim=1),
                labels=action_results["action_gt_labels"] # 정상적으로 사용 가능
            )
            
            if anchor is not None:
                contrastive_loss = self.triplet_loss(anchor, positive, negative)
                total_loss += self.lambda_contrastive_loss * contrastive_loss
                losses["contrastive_loss"] = contrastive_loss.detach()
                
    
        return total_loss, results, losses
        
        
    # def recover_hand(self, flatten_sample, flatten_hpose_25d_3d, weights, total_loss,verbose=False):
    #     hand_results, hand_losses={},{}
        
    #     joints3d_gt = flatten_sample[BaseQueries.JOINTS3D].cuda()
    #     hand_results["gt_joints3d"]=joints3d_gt
    #     hand_results["pred_joints3d"]=flatten_hpose_25d_3d["rep3d"].detach().clone()
    #     hand_results["pred_joints2d"]=flatten_hpose_25d_3d["rep2d"]
    #     hand_results["pred_jointsz"]=flatten_hpose_25d_3d["rep_absz"]
 
            
    #     hpose_loss=0.
        
    #     joints25d_gt = flatten_sample[TransQueries.JOINTSABS25D].cuda()
    #     hand_losses=compute_hand_loss(est2d=flatten_hpose_25d_3d["rep2d"],
    #                                 gt2d=joints25d_gt[:,:,:2],
    #                                 estz=flatten_hpose_25d_3d["rep_absz"],
    #                                 gtz=joints25d_gt[:,:,2:3],
    #                                 est3d=flatten_hpose_25d_3d["rep3d"],
    #                                 gt3d= joints3d_gt,
    #                                 weights=weights,
    #                                 is_single_hand=self.is_single_hand,
    #                                 pose_loss=self.pose_loss,
    #                                 verbose=verbose)

    #     hpose_loss+=hand_losses["recov_joints2d"]*self.lambda_hand_2d + hand_losses["recov_joints_absz"]*self.lambda_hand_z + hand_losses["recov_joint_angle"]*self.lambda_hand_z/10+hand_losses["recov_joint_NCJ"]*self.lambda_hand_z

    #     if total_loss is None:
    #         total_loss= hpose_loss
    #     else:
    #         total_loss += hpose_loss
                
    #     return hand_results, total_loss, hand_losses

    def predict_object(self,sample,features, weights, total_loss,verbose=False):
        olabel_feature=features
        out=self.obj_classification(olabel_feature)
        batch_size = features.shape[0] #2 X 128
        num_classes = 63
        olabel_results, olabel_losses={},{}
        
        obj_idx_list = sample[BaseQueries.OBJIDX]  # list of list[int], e.g. [[3], [1,4]]
        olabel_gts = torch.zeros((batch_size, num_classes), device=features.device)
        
        for i, obj_ids in enumerate(obj_idx_list):
            for obj_id in obj_ids:
                olabel_gts[i, obj_id] = 1.0 
                
        olabel_results["obj_gt_labels"]=olabel_gts
        olabel_results["obj_pred_labels"]=out["pred_labels"]
        olabel_results["obj_reg_possibilities"]=out["reg_possibilities"]
        
        # print(f"[GT Log] Object GT (shape): {olabel_gts.shape}")
        # print(f"[Pred Log] Pred Labels shape: {out['pred_labels'].shape}")
        
        bce_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
        olabel_loss = bce_loss_fn(out["reg_outs"], olabel_gts)  # [B, C]
        olabel_loss = torch.sum(olabel_loss, dim=1)  # sum over classes per sample
        olabel_loss = torch.mul(olabel_loss, weights.flatten())
        olabel_loss = torch.sum(olabel_loss) / torch.sum(weights)
        

        if total_loss is None:
            total_loss=self.lambda_action_loss*olabel_loss
        else:
            total_loss+=self.lambda_action_loss*olabel_loss
            olabel_losses["olabel_loss"]=olabel_loss
        return olabel_results, total_loss, olabel_losses
    

    def predict_handtype(self, sample, features, weights, total_loss, verbose=False):   # h2o용(양손데이터)
        hand_feature = features
        out=self.hand_type_classification(hand_feature)
        
        hlabel_results, hlabel_losses={},{}
        hlabel_gts_left = sample['hand_label_left'].cuda()
        hlabel_gts_right = sample['hand_label_right'].cuda()

        # =========== Focus on Right hand ===========
        hlabel_gts = hlabel_gts_right

        hlabel_results["hand_gt_labels"]=hlabel_gts
        hlabel_results["hand_pred_labels"]=out["pred_labels"]
        hlabel_results["hand_reg_possibilities"]=out["reg_possibilities"]
        
        
        # print(f"[GT Log]  handtype GT (shape): {hlabel_gts.shape}")
        # print(f"[Pred Log] Logits shape: {out['reg_outs'].shape}")


        hlabel_loss = torch_f.cross_entropy(out["reg_outs"],hlabel_gts,reduction='none')
        hlabel_loss = torch.mul(torch.flatten(hlabel_loss),torch.flatten(weights))
            
        hlabel_loss=torch.sum(hlabel_loss)/torch.sum(weights)

        if total_loss is None:
            total_loss=self.lambda_handtype_loss*hlabel_loss
        else:
            total_loss+=self.lambda_handtype_loss*hlabel_loss
            hlabel_losses["hlabel_loss"]=hlabel_loss
        return hlabel_results, total_loss, hlabel_losses


    def predict_action(self,sample,features,weights,total_loss=None,verbose=False):
        action_feature=features
        out=self.action_classification(action_feature)
        
        action_results, action_losses={},{}
        action_gt_labels=sample[BaseQueries.ACTIONIDX].cuda()[0::self.ntokens_action].clone()
        action_results["action_gt_labels"]=action_gt_labels
        action_results["action_pred_labels"]=out["pred_labels"]
        
        # print(f"[GT Log] Logits shape: {action_gt_labels.shape}")
        # print(f"[pred Log] Logits shape: {out['reg_outs'].shape}")


        action_results["action_reg_possibilities"]=out["reg_possibilities"]
        action_loss = torch_f.cross_entropy(out["reg_outs"],action_gt_labels,reduction='none')  
        action_loss = torch.mul(torch.flatten(action_loss),torch.flatten(weights)) 
        action_loss=torch.sum(action_loss)/torch.sum(weights) 

        if total_loss is None:
            total_loss=self.lambda_action_loss*action_loss
        else:
            total_loss+=self.lambda_action_loss*action_loss
        action_losses["action_loss"]=action_loss
        return action_results, total_loss, action_losses

    def _get_triplets_from_batch(self, embeddings, labels):
        """
        배치 내에서 Hard Triplet을 구성하는 헬퍼 함수.
        (Anchor-Positive 거리는 최대로, Anchor-Negative 거리는 최소로)
        """
        # embeddings: (B, 128), labels: (B,)
        dist_matrix = torch.cdist(embeddings, embeddings, p=2) # 유클리드 거리 행렬 계산
        
        anchors, positives, negatives = [], [], []
        
        for i in range(len(labels)):
            anchor_label = labels[i]
            
            # --- Positive 찾기 ---
            # 자기 자신을 제외하고 라벨이 같은 샘플들
            positive_mask = (labels == anchor_label) & (torch.arange(len(labels), device=labels.device) != i)
            if not torch.any(positive_mask):
                continue # Positive 후보가 없으면 건너뛰기

            # 같은 라벨 샘플 중 가장 거리가 먼 샘플 (Hard Positive)
            positive_dists = dist_matrix[i][positive_mask]
            hardest_positive_idx = torch.where(positive_mask)[0][torch.argmax(positive_dists)]
            
            # --- Negative 찾기 ---
            # 라벨이 다른 모든 샘플들
            negative_mask = (labels != anchor_label)
            if not torch.any(negative_mask):
                continue # Negative 후보가 없으면 건너뛰기

            # 다른 라벨 샘플 중 가장 거리가 가까운 샘플 (Hard Negative)
            negative_dists = dist_matrix[i][negative_mask]
            hardest_negative_idx = torch.where(negative_mask)[0][torch.argmin(negative_dists)]
            
            # 유효한 Triplet 쌍 추가
            anchors.append(embeddings[i])
            positives.append(embeddings[hardest_positive_idx])
            negatives.append(embeddings[hardest_negative_idx])

        if not anchors:
            return None, None, None
            
        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)
