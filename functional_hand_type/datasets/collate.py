"""
Inspired from https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
"""
import re
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from datasets.queries import BaseQueries, TransQueries 
np_str_obj_array_pattern = re.compile(r"[SaUO]") 


def meshreg_collate(batch, extend_queries=None):
    """
    Collate function for meshreg models. Handles variable-sized fields and multi-modal image tensors.
    """
    pop_queries = []
    for poppable_query in extend_queries or []:
        if poppable_query in [BaseQueries.OBJIDX, TransQueries.CAMINTR]:
            continue
        if poppable_query in batch[0]:
            pop_queries.append(poppable_query)

    for pop_query in pop_queries:
        # ✅ Case 1: Multi-modal list (like [rgb, depth, thermal])
        if isinstance(batch[0][pop_query], list) and isinstance(batch[0][pop_query][0], torch.Tensor):
            num_modals = len(batch[0][pop_query])
            modalwise_batches = [[] for _ in range(num_modals)]  # e.g., [[], [], []]

            for sample in batch:
                for i in range(num_modals):
                    modalwise_batches[i].append(sample[pop_query][i])

            # Stack modal-wise
            batch_modal = [torch.stack(modal, dim=0) for modal in modalwise_batches]
            # 저장: batch[pop_query] = [B, C, H, W] * 3 형태 유지
            for i, sample in enumerate(batch):
                sample[pop_query] = [modal[i] for modal in batch_modal]
        
        # ✅ Case 2: 일반 텐서 (기존 방식)
        elif isinstance(batch[0][pop_query], np.ndarray) or torch.is_tensor(batch[0][pop_query]):
            max_size = max([sample[pop_query].shape[0] for sample in batch])
            for sample in batch:
                pop_value = sample[pop_query]
                pop_value = np.concatenate([pop_value] * int(max_size / pop_value.shape[0] + 1))[:max_size]
                sample[pop_query] = pop_value

    # OBJIDX는 리스트로 유지
    obj_idx_list = [sample.pop(BaseQueries.OBJIDX, None) for sample in batch]

    # 기본 collate
    batch_collated = default_collate(batch)

    # 다시 붙이기
    batch_collated[BaseQueries.OBJIDX] = obj_idx_list

    return batch_collated



def seq_extend_flatten_collate(seq, extend_queries=None):
    batch=[]    
    seq_len = len(seq[0])#len(seq) is batch size, seq_len is num frames per sample

    for sample in seq:
        for seq_idx in range(seq_len):
            batch.append(sample[seq_idx])
    return meshreg_collate(batch,extend_queries)



def collate_with_rotation_feature(batch, extend_queries=None):
    """
    (시퀀스, 회전_텐서) 튜플로 구성된 배치를 처리하는 함수.
    """
    # 1. 시퀀스 샘플 리스트와 회전 특징 텐서를 분리합니다.
    list_of_sample_sequences = [item[0] for item in batch]
    rotation_features = [item[1] for item in batch]
    wrist_direction_features = [item[2] for item in batch]
    
    # 2. 기존 함수를 호출하여 시퀀스 샘플들을 처리합니다.
    collated_samples = seq_extend_flatten_collate(list_of_sample_sequences, extend_queries)

    # 3. 분리해 둔 회전 특징들을 하나의 배치 텐서로 합칩니다.
    collated_rotation_tensor = torch.stack(rotation_features, dim=0)
    collated_wrist_direction_tensor = torch.stack(wrist_direction_features, dim=0)
    
    # 4. 최종 배치 딕셔너리에 회전 특징 텐서를 추가합니다.
    collated_samples[TransQueries.ROTATION_FEATURE] = collated_rotation_tensor
    collated_samples[TransQueries.WRIST_DIRECTION_FEATURE] = collated_wrist_direction_tensor

    
    # print(collated_samples)

    return collated_samples