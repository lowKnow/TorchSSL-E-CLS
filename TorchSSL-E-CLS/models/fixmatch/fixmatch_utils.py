import gc

import torch
import torch.nn.functional as F
from train_utils import ce_loss


class Get_Scalar:
    def __init__(self, value):
        self.value = value

    def get_value(self, iter):
        return self.value

    def __call__(self, iter):
        return self.value


def compute_distances(query_samples, proto_bank):
    """
    计算每个查询样本与每个原型之间的欧氏距离
    Args:
    - query_samples: 查询样本的张量，形状为(batch_size, channel)
    - proto_bank: 原型字典，键为0~class_num，值为通道数的字典

    Returns:
    - distances: 欧氏距离的张量，形状为(batch_size, class_num)
    """

    batch_size = query_samples.size(0)
    class_num = len(proto_bank)
    channel = query_samples.size(1)

    # 初始化距离张量
    distances = torch.zeros(batch_size, class_num)

    # 计算每个查询样本与每个原型之间的欧氏距离
    for i, (class_label, prototypes) in enumerate(proto_bank.items()):
        # 获取当前类别的原型值
        proto_values = prototypes.unsqueeze(0).expand(batch_size, -1, -1)  # 扩展为(batch_size, num_prototypes, channel)

        # 计算欧氏距离的平方
        dists_squared = torch.pow(query_samples.unsqueeze(1) - proto_values, 2).sum(dim=2)

        # 计算欧氏距离
        dists = torch.sqrt(dists_squared)

        # 将距离存储在距离张量的相应位置
        distances[:, i] = dists.squeeze()
        del proto_values
        gc.collect()

    return distances


def consistency_loss(gpu, proto_bank, ft_x_ulb_s, logits_s, logits_w, name='ce', T=1.0, p_cutoff=0.0,
                     use_hard_labels=True):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    ft_x_ulb_s = ft_x_ulb_s.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')

    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(p_cutoff).float()
        select = max_probs.ge(p_cutoff).long()

        num_classes = 10
        dists = compute_distances(ft_x_ulb_s, proto_bank)
        dists = 1 / (dists + 1e-8)
        logits_dists = torch.softmax(dists, dim=1).to(gpu)
        del dists
        u_max_prob, u_label = torch.max(logits_dists, dim=-1)

        select_samples = ft_x_ulb_s[mask == 1]
        for class_label in range(num_classes):
            class_indices = torch.arange(select_samples.size(0))[u_label[mask == 1] == class_label]
            if len(class_indices) > 0:
                class_prototype = select_samples[class_indices].mean(dim=0)
                if not torch.isnan(class_prototype).any() and class_label in proto_bank:
                    proto_bank[class_label] = (0.999 * proto_bank[class_label] + (1 - 0.999) * class_prototype)
                elif not torch.isnan(class_prototype).any():
                    proto_bank[class_label] = class_prototype
                else:
                    pass

        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
            proto_loss = ce_loss(logits_dists, max_idx, use_hard_labels, reduction='none') * mask
            total_loss = masked_loss + proto_loss
        else:
            pseudo_label = torch.softmax(logits_w / T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return total_loss.mean(), mask.mean(), select, max_idx.long(), proto_bank

    else:
        assert Exception('Not Implemented consistency_loss')
