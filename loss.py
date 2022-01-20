import einops
import torch
import torch.nn.functional as F


def contrastive_loss(feats, t=0.07):
    feats = F.normalize(feats, dim=2)  # B x K x C
    scores = torch.einsum('aid, bjd -> abij', feats, feats)
    scores = einops.rearrange(scores, 'a b i j -> (a i) (b j)')

    # positive logits: Nx1
    pos_idx = einops.repeat(torch.eye(feats.size(1), dtype=torch.int, device=feats.device), 'i j -> (a i) (b j)', a=feats.size(0), b=feats.size(0))
    pos_idx.fill_diagonal_(0)
    l_pos = torch.gather(scores, 1, pos_idx.nonzero()[:, 1].view(scores.size(0), -1))
    rand_idx = torch.randint(1, l_pos.size(1), (l_pos.size(0), 1), device=feats.device)
    l_pos = torch.gather(l_pos, 1, rand_idx)

    # negative logits: NxK
    neg_idx = einops.repeat(1-torch.eye(feats.size(1), dtype=torch.int, device=feats.device), 'i j -> (a i) (b j)', a=feats.size(0), b=feats.size(0))
    l_neg = torch.gather(scores, 1, neg_idx.nonzero()[:, 1].view(scores.size(0), -1))
    # logits: Nx(1+K)
    logits = torch.cat([l_pos, l_neg], dim=1)

    # apply temperature
    logits /= t

    # labels: positive key indicators
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=scores.device)
    return F.cross_entropy(logits, labels)

def consistency_loss(masks, image, fg):
    # masks: N x R x H x W
    # image: N x 3 x H x W
    weighted_regions = masks.unsqueeze(2) * image.unsqueeze(1)  # N x R x 3 x H x W
    mask_sum = masks.sum(3).sum(2, keepdim=True)  # N x R x 1
    means = weighted_regions.sum(4).sum(3) / (mask_sum + 1e-5)  # N x R x 3
    diff_sq = (image.unsqueeze(1) - means.unsqueeze(3).unsqueeze(4))**2  # N x R x 3 x H x W
    loss = (diff_sq * masks.unsqueeze(2)*fg.unsqueeze(2)).sum(4).sum(3)  # N x R x 3
    loss /= (fg.unsqueeze(2).sum(4).sum(3) + 1e-5)  # N x R x 3
    return loss.sum(2).sum(1).mean()
