
import torch
from torch import nn

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.sim = nn.CosineSimilarity()
    def _eval_denom(self, z1, z2):
        cosine_vals = []
        for v in z1:
            cosine_vals.append(self.sim(v.view(1,-1), z2)/self.temperature)
        cos_batch = torch.cat(cosine_vals, dim=0).view(z1.shape[0], -1)
        denom = torch.sum(torch.exp(cos_batch),dim=1)
        return denom
    def _contrastive_loss(self, z1, z2, z3):
        num = torch.exp(self.sim(z1, z2)/self.temperature)
        if z3:
            denom = self._eval_denom(z1, z2) + self._eval_denom(z1, z3)
        else:
            denom = self._eval_denom(z1, z2)
        loss = -torch.mean(torch.log(num/denom))
        return loss
    def forward(self, z1, z2, z3=None):
        return self._contrastive_loss(z1, z2, z3)

if __name__=="__main__":
    pass