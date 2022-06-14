
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
    def _contrastive_loss(self, z1, z2):
        num = torch.exp(self.sim(z1, z2)/self.temperature)
        denom = self._eval_denom(z1, z2)
        loss = -torch.mean(torch.log(num/denom))
        return loss
    def forward(self, z1, z2):
        return self._contrastive_loss(z1, z2)



#NOTE: This loss might be wrong
class TstudentSupervisedContrastiveLoss(nn.Module):
    def __init__(self, v_fd=2):
        super(TstudentSupervisedContrastiveLoss, self).__init__()
        self.sim = nn.CosineSimilarity()
        self.v_fd = v_fd #5 #0.02 #0.05 # degree of freedom for student distribution
        
    def _eval_denom(self, z1, z2):
        cosine_vals = []
        for v in z1:
            cosine_vals.append(self.sim(v.view(1,-1), z2))
        cos_batch = torch.cat(cosine_vals, dim=0).view(z1.shape[0], -1)
        denom = torch.sum((1+(1/self.v_fd)*(1-cos_batch)**1)**(-(self.v_fd+1)/2), dim=1)
        return denom

    def _contrastive_loss(self, z1, z2):
        num = (1+(1/self.v_fd)*(1-self.sim(z1,z2))**1)**(-(self.v_fd+1)/2)
        denom = self._eval_denom(z1,z2)
        loss = -1*torch.mean(num/denom)
        return loss

    def forward(self, z1, z2):
        return self._contrastive_loss(z1, z2)

if __name__=="__main__":
    pass