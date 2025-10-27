import torch
from torch import Tensor

class SoftSort_p2(torch.nn.Module):
    def __init__(self, tau=1.0, hard=False):
        super(SoftSort_p2, self).__init__()
        self.hard = hard
        self.tau = tau

    def forward(self, scores: Tensor):
        """
        scores: elements to be sorted. Typical shape: batch_size x n
        """
        scores = scores.unsqueeze(-1)
        sorted = scores.sort(descending=False, dim=1)[0]
        pairwise_diff = ((scores.transpose(1, 2) - sorted) ** 2).neg() / self.tau
        P_hat = pairwise_diff.softmax(-1)

        if self.hard:
            P = torch.zeros_like(P_hat, device=P_hat.device)
            P.scatter_(-1, P_hat.topk(1, -1)[1], value=1)
            P_hat = (P - P_hat).detach() + P_hat
        return P_hat