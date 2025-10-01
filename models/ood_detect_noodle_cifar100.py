import torch
import torch.nn as nn
import torch.nn.functional as F
import models.densenet as dn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models.densenet as dn


def approx_top_k_singular_vectors(H, k, n_iter=3):
    # H: [B, d]
    B, d = H.shape
    Q = torch.randn(d, k, device=H.device)
    #give orthnormal 
    for _ in range(n_iter):
        Z = H.T @ (H @ Q)  # [d, k]
        Q, _ = torch.linalg.qr(Z)  # keep orthonormal
    return Q  # [d, k]

class OOD_Detection(nn.Module):
    def __init__(self, M, K, layers, fnet_type):
        super(OOD_Detection, self).__init__()
        self.M = M
        self.K = K
        self.rank_r =20
        self.device = torch.device('cuda:0')

        if fnet_type == 'densenet':
            self.fnet = dn.DenseNet3(
                layers, num_classes=K, growth_rate=12,
                reduction=0.5, bottleneck=True,
                dropRate=0.0, normalizer=None, r=1)
            self.feature_dim = self.fnet.in_planes
        
        #self.rank_r = max(1, int(0.2 * self.feature_dim))

        # Transformation matrix
        self.trans = sig_t(self.device, self.K)

    def forward(self, x):
        final_output = self.fnet(x)
        #final_output = F.softmax(final_output, dim=1)

        # Feature extraction
        h = self.fnet.features(x)  # [B, d, H, W]
        h = F.adaptive_avg_pool2d(h, (1, 1)).view(h.size(0), -1)  # [B, d]
        h = F.normalize(h, dim=1)  # perturb & normalize

        # Approximate top-k left singular vectors via power iteration
        Q = approx_top_k_singular_vectors(h, self.rank_r, n_iter=3)  # [d, r]
        h_ID = (h @ Q) @ Q.T  # projection onto top subspace [B, d]
        h_OOD = h - h_ID

        # Column sparse loss
        col_sparse_loss = torch.norm(h_OOD, dim=0).sum()

        # Output with transformation matrix
        t = self.trans()
        out = torch.mm(final_output, t)

        return out, col_sparse_loss

    def feature_list(self, x):
        score, feature_list = self.fnet.feature_list(x)
        h = feature_list[-1]
        h = F.avg_pool2d(h, 8).view(h.size(0), -1)
        h = F.normalize(h, dim=1)

        # Use same power iteration to approximate top subspace
        Q = approx_top_k_singular_vectors(h, self.rank_r, n_iter=3)
        h_ID = (h @ Q) @ Q.T

        return score, h_ID

    
    def feature_list_val(self, x):
        """
        Return the model's feature list, final output, and matrices A.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width].

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
                Final output, feature list, and A.
        """
        # Compute final output and feature list
        score, feature_list = self.fnet.feature_list(x)
        h = feature_list[-1]  # The penultimate feature map (Shape: [batch_size, 342, 8, 8])
        
        # Apply Global Average Pooling to penultimate features
        #h = F.adaptive_avg_pool2d(h, (1, 1))  # GAP reduces [batch_size, 342, 8, 8] -> [batch_size, 342, 1, 1]
        h = F.avg_pool2d(h, 8)
        h = h.view(h.size(0), -1)  # Flatten to [batch_size, 342]
        epsilon = 1e-6
        #h = h + epsilon * torch.randn_like(h)
        h = F.normalize(h, dim=1)  # Normalize to unit norm per example
        
        return score,h


    def get_trans_matrices(self):
        """
        Return the model's feature list, final output, and matrices A.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width].

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
                Final output, feature list, and A.
        """
        
        # Compute A (softmax normalization of self.P)
        A = F.softmax(self.P, dim=1)  # Shape: [M, K, K]

        return  A
class sig_t(nn.Module):
    def __init__(self, device, num_classes, init=2):
        super(sig_t, self).__init__()

        self.register_parameter(name='w', param=nn.parameter.Parameter(-init*torch.ones(num_classes, num_classes)))

        self.w.to(device)

        co = torch.ones(num_classes, num_classes)
        ind = np.diag_indices(co.shape[0])
        co[ind[0], ind[1]] = torch.zeros(co.shape[0])
        self.co = co.to(device)
        self.identity = torch.eye(num_classes).to(device)


    def forward(self):
        sig = torch.sigmoid(self.w)
        T = self.identity.detach() + sig*self.co.detach()
        T = F.normalize(T, p=1, dim=1)

        return T     


