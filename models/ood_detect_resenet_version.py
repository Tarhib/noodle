import torch
import torch.nn as nn
import torch.nn.functional as F
import models.densenet as dn
from models.resnet_in import resnet
from models.resnet import ResNet50
from models.resnet_custom import ResNet18,ResNet34

class OOD_Detection(nn.Module):
    def __init__(self, M, K, layers, fnet_type):
        super(OOD_Detection, self).__init__()
        self.M = M
        self.K = K

        if fnet_type == 'densenet':
            # Instantiate DenseNet3. Ensure that DenseNet3 sets the attribute 'in_planes'
            self.fnet = dn.DenseNet3(layers, num_classes=K, growth_rate=12, reduction=0.5,
                                     bottleneck=True, dropRate=0.0, normalizer=None, r=1)
            final_feature_dim = self.fnet.in_planes  # expected feature dim from DenseNet
        elif fnet_type == 'resnet50':
            self.fnet = ResNet50(r = 1, num_class = K)
            # For ResNet50, enforce the expected feature dimension for the penultimate layer
            final_feature_dim = 512 * self.fnet.block.expansion  # expansion is usually 1 for ResNet18
        elif fnet_type == 'resnet18':
            self.fnet = ResNet18(num_classes=K)
            # For ResNet18, typically the last feature map has 512 channels (or 512*expansion)
            final_feature_dim = 512 * self.fnet.block.expansion  # expansion is usually 1 for ResNet18
        elif fnet_type == 'resnet34':
            self.fnet = ResNet34(num_classes=K)
            # For ResNet18, typically the last feature map has 512 channels (or 512*expansion)
            final_feature_dim = 512 * self.fnet.block.expansion  # expansion is usually 1 for ResNet18
        else:
            raise ValueError("Unsupported fnet_type provided.")

        # Define linear layers using the determined final feature dimension
        self.linear = nn.Linear(final_feature_dim, M * K, bias=True)
        self.linear_d = nn.Linear(final_feature_dim, M, bias=True)

        # Parameter for mixture weights (initialized as a uniform distribution)
        self.param_d = nn.Parameter(torch.full((M,), 1.0 / M), requires_grad=True)
        # Parameter B is a stack of identity matrices (one per mixture component)
        self.param_B = nn.Parameter(torch.stack([torch.eye(K) for _ in range(M)]), requires_grad=True)

    def forward(self, x):
        # Extract features using the backbone network
        _ = self.fnet(x)  # final classifier output (unused here)
        h = self.fnet.features(x)  # penultimate feature map
        assert not torch.isnan(h).any(), f"NaN detected in features"

        # Global Average Pooling to get a vector per sample
        h = F.adaptive_avg_pool2d(h, (1, 1))
        h = h.view(h.size(0), -1)  # shape: [batch, final_feature_dim]

        # Transform features using the linear layer and reshape to [batch, M, K]
        h_tilde = self.linear(h).view(h.size(0), self.M, self.K)
        # Apply softmax along the K-dimension for each of the M components
        h_softmax = F.softmax(h_tilde, dim=2)
        h_softmax_flat = h_softmax.view(h.size(0), self.M * self.K)

        # Mixture weights d (softmax so they sum to 1)
        d = F.softmax(self.param_d, dim=0)
        # Normalize B (each of the M blocks of shape [K, K]) row-wise
        B_normalized = F.softmax(self.param_B, dim=1)
        # Reshape B_normalized into a matrix of shape [K, M*K]
        B_flat_normalized = B_normalized.transpose(0, 1).reshape(self.K, self.M * self.K)
        # Expand d to match dimensions
        d_expanded = d.repeat_interleave(self.K)  # shape: [M*K]
        # Weight each block in B
        B_weighted = B_flat_normalized * d_expanded.unsqueeze(0)
        # Compute final prediction Bf_x using the weighted B
        Bf_x = torch.matmul(h_softmax_flat, B_weighted.T)
        ones_tensor = torch.ones(Bf_x.size(0), device=Bf_x.device)
        assert torch.allclose(Bf_x.sum(dim=1), ones_tensor, atol=1e-3), "Bf_x does not sum to 1"

        # For the regularizer, project the flattened softmax outputs
        p_theta_x_star_flat = h_softmax_flat  # shape: [batch, M*K]
        epsilon = 1e-3
        identity = torch.eye(B_normalized.size(-1), device=B_normalized.device)
        B_inv = torch.inverse(B_normalized + epsilon * identity)
        # Construct W by reshaping the inverse of B blocks
        W = B_inv.transpose(0, 1).reshape(self.K, self.M * self.K)
        I = torch.eye(W.shape[0], device=W.device)
        P = W.t() @ torch.linalg.inv(W @ W.t() + epsilon * I) @ W
        # Projection of the softmax output
        project_h_softmax_flat = p_theta_x_star_flat @ P
        # Relative L2 difference as regularizer
        reg_loss = torch.norm(p_theta_x_star_flat - project_h_softmax_flat, p=2, dim=1) / \
                   (torch.norm(p_theta_x_star_flat, p=2, dim=1) + 1e-8)
        reg_loss = reg_loss.mean()

        return Bf_x, B_normalized, h_softmax, reg_loss

    def feature_list(self, x):
        """
        Returns the final output (score), and the penultimate features.
        """
        # Get the classifier output and features from the backbone
        score = self.fnet(x)
        h = self.fnet.features(x)
        assert not torch.isnan(h).any(), f"NaN detected in features"
        h = F.adaptive_avg_pool2d(h, (1, 1))
        h = h.view(h.size(0), -1)
        
        h_tilde = self.linear(h).view(h.size(0), self.M, self.K)
        h_softmax = F.softmax(h_tilde, dim=2)
        h_softmax_flat = h_softmax.view(h.size(0), self.M * self.K)
        
        d = F.softmax(self.param_d, dim=0)
        B_normalized = F.softmax(self.param_B, dim=1)
        B_flat_normalized = B_normalized.transpose(0, 1).reshape(self.K, self.M * self.K)
        d_expanded = d.repeat_interleave(self.K)
        B_weighted = B_flat_normalized * d_expanded.unsqueeze(0)
        Bf_x = torch.matmul(h_softmax_flat, B_weighted.T)
        p_theta_x_star_flat = h_softmax_flat
        
        epsilon = 1e-3
        identity = torch.eye(B_normalized.size(-1), device=B_normalized.device)
        B_inv = torch.inverse(B_normalized + epsilon * identity)
        W = B_inv.transpose(0, 1).reshape(self.K, self.M * self.K)
        I = torch.eye(W.shape[0], device=W.device)
        P = W.t() @ torch.linalg.inv(W @ W.t() + epsilon * I) @ W
        project_h_softmax_flat = p_theta_x_star_flat @ P
        reg_loss = torch.norm(p_theta_x_star_flat - project_h_softmax_flat, p=2, dim=1) / \
                   (torch.norm(p_theta_x_star_flat, p=2, dim=1) + 1e-8)
        reg_loss = reg_loss.mean()
        
        # Here, 'score' is returned as the final classifier output,
        # and 'h' as the penultimate features.
        return Bf_x, h

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
        


class SigT(nn.Module):
    def __init__(self, device, M, K, init=2):
        super(SigT, self).__init__()
        self.M = M
        self.K = K

        # Learnable parameter for the confusion matrix
        self.register_parameter(name='w', param=nn.parameter.Parameter(-init * torch.ones(M, K)))
        self.w.to(device)

        # Coefficient matrix to allow interaction between classes
        co = torch.ones(M, K)
        self.co = co.to(device)

        # Identity matrix for diagonal elements
        self.identity = torch.eye(M).to(device)

    def forward(self):
       sig = torch.sigmoid(self.w)

      # Adjust identity matrix to have the same shape as w
       identity = torch.zeros_like(self.w).to(self.identity.device)
       min_dim = min(self.M, self.K)
       identity[:min_dim, :min_dim] = torch.eye(min_dim).to(self.identity.device)

        # Compute the transformation matrix T
       T = identity.detach() + sig * self.co.detach()

        # Normalize T along rows to ensure probabilities sum to 1
       T = F.normalize(T, p=1, dim=1)

       return T
