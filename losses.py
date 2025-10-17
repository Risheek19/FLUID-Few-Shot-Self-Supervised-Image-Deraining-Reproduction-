import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# -----------------------------------------------------------
# 1️⃣ Binary Cross Entropy Loss (for PEN)
# -----------------------------------------------------------
class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        return self.bce(pred, target)


# -----------------------------------------------------------
# 2️⃣ Mean Squared Error Loss (for SSN)
# -----------------------------------------------------------
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.mse(pred, target)


# -----------------------------------------------------------
# 3️⃣ Total Variation Loss (for SSN - smoothness regularizer)
# -----------------------------------------------------------
class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, img):
        """Compute isotropic Total Variation (TV) loss."""
        batch_size = img.size()[0]
        h_tv = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
        w_tv = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
        return (h_tv + w_tv) / batch_size


# -----------------------------------------------------------
# 4️⃣ VGG Perceptual Loss (GPU-safe)
# -----------------------------------------------------------
class VGGPerceptualLoss(nn.Module):
    def __init__(self, device=None, resize=True, layers=['relu2_2']):
        """
        Computes perceptual loss based on feature maps from VGG16.
        Moves all feature extraction blocks to GPU if available.
        """
        super(VGGPerceptualLoss, self).__init__()

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # ✅ Load pretrained VGG16 features
        vgg_features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.eval()

        self.selected_layers = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15,
            'relu4_3': 22,
        }

        self.layers = layers
        self.blocks = nn.ModuleDict()
        for l in layers:
            end_layer = self.selected_layers[l]
            block = vgg_features[:end_layer + 1].to(self.device)  # ✅ move to device
            block.eval()
            self.blocks[l] = block

        for block in self.blocks.values():
            for param in block.parameters():
                param.requires_grad = False

        self.resize = resize

    def forward(self, img1, img2):
        """Compute perceptual loss between two images."""
        def vgg_normalize(x):
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            return (x - mean) / std

        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        if self.resize:
            img1 = F.interpolate(img1, size=(224, 224), mode='bilinear', align_corners=False)
            img2 = F.interpolate(img2, size=(224, 224), mode='bilinear', align_corners=False)

        img1 = vgg_normalize(img1)
        img2 = vgg_normalize(img2)

        loss = 0.0
        for l in self.layers:
            f1 = self.blocks[l](img1)
            f2 = self.blocks[l](img2)
            loss += F.mse_loss(f1, f2)

        return loss


# -----------------------------------------------------------
# 5️⃣ Combined SSN Loss (as per paper)
# -----------------------------------------------------------
class SSNLoss(nn.Module):
    def __init__(self, lambda_tv=1e-3, lambda_vgg=0.04, device=None):
        super(SSNLoss, self).__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mse_loss = MSELoss()
        self.tv_loss = TotalVariationLoss()
        self.vgg_loss = VGGPerceptualLoss(device=self.device)
        self.lambda_tv = lambda_tv
        self.lambda_vgg = lambda_vgg

    def forward(self, pred, prior, rainy):
        """
        pred  : Derained output (I_hat)
        prior : Prior generated image (I_rho)
        rainy : Input rainy image (I_NL)
        """
        pred, prior, rainy = pred.to(self.device), prior.to(self.device), rainy.to(self.device)

        l_mse = self.mse_loss(pred, prior)
        l_tv = self.tv_loss(pred)
        l_vgg = self.vgg_loss(rainy, pred)

        total_loss = 0.1 * l_mse + self.lambda_tv * l_tv + self.lambda_vgg * l_vgg
        return total_loss, {'MSE': l_mse.item(), 'TV': l_tv.item(), 'VGG': l_vgg.item()}
