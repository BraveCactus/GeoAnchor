import torch
import torch.nn.functional as F
from transformers import AutoModel

class Dinov2EmbendingExtractor:
    def __init__(self, model_name="facebook/dinov2-small", gem_p=3.0, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.gem_p = gem_p
        
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        print(f"Загружена модель {model_name} на {self.device}")

    def gem_pooling(self, x, p=3.0):    
        """
        Generalized Mean Pooling: GeM(x, p) = (1/N * Σ xi^p)^(1/p)
        
        Args:
            x: torch.Tensor [batch_size, num_tokens, hidden_dim]
            p: float, степень для pooling
        
        Returns:
            torch.Tensor [batch_size, hidden_dim]
        """
        
        return x.clamp(min=1e-6).pow(p).mean(dim=1).pow(1./p)
    
    def extract_embedding(self, img_tensor):
        """
        Извлекает эмбеддинг из тензора изображения
        
        Args:
            img_tensor: torch.Tensor [3, H, W] или [batch_size, 3, H, W]
                       H,W должны быть 518 для DINOv2
        
        Returns:
            np.array: эмбеддинг (384 чисел для dinov2-small)
        """        
        img_tensor = img_tensor.to(self.device)        
        
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)
        elif len(img_tensor.shape) != 4:
            raise ValueError(f"Некорректная размерность тензора: {img_tensor.shape}")
        
        
        if img_tensor.shape[-2] != 518 or img_tensor.shape[-1] != 518:
            print(f"Предупреждение: Изображение имеет размер {img_tensor.shape[-2:]} вместо 518×518")            
            if img_tensor.shape[-2] != 518 or img_tensor.shape[-1] != 518:
                print(f"  Автоматический ресайз до 518×518")
                img_tensor = F.interpolate(img_tensor, size=(518, 518), mode='bilinear', align_corners=False)
        
        with torch.no_grad():            
            outputs = self.model(img_tensor)            
            
            all_tokens = outputs.last_hidden_state[:, 1:, :]  # [batch_size, num_tokens-1, hidden_dim]            
            
            gem_embedding = self.gem_pooling(all_tokens, p=self.gem_p)            
            
            embedding_np = gem_embedding.cpu().numpy()
        
        return embedding_np[0] if embedding_np.shape[0] == 1 else embedding_np