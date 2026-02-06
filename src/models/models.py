import torch
from transformers import AutoModel
import torch.nn.functional as F

class Dinov2EmbendingExtractor:
    def __init__(self, model_name = "facebook/dinov2-small", gem_p =3.0):
        self.model = AutoModel.from_pretrained(model_name)
        self.gem_p = gem_p

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def gem_pooling(self, x, p=3.0):    
        # GeM(x, p) = (1/N * Σ xi^p)^(1/p)
        batch_size, num_tokes, hidden_dim = x.shape

        x_pow = x.clamp(min=1e-6).pow(p)
        mean_pooled = x_pow.mean(dim=1)

        gem_result = mean_pooled.pow(1.0 / p)
        return gem_result 
    
    def extract_embedding(self, img_tensor):
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0) # Переводим изображение в батч
        else:
            print(f"Тензор некорректной размерности")

        with torch.no_grad():
            outputs = self.model(img_tensor)

            all_tokens = outputs.last_hidden_state

            gem_embedding = self.gem_pooling(all_tokens, p=self.gem_p)

            gem_embedding = gem_embedding.squeeze(1)

            embedding_np = gem_embedding.numpy().flatten()
        
        return embedding_np


