import torch
import time
from torchvision.transforms import v2
from torchvision import datasets
from cnn import CNN

class Treinar:
    def __init__(self):
        global cnn
        transforms = self.transformacoes(224, 224)

        dados_treino, dados_validacao, dados_teste = self.carregar_datasets(transforms)
        
        self.cnn = CNN(dados_treino, dados_validacao, dados_teste, 8)
    
    def carregar_datasets(self, transforms):
        """Carrega os datasets de treino, validação e teste."""
        return (
            datasets.ImageFolder('./data/resumido/train/', transform=transforms['train']),
            datasets.ImageFolder('./data/resumido/validation/', transform=transforms['test']),
            datasets.ImageFolder('./data/resumido/test/', transform=transforms['test'])
        )
        
    def treinar(self, model_name, epochs, learning_rate, weight_decay, replicacoes):
        """Executa o treinamento do modelo e retorna os resultados"""
        inicio = time.time()
        acc_media, rep_max = self.cnn.create_and_train_cnn(model_name, epochs, learning_rate, weight_decay, replicacoes)
        duracao = time.time() - inicio
        return f"{model_name}-{epochs}-{learning_rate}-{weight_decay}-Acurácia média: {acc_media} - Melhor replicação: {rep_max} - Tempo: {duracao}"  

    def transformacoes(self, height, width):
        """Define as transformações para pré-processamento das imagens."""
        return {
            'train': v2.Compose([
                v2.Resize((height, width)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            'test': v2.Compose([
                v2.Resize((height, width)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        }