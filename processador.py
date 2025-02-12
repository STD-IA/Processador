import configparser
import socket
import time
import torch
import xmlrpc.client
from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
from torchvision import datasets
from torchvision.transforms import v2
from cnn import CNN

class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)

def treinar(model_name, epochs, learning_rate, weight_decay, replicacoes):
    """Executa o treinamento do modelo e retorna métricas."""
    inicio = time.time()
    acc_media, rep_max = cnn.create_and_train_cnn(model_name, epochs, learning_rate, weight_decay, replicacoes)
    duracao = time.time() - inicio
    return f"{model_name}-{epochs}-{learning_rate}-{weight_decay}-Acurácia média: {acc_media} - Melhor replicação: {rep_max} - Tempo: {duracao}"  

def transformacoes(height, width):
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

def carregar_datasets(transforms):
    """Carrega os datasets de treino, validação e teste."""
    return (
        datasets.ImageFolder('./data/resumido/train/', transform=transforms['train']),
        datasets.ImageFolder('./data/resumido/validation/', transform=transforms['test']),
        datasets.ImageFolder('./data/resumido/test/', transform=transforms['test'])
    )

def config_processamento(config):
    """Configura o número de threads para processamento no PyTorch."""
    num_threads = config.get('Self', 'num_threads')
    if num_threads.startswith('auto'):
        num_threads = torch.get_num_threads() * int(num_threads[4:])
    else:
        num_threads = int(num_threads)
    torch.set_num_threads(num_threads)

def config_ip(config):
    """Obtém o IP do servidor com base na configuração."""
    ip = config.get('Self', 'IP')
    if ip == 'auto':
        return socket.gethostbyname(socket.gethostname())
    return ip

def registrar_servidor(name_server_ip, name_server_port, server_ip, server_port):
    """Registra o servidor no Servidor de Nomes."""
    try:
        client = xmlrpc.client.ServerProxy(f"http://{name_server_ip}:{name_server_port}")
        return client.cadastrar_servidor(server_ip, server_port)
    except Exception as e:
        return f"Erro ao registrar no Servidor de Nomes: {e}"

def main():
    global cnn
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    # Configuração do PyTorch
    config_processamento(config)
    
    # Inicialização do modelo
    transforms = transformacoes(224, 224)

    dados_treino, dados_validacao, dados_teste = carregar_datasets(transforms)
    
    cnn = CNN(dados_treino, dados_validacao, dados_teste, 8)
    
    # Configuração do servidor
    server_ip = config_ip(config)
    server = SimpleXMLRPCServer((server_ip, 0), requestHandler=RequestHandler)
    server_port = server.socket.getsockname()[1]
    print(f"Servidor iniciado em {server_ip}:{server_port}")
    
    # Registrar função de treinamento
    server.register_function(treinar)
    print("Função de treinamento registrada.")
    
    # Registro no Servidor de Nomes
    name_server_ip = config.get('NameServer', 'IP')
    name_server_port = config.getint('NameServer', 'Port')
    print("Registrando no Servidor de nomes...")
    print(registrar_servidor(name_server_ip, name_server_port, server_ip, server_port))
    
    # Iniciar loop de requisições
    print("Servidor pronto para receber requisições.")
    server.serve_forever()

if __name__ == "__main__":
    main()