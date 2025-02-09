import configparser
import torch
from torchvision import datasets
from torchvision.transforms import v2
import xmlrpc
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
import socket
import time
from cnn import CNN

cnn = 0

class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2')

#Função de treinamento

def treinar(model_name,epochs,learning_rate,weight_decay,replicacoes):
    inicio = time.time()
    acc_media, rep_max = cnn.create_and_train_cnn(model_name,epochs,learning_rate,weight_decay,replicacoes)
    fim = time.time()
    duracao = fim - inicio
    return f"{model_name}-{epochs}-{learning_rate}-{weight_decay}-Acurácia média: {acc_media} - Melhor replicação: {rep_max} - Tempo:{duracao}"


# Funções complementares

def define_transforms(height, width):
    data_transforms = {
        'train': v2.Compose([
            v2.Resize((height, width)),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'test': v2.Compose([
            v2.Resize((height, width)),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    }
    return data_transforms


def read_images(data_transforms):
    train_data = datasets.ImageFolder('./data/resumido/train/', transform=data_transforms['train'])
    validation_data = datasets.ImageFolder('./data/resumido/validation/', transform=data_transforms['test'])
    test_data = datasets.ImageFolder('./data/resumido/test/', transform=data_transforms['test'])
    return train_data, validation_data, test_data

# Main

if __name__ == "__main__":
    # Criar CNN
    data_transforms = define_transforms(224, 224)
    train_data, validation_data, test_data = read_images(data_transforms)
    cnn = CNN(train_data, validation_data, test_data, 8)
    # Ler arquivo de configuração
    config = configparser.ConfigParser()
    config.read('config.ini')
    # Configurar torch para usar todos os nucelos do processador
    num_threads = config.get('Self','num_threads')
    if(num_threads[0:4]=='auto'):
        num_threads = torch.get_num_threads()*int(num_threads[4:])
    else:
        num_threads = int(num_threads)
    torch.set_num_threads(num_threads)
    # Inicializar servidor
    print("Running Server...")
    # Obter IP
    myIP = config.get('Self','IP')
    if(myIP == 'auto'):
        hostname = socket.gethostname()
        myIP = socket.gethostbyname(hostname)
        print('Host name:', hostname, '\nIP:', myIP)
    server = SimpleXMLRPCServer((myIP, 0), requestHandler=RequestHandler)
    myPort = server.socket.getsockname()[1]
    print(f"Server started at port {myPort}")
    # Registrar função
    print("Registring function...")
    server.register_function(treinar)
    print("Function registered.")
    # Se cadastrar no SDN
    ip = config.get('NameServer','IP')
    port = config.getint('NameServer','Port')
    print("Registering at the Name Server...")
    client = xmlrpc.client.ServerProxy(f"http://{ip}:{str(port)}")
    print(client.cadastrar_servidor(myIP, myPort))
    # Request Loop
    print("Now serving. Waiting requests...")
    server.serve_forever()
