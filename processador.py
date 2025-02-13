import configparser
import socket
import torch
import xmlrpc.client
from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
from treinador import Treinar

class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)

class Processador:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        
        # Configuração do PyTorch
        self.config_processamento(config)
        
        # Inicialização do treinamento do modelo
        treinadorObj = Treinar()

        # Configuração do servidor
        server_ip = self.config_ip(config)
        server = SimpleXMLRPCServer((server_ip, 0), requestHandler=RequestHandler)
        server_port = server.socket.getsockname()[1]
        print(f"Servidor iniciado em {server_ip}:{server_port}")
        
        # Registrar função de treinamento
        server.register_function(treinadorObj.treinar)
        print("Função de treinamento registrada.")
        
        # Registro no Servidor de Nomes
        name_server_ip = config.get('NameServer', 'IP')
        name_server_port = config.getint('NameServer', 'Port')
        print("Registrando no Servidor de nomes...")

        resultado_registro_servidor = self.registrar_servidor(name_server_ip, name_server_port, server_ip, server_port)
        print(resultado_registro_servidor)
        
        # Iniciar loop de requisições
        print("Servidor pronto para receber requisições.")
        server.serve_forever()

    def config_processamento(self, config):
        """Configura o número de threads para processamento no PyTorch."""
        num_threads = config.get('Self', 'num_threads')

        if num_threads.startswith('auto'):
            num_threads = torch.get_num_threads() * int(num_threads[4:])
        else:
            num_threads = int(num_threads)

        torch.set_num_threads(num_threads)

    def config_ip(self, config):
        """Obtém o IP do servidor com base na configuração."""
        ip = config.get('Self', 'IP')

        if ip == 'auto':
            return socket.gethostbyname(socket.gethostname())
        return ip

    def registrar_servidor(self, name_server_ip, name_server_port, server_ip, server_port):
        """Registra o servidor no Servidor de Nomes."""
        try:
            client = xmlrpc.client.ServerProxy(f"http://{name_server_ip}:{name_server_port}")
            return client.cadastrar_servidor(server_ip, server_port)
        except Exception as e:
            return f"Erro ao registrar no Servidor de Nomes: {e}"

if __name__ == "__main__":
    Processador()