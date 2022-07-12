#MOVER PARA ARQUIVO DE CONFIGURAÇÃO, CRIANDO FUNÇÃO PARA CARREGAR ARQUIVO
conf = {
    "ORIGIN_DIR": "/home/jolima/Documentos/multilabel-classification/",
    "DESTINY_DIR": "/home/jolima/Documentos/multilabel-classification/",
    "TERM_SIZE_STD": 8,
    "RAM_STD": 16,
    "MIN_TERM_SIZE": 3,
    "MAX_TERM_SIZE": 32,
    "MIN_RAM_SIZE": 8,
    "MAX_RAM_SIZE": 16,
    "SEEDS": 2,
    "IGNORE_ZERO_WSD": False,
    "SAMPLE": 100,
    "Y_COLUMNS":[]
}


class Configuration():
    def __init__(self):
        self._config = conf

    def get_property(self, property_name):
        return self._config.get(property_name)


