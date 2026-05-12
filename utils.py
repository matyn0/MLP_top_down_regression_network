class Tee:
    """Zapisuje jeden print vystup naraz do viacerych streamov."""

    def __init__(self, *streams):
        self.streams = streams  # napr. terminal a otvoreny log subor

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def count_trainable_params(model):
    """Spocita vsetky trenovatelne vahy a biasy v MLP modeli."""
    weights = sum(w.size for w in model.coefs_)  # vahove matice medzi vrstvami
    biases = sum(b.size for b in model.intercepts_)  # bias vektory jednotlivych vrstiev

    return weights + biases
