class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def count_trainable_params(model):
    weights = sum(w.size for w in model.coefs_)
    biases = sum(b.size for b in model.intercepts_)
    return weights + biases
