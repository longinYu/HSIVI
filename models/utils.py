import torch

_MODELS = {}


def register_model(cls=None, *, name=None):
    '''A decorator for registering model classes.'''

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(
                'Already registered model with name: %s' % local_name)
        _MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_model(name):
    return _MODELS[name]


def create_model(config):
    model_name = config.name
    score_model = get_model(model_name)(config)
    return score_model


def get_model_fn(model, train=False):
    def model_fn(x, labels):
        if not train:
            model.eval()
            return model(x, labels)
        else:
            model.train()
            return model(x, labels)

    return model_fn

