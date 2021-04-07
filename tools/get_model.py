import torch


def load_torchvision(model_name):
    """
    Given a model name, returns a Torchvision model in eval mode as well
    as an example input.
    """
    # Lazy import as torchvision may not be required.
    import torchvision

    with torch.no_grad():
        if model_name.startswith("inception"):
            height = width = 299
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            height = width = 224
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        input_shape = [1, 3, height, width]
        input_data = torch.randn(input_shape).float()
        for channel in range(3):
            input_data[:, channel] -= mean[channel]
            input_data[:, channel] /= std[channel]

        if model_name.startswith("googlenet"):
            model = getattr(torchvision.models, model_name)(pretrained=True, aux_logits=True)
        else:
            model = getattr(torchvision.models, model_name)(pretrained=True)
        model = model.float().eval()
        return model, [input_data]


def load_pretrainedmodels(model_name):
    """
    Given a model name, returns a pretrainedmodels.pytorch model in eval
    mode as well as an example input.

    Available at: https://github.com/Cadene/pretrained-models.pytorch
    """

    # Lazy import as torchvision may not be required.
    import pretrainedmodels

    model = getattr(pretrainedmodels, model_name)().float().eval()
    input_shape = [1, *model.input_size]
    input_data = torch.rand(input_shape).float() * 256
    for channel in range(3):
        input_data[:, channel] -= model.mean[channel]
        input_data[:, channel] /= model.std[channel]
    return model, [input_data]


def load_torchtransformers(model_name):
    """
    Given a model name, returns a pytorch transformers model in eval
    mode.

    Available at: https://github.com/huggingface/transformers
    """

    # There are two versions of huggingface, support both
    try:
        import pytorch_transformers
    except ModuleNotFoundError:
        import transformers as pytorch_transformers

    if model_name == "bert":
        tokenizer = pytorch_transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        model = pytorch_transformers.BertModel.from_pretrained('bert-base-uncased', torchscript=True)
        input_data = torch.tensor([tokenizer.encode(text="Here is some text to encode", add_special_tokens=True)])
    elif model_name == "transformer_xl":
        tokenizer = pytorch_transformers.TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
        model = pytorch_transformers.TransfoXLModel.from_pretrained('transfo-xl-wt103', torchscript=True)
        input_data = torch.tensor([tokenizer.encode(text="Here is some text to encode", add_special_tokens=True)])
    else: 
        raise ValueError(f'{model_name} is not supported. Unknown model name.')

    model = model.eval()
    return model, [input_data]


def load_deepspeech(model_name):
    """ Load DeepSpeech LSTM model from GitHub repo. 

    Unfortunately TVM does not currently support LSTM operators in the PyTorch front-end.
    This is also the case for most other frontends.

        Pytorch frontend missing: ['aten::_pad_packed_sequence', 'aten::lstm',
        'aten::_pack_padded_sequence', 'aten::masked_fill', 'aten::fill_',
        'aten::narrow']
    """

    # For reference:
    # from deepspeech_pytorch.model import DeepSpeech
    # from torch.utils.model_zoo import load_url
    # import torch.onnx

    # pretrained_url = 'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/an4_pretrained_v2.pth'
    # params = load_url(pretrained_url)
    # model = DeepSpeech.load_model_package(params)
    # model.eval()
    # input_sizes = (1, 1, 161, 753)
    # input_data = torch.randn(*input_sizes).float()
    # input_sizes = torch.IntTensor([161]).int()
    # model(input_data, input_sizes)
    # return model, [input_data, input_sizes]

    raise NotImplementedError("TVM pytorch frontend doesn't support all the required "
                              "operators for this model.")


def load_simple_transformer(model_name):
    """
    A simple transformer from pytorch.
    """
    model = torch.nn.Transformer(nhead=2, num_encoder_layers=1, num_decoder_layers=1)
    model = model.eval()
    src = torch.rand((10, 32, 512))
    tgt = torch.rand((20, 32, 512))
    return model, [src, tgt]


def load_single_operators(operator_name):
    """
    Single operators.
    """
    if operator_name == "matmul1":
        def compute(x, y):
            return torch.matmul(x, y)
        inp = torch.rand((100, 30, 40))
        inp2 = torch.rand((40, 50))
        return compute, [inp, inp2]
    elif operator_name == "matmul2":
        def compute(x, y):
            return torch.matmul(x, y)
        inp = torch.rand((30, 30, 30))
        inp2 = torch.rand((30, 30, 30))
        return compute, [inp, inp2]
    elif operator_name == "convolution1":
        model = torch.nn.Conv2d(144, 32, 1)
        model = model.eval()
        inp = torch.rand((1, 144, 28, 28))
        return model, [inp]
    elif operator_name == "convolution2":
        model = torch.nn.Conv2d(16, 33, 3, stride=2)
        model = model.eval()
        inp = torch.rand((20, 16, 50, 100))
        return model, [inp]
    else:
        raise ValueError(f"Operator name {operator_name} not recognised.")


def get_model(model_name, type):
    """
    Get a PyTorch model by type and name. Returns PyTorch trace and input shape dict.
    """

    MODEL_MAP = {"torchvision":       (["*"], load_torchvision),
                 "torchtransformers": (["bert", "transformer_xl"], load_torchtransformers),
                 "github":            (["deepspeech"], load_deepspeech),
                 "custom":            (["simple_transformer"], load_simple_transformer),
                 "op":                (["matmul1", "matmul2", "convolution1", "convolution2"], load_single_operators)}

    if type not in MODEL_MAP:
        raise ValueError(f'{type} is not supported. Unknown type name.')

    model_map_item = MODEL_MAP[type]
    supported_model_names = model_map_item[0]

    if model_name not in supported_model_names and \
            (len(supported_model_names) and supported_model_names[0] != "*"):
        raise ValueError(f'{model_name} is not supported. Unknown model name.')

    baseline_model, baseline_input = model_map_item[1](model_name)

    # Extract model to PyTorch graph
    if torch.cuda.is_available():
        if isinstance(baseline_model, torch.nn.Module):
            baseline_model = baseline_model.cuda()
        baseline_input = [inp.cuda() for inp in baseline_input]

    trace = torch.jit.trace(baseline_model, baseline_input)
    if isinstance(baseline_model, torch.nn.Module):
        trace = trace.float().eval()

        if torch.cuda.is_available():
            trace = trace.cuda()
        else:
            trace = trace.cpu()

    input_names = ["input{}".format(idx) for idx, inp in enumerate(baseline_input)]
    input_shapes = list(zip(input_names, [inp.shape for inp in baseline_input]))
    return trace, input_shapes
