#!/usr/bin/env python3
import torch


def load_torchvision(model_name):
    """Given a model name, returns a Torchvision model in eval mode as well
    as an example input."""
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
    """Given a model name, returns a pretrainedmodels.pytorch model in eval
    mode as well as an example input."""
    import pretrainedmodels  # https://github.com/Cadene/pretrained-models.pytorch

    model = getattr(pretrainedmodels, model_name)().float().eval()
    input_shape = [1, *model.input_size]
    input_data = torch.rand(input_shape).float() * 256
    for channel in range(3):
        input_data[:, channel] -= model.mean[channel]
        input_data[:, channel] /= model.std[channel]
    return model, [input_data]


def load_torchtransformers(model_name):
    """Given a model name, returns a pytorch transformers model in eval
    mode. Provided by HuggingFace: https://github.com/huggingface/transformers """
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
        raise ValueError('Model name unknown.')

    model = model.eval()

    return model, [input_data]


def load_deepspeech():
    """ Load DeepSpeech LSTM model from GitHub repo. 
    
    Pytorch frontend missing: ['aten::_pad_packed_sequence', 'aten::lstm', 
    'aten::_pack_padded_sequence', 'aten::masked_fill', 'aten::fill_', 
    'aten::narrow']
    """
    raise NotImplementedError("TVM pytorch frontend doesn't support all the required operators for this model.")

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


def load_simple_transformer():
    """ A simple transformer from pytorch. """
    model = torch.nn.Transformer(nhead=2, num_encoder_layers=1, num_decoder_layers=1)
    model = model.eval()
    src = torch.rand((10, 32, 512))
    tgt = torch.rand((20, 32, 512))
    return model, [src, tgt]


def get_model(model_name, type, input_data=[]):
    """Assert that the output of a compiled model matches with that of its
    baseline."""

    # nasty but it'll do for now
    if type == "torchvision":
        baseline_model, baseline_input = load_torchvision(model_name)
    elif type == "torchtransformers":
        baseline_model, baseline_input = load_torchtransformers(model_name)
    elif type == "github":
        if model_name == "deepspeech":
            baseline_model, baseline_input = load_deepspeech()
        else:
            assert False, "Unexpected model name"
    elif type == "custom":
        if model_name == "simple_transformer":
            baseline_model, baseline_input = load_simple_transformer()
        else:
            assert False, "Unexpected model name"
    else:
        assert False, "Unexpected type"

    if torch.cuda.is_available():
        if isinstance(baseline_model, torch.nn.Module):
            baseline_model = baseline_model.cuda()
        baseline_input = [inp.cuda() for inp in baseline_input]

    with torch.no_grad():
        baseline_outputs = baseline_model(*baseline_input)

    if isinstance(baseline_outputs, tuple):
        baseline_outputs = tuple(out.cpu().numpy() for out in baseline_outputs)
    else:
        baseline_outputs = (baseline_outputs.cpu().numpy(),)

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
