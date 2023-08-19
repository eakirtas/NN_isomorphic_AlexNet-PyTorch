import torch as T

from .nn_conv1d import NNConv1d
from .nn_conv2d import NNConv2d
from .nn_layer import NNLinear


def nn_transformation(model, dtype=T.cuda.FloatTensor) -> T.nn.Module:
    """
    Receives a regular models and return a fully non-negative one.

    param model: The model that will be transformed into non-negative one
    type model: T.nn.Module
    param dtype: Type of model
    type mode: Type
    """

    # Create lists for the new biases and activation shifting points
    b_new_list = [None for _ in range(len(model.layers))]
    act_shift_list = [None for _ in range(len(model.layers))]

    # Get the non-negative replica of model
    nn_model = model.get_nn_net()

    for i in range(len(model.layers)):
        cur_layer = model.layers[i]
        classname = cur_layer.__class__.__name__

        if classname.find('Linear') != -1:
            sum_dim = 1
        elif classname.find('Conv2d') != -1:
            sum_dim = (1, 2, 3)
        elif classname.find('Conv1d') != -1:
            sum_dim = (1, 2)
        else:
            raise Exception("This layer type cannot be converted")

        # Get the non-negative weights
        w_neg = T.clamp_max(cur_layer.weight, 0).type(dtype)

        # Calculated b_tilde
        b_tilde = cur_layer.bias.type(dtype) - model.alpha[i].type(
            dtype) * T.sum(T.abs(w_neg), dim=sum_dim).type(dtype)

        b_new, act_shift = calc_b_new(cur_layer, b_tilde, dtype)

        b_new_list[i] = b_new.type(dtype)
        act_shift_list[i] = act_shift.type(dtype)

        # Create two different tensors one for positive and one for negative biases
        w_neg_abs = T.abs(w_neg).type(dtype)
        w_pos = T.clamp_min(cur_layer.weight, 0).type(dtype)

        if classname.find('Linear') != -1:
            new_layer = NNLinear(
                cur_layer.in_features,
                cur_layer.out_features,
                w_pos,
                w_neg_abs,
                b_new,
                model.alpha[i],
                act_shift,
            )
        elif classname.find('Conv2d') != -1:
            new_layer = NNConv2d(
                cur_layer.in_channels,
                cur_layer.out_channels,
                cur_layer.kernel_size,
                w_pos,
                w_neg_abs,
                b_new,
                model.alpha[i],
                act_shift,
                stride=cur_layer.stride,
                padding=cur_layer.padding,
                dilation=cur_layer.dilation,
                groups=cur_layer.groups,
                padding_mode=cur_layer.padding_mode,
            )
        elif classname.find('Conv1d') != -1:
            new_layer = NNConv1d(
                cur_layer.in_channels,
                cur_layer.out_channels,
                cur_layer.kernel_size,
                w_pos,
                w_neg_abs,
                b_new,
                model.alpha[i],
                act_shift,
                stride=cur_layer.stride,
                padding=cur_layer.padding,
                dilation=cur_layer.dilation,
                groups=cur_layer.groups,
                padding_mode=cur_layer.padding_mode,
                # device=cur_layer.device(),
                # dtype=cur_layer.dtype
            )
        else:
            raise Exception("This layer type cannot be converted")
        nn_model.add_layer(new_layer)
    return nn_model


def calc_b_new(layer, b_tilde, dtype):
    with T.no_grad():
        act_shift = T.max(T.abs(b_tilde)).type(dtype)

        # For b_tilde < 0 replace biases with b_tilde
        b_new = T.scatter(layer.bias.type(dtype), 0,
                          T.where(b_tilde < 0)[0],
                          b_tilde[b_tilde < 0]).type(dtype)

        # Add the activation shifting point to the b_tilde < 0 case
        b_new = T.scatter_add(-T.abs(b_new), 0,
                              T.where(b_tilde < 0)[0],
                              act_shift.expand_as(
                                  T.where(b_tilde < 0)[0])).type(dtype)

        # Add the activation shifting point to b_tilde > 0
        b_new = T.scatter_add(b_new, 0,
                              T.where(b_tilde > 0)[0],
                              act_shift.expand_as(
                                  T.where(b_tilde > 0)[0])).type(dtype)

    return b_new, act_shift
