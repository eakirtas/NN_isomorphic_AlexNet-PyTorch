import torch as T


@T.no_grad()
def exp_init(m, lambd):
    if isinstance(m, T.nn.Conv1d) or isinstance(m, T.nn.Conv2d) or isinstance(
            m, T.nn.Linear):
        m.weight.exponential_(lambd=lambd)


@T.no_grad()
def fill_bias(m, constant):
    if isinstance(m, T.nn.Conv1d) or isinstance(m, T.nn.Conv2d) or isinstance(
            m, T.nn.Linear):
        T.nn.init.constant_(m.bias, constant)
