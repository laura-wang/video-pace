'''
set different learning rate for different parameters

'''


def get_1x_lr_params(model):
    """
    This generator returns all the parameters of weights
    """

    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            #print (name, param.data)

            yield param


def get_2x_lr_params(model):
    """
    This generator returns all the parameters of bias
    """

    for name, param in model.named_parameters():
        if 'bias' in name and param.requires_grad:
            #print (name, param.data)

            yield param