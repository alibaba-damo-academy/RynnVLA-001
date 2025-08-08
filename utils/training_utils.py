
def requires_grad(model, flag=True, trainable_param_list=None):
    for name, p in model.named_parameters():
        if trainable_param_list is None:
            p.requires_grad = flag
        else:
            if name.replace('module.', '') in trainable_param_list:
                p.requires_grad = flag
            else:
                p.requires_grad = (not flag)