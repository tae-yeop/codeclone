

def batch_cosine_sim(x, y):
    if type(x) is list:
        x = torch.cat(x, dim=0)
    if type(y) is list:
        y = torch.cat(y, dim=0)

    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    similarity = x @ y.T
    return similarity



def isinstance_str(x, cls_name):
    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    return False