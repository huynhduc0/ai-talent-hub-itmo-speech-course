import torch

def calc_params_flops(model, input_size=(1, 16000)):
    # Very basic size computation 
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    try:
        from thop import profile
        # Get random input sample 
        dummy_in = torch.randn(*input_size).to(next(model.parameters()).device)
        macs, _ = profile(model, inputs=(dummy_in, ), verbose=False)
        flops = macs * 2 # One mac translates to 2 flops usually
    except ImportError:
        flops = 0
    return params, flops
