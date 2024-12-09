import torch
import torch_pruning as tp
from utils.models.models import resnet34,resnet18,LSTMModel,GRUModel,Mamba,RetNet
from utils.models.models import Residual_Conv_GRU, Residual_Conv_LSTM, Residual_ConvTransformer,Residual_conv_retnet, Residual_Conv_Mamba

nleads = 12

def prune(args, path, device, model):
    # print(path)
    # model = model.load_state_dict(torch.load(path), map_location=device)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    # print(model)
    model.to(device)
    model.eval()
    example_inputs = torch.randn(1, 12, 15000).to(device)


    # 1. Importance criterion
    imp = tp.importance.GroupNormImportance(p=2) # or GroupTaylorImportance(), GroupHessianImportance(), etc.

    # 2. Initialize a pruner with the model and the importance criterion
    ignored_layers = []
    for m in model.modules():
        # print(f'module : {m}')

        if isinstance(m, Residual_Conv_GRU) :
            print(f'module ignored: {m}')
            ignored_layers.append(m)


        # for cpsc_2018
        if isinstance(m, torch.nn.Linear) and m.out_features == 9:
            print(f'module ignored: {m}')
            ignored_layers.append(m) # DO NOT prune the final classifier!
        
        # for ptb-xl
        if isinstance(m, torch.nn.Linear) and m.out_features == 5:
            print(f'module ignored: {m}')
            ignored_layers.append(m)

        # for shaoxing-ninbo
        if isinstance(m, torch.nn.Linear) and m.out_features == 63: 
            print(f'module ignored: {m}')
            ignored_layers.append(m)

        if isinstance(m, torch.nn.GRU) : 
            print(f'module ignored: {m}')
            ignored_layers.append(m)

        if isinstance(m, torch.nn.LSTM) :
            print(f'module ignored: {m}')
            ignored_layers.append(m)

    pruner = tp.pruner.MetaPruner( # We can always choose MetaPruner if sparse training is not required.
        model,
        example_inputs,
        importance=imp,
        pruning_ratio=0.5, # remove 50% channels, 
        # pruning_ratio_dict = {model.conv1: 0.2, model.layer2: 0.8}, # customized pruning ratios for layers or blocks
        ignored_layers=ignored_layers,
    ) 

    # 3. Prune
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    pruner.step()
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"MACs: {base_macs/1e9} G -> {macs/1e9} G, #Params: {base_nparams/1e6} M -> {nparams/1e6} M")

    return model



# import torch
# import torch_pruning as tp

# def prune(args, path, device, model):
#     print(path)
#     # Load model state
#     state_dict = torch.load(path, map_location=device)
#     model.load_state_dict(state_dict)
#     model.to(device)
#     model.eval()

#     # Ensure RNN weights are flattened (for GRU/LSTM layers)
#     for m in model.modules():
#         if isinstance(m, torch.nn.GRU) or isinstance(m, torch.nn.LSTM):
#             m.flatten_parameters()

#     # Create example inputs
#     example_inputs = torch.randn(1, 12, 15000).to(device)

#     # Define the importance criterion
#     imp = tp.importance.GroupNormImportance(p=2)

#     # Initialize ignored layers (like final classifiers or GRU/LSTM layers)
#     ignored_layers = []
#     for m in model.modules():
#         if isinstance(m, torch.nn.Linear) and m.out_features == args.num_classes:
#             ignored_layers.append(m)  # Do not prune the final classifier
#         if isinstance(m, torch.nn.GRU) or isinstance(m, torch.nn.LSTM):
#             ignored_layers.append(m)  # Do not prune GRU/LSTM layers

#     # Initialize the pruner
#     pruner = tp.pruner.MetaPruner(
#         model,
#         example_inputs,
#         importance=imp,
#         pruning_ratio=0.5,  # Prune 50% of the channels
#         ignored_layers=ignored_layers
#     )

#     # Custom operation counting wrapper without deepcopy
#     def count_ops_and_params_no_deepcopy(model, example_inputs):
#         # Backup original forward methods for GRU/LSTM
#         original_forward_methods = {}

#         # Define a forward wrapper that skips counting and returns placeholder outputs
#         def forward_wrapper(self, *args, **kwargs):
#             # Create a placeholder tensor of the expected output shape
#             batch_size, seq_length, _ = args[0].shape
#             hidden_size = self.hidden_size
#             # Return a placeholder output of the same shape as the GRU would produce
#             placeholder_output = torch.zeros(batch_size, seq_length, hidden_size).to(args[0].device)
#             placeholder_hidden = torch.zeros(self.num_layers, batch_size, hidden_size).to(args[0].device)
#             return placeholder_output, placeholder_hidden

#         # Temporarily replace forward method of GRU/LSTM layers
#         for m in model.modules():
#             if isinstance(m, torch.nn.GRU) or isinstance(m, torch.nn.LSTM):
#                 original_forward_methods[m] = m.forward  # Backup original forward
#                 m.forward = forward_wrapper.__get__(m)  # Replace forward method

#         # Count operations and parameters excluding GRU/LSTM layers
#         base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)

#         # Restore original forward method after counting
#         for m, original_forward in original_forward_methods.items():
#             m.forward = original_forward  # Restore original forward

#         return base_macs, base_nparams

#     # Perform operation counting without deepcopy
#     base_macs, base_nparams = count_ops_and_params_no_deepcopy(model, example_inputs)
#     pruner.step()
#     macs, nparams = count_ops_and_params_no_deepcopy(model, example_inputs)

#     print(f"MACs: {base_macs / 1e9} G -> {macs / 1e9} G, #Params: {base_nparams / 1e6} M -> {nparams / 1e6} M")

#     return model
