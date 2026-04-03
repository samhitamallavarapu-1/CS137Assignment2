import torch
x = torch.load("part2_outputs/confidently_correct/01_confidently_correct_tidx_41565_full_saliency.pt", map_location="cpu")
print(type(x))
print(x.shape)