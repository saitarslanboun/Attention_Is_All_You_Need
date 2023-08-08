import torch.nn as nn

class AddAndNorm(nn.Module):
        def __init__(self, d_model, P_drop):
                super(AddAndNorm, self).__init__()
                self.layer_norm = nn.LayerNorm(d_model)
                self.dropout = nn.Dropout(P_drop)

        def forward(self, x, y):
                output = x + y
                output = self.layer_norm(output)
                output = self.dropout(output)
                return output
