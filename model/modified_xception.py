import torch
from model.xception_flows import EntryFlow, MiddleFlow, ExitFlow

class ModifiedXception(torch.nn.Module):
    def __init__(self):
        super(ModifiedXception, self).__init__()
        self.entry_flow = EntryFlow()
        self.middle_flow = torch.nn.Sequential(*[MiddleFlow() for _ in range(16)])
        self.exit_flow = ExitFlow()

    def forward(self, x):
        x, x_out = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        return x, x_out
