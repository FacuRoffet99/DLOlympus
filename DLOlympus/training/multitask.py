# Helper function to create each loss function
def create_loss_func(loss, start, end, idx):
    def loss_func(inp, *args):
        return loss(inp[:, start:end], args[idx])
    return loss_func