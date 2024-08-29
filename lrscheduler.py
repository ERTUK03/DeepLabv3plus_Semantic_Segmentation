from torch.optim.lr_scheduler import _LRScheduler

class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, max_iter, base_lr=0.007, power=0.9, last_epoch=-1):
        self.max_iter = max_iter
        self.base_lr = base_lr
        self.power = power
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        iter = self.last_epoch + 1
        lr = self.base_lr * (1 - iter / self.max_iter) ** self.power
        return [lr for _ in self.optimizer.param_groups]
