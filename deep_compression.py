import torch
from torch.optim import SGD
from functools import reduce

class MaskedSGD(SGD):
    r"""Implements the sparsity part of the Deep Compression algorithm of Han et al:
    https://arxiv.org/abs/1510.00149v5
    """
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            if not 'masks' in group:
                group['masks'] = [None]*len(group['params'])

            for m,p in zip(group['masks'],group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = p.data.new().resize_as_(p.data).zero_()
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)
                # apply mask if we have one
                if m is not None:
                    p.data.mul_(m.data)

        return loss

    def sparsify(self):
        """Sets a mask that will zero all parameters below 0.02 while this
        optimizer continues to be used."""
        for group in self.param_groups:
            group['masks'] = []
            for p in group['params']:
                group['masks'].append(torch.gt(torch.abs(p),0.02).float())

    def sparsity(self):
        """Sets a mask that will zero all parameters below 0.02 while this
        optimizer continues to be used."""
        nonzero = 0
        total = 0
        for group in self.param_groups:
            for m in group['masks']:
                nonzero += m.sum()
                total += reduce(lambda a,b: a*b, m.size())
        return float(nonzero.data.cpu().numpy())/float(total)

    def maybe_sparsify(self, batch_index, threshold):
        # only sparsify once, when over the threshold
        if batch_index > threshold and not self.__dict__.get('sparsified', False):
            self.sparsify()
            self.sparsified = True
