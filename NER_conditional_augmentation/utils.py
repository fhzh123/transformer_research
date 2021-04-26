import re
import emoji
from torch import optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, LambdaLR

def encoding_text(list_x, tokenizer, max_len):

    emojis = ''.join(emoji.UNICODE_EMOJI.keys())
    pattern = re.compile(r'<[^>]+>')

    def clean(x):
        x = pattern.sub(' ', x)
        x = x.strip()
        return x

    encoded_text_list = list_x.map(lambda x: tokenizer.encode(
        clean(str(x)),
        max_length=max_len,
        truncation=True
    ))
    return encoded_text_list

def optimizer_select(model, args):
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(optimizer, lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-8)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.w_decay)
    else:
        raise Exception("Choose optimizer in ['AdamW', 'Adam', 'SGD']")
    return optimizer

def shceduler_select(optimizer, dataloader_dict, args):
    if args.scheduler == 'constant':
        scheduler = StepLR(optimizer, step_size=len(dataloader_dict['train']), gamma=1)
    elif args.scheduler == 'warmup':
        scheduler = WarmupLinearSchedule(optimizer, 
                                        warmup_steps=int(len(dataloader_dict['train'])*args.n_warmup_epochs), 
                                        t_total=len(dataloader_dict['train'])*args.num_epochs)
    elif args.scheduler == 'reduce_train':
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=int(len(dataloader_dict['train'])*1.5),
                                      factor=0.5)
    elif args.scheduler == 'reduce_valid':
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5)
    elif args.scheduler == 'lambda':
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: args.lr_lambda ** epoch)
    else:
        raise Exception("Choose shceduler in ['constant', 'warmup', 'reduce_train', 'reduce_valid', 'lambda']")
    return scheduler

class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return max(1e-7, float(step) / float(max(1, self.warmup_steps)))
        return max(1e-7, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))
