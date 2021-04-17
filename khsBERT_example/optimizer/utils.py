from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, LambdaLR
from .learning_rate import WarmupLinearSchedule

def shceduler_select(optimizer, dataloader_dict, args):
    if args.scheduler == 'constant':
        scheduler = StepLR(optimizer, step_size=len(dataloader_dict['train'])*999999, gamma=0.1)
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