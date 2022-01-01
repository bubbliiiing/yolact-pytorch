import torch
from tqdm import tqdm

from utils.utils import get_lr


def fit_one_epoch(model_train, model, multi_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda):
    total_loss  = 0
    val_loss    = 0

    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images, targets, masks_gt, num_crowds = batch[0], batch[1], batch[2], batch[3]
            with torch.no_grad():
                if cuda:
                    images      = images.cuda()
                    targets     = [ann.cuda() for ann in targets]
                    masks_gt    = [mask.cuda() for mask in masks_gt]
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs = model_train(images)
            #----------------------#
            #   计算损失
            #----------------------#
            losses  = multi_loss(outputs, targets, masks_gt, num_crowds)
            losses  = {k: v.mean() for k, v in losses.items()}  # Mean here because Dataparallel.
            loss    = sum([losses[k] for k in losses])

            #----------------------#
            #   反向传播
            #----------------------#
            loss.backward()
            optimizer.step()

            total_loss += loss
            
            pbar.set_postfix(**{'total_loss': total_loss.item() / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets, masks_gt, num_crowds = batch[0], batch[1], batch[2], batch[3]
            with torch.no_grad():
                if cuda:
                    images      = images.cuda()
                    targets = [ann.cuda() for ann in targets]
                    masks_gt    = [mask.cuda() for mask in masks_gt]

                optimizer.zero_grad()
                #----------------------#
                #   前向传播
                #----------------------#
                outputs = model_train(images)
                #----------------------#
                #   计算损失
                #----------------------#
                losses      = multi_loss(outputs, targets, masks_gt, num_crowds)
                losses      = {k: v.mean() for k, v in losses.items()}
                loss        = sum([losses[k] for k in losses])

                val_loss += loss
                
                pbar.set_postfix(**{'total_loss': val_loss.item() / (iteration + 1), 
                                    'lr'        : get_lr(optimizer)})
                pbar.update(1)

    print('Finish Validation')
    
    loss_history.append_loss(total_loss / epoch_step, val_loss / epoch_step_val)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
    torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val))
