# coding:utf8
import models
from sets import *
from data.dataset_2d import Brats17
from torch.utils.data import DataLoader
import torch as t
from tqdm import tqdm
import numpy
import time


############################################################################
def val(model, dataloader):

    model.eval()
    # val_meter=AverageMeter()
    val_losses, dcs = [], []
    # criterion = t.nn.CrossEntropyLoss()
    for ii, data in enumerate(dataloader):
        input, label = data
        val_input = Variable(input.cuda())
        val_label = Variable(label.cuda())
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
            model = model.cuda()
        outputs = model(val_input)
        pred = outputs.data.max(1)[1].cpu().numpy().squeeze()
        gt = val_label.data.cpu().numpy().squeeze()

        # print(pred.shape)
        # print(gt.shape)
        for i in range(gt.shape[0]):
            # print(i)
            dc, val_loss = calc_dice(gt[i, :, :, :], pred[i, :, :, :])
            dcs.append(dc)
            val_losses.append(val_loss)
        # for gt_, pred_ in zip(gt, pred):
        # gts.append(gt_)
        # preds.append(pred_)
    # score,cc,acc=scores(gts, preds, n_class=classes)
    model.train()
    return np.mean(dcs), np.mean(val_losses)


############################################################################


############################################################################
print('train:')
lr = 0.00001  # opt.lr
batch_size = 4
print('batch_size:', batch_size, 'lr:', lr)

plt_list = []

model = getattr(models, 'U_Net')()

if opt.use_gpu:
    model.cuda()
train_data = Brats17(opt.train_data_root, train=True)
val_data = Brats17(opt.train_data_root, train=False, val=True)
val_dataloader = DataLoader(val_data, 4, shuffle=False, num_workers=opt.num_workers)

# criterion = get_loss_criterion('DiceLoss')
# weight = torch.FloatTensor([1, 2, 2, 2, 2])
criterion = t.nn.CrossEntropyLoss()  # weight=weight
# criterion = get_loss_criterion(config)
# criterion = FocalLoss2d()
# criterion = DiceLoss()
# criterion = FocalLoss(5)
# criterion = DiceLoss2D()
if opt.use_gpu:
    criterion = criterion.cuda()

loss_meter = AverageMeter()
previous_loss = 1e+20

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=opt.num_workers)
optimizer = t.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)

# train
for epoch in range(opt.max_epoch):

    loss_meter.reset()
    # confusion_matrix.reset()

    for ii, (data, label) in tqdm(enumerate(train_dataloader), total=len(train_data)):

        # print(data.shape,label.shape)
        # train model 
        input = Variable(data)
        target = Variable(label)

        if opt.use_gpu:
            input = input.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        # model = model.cuda()
        score = model(input)
        # print('aa')
        # print(score.shape)
        # print(target.shape)
        loss = criterion(score, target)

        loss.backward()
        optimizer.step()

        # meters update and visualize
        # loss_meter.add(loss.data[0])
        loss_meter.update(loss.item())

        if ii % 5 == 1:
            plt_list.append(loss_meter.val)
        if ii % 50 == 1:
            print('train-loss-avg:', loss_meter.avg, 'train-loss-each:', loss_meter.val)

    if epoch % 10 == 1:
        if 1 > 0:
            # acc,val_loss = val(model,val_dataloader)
            acc = 0
            val_loss = 0
            # if acc > pre_acc:
            # prefix = 'checkpoints/' + str(loss_meter.avg)+'_'+str(loss_meter.val) + '_'
            prefix = '/userhome/save/path/check/pth/' + str(acc) + '_4444_' + str(val_loss) + '_' + str(
                lr) + '_' + str(batch_size) + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
            t.save(model.state_dict(), name)

            name_loss = time.strftime('/userhome/save/path/check/plt/' + '%m%d_%H:%M:%S.npy')
            numpy.save(name_loss, plt_list)

        # if ii%100=7
        # numpy.save(time.strftime('checkpoints/plt/' + str(loss_meter) + '__%m%d_%H:%M:%S.npy'), loss_meter)
        # model.save(str(acc)+'_'+str(val_loss))
        # pre_acc=acc
        # vis.plot('dice-coefficent',acc)
        # vis.plot('val_loss',val_loss)

    # update learning rate
    # if loss_meter.avg > previous_loss:
    #   lr = lr * opt.lr_decay
    #    for param_group in optimizer.param_groups:
    #        param_group['lr'] = lr

    print('batch_size:', batch_size, 'lr:', lr)
    # train_dataloader = DataLoader(train_data,batch_size,shuffle=True,num_workers=opt.num_workers)
    # optimizer = t.optim.Adam(model.parameters(),lr = lr,weight_decay = opt.weight_decay)
    # previous_loss = loss_meter.avg
############################################################################
