
import time
# import os
# import datetime
import torch
import math

import torch.utils.data as data_utils

from constants import CONSTRAINTS_PATH, NUM_LABELS, NUM_REQ
from modules import  AverageMeter
from data import custum_collate
from modules.solver import get_optim
from req_handler import createIs, createMs
from val import validate
from modules import utils
logger = utils.get_logger(__name__)


def train(args, net, train_dataset, val_dataset):

    optimizer, scheduler, solver_print_str = get_optim(args, net)

    if args.TENSORBOARD:
        from tensorboardX import SummaryWriter

    source_dir = args.SAVE_ROOT+'/source/' # where to save the source
    utils.copy_source(source_dir)

    args.START_EPOCH = 1
    if args.RESUME>0:
        # raise Exception('Not implemented')
        args.START_EPOCH = args.RESUME + 1
        # args.iteration = args.START_EPOCH
        for _ in range(args.RESUME):
            scheduler.step()
        model_file_name = '{:s}/model_{:06d}.pth'.format(args.SAVE_ROOT, args.RESUME)
        optimizer_file_name = '{:s}/optimizer_{:06d}.pth'.format(args.SAVE_ROOT, args.RESUME)
        # sechdular_file_name = '{:s}/optimizer_{:06d}.pth'.format(args.SAVE_ROOT, args.START_EPOCH)
        net.load_state_dict(torch.load(model_file_name))
        optimizer.load_state_dict(torch.load(optimizer_file_name))

    if args.TENSORBOARD:
        log_dir = '{:s}/log-lo_tboard-{}-{date:%m-%d-%H-%M-%S}_logic-{logic:s}_req-weight-{weight}'.format(args.log_dir, args.MODE, date=args.DATETIME_NOW, logic=str(args.LOGIC), weight=args.req_loss_weight)
        # log_dir = '{:s}/log-lo_tboard-{}_logic-{logic:s}_req-weight-{weight}'.format(args.log_dir, args.MODE, logic=str(args.LOGIC), weight=args.req_loss_weight)
        args.sw = SummaryWriter(log_dir)

    logger.info(str(net))
    logger.info(solver_print_str)


    logger.info(train_dataset.print_str)
    logger.info(val_dataset.print_str)
    epoch_size = len(train_dataset) // args.BATCH_SIZE
    args.MAX_ITERS = args.MAX_EPOCHS*epoch_size

    for arg in sorted(vars(args)):
        logger.info(str(arg)+': '+str(getattr(args, arg)))

    logger.info('EXPERIMENT NAME:: ' + args.exp_name)


    logger.info('Training FPN with {} + {} as backbone '.format(args.ARCH, args.MODEL_TYPE))


    train_data_loader = data_utils.DataLoader(train_dataset, args.BATCH_SIZE, num_workers=args.NUM_WORKERS,
                                  shuffle=True, pin_memory=True, collate_fn=custum_collate, drop_last=True)

    val_data_loader = data_utils.DataLoader(val_dataset, args.BATCH_SIZE, num_workers=args.NUM_WORKERS,
                                            shuffle=False, pin_memory=True, collate_fn=custum_collate)

    # TO REMOVE
    # net.eval()
    # run_val(args, val_data_loader, val_dataset, net, 0, 0)

    iteration = 0
    for epoch in range(args.START_EPOCH, args.MAX_EPOCHS + 1):
        net.train()

        if args.FBN:
            if args.MULTI_GPUS:
                net.module.backbone.apply(utils.set_bn_eval)
            else:
                net.backbone.apply(utils.set_bn_eval)
        iteration = run_train(args, train_data_loader, net, optimizer, epoch, iteration)

        if epoch % args.VAL_STEP == 0 or epoch == args.MAX_EPOCHS:
            net.eval()
            run_val(args, val_data_loader, val_dataset, net, epoch, iteration)

        scheduler.step()

def run_train(args, train_data_loader, net, optimizer, epoch, iteration):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loc_losses = AverageMeter()
    cls_losses = AverageMeter()
    req_losses = AverageMeter()
    torch.cuda.synchronize()
    start = time.perf_counter()

    if args.LOGIC is not None:
        # Read constraints from file and create the Ms and Is matrices
        Iplus_np, Iminus_np = createIs(CONSTRAINTS_PATH, NUM_LABELS)
        Mplus_np, Mminus_np = createMs(CONSTRAINTS_PATH, NUM_LABELS)

        Iplus, Iminus = torch.from_numpy(Iplus_np).float(), torch.from_numpy(Iminus_np).float()
        Mplus, Mminus = torch.from_numpy(Mplus_np).float(), torch.from_numpy(Mminus_np).float()

        if args.LOGIC == "Product":
            # These are already the negated literals
            # matrix of negative appearances in the conjunction
            Cminus = Iminus + torch.transpose(Mplus, 0, 1)
            # matrix of positive appearances in the conjunction
            Cplus = Iplus + torch.transpose(Mminus, 0, 1)
        else: # elif args.LOGIC == "Godel" or args.LOGIC == "Lukasiewicz":
            # These are the literals as they appear in the disjunction
            # Matrix of the positive appearances in the disjunction
            Cplus = Iminus + torch.transpose(Mplus, 0, 1)
            # matrix of negative appearances in the conjunction
            Cminus = Iplus + torch.transpose(Mminus, 0, 1)

        if args.MULTI_GPUS:
            # Since we are splitting the foarward call on multiple GPUs, whatever we pass to the forward call
            # gets splitted along the 0 dimension. In order to have a replication and not a splitting we replicate
            # the matrices along the newly generated dimension 0.
            # Iplus, Iminus = Iplus.unsqueeze(0), Iminus.unsqueeze(0)
            # Mplus, Mminus = Mplus.unsqueeze(0), Mminus.unsqueeze(0)
            Cplus, Cminus = Cplus.unsqueeze(0), Cminus.unsqueeze(0)

            # Iplus = Iplus.expand(torch.cuda.device_count(), NUM_REQ, NUM_LABELS)
            # Iminus = Iminus.expand(torch.cuda.device_count(), NUM_REQ, NUM_LABELS)
            # Mplus = Mplus.expand(torch.cuda.device_count(), NUM_LABELS, NUM_REQ)
            # Mminus = Mminus.expand(torch.cuda.device_count(), NUM_LABELS, NUM_REQ)
            Cplus = Cplus.expand(torch.cuda.device_count(), NUM_REQ, NUM_LABELS)
            Cminus = Cminus.expand(torch.cuda.device_count(), NUM_REQ, NUM_LABELS)

    # for internel_iter, (images, gt_boxes, gt_labels, ego_labels, counts, img_indexs, wh) in enumerate(train_data_loader):
    for internel_iter, (images, gt_boxes, gt_labels, counts, img_indexs, wh, _, _) in enumerate(train_data_loader):
        iteration += 1
        images = images.cuda(0, non_blocking=True)
        gt_boxes = gt_boxes.cuda(0, non_blocking=True)
        gt_labels = gt_labels.cuda(0, non_blocking=True)
        counts = counts.cuda(0, non_blocking=True)
        # ego_labels = ego_labels.cuda(0, non_blocking=True)

        if args.LOGIC is not None:
            # Iplus = Iplus.cuda(0, non_blocking=True)
            # Iminus = Iminus.cuda(0, non_blocking=True)
            # Mplus = Mplus.cuda(0, non_blocking=True)
            # Mminus = Mminus.cuda(0, non_blocking=True)
            Cplus = Cplus.cuda(0, non_blocking=True)
            Cminus = Cminus.cuda(0, non_blocking=True)

        # forward
        torch.cuda.synchronize()
        data_time.update(time.perf_counter() - start)

        # print(images.size(), anchors.size())
        optimizer.zero_grad()
        # pdb.set_trace()


        if args.LOGIC is None:
            loc_loss, conf_loss = net(images, gt_boxes, gt_labels, counts, img_indexs)
            # Mean over the losses computed on the different GPUs
            loc_loss, conf_loss = loc_loss.mean(), conf_loss.mean()
            loss = loc_loss + conf_loss
        else:
            loc_loss, conf_loss, req_loss = net(images, gt_boxes, gt_labels, counts, img_indexs, logic=args.LOGIC, Cplus=Cplus, Cminus=Cminus)
            # Mean over the losses computed on the different GPUs
            loc_loss, conf_loss, req_loss = loc_loss.mean(), conf_loss.mean(), req_loss.mean()

            # If a t-norm is used, the regularisation term <req_loss> gives the
            # degree of constraint satisfaction of the neural predictions w.r.t. that t-norm.
            # To customise this term, changes should be made in the method
            # <logical_requirements_loss> (found in <modules/req_losses.py>),
            # which is called in the forward method of the
            # <FocalLoss> class (found in <modules/detection_loss.py>).
            loss = loc_loss + conf_loss + args.req_loss_weight * req_loss

        loss.backward()
        optimizer.step()

        if math.isnan(loc_loss.item()) or loc_loss.item()>300:
            lline = '\n\n\n We got faulty LOCATION loss {} {} \n\n\n'.format(loc_loss, conf_loss)
            logger.info(lline)
            loc_loss = loc_loss*0+20.0
        if math.isnan(conf_loss.item()) or  conf_loss.item()>300:
            lline = '\n\n\n We got faulty CLASSIFICATION loss {} {} \n\n\n'.format(loc_loss, conf_loss)
            logger.info(lline)
            conf_loss = conf_loss*0+20.0

        loc_losses.update(loc_loss.item())
        cls_losses.update(conf_loss.item())
        if args.LOGIC is None:
            losses.update(loc_loss.item() + conf_loss.item())
        else:
            req_losses.update(req_loss.item())
            losses.update(loc_loss.item() + conf_loss.item() + req_loss.item())  # do not multiply by req weight, so exp are comparable

        torch.cuda.synchronize()
        batch_time.update(time.perf_counter() - start)
        start = time.perf_counter()

        if internel_iter % args.LOG_STEP == 0 and iteration > args.LOG_START and internel_iter>0:
            if args.TENSORBOARD:
                loss_group = dict()
                loss_group['Classification'] = cls_losses.val
                loss_group['Localisation'] = loc_losses.val
                loss_group['Requirements'] = req_losses.val
                loss_group['Overall'] = losses.val
                args.sw.add_scalars('Losses', loss_group, iteration)

            print_line = 'Iteration [{:d}/{:d}]{:06d}/{:06d} loc-loss {:.2f}({:.2f}) cls-loss {:.2f}({:.2f}) req-loss {:.2f}({:.2f}) ' \
                        'overall-loss {:.2f}({:.2f}) DataTime {:0.2f}({:0.2f}) Timer {:0.2f}({:0.2f})'.format( epoch,
                        args.MAX_EPOCHS, iteration, args.MAX_ITERS, loc_losses.val, loc_losses.avg, cls_losses.val,
                        cls_losses.avg, req_losses.val, req_losses.avg, losses.val, losses.avg,
                        10*data_time.val, 10*data_time.avg, 10*batch_time.val, 10*batch_time.avg)

            logger.info(print_line)
            if internel_iter % (args.LOG_STEP*20) == 0:
                logger.info(args.exp_name)
    logger.info('Saving state, epoch:' + str(epoch))
    torch.save(net.state_dict(), '{:s}/model_{:06d}.pth'.format(args.SAVE_ROOT, epoch))
    torch.save(optimizer.state_dict(), '{:s}/optimizer_{:06d}.pth'.format(args.SAVE_ROOT, epoch))

    return iteration


def run_val(args, val_data_loader, val_dataset, net, epoch, iteration):
        torch.cuda.synchronize()
        tvs = time.perf_counter()

        mAP, ap_all, ap_strs = validate(args, net, val_data_loader, val_dataset, epoch)
        label_types = args.label_types #+ ['ego_action']
        all_classes = args.all_classes #+ [args.ego_classes]
        mAP_group = dict()

        # for nlt in range(args.num_label_types+1):
        for nlt in range(args.num_label_types):
            for ap_str in ap_strs[nlt]:
                logger.info(ap_str)
            ptr_str = '\n{:s} MEANAP:::=> {:0.5f}'.format(label_types[nlt], mAP[nlt])
            logger.info(ptr_str)

            if args.TENSORBOARD:
                mAP_group[label_types[nlt]] = mAP[nlt]
                # args.sw.add_scalar('{:s}mAP'.format(label_types[nlt]), mAP[nlt], iteration)
                class_AP_group = dict()
                for c, ap in enumerate(ap_all[nlt]):
                    class_AP_group[all_classes[nlt][c]] = ap
                args.sw.add_scalars('ClassAP-{:s}'.format(label_types[nlt]), class_AP_group, epoch)

        if args.TENSORBOARD:
            args.sw.add_scalars('mAPs', mAP_group, epoch)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prt_str = '\nValidation TIME::: {:0.3f}\n\n'.format(t0-tvs)
        logger.info(prt_str)




