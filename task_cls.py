import os, sys, glob, time
import numpy as np
import logging
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

try:
    from tensorboardX import SummaryWriter
    import utils
    import models
    import datasets
except (ImportError, RuntimeError, FileNotFoundError) as e:
    print('import project module error', e)

dali_enable = True
try:
    if torch.cuda.is_available():
        from nvidia.dali.plugin.pytorch import DALIClassificationIterator
        from nvidia.dali.pipeline import Pipeline
        import nvidia.dali.ops as ops
        import nvidia.dali.types as types
    else:
        dali_enable = False
except ImportError:
    dali_enable = False

try:
    from apex import amp
    apex_enable=True
except ImportError:
    apex_enable=False

try:
    import plugin
    plugin_enable=True
except (ImportError, RuntimeError, FileNotFoundError) as e:
    plugin_enable=False

def get_parser(parser=None):
    # default parameters for various projects
    if parser is None:
        parser = utils.get_parser()

    parser.add_argument('--num_classes', default=None, type=int)
    parser.add_argument('--input_size', default=None, type=int)

    # custom parameters for quantization related projects
    parser.add_argument('--base', default=1, type=int, help='base used in GroupNet') 
    parser.add_argument('--width_alpha', default=1.0, type=float, help='channel alpha')
    parser.add_argument('--block_alpha', default=1.0, type=float)
    parser.add_argument('--se_reduction', default=16, type=int, help='ratio in Squeeze-Excition Module')
    parser.add_argument('--stem_kernel', default=1, type=int)
    parser.add_argument('--order', default='none', type=str)
    parser.add_argument('--policy', default='none', type=str)

    # config for activation quantization
    parser.add_argument('--fm_bit', default=None, type=float)
    parser.add_argument('--fm_level', default=None, type=int, help="default of quantization level=2^bit - 1")
    parser.add_argument('--fm_half_range', action='store_false', default=True, help='real domain or non-positive range')
    parser.add_argument('--fm_separator', default=0.38, type=float)
    parser.add_argument('--fm_correlate', default=-1, type=float)
    parser.add_argument('--fm_ratio', default=1, type=float)
    parser.add_argument('--fm_scale', default=0.5, type=float)
    parser.add_argument('--fm_enable', action='store_true', default=False, help='enable quantization or not')
    parser.add_argument('--fm_boundary', default=None, type=float)
    parser.add_argument('--fm_quant_group', default=None, type=int)
    # advanced options for gradient control / normalization / debug
    parser.add_argument('--fm_adaptive', default='none', type=str, choices=['none', 'var', 'mean', 'mean-var', 'var-mean'])
    parser.add_argument('--fm_custom', default='none', type=str, choices=['none', 'channel', 'resolution'])
    parser.add_argument('--fm_grad_type', default='none', type=str, choices=['none', 'STE', 'Triangle', 'STE-scale'])
    parser.add_argument('--fm_grad_scale', default='none', type=str, choices=['none', 'fan-scale', 'scale-fan', 'element-scale', 'scale-element'])

    # config for weight quantization
    parser.add_argument('--wt_bit', default=None, type=float)
    parser.add_argument('--wt_level', default=None, type=int)
    parser.add_argument('--wt_half_range', action='store_true', default=False)
    parser.add_argument('--wt_separator', default=0.38, type=float)
    parser.add_argument('--wt_correlate', default=-1, type=float)
    parser.add_argument('--wt_ratio', default=1, type=float)
    parser.add_argument('--wt_scale', default=0.5, type=float)
    parser.add_argument('--wt_enable', action='store_true', default=False)
    parser.add_argument('--wt_boundary', default=None, type=float)
    parser.add_argument('--wt_quant_group', default=None, type=int)
    parser.add_argument('--wt_adaptive', default='none', type=str, choices=['none', 'var', 'mean', 'mean-var', 'var-mean'])
    parser.add_argument('--wt_grad_type', default='none', type=str, choices=['none', 'STE', 'STE-scale'])
    parser.add_argument('--wt_grad_scale', default='none', type=str, choices=['none', 'fan-scale', 'scale-fan', 'element-scale', 'scale-element'])

    # config for output quantization
    parser.add_argument('--ot_bit', default=None, type=float)
    parser.add_argument('--ot_level', default=None, type=int)
    parser.add_argument('--ot_half_range', action='store_true', default=False)
    parser.add_argument('--ot_separator', default=0.38, type=float)
    parser.add_argument('--ot_correlate', default=-1, type=float)
    parser.add_argument('--ot_ratio', default=1, type=float)
    parser.add_argument('--ot_scale', default=0.5, type=float)
    parser.add_argument('--ot_enable', action='store_true', default=False)
    parser.add_argument('--ot_boundary', default=None, type=float)
    parser.add_argument('--ot_quant_group', default=None, type=int)
    parser.add_argument('--ot_adaptive', default='none', type=str, choices=['none', 'var', 'mean', 'mean-var', 'var-mean'])
    parser.add_argument('--ot_grad_type', default='none', type=str, choices=['none', 'STE', 'STE-scale'])
    parser.add_argument('--ot_grad_scale', default='none', type=str, choices=['none', 'fan-scale', 'scale-fan', 'element-scale', 'scale-element'])
    parser.add_argument('--ot_independent_parameter', action='store_true', default=False, help="independent or shared parameters")

    # re-init the model to pre-calculate some initial value
    parser.add_argument('--re_init', action='store_true', default=False)

    # proxquant
    parser.add_argument('--proxquant_step', '--ps', default=5, type=int)

    # mixup data augment
    parser.add_argument('--mixup_alpha', default=0.7, type=float)
    parser.add_argument('--mixup_enable', default=False, action='store_true')

    parser.add_argument('--padding_after_quant', action='store_true', default=False)

    # record / debug runtime information
    parser.add_argument('--probe_iteration', default=1, type=int)
    parser.add_argument('--probe_index', default=[], type=int, nargs='+')
    parser.add_argument('--probe_list', default='', type=str)

    # label-smooth
    parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')

    # specific custom learning rate or weight decay for certain parameters
    parser.add_argument('--custom_decay_list', default='', type=str)
    parser.add_argument('--custom_decay', default=0.02, type=float)
    parser.add_argument('--custom_lr_list', default='', type=str)
    parser.add_argument('--custom_lr', default=1e-5, type=float)

    # gloabl buffer
    parser.add_argument('--global_buffer', default=dict(), type=dict)
    return parser

def get_parameter():
    parser = get_parser()
    args = parser.parse_args()

    if isinstance(args.lr_custom_step, str):
        args.lr_custom_step = [int(x) for x in args.lr_custom_step.split(',')]
    if isinstance(args.keyword, str):
        args.keyword = [x.strip() for x in args.keyword.split(',')]
    if isinstance(args.custom_decay_list, str):
        args.custom_decay_list = [x.strip() for x in args.custom_decay_list.split(',')]
    if isinstance(args.custom_lr_list, str):
        args.custom_lr_list = [x.strip() for x in args.custom_lr_list.split(',')]
    if isinstance(args.probe_list, str):
        args.probe_list = [x.strip() for x in args.probe_list.split(',')]
    return args

def main(args=None):
    if args is None:
        args = get_parameter()

    if args.dataset == 'dali' and not dali_enable:
        args.case = args.case.replace('dali', 'imagenet')
        args.dataset = 'imagenet'
        args.workers = 12
        
    # log_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    model_arch = args.model
    model_name = model_arch
    if args.evaluate:
        log_suffix = 'eval-' + model_arch + '-' + args.case
    else:
        log_suffix = model_arch + '-' + args.case
    utils.setup_logging(os.path.join(args.log_dir, log_suffix + '.txt'), resume=args.resume)

    logging.info("current folder: %r", os.getcwd())
    logging.info("alqnet plugins: %r", plugin_enable)
    logging.info("apex available: %r", apex_enable)
    logging.info("dali available: %r", dali_enable)
    for x in vars(args):
        logging.info("config %s: %r", x, getattr(args, x))

    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and len(args.device_ids) > 0:
        args.device_ids = [x for x in args.device_ids if x < torch.cuda.device_count() and x >= 0]
        if len(args.device_ids) == 0:
            args.device_ids = None
        else:
            logging.info("training on %d gpu", len(args.device_ids))
    else:
        args.device_ids = None

    if args.device_ids is not None:
        torch.cuda.manual_seed_all(args.seed)
        cudnn.benchmark = True
        torch.backends.cudnn.deterministic=True #https://github.com/pytorch/pytorch/issues/8019
    else:
        logging.info("no gpu available, try CPU version, lots of functions limited")
        #return

    if model_name in models.model_zoo:
        model, args = models.get_model(args)
    else:
        logging.error("model(%s) not support, available models: %r" % (model_name, models.model_zoo))
        return
    criterion = nn.CrossEntropyLoss()
    if 'label-smooth' in args.keyword:
        criterion_smooth = utils.CrossEntropyLabelSmooth(args.num_classes, args.label_smooth)

    # load policy for initial phase
    models.policy.deploy_on_init(model, getattr(args, 'policy', ''))
    # load policy for epoch updating
    epoch_policies = models.policy.read_policy(getattr(args, 'policy', ''), section='epoch')
    # print model
    logging.info("models: %r" % model)
    logging.info("epoch_policies: %r" % epoch_policies)

    utils.check_folder(args.weights_dir)
    args.weights_dir = os.path.join(args.weights_dir, model_name)
    utils.check_folder(args.weights_dir)
    args.resume_file = os.path.join(args.weights_dir, args.case + "-" + args.resume_file)
    args.pretrained = os.path.join(args.weights_dir, args.pretrained)
    epoch = 0
    lr = args.lr
    best_acc = 0
    scheduler = None
    checkpoint = None
    # resume training
    if args.resume:
        if utils.check_file(args.resume_file):
            logging.info("resuming from %s" % args.resume_file)
            if torch.cuda.is_available():
                checkpoint = torch.load(args.resume_file)
            else:
                checkpoint = torch.load(args.resume_file, map_location='cpu')
            if 'epoch' in checkpoint:
                epoch = checkpoint['epoch']
                logging.info("resuming ==> last epoch: %d" % epoch)
                epoch = epoch + 1
                logging.info("updating ==> epoch: %d" % epoch)
            if 'best_acc' in checkpoint:
                best_acc = checkpoint['best_acc']
                logging.info("resuming ==> best_acc: %f" % best_acc)
            if 'learning_rate' in checkpoint:
                lr = checkpoint['learning_rate']
                logging.info("resuming ==> learning_rate: %f" % lr)
            if 'state_dict' in checkpoint:
                utils.load_state_dict(model, checkpoint['state_dict'])
                logging.info("resumed from %s" % args.resume_file)
        else:
            logging.info("warning: *** resume file not exists({})".
                format(args.resume_file))
            args.resume = False
    else:
        if utils.check_file(args.pretrained):
            logging.info("load pretrained from %s" % args.pretrained)
            if torch.cuda.is_available():
                checkpoint = torch.load(args.pretrained)
            else:
                checkpoint = torch.load(args.pretrained, map_location='cpu')
            logging.info("load pretrained ==> last epoch: %d" % checkpoint.get('epoch', 0))
            logging.info("load pretrained ==> last best_acc: %f" % checkpoint.get('best_acc', 0))
            logging.info("load pretrained ==> last learning_rate: %f" % checkpoint.get('learning_rate', 0))
            #if 'learning_rate' in checkpoint:
            #    lr = checkpoint['learning_rate']
            #    logging.info("resuming ==> learning_rate: %f" % lr)
            try:
                utils.load_state_dict(model, checkpoint.get('state_dict', checkpoint.get('model', checkpoint)))
            except RuntimeError as err:
                logging.info("Loading pretrained model failed %r" % err)
        else:
            logging.info("no pretrained file exists({}), init model with default initlizer".
                format(args.pretrained))

    if args.device_ids is not None:
        torch.cuda.set_device(args.device_ids[0])
        if not isinstance(model, nn.DataParallel) and len(args.device_ids) > 1:
            model = nn.DataParallel(model, args.device_ids).cuda()
        else:
            model = model.cuda()
        criterion = criterion.cuda()
        if 'label-smooth' in args.keyword:
            criterion_smooth = criterion_smooth.cuda()

    if 'label-smooth' in args.keyword:
        train_criterion = criterion_smooth
    else:
        train_criterion = criterion

    # move after to_cuda() for speedup
    if args.re_init and not args.resume:
        for m in model.modules():
            if hasattr(m, 'init_after_load_pretrain'):
                m.init_after_load_pretrain()

    # dataset
    data_path = args.root
    dataset = args.dataset
    logging.info("loading dataset with batch_size {} and val-batch-size {}. "
        "dataset: {}, resolution: {}, path: {}".
        format(args.batch_size, args.val_batch_size, dataset, args.input_size, data_path))

    if args.val_batch_size < 1:
        val_loader = None
    else:
        if args.evaluate:
            val_batch_size = (args.batch_size // 100) * 100
            if val_batch_size > 0:
                args.val_batch_size = val_batch_size
            logging.info("update val_batch_size to %d in evaluate mode" % args.val_batch_size)
        val_loader = datasets.data_loader(args.dataset)('val', args)

    if args.evaluate and val_loader is not None:
        if args.fp16 and torch.backends.cudnn.enabled and apex_enable and args.device_ids is not None:
            logging.info("training with apex fp16 at opt_level {}".format(args.opt_level))
        else:
            args.fp16 = False
            logging.info("training without apex")

        if args.fp16:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) # 
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

        logging.info("evaluate the dataset on pretrained model...")
        result = validate(val_loader, model, criterion, args)
        top1, top5, loss = result
        logging.info('evaluate accuracy on dataset: top1(%f) top5(%f)' %(top1, top5))
        return

    train_loader = datasets.data_loader(args.dataset)('train', args)
    if isinstance(train_loader, torch.utils.data.dataloader.DataLoader):
        train_length = len(train_loader)
    else:
        train_length = getattr(train_loader, '_size', 0) / getattr(train_loader, 'batch_size', 1)

    # sample several iteration / epoch to calculate the initial value of quantization parameters
    if args.stable_epoch > 0 and args.stable <= 0:
        args.stable = train_length * args.stable_epoch
        logging.info("update stable: %d" % args.stable)

    # fix learning rate at the beginning to warmup
    if args.warmup_epoch > 0 and args.warmup <= 0:
        args.warmup = train_length * args.warmup_epoch
        logging.info("update warmup: %d" % args.warmup)

    params_dict = dict(model.named_parameters())
    params = []
    quant_wrapper = []
    for key, value in params_dict.items():
        #print(key)
        if 'quant_weight' in key and 'quant_weight' in args.custom_lr_list:
            to_be_quant = key.split('.quant_weight')[0] + '.weight'
            if to_be_quant not in quant_wrapper:
                quant_wrapper += [to_be_quant]
    if len(quant_wrapper) > 0 and args.verbose:
        logging.info("quant_wrapper: {}".format(quant_wrapper))

    for key, value in params_dict.items():
        shape = value.shape
        custom_hyper = dict()
        custom_hyper['params'] = value
        if value.requires_grad == False:
            continue

        found = False
        for i in args.custom_decay_list:
            if i in key and len(i) > 0:
              found = True
              break
        if found:
            custom_hyper['weight_decay'] = args.custom_decay
        elif (not args.decay_small and args.no_decay_small) and ((len(shape) == 4 and shape[1] == 1) or (len(shape) == 1)):
            custom_hyper['weight_decay'] = 0.0

        found = False
        for i in args.custom_lr_list:
            if i in key and len(i) > 0:
              found = True
              break
        if found:
            #custom_hyper.setdefault('lr_constant', args.custom_lr) # 2019.11.25
            custom_hyper['lr'] = args.custom_lr
        elif key in quant_wrapper:
            custom_hyper.setdefault('lr_constant', args.custom_lr)
            custom_hyper['lr'] = args.custom_lr
           
        params += [custom_hyper]

        if 'debug' in args.keyword:
            logging.info("{}, decay {}, lr {}, constant {}".
                    format(key, custom_hyper.get('weight_decay', "default"), custom_hyper.get('lr', "default"), custom_hyper.get('lr_constant', "No") ))

    optimizer = None
    if args.optimizer == "ADAM":
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    if args.resume and checkpoint is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except RuntimeError as error:
            logging.info("Restore optimizer state failed %r" % error)

    if args.fp16 and torch.backends.cudnn.enabled and apex_enable and args.device_ids is not None:
        logging.info("training with apex fp16 at opt_level {}".format(args.opt_level))
    else:
        args.fp16 = False
        logging.info("training without apex")

    if args.sync_bn:
        logging.info("sync_bn to be supported, currently not yet")

    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
        if args.resume and checkpoint is not None:
            try:
                amp.load_state_dict(checkpoint['amp'])
            except RuntimeError as error:
                logging.info("Restore amp state failed %r" % error)

    # start tensorboard as late as possible
    if args.tensorboard and not args.evaluate:
        tb_log = os.path.join(args.log_dir, log_suffix)
        args.tensorboard = SummaryWriter(tb_log, filename_suffix='.' + log_suffix)
    else:
        args.tensorboard = None

    logging.info("start to train network " + model_name + ' with case ' + args.case)
    while epoch < (args.epochs + args.extra_epoch):
        if 'proxquant' in args.keyword:
            if args.proxquant_step < 10:
                if args.lr_policy in ['sgdr', 'sgdr_step', 'custom_step']:
                    index = len([x for x in args.lr_custom_step if x <= epoch])
                    for m in model.modules():
                        if hasattr(m, 'prox'):
                            m.prox = 1.0 - 1.0 / args.proxquant_step * (index + 1)
            else:
                for m in model.modules():
                    if hasattr(m, 'prox'):
                        m.prox = 1.0 - 1.0 / args.proxquant_step * epoch
                        if m.prox < 0:
                            m.prox = 0
        if epoch < args.epochs:
            lr, scheduler = utils.setting_learning_rate(optimizer, epoch, train_length, checkpoint, args, scheduler)
        if lr is None:
            logging.info('lr is invalid at epoch %d' % epoch)
            return
        else:
            logging.info('[epoch %d]: lr %e', epoch, lr)

        loss = 0
        top1, top5, eloss = 0, 0, 0
        is_best = top1 > best_acc
        # leverage policies on epoch
        models.policy.deploy_on_epoch(model, epoch_policies, epoch, optimizer=optimizer, verbose=logging.info)

        if 'lr-test' not in args.keyword: # otherwise only print the learning rate in each epoch
            # training
            loss = train(train_loader, model, train_criterion, optimizer, args, scheduler, epoch, lr)
            #for i in range(train_length):
            #  scheduler.step()
            logging.info('[epoch %d]: train_loss %.3f' % (epoch, loss))

            # validate
            top1, top5, eloss = 0, 0, 0
            top1, top5, eloss = validate(val_loader, model, criterion, args)
            is_best = top1 > best_acc
            if is_best:
                best_acc = top1
            logging.info('[epoch %d]: test_acc %f %f, best top1: %f, loss: %f', epoch, top1, top5, best_acc, eloss)

        if args.tensorboard is not None:
            args.tensorboard.add_scalar(log_suffix + '/train-loss', loss, epoch)
            args.tensorboard.add_scalar(log_suffix + '/eval-top1', top1, epoch)
            args.tensorboard.add_scalar(log_suffix + '/eval-top5', top5, epoch)
            args.tensorboard.add_scalar(log_suffix + '/lr', lr, epoch)

        utils.save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : None if scheduler is None else scheduler.state_dict(),
                'best_acc': best_acc,
                'learning_rate': lr,
                'amp': None if not args.fp16 else amp.state_dict(),
                }, is_best, args)

        epoch = epoch + 1
        if epoch == 1:
            logging.info(utils.gpu_info())

def train(loader, model, criterion, optimizer, args, scheduler, epoch, lr):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter()

    if isinstance(loader, torch.utils.data.dataloader.DataLoader):
        length = len(loader)
    else:
        length = getattr(loader, '_size', 0) / getattr(loader, 'batch_size', 1)
    model.train()
    if 'less_bn' in args.keyword:
        utils.custom_state(model)

    end = time.time()
    for i, data in enumerate(loader):
        if isinstance(data, list) and isinstance(data[0], dict):
            input = data[0]['data']
            target = data[0]['label'].squeeze()
        else:
            input, target = data
        data_time.update(time.time() - end)

        if args.device_ids is not None:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True).long()
        
        if args.mixup_enable:
            input, target_a, target_b, lam = utils.mixup_data(input, target, args.mixup_alpha, use_cuda=(args.device_ids is not None))

        if 'sgdr' in args.lr_policy and scheduler is not None and torch.__version__ < "1.0.4" and epoch < args.epochs:
            scheduler.step()
            for group in optimizer.param_groups:
                if 'lr_constant' in group:
                    group['lr'] = group['lr_constant']
            lr_list = scheduler.get_lr()
            if isinstance(lr_list, list):
                lr = lr_list[0]

        outputs = model(input)
        if isinstance(outputs, dict) and hasattr(model, '_out_features'):
            outputs = outputs[model._out_features[0]]

        if args.mixup_enable:
            mixup_criterion = lambda pred, target, \
                    lam: (-F.log_softmax(pred, dim=1) * torch.zeros(pred.size()).cuda().scatter_(1, target.data.view(-1, 1), lam.view(-1, 1))) \
                    .sum(dim=1).mean()
            loss = utils.mixup_criterion(target_a, target_b, lam)(mixup_criterion, outputs)
        else:
            loss = criterion(outputs, target)

        if 'quant_loss' in args.global_buffer:
            loss += args.global_buffer['quant_loss']
            args.global_buffer.pop('quant_loss')

        if i % args.iter_size == 0:
            optimizer.zero_grad()

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if i % args.iter_size == (args.iter_size - 1):
            if args.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            iterations = epoch * length + i
            if args.wakeup > iterations:
                for param_group in optimizer.param_groups:
                    if param_group.get('lr_constant', None) is not None:
                        continue
                    param_group['lr'] = param_group['lr'] * (1.0 / args.wakeup) * iterations
                logging.info('train {}/{}, change learning rate to lr * {}'.format(i, length, iterations / args.wakeup))
            if iterations >= args.warmup:
                optimizer.step()

        if 'sgdr' in args.lr_policy and scheduler is not None and torch.__version__ > "1.0.4" and epoch < args.epochs:
            scheduler.step()
            for group in optimizer.param_groups:
                if 'lr_constant' in group:
                    group['lr'] = group['lr_constant']
            lr_list = scheduler.get_lr()
            if isinstance(lr_list, list):
                lr = lr_list[0]

        losses.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.report_freq == 0:
            logging.info('train %d/%d, loss:%.3f(%.3f), batch time:%.2f(%.2f), data load time: %.2f(%.2f)' %
              (i, length, losses.val, losses.avg, batch_time.val, batch_time.avg, data_time.val, data_time.avg))

        if epoch == 0 and i == 10:
            logging.info(utils.gpu_info())
        if args.delay > 0:
            time.sleep(args.delay)

        input = None
        target = None
        data = None

    if 'dali' in args.dataset:
        loader.reset()

    return losses.avg

def validate(loader, model, criterion, args):
    if loader is None:
        logging.info('eval_loader is None, skip validate')
        return 0, 0

    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    if isinstance(loader, torch.utils.data.dataloader.DataLoader):
        length = len(loader)
    else:
        length = getattr(loader, '_size', 0) / getattr(loader, 'batch_size', 1)
    model.eval()

    with torch.no_grad():
        end = time.time()
        for step, data in enumerate(loader):
            if isinstance(data, list) and isinstance(data[0], dict):
                input = data[0]['data']
                target = data[0]['label'].squeeze()
            else:
                input, target = data

            if args.device_ids is not None:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True).long()

            outputs = model(input)
            if isinstance(outputs, dict) and hasattr(model, '_out_features'):
                outputs = outputs[model._out_features[0]]

            loss = criterion(outputs, target)
 
            prec1, prec5 = utils.accuracy(outputs, target, topk=(1, 5))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
            losses.update(loss.item(), input.size(0))

            if step % args.report_freq == 0:
                logging.info('test %d/%d %.3f %.3f' % (step, length, top1.avg, top5.avg))

            input = None
            target = None
            data = None
        logging.info("evaluation time: %.3f s" % (time.time() - end))

    if 'dali' in args.dataset:
        loader.reset()

    return top1.avg, top5.avg, losses.avg
 

if __name__ == '__main__':
    main()

