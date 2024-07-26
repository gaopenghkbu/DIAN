from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from dian import embed_net
from utils import *
from loss import OriTripletLoss, Frequency_ContrastiveLoss, PLoss
from tensorboardX import SummaryWriter
from random_erasing import *
from lightning.pytorch import seed_everything, Trainer
from torchinfo import summary
import threading
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='nnnnn_save_model2/', type=str, help='model save path')
parser.add_argument('--save_epoch', default=20, type=int, metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str, help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=6, type=int, metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=4, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--margin', default=0.3, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--num_pos', default=4, type=int, help='num of pos per identity in each modality')
parser.add_argument('--trial', default=2, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=42, type=int, metavar='t', help='random seed')
parser.add_argument('--gpu', default='2', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--lambda_1', default=0.5, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--lambda_2', default=0.01, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--name', default='best_model', type=str, help='learning rate, 0.00035 for adam')

args = parser.parse_args()
print(threading.current_thread().name)

#set_seed(args.seed)
seed_everything(args.seed)
train = Trainer(accelerator='gpu')

dataset = args.dataset
if dataset == 'sysu':
    data_path = '../../paper/dataset/SYSU-MM01/'
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]  # thermal to visible
    pool_dim = 2048
elif dataset == 'regdb':
    data_path = '../../paper/dataset/RegDB/'
    log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1]  # visible to thermal
    pool_dim = 2048


checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = dataset
suffix = suffix + '_deen_p{}_n{}_lr_{}_seed_{}__name{}'.format(args.num_pos, args.batch_size, args.lr, args.seed, args.name)


if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])



transform_sysu_color = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomGrayscale(p=0.5),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    RandomErasing(probability = args.erasing_p, sl = 0.2, sh = 0.8, r1 = 0.3, mean=[0.485, 0.456, 0.406]),
    #ChannelExchange(gray = 2)
])

transform_regdb = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomGrayscale(p=0.5),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    RandomErasing(probability = args.erasing_p, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.485, 0.456, 0.406]),
])
transform_llcm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomGrayscale(p=0.5),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    RandomErasing(probability = args.erasing_p, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.485, 0.456, 0.406]),
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()
if dataset == 'sysu':
    # training set
    # print('transform_sysu_color ********************',transform_sysu_color)
    # print('transform_sysu_thermal ********************',transform_sysu_thermal)
    trainset = SYSUData(data_path, transform=transform_sysu_color)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_regdb)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')



gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
net = embed_net(n_class, dataset, arch=args.arch)
net.cuda()

summary(net, input_size=[(args.batch_size, 3, args.img_h, args.img_w), (args.batch_size, 3, args.img_h, args.img_w)])

# print(net)
if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = 0 #checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
criterion_id = nn.CrossEntropyLoss()

loader_batch = args.batch_size * args.num_pos
criterion_tri= OriTripletLoss(batch_size=loader_batch, margin=args.margin)

criterion_ctri = ctri_loss(margin=0.2)
criterion_bl = BLLoss(batch_size=args.batch_size)

criterion_id.to(device)
criterion_tri.to(device)
criterion_ctri.to(device)
criterion_bl.to(device)

if args.optim == 'sgd':
    ignored_params =   list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.classifier.parameters())) \
               

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())


    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr},
        
        ],
        weight_decay=5e-4, momentum=0.9, nesterov=True)


# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 80:
        lr = args.lr * 0.1
    elif epoch >= 80:
        lr = args.lr * 0.01
    elif epoch >= 120:
        lr = args.lr * 0.001
    elif epoch >= 160:
        lr = args.lr * 0.0001
        
    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr


def train(epoch):

    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    #con_loss = AverageMeter()
    cpm_loss = AverageMeter()
    ort_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):

        labs = torch.cat((label1, label2, label1, label2), 0)
        labels = torch.cat((label1, label2, label1, label2, label1, label2, label1, label2), 0)
        #labels = torch.cat((label1, label2), 0)


        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())

        labs = Variable(labs.long().cuda())
        labels = Variable(labels.long().cuda())

        data_time.update(time.time() - end)

        feat1, out1, _ = net(input1, input2)
        #print('feat1=====', feat1.shape)
        #print('labs=====', labs.shape)
        loss_id = criterion_id(out1, labels)
        
        loss_tri = criterion_tri(feat1, labels)
        
        ft0, ft1, ft3, ft5 = torch.chunk(feat1, 4, 0)
        
        loss_cpm = (criterion_ctri(torch.cat((ft1, ft3), 0), labs) + criterion_ctri(torch.cat((ft3, ft5), 0), labs) + criterion_ctri(torch.cat((ft0, ft3), 0), labs))* args.lambda_1
        loss_ort = (criterion_bl(torch.cat((ft1, ft3), 0), labs) + criterion_bl(torch.cat((ft3, ft5), 0), labs) + criterion_bl(torch.cat((ft0, ft3), 0), labs))* args.lambda_1
   
        loss = loss_id + loss_tri + loss_cpm + loss_ort
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update P
        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        tri_loss.update(loss_tri.item(), 2 * input1.size(0))
        #con_loss.update(loss_con.item(), 2 * input1.size(0))
        cpm_loss.update(loss_cpm.item(), 2 * input1.size(0))
        ort_loss.update(loss_ort.item(), 2 * input1.size(0))
        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Loss:{train_loss.val:.3f} '
                  'iLoss:{id_loss.val:.3f} '
                  'TLoss:{tri_loss.val:.3f} '
                  'CLoss:{cpm_loss.val:.3f} '
                  'BLoss:{ort_loss.val:.3f} '
                  .format(
                epoch, batch_idx, len(trainloader),
                train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss, cpm_loss=cpm_loss, ort_loss=ort_loss))

    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('tri_loss', tri_loss.avg, epoch)

    writer.add_scalar('ctri_loss', cpm_loss.avg, epoch)
    writer.add_scalar('bl_loss', ort_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)


def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat1 = np.zeros((ngall, pool_dim))
    gall_feat2 = np.zeros((ngall, pool_dim))
    gall_feat3 = np.zeros((ngall, pool_dim))
    gall_feat4 = np.zeros((ngall, pool_dim))
    gall_feat5 = np.zeros((ngall, pool_dim))
    gall_feat6 = np.zeros((ngall, pool_dim))
    gall_feat7 = np.zeros((ngall, pool_dim))
    gall_feat8 = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input, test_mode[0])
            gall_feat1[ptr:ptr + batch_num, :] = feat[:batch_num].detach().cpu().numpy()
            gall_feat2[ptr:ptr + batch_num, :] = feat_att[:batch_num].detach().cpu().numpy()
            gall_feat3[ptr:ptr + batch_num, :] = feat[batch_num:batch_num*2].detach().cpu().numpy()
            gall_feat4[ptr:ptr + batch_num, :] = feat_att[batch_num:batch_num*2].detach().cpu().numpy()
            gall_feat5[ptr:ptr + batch_num, :] = feat[batch_num*2:batch_num*3].detach().cpu().numpy()
            gall_feat6[ptr:ptr + batch_num, :] = feat_att[batch_num*2:batch_num*3].detach().cpu().numpy()
            gall_feat7[ptr:ptr + batch_num, :] = feat_att[batch_num*3:].detach().cpu().numpy()
            gall_feat8[ptr:ptr + batch_num, :] = feat_att[batch_num*3:].detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat1 = np.zeros((nquery, pool_dim))
    query_feat2 = np.zeros((nquery, pool_dim))
    query_feat3 = np.zeros((nquery, pool_dim))
    query_feat4 = np.zeros((nquery, pool_dim))
    query_feat5 = np.zeros((nquery, pool_dim))
    query_feat6 = np.zeros((nquery, pool_dim))
    query_feat7 = np.zeros((nquery, pool_dim))
    query_feat8 = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input, test_mode[1])
            query_feat1[ptr:ptr + batch_num, :] = feat[:batch_num].detach().cpu().numpy()
            query_feat2[ptr:ptr + batch_num, :] = feat_att[:batch_num].detach().cpu().numpy()
            query_feat3[ptr:ptr + batch_num, :] = feat[batch_num:batch_num*2].detach().cpu().numpy()
            query_feat4[ptr:ptr + batch_num, :] = feat_att[batch_num:batch_num*2].detach().cpu().numpy()
            query_feat5[ptr:ptr + batch_num, :] = feat[batch_num*2:batch_num*3].detach().cpu().numpy()
            query_feat6[ptr:ptr + batch_num, :] = feat_att[batch_num*2:batch_num*3].detach().cpu().numpy()
            query_feat7[ptr:ptr + batch_num, :] = feat_att[batch_num*3:].detach().cpu().numpy()
            query_feat8[ptr:ptr + batch_num, :] = feat_att[batch_num*3:].detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat1 = np.matmul(query_feat1, np.transpose(gall_feat1))
    distmat2 = np.matmul(query_feat2, np.transpose(gall_feat2))
    distmat3 = np.matmul(query_feat3, np.transpose(gall_feat3))
    distmat4 = np.matmul(query_feat4, np.transpose(gall_feat4))
    distmat5 = np.matmul(query_feat5, np.transpose(gall_feat5))
    distmat6 = np.matmul(query_feat6, np.transpose(gall_feat6))
    distmat7 = np.matmul(query_feat7, np.transpose(gall_feat7))
    distmat8 = np.matmul(query_feat8, np.transpose(gall_feat8))
    # distmat9 = np.matmul(query_feat6, np.transpose(gall_feat9))
    distmat9 = distmat1 + distmat2 + distmat3 + distmat4 + distmat5 + distmat6 + distmat7 + distmat8
    distmat11 = distmat2 + distmat4 + distmat6 +distmat8
    # distmat7 = distmat1 + distmat2
    # evaluation
    if dataset == 'regdb':
        cmc2, mAP2, mINP2 = eval_regdb(-distmat2, query_label, gall_label)
        cmc4, mAP4, mINP4 = eval_regdb(-distmat4, query_label, gall_label)
        cmc6, mAP6, mINP6 = eval_regdb(-distmat6, query_label, gall_label)
        cmc8, mAP8, mINP8 = eval_regdb(-distmat8, query_label, gall_label)
        cmc9, mAP9, mINP9 = eval_regdb(-distmat9, query_label, gall_label)
        cmc11, mAP11, mINP11 = eval_regdb(-distmat11, query_label, gall_label)
    elif dataset == 'sysu':
        # cmc1, mAP1, mINP1 = eval_sysu(-distmat1, query_label, gall_label, query_cam, gall_cam)
        cmc2, mAP2, mINP2 = eval_sysu(-distmat2, query_label, gall_label, query_cam, gall_cam)
        cmc4, mAP4, mINP4 = eval_sysu(-distmat4, query_label, gall_label, query_cam, gall_cam)
        cmc6, mAP6, mINP6 = eval_sysu(-distmat6, query_label, gall_label, query_cam, gall_cam)
        cmc8, mAP8, mINP8 = eval_sysu(-distmat8, query_label, gall_label, query_cam, gall_cam)
        cmc9, mAP9, mINP9 = eval_sysu(-distmat9, query_label, gall_label, query_cam, gall_cam)
        cmc11, mAP11, mINP11 = eval_sysu(-distmat11, query_label, gall_label, query_cam, gall_cam)
        # cmc7, mAP7, mINP7 = eval_sysu(-distmat7, query_label, gall_label, query_cam, gall_cam)
   
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    # return cmc1, mAP1, mINP1, cmc2, mAP2, mINP2
    return cmc2, mAP2, mINP2, cmc4, mAP4, mINP4, cmc6, mAP6, mINP6, cmc8, mAP8, mINP8, cmc9, mAP9, mINP9, cmc11, mAP11, mINP11
    # return cmc1, mAP1, mINP1, cmc2, mAP2, mINP2, cmc7, mAP7, mINP7

cmc2, mAP2, mINP2, cmc4, mAP4, mINP4, cmc6, mAP6, mINP6, cmc8, mAP8, mINP8, cmc9, mAP9, mINP9, cmc11, mAP11, mINP11 = test(0)
print('test ok')
# training
print('==> Start Training...')
print('Start Training Time: ', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
for epoch in range(start_epoch, 150 - start_epoch):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, \
                              trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    print(epoch)
    print(trainset.cIndex)
    print(trainset.tIndex)

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
                                  sampler=sampler, num_workers=args.workers, drop_last=True)

    # training
    train(epoch)

    if epoch >= 100 and epoch % 1 == 0:
        print('Test Epoch: {}'.format(epoch))
    
        # testing
        cmc2, mAP2, mINP2, cmc4, mAP4, mINP4, cmc6, mAP6, mINP6, cmc8, mAP8, mINP8, cmc9, mAP9, mINP9, cmc11, mAP11, mINP11 = test(epoch)
        # save model
        if cmc9[0] > best_acc:  # not the real best for sysu-mm01
            best_acc = cmc9[0]
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'cmc': cmc9,
                'mAP': mAP9,
                'mINP': mINP9,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')
                    
            state = {
                'net': net.state_dict(),
                'cmc': cmc11,
                'mAP': mAP11,
                'mINP': mINP11,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_11best.t')
    
        # print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        #     cmc1[0], cmc1[4], cmc1[9], cmc1[19], mAP1, mINP1))
        print('0-POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc2[0], cmc2[4], cmc2[9], cmc2[19], mAP2, mINP2))
        print('1-POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc4[0], cmc4[4], cmc4[9], cmc4[19], mAP4, mINP4))
        print('3-POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc6[0], cmc6[4], cmc6[9], cmc6[19], mAP6, mINP6))
        print('5-POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc8[0], cmc8[4], cmc8[9], cmc8[19], mAP8, mINP8))
        print('A-POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc9[0], cmc9[4], cmc9[9], cmc9[19], mAP9, mINP9))
        # print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        #     cmc7[0], cmc7[4], cmc7[9], cmc7[19], mAP7, mINP7))
        print('N-POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc11[0], cmc11[4], cmc11[9], cmc11[19], mAP11, mINP11))
        print('Best Epoch [{}]'.format(best_epoch))

print('End Training Time: ', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
