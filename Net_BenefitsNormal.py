import sys
import os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from utils.config import img_param_init, set_random_seed
from utils.prepare_data_dg_clip import *
import copy
import argparse
from nets.models import ClipModelat
import torch.optim as optim
import torch
import numpy as np
from utils.training import train
from utils.testing import test, Glotest
from utils.aggregation import communication
from adaptation import LMMDLoss
from  adaptation import AdversarialLoss
from nets.models import MLP
from nets.MLPs import ImageMlp, TextMlp
from kan import KAN


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='BrainTumor')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--datapercent', type=float,
                        default=6e-1, help='data percent to use')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--root_dir', type=str, default='./data/')
    parser.add_argument('--iters', type=int, default=100,
                        help='iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=1,
                        help='optimization iters in local worker between communication')
    parser.add_argument('--mode', type=str, default='FedAtImg')
    parser.add_argument('--net', type=str, default='ViT-B/32',
                        help='[RN50 | RN101 | RN50x4 | RN50x16 | RN50x64 | ViT-B/32 | ViT-B/16 | ViT-L/14 | ViT-L/14@336px]')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--n_clients', type=int, default=20)
    parser.add_argument('--n_iter', type=int, default=50)
    parser.add_argument('--test_envs', type=int, nargs='+', default=[3])
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.98)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--step', type=float, default=0)
    parser.add_argument('--aggmode', type=str, default='att')
    parser.add_argument('--weight_decay', type=float, default=0.02)
    parser.add_argument('--method', type=str, default='ours')
    parser.add_argument('--temp', type=float, default=0.5)
    args = parser.parse_args()
    args.random_state = np.random.RandomState(1)
    set_random_seed(args.seed)
    args.n_clients = 4

    args = img_param_init(args)
    os.makedirs('./data/', exist_ok=True)
    server_model = ClipModelat(
        args.net, attention=True, freezepy=True)

    train_loaders, val_loaders, test_loaders, test_train, train_test_loaders = get_data(
        args.dataset)(args, server_model)
    server_model.initdgatal(test_loaders[3])
    client_num = len(test_loaders)
    sclient_num = client_num-len(args.test_envs)
    client_weights = [0.33333333  for i in range(sclient_num)]
    client_weights.append(0.33333333333)
    models = [copy.deepcopy(server_model)for idx in range(client_num)]
    for i in range(client_num):
        models[i].model.to(device)
        models[i].fea_attn.to(device)
    num_params = sum(p.numel() for p in server_model.parameters())
    print(f"The network has {num_params} parameters.")
    best_changed = False
    server_model_pre = server_model
    best_acc = 0
    finalrecord = ''
    logrecord = ''
    log = []
    log2 = []
    log3 = []
    log4 = []
    previous_nets = models
    adv_loss = AdversarialLoss()
    adv = [AdversarialLoss() for idx in range(client_num)]
    # mlp = [KAN(layers_hidden=[512, 256, 4]).to(device) for idx in range(client_num)]
    mlp = [MLP(hidden_size=512, input_size=512, num_classes=4).to(device) for idx in range(client_num) ]
    num_params = sum(p.numel() for p in mlp[0].parameters())
    print(f"The MLP has {num_params} parameters.")

    # optimizers = [optim.Adam(params=[{'params': models[idx].parameters()}], lr=args.lr, betas=(
    #     args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay) for idx in range(client_num)]
    optimizers = [ optim.AdamW(
        params=[ {'params': models[ idx ].fea_attn.parameters()},{'params':mlp[idx].parameters()}], lr=args.lr, betas=(
            args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay) for idx in range(client_num) ]

    net_benefits = []
    net_benefits2 = [ ]
    net_benefits3 = [ ]
    net_benefits4 = [ ]
    tpr = [ ]
    tpr2 = [ ]
    tpr3 = [ ]
    tpr4 = [ ]
    fpr = [ ]
    fpr2 = [ ]
    fpr3 = [ ]
    fpr4 = [ ]
    net_benefits_all = []
    net_benefits_none = []
    best_testing_acc = [0, 0, 0, 0]
    if not os.path.exists('./SavedModel/FedCLIP'):
        os.makedirs('./SavedModel/FedCLIP')

    for a_iter in range(args.iters): #All Epoch

        for wi in range(args.wk_iters): #Each client local training epoch
            print("============ Train epoch {} ============".format(
                wi + a_iter * args.wk_iters))
            logrecord += 'Train epoch:%d\n' % (wi + a_iter * args.wk_iters)
            for client_idx, model in enumerate(models):
                if client_idx in args.test_envs:
                    pass
                else:
                    train(
                        args, model, train_loaders[client_idx], optimizers[client_idx], device, test_train[0], adv[client_idx], server_model_pre, previous_nets[client_idx], mlp[client_idx])

        args.step += 1


        with torch.no_grad():
            server_model_pre = server_model
            previous_nets = models
            server_model, models = communication(
                args, server_model, models, client_weights)

            # Validation
            Val_Acc = [0, 0, 0]
            Val_mean = 0
            for client_idx in range(client_num):
                if client_idx in args.test_envs:
                    pass
                else:
                    test_acc, bacc, f1, precision, recall, benefits, tp, fp, all, none = Glotest(args, models[ client_idx ],
                                                                           val_loaders[ client_idx ], device,
                                                                           )
                    if client_idx == 0:
                        Val_Acc[0] = test_acc
                        if test_acc > best_testing_acc[ client_idx ]:
                            best_testing_acc[ client_idx ] = test_acc
                        log.append([test_acc, bacc, f1, precision, recall])
                    elif client_idx == 1:
                        Val_Acc[1] = test_acc
                        if test_acc > best_testing_acc[ client_idx ]:
                            best_testing_acc[ client_idx ] = test_acc
                            net_benefits2 = benefits
                            tpr2 = tp
                            fpr2 = fp
                        log2.append([test_acc, bacc, f1, precision, recall])
                    elif client_idx == 2:
                        Val_Acc[2] = test_acc
                        if test_acc > best_testing_acc[ client_idx ]:
                            best_testing_acc[ client_idx ] = test_acc
                            net_benefits3 = benefits
                            tpr3 = tp
                            fpr3 = fp
                        log3.append([test_acc, bacc, f1, precision, recall])


                    if (Val_Acc[0]+Val_Acc[1]+Val_Acc[2]) / 3 > Val_mean:
                        Val_mean = (Val_Acc[ 0 ] + Val_Acc[ 1 ] + Val_Acc[ 2 ]) / 3
                        torch.save(models[0].fea_attn.state_dict(), './SavedModel/FedCLIP/Client1.pth')
                        torch.save(models[1].fea_attn.state_dict(), './SavedModel/FedCLIP/Client2.pth')
                        torch.save(models[2].fea_attn.state_dict(), './SavedModel/FedCLIP/Client3.pth')
                    print(' Test site-{:d}| Validation Acc: {:.4f} | Bacc: {:.4f}'.format(client_idx, test_acc, bacc))

    # Testing
    TestingMetrics = [[],[],[],[]]
    with torch.no_grad():
        models[0].fea_attn.load_state_dict(torch.load('./SavedModel/FedCLIP/Client1.pth'))
        models[1].fea_attn.load_state_dict(torch.load('./SavedModel/FedCLIP/Client2.pth'))
        models[2].fea_attn.load_state_dict(torch.load('./SavedModel/FedCLIP/Client3.pth'))
        server_model, models = communication(
            args, server_model, models, client_weights)
    for client_idx in range(client_num):
        if client_idx in args.test_envs:
            test_acc, bacc, f1, precision, recall, benefits, tp, fp, all, none = Glotest(args, server_model,
                                                                      test_loaders[ client_idx ], device)
            net_benefits4 = benefits
            tpr4 = tp
            fpr4 = fp
            net_benefits_all = all
            net_benefits_none = none
            TestingMetrics[3].append([test_acc, bacc, f1, precision, recall])
            print(' Test site-{:d}| Test Acc: {:.4f} | Bacc: {:.4f}'.format(client_idx, test_acc, bacc))
        else:
            if client_idx == 0:
                test_acc, bacc, f1, precision, recall, benefits, tp, fp, all, none = Glotest(args, models[ client_idx ],
                                                                       test_loaders[ client_idx ], device,
                                                                       )
                net_benefits = benefits
                tpr = tp
                fpr = fp
                TestingMetrics[0].append([test_acc, bacc, f1, precision, recall])
                print(' Test site-{:d}| Test Acc: {:.4f} | Bacc: {:.4f}'.format(client_idx, test_acc, bacc))
            elif client_idx == 1:
                test_acc, bacc, f1, precision, recall, benefits, tp, fp, all, none = Glotest(args, models[ client_idx ],
                                                                       test_loaders[ client_idx ], device,
                                                                       )
                net_benefits2 = benefits
                tpr2 = tp
                fpr2 = fp
                TestingMetrics[1].append([test_acc, bacc, f1, precision, recall])
                print(' Test site-{:d}| Test Acc: {:.4f} | Bacc: {:.4f}'.format(client_idx, test_acc, bacc))
            elif client_idx == 2:
                test_acc, bacc, f1, precision, recall, benefits, tp, fp, all, none = Glotest(args, models[ client_idx ],
                                                                       test_loaders[ client_idx ], device,
                                                                       )
                net_benefits3 = benefits
                tpr3 = tp
                fpr3 = fp
                TestingMetrics[2].append([test_acc, bacc, f1, precision, recall])
                print(' Test site-{:d}| Test Acc: {:.4f} | Bacc: {:.4f}'.format(client_idx, test_acc, bacc))


    dataset = '/BrainTumor/'
    SaveFile = '/lr=5e-5/'
    method = '/FedCLIP/'
    FileName = [ 'ValidationMetricsClient1.csv', 'ValidationMetricsClient2.csv', 'ValidationMetricsClient3.csv',
                 'ValidationMetricsGlobal.csv',
                 'NetBenefitsClient1.csv', 'NetBenefitsClient2.csv', 'NetBenefitsClient3.csv',
                 'NetBenefitsGlobal.csv', 'TPRClient1.csv', 'TPRClient2.csv', 'TPRClient3.csv',
                 'TPRGlobal.csv', 'FPRClient1.csv', 'FPRClient2.csv', 'FPRClient3.csv',
                 'FPRGlobal.csv', 'NetBenefitsAll.csv', 'NetBenefitsNone.csv',
                 'TestingMetricsClient1.csv','TestingMetricsClient2.csv','TestingMetricsClient3.csv',
                 'TestingMetricsGlobal.csv']
    logs = [ [log, log2, log3, log4], [net_benefits, net_benefits, net_benefits, net_benefits],
           [tpr, tpr2, tpr3, tpr4], [fpr, fpr2, fpr3, fpr4], [net_benefits_all, net_benefits_none],
             [TestingMetrics[0], TestingMetrics[1], TestingMetrics[2], TestingMetrics[3]]]
    length = [4,4,4,4,2,4]
    k = 0
    for i in range(0, 6):
        if not os.path.exists('./results' + dataset + method + SaveFile):
            os.makedirs('./results' + dataset + method + SaveFile)
        for j in range(0, length[i]):
            np_log = np.array(logs[i][j], dtype=float)
            loc = './results' + dataset + method + SaveFile + FileName[k]
            k += 1
            np.savetxt(loc, np_log, delimiter=',', fmt='%.6f')