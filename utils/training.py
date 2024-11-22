import copy
from nets.PromptCLIP import TextEncoder
import torch
from adaptation import LMMDLoss, MMDLoss, CORAL
from utils.clip_util import AverageMeter
import utils.clip_util as clu
import torch.nn as nn
import clip
from utils.loss_function import CrossEntropyLabelSmooth
from utils.clip_util import convert_models_to_fp32
from utils.clip_util import FocalLossWithSmoothing, LinearDiscriminantLoss, FocalLoss
from tqdm import tqdm
from adaptation import CORAL
from torch.autograd import Variable
import numpy as np
from torch.nn import KLDivLoss
from nets.MLPs import ImageMlp, TextMlp
import torch.nn.functional as F
from nets.LinearProbeV2 import compute_centroids,clip_classifier,compute_centroids_alpha,calculate_lr_alpha,calculate_init_alpha,calculate_lr_w
import math
from nets.CoCoOpCLIP import TextEncoderCocoOp

lmmd_loss = LMMDLoss()

def set_grad(model): # fine-tune head and freeze other layers
    # For visual encoder
    for param in model.visual.conv1.parameters():
        param.requires_grad = False
    for param in model.visual.ln_pre.parameters():
        param.requires_grad = False
    for layer in model.visual.transformer.resblocks:
        for param in layer.parameters():
            param.requires_grad = False
    for param in model.visual.ln_post.parameters(): #ln head in visual
        param.requires_grad = True

    # for text encoder
    for layer in model.transformer.resblocks:
        for param in layer.parameters():
            param.requires_grad = False
    for param in model.token_embedding.parameters():
        param.requires_grad = False
    for param in model.ln_final.parameters(): # ln head in text
        param.requires_grad = True

def set_grad2(model):
    # For visual encoder
    for param in model.visual.conv1.parameters():
        param.requires_grad = True
    for param in model.visual.ln_pre.parameters():
        param.requires_grad = True
    for layer in model.visual.transformer.resblocks:
        for param in layer.parameters():
            param.requires_grad = True
    for param in model.visual.ln_post.parameters(): #ln head in visual
        param.requires_grad = False

    # for text encoder
    for layer in model.transformer.resblocks:
        for param in layer.parameters():
            param.requires_grad = True
    for param in model.token_embedding.parameters():
        param.requires_grad = True
    for param in model.ln_final.parameters(): # ln head in text
        param.requires_grad = False

def set_parameters(model, previous_nets, coef_self):
    for new_param, old_param in zip(previous_nets.model.parameters(), model.model.parameters()):
        old_param.data = (new_param.data + coef_self * old_param.data).clone() # add learned knowledge for each client
        # params_i = mu_params + coef_self * params_i

def e(x, sigma):
    return math.exp(-x / sigma) / sigma

def weight_flatten(model):
    params = []
    for u in model.parameters():
        params.append(u.view(-1))
    params = torch.cat(params)

    return params

def totrain(model):
    model.model.train()
    model.fea_attn.train()

def totrainMLP(mlp):
    mlp.train()

def train(args, model, data_loader, optimizer, device, testloader, mmd_loss, server_model, previous_nets, mlp):
    totrain(model)
    totrainMLP(mlp)
    # texts = model.labels
    # t_features = clu.get_text_features_list(texts, model.model).float() #update each round based on the model
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    # loss_img = FocalLossWithSmoothing(num_classes=32)
    # loss_txt = FocalLossWithSmoothing(num_classes=32)
    loss_cla = nn.CrossEntropyLoss()
    kl_loss = KLDivLoss(reduction='batchmean', log_target=True)
    LD_loss = LinearDiscriminantLoss(num_classes=args.num_classes)
    train_loss_clf = AverageMeter()
    prox_loss_clf = AverageMeter()
    train_loss_transfer = AverageMeter()
    train_loss_transfer2 = AverageMeter()
    print(len(data_loader), len(testloader))
    source_data = iter(data_loader)
    source_data2 = iter(data_loader)
    target_data = iter(testloader)
    loss_all = 0
    if args.method == 'ours':
        # for _ in (range(0, args.n_iter)):
        for _ in tqdm(range(0, len(data_loader))):
            image, text, label = next(source_data)  # .next()
            if len(text) > 1:
                image = image.to(device)
                text = text.to(device)
                image_features = model.model.encode_image(image).float()
                text_features = model.model.encode_text(text).float()
                image_features_att = model.fea_attn(image_features)
                image_features = torch.mul(image_features_att, image_features)

                image_features = image_features / \
                image_features.norm(dim=1, keepdim=True)
                text_features = text_features / \
                text_features.norm(dim=1, keepdim=True)

                logit_scale = model.model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                ground_truth = torch.arange(
                len(image), dtype=torch.long, device=device)

                cla_loss = (loss_img(logits_per_image, ground_truth) +
                    loss_txt(logits_per_text, ground_truth)) / 2


                train_loss_clf.update(cla_loss.item())
                # train_loss_transfer.update(0.5 * loss_m.item())
                optimizer.zero_grad()
                cla_loss.backward()
                optimizer.step()
        print("cla loss: ", train_loss_clf.avg, 'trans loss:', train_loss_transfer.avg,'text KL loss:', prox_loss_clf.avg)
    if args.method == 'fedprox':
        for _ in tqdm(range(0, len(data_loader))):
            image, text, label = next(source_data)  # .next()
            if len(text) > 1:
                image = image.to(device)
                text = text.to(device)
                image_features = model.model.encode_image(image).float()
                text_features = model.model.encode_text(text).float()
                image_features = image_features / \
                                 image_features.norm(dim=1, keepdim=True)
                text_features = text_features / \
                                text_features.norm(dim=1, keepdim=True)
                logit_scale = model.model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                ground_truth = torch.arange(
                    len(image), dtype=torch.long, device=device)

                loss = (loss_img(logits_per_image, ground_truth) +
                        loss_txt(logits_per_text, ground_truth)) / 2
                train_loss_clf.update(loss.item())
                # print(loss)
                # loss_all += loss
                if args.step > 0:
                    w_diff = torch.tensor(1e-10, device=device)
                    for w, w_t in zip(server_model.parameters(), model.parameters()):
                        w_diff += torch.pow(torch.norm(w - w_t), 2).float()  # model difference
                        # print(w_diff)
                    w_diff = torch.sqrt(w_diff)
                    train_loss_transfer.update((1e-2 / 2. * w_diff).item())
                    loss += 1e-2 / 2. * w_diff  # dif loss
                    # print(loss)
                optimizer.zero_grad()
                loss.backward()
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
        print("cla loss: ", train_loss_clf.avg, 'w_diff loss: ', train_loss_transfer.avg)
    if args.method == 'moon':
        cos = torch.nn.CosineSimilarity(dim=-1)
        criterion = nn.CrossEntropyLoss()
        mu = 1
        for _ in tqdm(range(0, len(data_loader))):
            image, text, label = next(source_data)  # .next()
            if len(text) > 1:
                optimizer.zero_grad()
                image = image.to(device)
                text = text.to(device)
                image_features = model.model.encode_image(image).float()
                image_features_glo = server_model.model.encode_image(image).float()
                text_features = model.model.encode_text(text).float()
                image_features = image_features / \
                                 image_features.norm(dim=1, keepdim=True)
                text_features = text_features / \
                                text_features.norm(dim=1, keepdim=True)
                image_features_glo = image_features_glo / \
                                     image_features_glo.norm(dim=1, keepdim=True)
                logit_scale = model.model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                ground_truth = torch.arange(
                    len(image), dtype=torch.long, device=device)

                loss = (loss_img(logits_per_image, ground_truth) +
                        loss_txt(logits_per_text, ground_truth)) / 2
                train_loss_clf.update(loss.item())
                # MOON contrastive loss below, we refered the original codes, it needs [logits_per_image] to measure.
                # Model-Contrastive Federated Learning
                posi = cos(image_features, image_features_glo)  # pro1, pro2
                logits = posi.reshape(-1, 1)
                if args.step > 0:
                    image_features_pre = previous_nets.model.encode_image(image).float()
                    nega = cos(image_features, image_features_pre)  # pro1, pro3
                    logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
                    logits /= args.temp
                    labels = torch.zeros(image.size(0)).cuda().long()
                    loss += mu * criterion(logits, labels)
                    train_loss_transfer.update(mu * criterion(logits, labels))
                loss.backward()
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
        print("cla loss: ", train_loss_clf.avg, 'MOON loss: ', train_loss_transfer.avg)
    if args.method == 'fedfocal':
        loss_img = FocalLossWithSmoothing(num_classes=32) # Batch size
        loss_txt = FocalLossWithSmoothing(num_classes=32)
        for _ in tqdm(range(0, len(data_loader))):
            image, text, label = next(source_data)  # .next()
            image = image.to(device)
            text = text.to(device)
            image_features = model.model.encode_image(image).float()
            text_features = model.model.encode_text(text).float()
            image_features = image_features / \
                             image_features.norm(dim=1, keepdim=True)
            text_features = text_features / \
                            text_features.norm(dim=1, keepdim=True)
            logit_scale = model.model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            ground_truth = torch.arange(
                len(image), dtype=torch.long, device=device)

            loss = (loss_img(logits_per_image, ground_truth) +
                        loss_txt(logits_per_text, ground_truth)) / 2
            train_loss_clf.update(loss.item())
            loss.backward()
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
        print("cla loss: ", train_loss_clf.avg)
    if args.method == 'fedclip':
        loss_mmd = MMDLoss()
        loss_mse = nn.MSELoss()
        loss_kl = nn.KLDivLoss(log_target=False, reduction='batchmean')
        texts = model.labels
        text_features_all = clu.get_text_features_list(texts, model.model).float()
        text_features_all /= text_features_all.norm(dim=-1, keepdim=True)
        # for _ in (range(0, args.n_iter)):
        for _ in tqdm(range(0, len(data_loader))):
            image, text, label = next(source_data)  # .next()
            if len(text) > 1:
                image = image.to(device)
                text = text.to(device)
                label = label.to(device)
                image_features = model.model.encode_image(image).float()
                text_features = model.model.encode_text(text).float()
                image_features_att = model.fea_attn(image_features)
                image_features = torch.mul(image_features_att, image_features)
                image_features = image_features / \
                image_features.norm(dim=1, keepdim=True)
                text_features = text_features / \
                text_features.norm(dim=1, keepdim=True)


                logit_scale = model.model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                ground_truth = torch.arange(
                len(image), dtype=torch.long, device=device)


                cla_loss = (loss_img(logits_per_image, ground_truth) +
                    loss_txt(logits_per_text, ground_truth))/ 2


                outputs = mlp(image_features) # MLP loss
                mlp_loss = loss_cla(outputs, label)
                train_loss_transfer.update(mlp_loss.item())


                # Sim loss
                similarity = (100.0 * image_features @ text_features_all.T).softmax(dim=-1) # Logits based on similarity
                # loss_sim = loss_mse(outputs, similarity) # Sim loss

                # print(F.softmax(outputs, dim=0), F.softmax(similarity, dim=0))
                # loss_sim = loss_kl(F.softmax(outputs, dim=0), similarity)
                # print(loss_kl(F.log_softmax(similarity,dim=0), F.softmax(outputs, dim=0)))
                loss_sim = (loss_kl(F.log_softmax(outputs, dim=-1), F.softmax(similarity,dim=-1)) +
                            loss_kl(F.log_softmax(similarity,dim=-1), F.softmax(outputs, dim=-1)))/2
                # loss_sim = loss_mmd(F.softmax(outputs, dim=0), F.softmax(similarity, dim=0))

                # loss_sim = (loss_mi(F.softmax(outputs,dim=0))+loss_mi(F.softmax(similarity,dim=0))) / 2
                # loss_mse = (loss_sim2(outputs, similarity) + loss_sim2(similarity, outputs)) / 2

                # total loss
                loss = cla_loss + mlp_loss + 0.01 * loss_sim
                # loss = cla_loss + mlp_loss

                prox_loss_clf.update(0.01 * loss_sim.item())
                train_loss_clf.update(cla_loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print("attn loss: ", train_loss_clf.avg, 'mlp loss:', train_loss_transfer.avg, 'sim loss:', prox_loss_clf.avg)

    if args.method == 'fedavg':
        for _ in tqdm(range(0, len(data_loader))):
            image, text, label = next(source_data)  # .next()
            # image_t, _, _ = next(target_data)  # .next()
            if len(text) > 1:
                image = image.to(device)
                text = text.to(device)
                # image_t = image_t.to(device)
                image_features = model.model.encode_image(image).float()
                text_features = model.model.encode_text(text).float()

                image_features = image_features / \
                image_features.norm(dim=1, keepdim=True)
                text_features = text_features / \
                text_features.norm(dim=1, keepdim=True)

                logit_scale = model.model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t() # S
                logits_per_text = logits_per_image.t()

                ground_truth = torch.arange(
                len(image), dtype=torch.long, device=device)

                cla_loss = (loss_img(logits_per_image, ground_truth) +
                    loss_txt(logits_per_text, ground_truth)) / 2

                train_loss_clf.update(cla_loss.item())
                optimizer.zero_grad()
                cla_loss.backward()
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
        print("cla loss: ", train_loss_clf.avg)
    if args.method == 'fedmlp': #freeze CLIP and only finetune MLP
        # for _ in (range(0, args.n_iter)):
        for _ in tqdm(range(0, len(data_loader))):
            image, text, label = next(source_data)  # .next()
            # image_t, _, _ = next(target_data)  # .next()
            if len(text) > 1:
                image = image.to(device)
                text = text.to(device)
                image_features = model.model.encode_image(image).float()
                label = label.to(device)
                image_features = image_features / \
                image_features.norm(dim=1, keepdim=True)


                outputs = mlp(image_features)
                # print(outputs)
                cla_kan = loss_cla(outputs, label)
                train_loss_transfer.update(cla_kan.item())

                loss = cla_kan # only fine-tune mlp
                optimizer.zero_grad()
                loss.backward()
                # convert_models_to_fp32(model)
                optimizer.step()
                # clip.model.convert_weights(model)
        print("cla loss: ", train_loss_transfer.avg)

    if args.method == 'clip':
        # for _ in (range(0, args.n_iter)):
        for _ in tqdm(range(0, len(data_loader))):
            image, text, label = next(source_data)  # .next()
            if len(text) > 1:
                image = image.to(device)
                text = text.to(device)
                image_features = model.model.encode_image(image).float()
                text_features = model.model.encode_text(text).float()
                image_features_att = model.fea_attn(image_features)
                image_features = torch.mul(image_features_att, image_features)

                image_features = image_features / \
                image_features.norm(dim=1, keepdim=True)
                text_features = text_features / \
                text_features.norm(dim=1, keepdim=True)

                logit_scale = model.model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                ground_truth = torch.arange(
                len(image), dtype=torch.long, device=device)

                cla_loss = (loss_img(logits_per_image, ground_truth) +
                    loss_txt(logits_per_text, ground_truth)) / 2


                train_loss_clf.update(cla_loss.item())
                optimizer.zero_grad()
                cla_loss.backward()
                optimizer.step()
        print("cla loss: ", train_loss_clf.avg)
    if args.method == 'fedlast':
        for _ in tqdm(range(0, len(data_loader))):
            image, text, label = next(source_data)  # .next()
            # image_t, _, _ = next(target_data)  # .next()
            if len(text) > 1:
                image = image.to(device)
                text = text.to(device)
                image_features = model.model.encode_image(image).float()
                text_features = model.model.encode_text(text).float()

                image_features = image_features / \
                image_features.norm(dim=1, keepdim=True)
                text_features = text_features / \
                text_features.norm(dim=1, keepdim=True)

                logit_scale = model.model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                ground_truth = torch.arange(
                len(image), dtype=torch.long, device=device)

                cla_loss = (loss_img(logits_per_image, ground_truth) +
                    loss_txt(logits_per_text, ground_truth)) / 2

                train_loss_clf.update(cla_loss.item())
                optimizer.zero_grad()
                cla_loss.backward()
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
        print("cla loss: ", train_loss_clf.avg)
    if args.method == 'CLIPFC': #freeze CLIP and only finetune FC
        for _ in tqdm(range(0, len(data_loader))):
            image, text, label = next(source_data)  # .next()
            if len(text) > 1:
                image = image.to(device)
                text = text.to(device)
                image_features = model.model.encode_image(image).float()
                label = label.to(device)
                image_features = image_features / \
                image_features.norm(dim=1, keepdim=True)

                outputs = mlp(image_features)
                # print(outputs)
                loss = loss_cla(outputs, label)
                train_loss_transfer.update(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print("cla loss: ", train_loss_transfer.avg)
    if args.method == 'fedamp':
        lamda = 1.0
        alphaK = 1.0
        sigma = 1.0
        for _ in tqdm(range(0, len(data_loader))):
            image, text, label = next(source_data)  # .next()
            if len(text) > 1:
                image = image.to(device)
                text = text.to(device)
                image_features = model.model.encode_image(image).float()
                text_features = model.model.encode_text(text).float()

                image_features = image_features / \
                                 image_features.norm(dim=1, keepdim=True)
                text_features = text_features / \
                                text_features.norm(dim=1, keepdim=True)

                logit_scale = model.model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                ground_truth = torch.arange(
                    len(image), dtype=torch.long, device=device)

                loss = (loss_img(logits_per_image, ground_truth) +
                        loss_txt(logits_per_text, ground_truth)) / 2

                params = weight_flatten(model.model)
                params_ = weight_flatten(previous_nets.model)
                sub = params - params_

                loss += lamda / alphaK / 2 * torch.dot(sub, sub)
                train_loss_transfer.update((lamda / alphaK / 2 * torch.dot(sub, sub)).item())
                train_loss_clf.update(loss.item())
                optimizer.zero_grad()
                loss.backward()
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
        print("cla loss: ", train_loss_clf.avg - train_loss_transfer.avg, 'amp loss: ', train_loss_transfer.avg)
    if args.method == 'fedrep': # here we assume step and epoch for head and base are set to 1
        # phase 1, fine-tune clip head
        set_grad(model.model)
        # for _ in tqdm(range(0, len(data_loader))):
        for batch in tqdm(data_loader):
            # image, text, label = next(source_data)  # .next()
            image, text, label = batch
            if len(text) > 1:
                image = image.to(device)
                text = text.to(device)
                image_features = model.model.encode_image(image).float()
                text_features = model.model.encode_text(text).float()

                image_features = image_features / \
                image_features.norm(dim=1, keepdim=True)
                text_features = text_features / \
                text_features.norm(dim=1, keepdim=True)

                logit_scale = model.model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                ground_truth = torch.arange(
                len(image), dtype=torch.long, device=device)

                cla_loss = (loss_img(logits_per_image, ground_truth) +
                    loss_txt(logits_per_text, ground_truth)) / 2

                train_loss_clf.update(cla_loss.item())
                optimizer.zero_grad()
                cla_loss.backward()
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
        print("phase 1 cla loss: ", train_loss_clf.avg)
        set_grad2(model.model)
        # for _ in tqdm(range(0, len(data_loader))):
        for batch in tqdm(data_loader):
            # image, text, label = next(source_data2)  # .next()
            image, text, label = batch
            if len(text) > 1:
                image = image.to(device)
                text = text.to(device)
                image_features = model.model.encode_image(image).float()
                text_features = model.model.encode_text(text).float()

                image_features = image_features / \
                                 image_features.norm(dim=1, keepdim=True)
                text_features = text_features / \
                                text_features.norm(dim=1, keepdim=True)

                logit_scale = model.model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                ground_truth = torch.arange(
                    len(image), dtype=torch.long, device=device)

                cla_loss = (loss_img(logits_per_image, ground_truth) +
                            loss_txt(logits_per_text, ground_truth)) / 2

                train_loss_clf.update(cla_loss.item())
                optimizer.zero_grad()
                cla_loss.backward()
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
        print("phase 2 cla loss: ", train_loss_clf.avg)

    if args.method == 'pfedclip':
        loss_mmd = MMDLoss()
        loss_mse = nn.MSELoss()
        loss_kl = nn.KLDivLoss(log_target=False, reduction='batchmean')
        texts = model.labels
        text_features_all = clu.get_text_features_list(texts, model.model).float()
        text_features_all /= text_features_all.norm(dim=-1, keepdim=True)
        # for _ in (range(0, args.n_iter)):
        for _ in tqdm(range(0, len(data_loader))):
            image, text, label = next(source_data)  # .next()
            if len(text) > 1:
                image = image.to(device)
                text = text.to(device)
                label = label.to(device)
                image_features = model.model.encode_image(image).float()
                text_features = model.model.encode_text(text).float()
                image_features_att = model.fea_attn(image_features)
                image_features = torch.mul(image_features_att, image_features)
                image_features = image_features / \
                image_features.norm(dim=1, keepdim=True)
                text_features = text_features / \
                text_features.norm(dim=1, keepdim=True)


                logit_scale = model.model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                ground_truth = torch.arange(
                len(image), dtype=torch.long, device=device)


                cla_loss = (loss_img(logits_per_image, ground_truth) +
                    loss_txt(logits_per_text, ground_truth))/ 2


                outputs = mlp(image_features) # MLP loss
                mlp_loss = loss_cla(outputs, label)
                train_loss_transfer.update(mlp_loss.item())


                # Sim loss
                similarity = (100.0 * image_features @ text_features_all.T).softmax(dim=-1) # Logits based on similarity
                # loss_sim = loss_mse(outputs, similarity) # Sim loss

                # print(F.softmax(outputs, dim=0), F.softmax(similarity, dim=0))
                # loss_sim = loss_kl(F.softmax(outputs, dim=0), similarity)
                # print(loss_kl(F.log_softmax(similarity,dim=0), F.softmax(outputs, dim=0)))
                loss_sim = (loss_kl(F.log_softmax(outputs, dim=-1), F.softmax(similarity,dim=-1)) +
                            loss_kl(F.log_softmax(similarity,dim=-1), F.softmax(outputs, dim=-1)))/2
                # loss_sim = loss_mmd(F.softmax(outputs, dim=0), F.softmax(similarity, dim=0))

                # loss_sim = (loss_mi(F.softmax(outputs,dim=0))+loss_mi(F.softmax(similarity,dim=0))) / 2
                # loss_mse = (loss_sim2(outputs, similarity) + loss_sim2(similarity, outputs)) / 2

                # total loss
                loss = cla_loss + mlp_loss + 0.01 * loss_sim
                # loss = cla_loss + mlp_loss

                prox_loss_clf.update(0.01 * loss_sim.item())
                train_loss_clf.update(cla_loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print("attn loss: ", train_loss_clf.avg, 'mlp loss:', train_loss_transfer.avg, 'sim loss:', prox_loss_clf.avg)

    if args.method == 'promptFL':
        tokenized_prompts = mlp.tokenized_prompts #
        text_encoder = TextEncoder(clip_model=model.model).to('cuda') # freezed
        for _ in tqdm(range(0, len(data_loader))):
            image, text, label = next(source_data)  # .next()
            if len(text) > 1:
                image = image.to(device)
                label = label.to(device)
                image_features = model.model.encode_image(image).float()
                prompts = mlp() #prompt learner
                text_features = text_encoder(prompts, tokenized_prompts).float()

                image_features = image_features / \
                image_features.norm(dim=1, keepdim=True)
                text_features = text_features / \
                text_features.norm(dim=1, keepdim=True)


                logit_scale = model.model.logit_scale.exp()
                logits = logit_scale * image_features @ text_features.t()

                cla_loss = F.cross_entropy(logits, label)


                train_loss_clf.update(cla_loss.item())
                optimizer.zero_grad()
                cla_loss.backward()
                optimizer.step()
        print("cla loss: ", train_loss_clf.avg)

    # if args.method == 'FAA-CLIP':
    #     # for _ in tqdm(range(0, len(data_loader))):
    #     for _ in tqdm(range(0, args.n_iter)):
    #         image, text, label = next(source_data)  # .next()
    #         image_t, _, _ = next(target_data)  # .next()
    #         if len(text) > 1:
    #             image = image.to(device)
    #             text = text.to(device)
    #             image_t = image_t.to(device)
    #             image_features = model.model.encode_image(image).float()
    #             text_features = model.model.encode_text(text).float()
    #             image_features_att = model.fea_attn(image_features)
    #             image_features = torch.mul(image_features_att, image_features)
    #             test_features = model.model.encode_image(image_t).float()
    #             # with torch.no_grad():
    #             test_features_att = model.fea_attn(test_features)
    #             test_features = torch.mul(test_features_att, test_features)
    #
    #             image_features = image_features / \
    #                              image_features.norm(dim=1, keepdim=True)
    #             text_features = text_features / \
    #                             text_features.norm(dim=1, keepdim=True)
    #             test_features = test_features / \
    #                             test_features.norm(dim=1, keepdim=True)
    #
    #             loss_m = mmd_loss(image_features, test_features)
    #             logit_scale = model.model.logit_scale.exp()
    #             logits_per_image = logit_scale * image_features @ text_features.t()
    #             logits_per_text = logits_per_image.t()
    #
    #             ground_truth = torch.arange(
    #                 len(image), dtype=torch.long, device=device)
    #
    #             cla_loss = (loss_img(logits_per_image, ground_truth) +
    #                         loss_txt(logits_per_text, ground_truth)) / 2
    #             loss = cla_loss + 0.5 * loss_m
    #
    #             train_loss_clf.update(cla_loss.item())
    #             train_loss_transfer.update(0.5 * loss_m.item())
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #     print("cla loss: ", train_loss_clf.avg, 'trans loss:', train_loss_transfer.avg)
    #
    if args.method == 'FAA-CLIP-MLP':
        texts = model.labels
        text_features_all = clu.get_text_features_list(texts, model.model).float()
        text_features_all /= text_features_all.norm(dim=-1, keepdim=True)
        for _ in tqdm(range(0, len(data_loader))):
        # for _ in tqdm(range(0, args.n_iter)):
            image, text, label = next(source_data)  # .next()
            image_t, _, _ = next(target_data)  # .next()
            if len(text) > 1:
                image = image.to(device)
                label = label.to(device)
                text = text.to(device)
                image_t = image_t.to(device)
                image_features = model.model.encode_image(image).float()
                text_features = model.model.encode_text(text).float()
                image_features_att = model.fea_attn(image_features)
                image_features = torch.mul(image_features_att, image_features)
                test_features = model.model.encode_image(image_t).float()
                with torch.no_grad():
                    test_features_att = model.fea_attn(test_features)
                    test_features = torch.mul(test_features_att, test_features)

                image_features = image_features / \
                                 image_features.norm(dim=1, keepdim=True)
                text_features = text_features / \
                                text_features.norm(dim=1, keepdim=True)
                test_features = test_features / \
                                test_features.norm(dim=1, keepdim=True)


                logit_scale = model.model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                outputs = mlp(image_features)  # MLP loss
                logits_per_text = logits_per_image.t()
                ground_truth = torch.arange(
                    len(image), dtype=torch.long, device=device)

                cla_loss = (loss_img(logits_per_image, ground_truth) +
                            loss_txt(logits_per_text, ground_truth)) / 2
                mlp_loss = loss_cla(outputs, label)
                prox_loss_clf.update(mlp_loss.item())
                loss_m = mmd_loss(image_features, test_features)
                loss_lmmd = lmmd_loss(image_features, test_features, label, (100.0 * test_features @ text_features_all.t()).softmax(dim=-1))
                # cla_loss = F.cross_entropy(logits_per_image, label)

                loss = cla_loss + 0.5 * loss_m + mlp_loss + 0.1 * loss_lmmd
                # loss = cla_loss + 0.5 * loss_m + mlp_loss
                train_loss_clf.update(cla_loss.item())
                train_loss_transfer.update(0.5 * loss_m.item())
                train_loss_transfer2.update(0.1 * loss_lmmd.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print("cla loss: ", train_loss_clf.avg, 'adv loss:', train_loss_transfer.avg,'lmmd loss:', train_loss_transfer2.avg,
              'mlp loss:', prox_loss_clf.avg)

    if args.method == 'FAA-CLIP':
        for _ in tqdm(range(0, len(data_loader))):
        # for _ in tqdm(range(0, args.n_iter)):
            image, text, label = next(source_data)  # .next()
            image_t, _, _ = next(target_data)  # .next()
            if len(text) > 1:
                image = image.to(device)
                text = text.to(device)
                image_t = image_t.to(device)
                image_features = model.model.encode_image(image).float()
                text_features = model.model.encode_text(text).float()
                test_features = model.model.encode_image(image_t).float()

                merged_features = torch.cat((image_features, test_features), dim=0)
                merged_features_att = model.fea_attn(merged_features)
                merged_features = torch.mul(merged_features_att, merged_features)

                image_features, test_features = merged_features.chunk(2, dim=0)

                # image_features_att = model.fea_attn(image_features)
                # image_features = torch.mul(image_features_att, image_features)
                #
                # # with torch.no_grad():
                # test_features_att = model.fea_attn(test_features)
                # test_features = torch.mul(test_features_att, test_features)

                image_features = image_features / \
                                 image_features.norm(dim=1, keepdim=True)
                text_features = text_features / \
                                text_features.norm(dim=1, keepdim=True)

                test_features = test_features / \
                                test_features.norm(dim=1, keepdim=True)


                loss_m = mmd_loss(image_features, test_features)
                # loss_m += 10. * CORAL(image_features, test_features)
                logit_scale = model.model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                ground_truth = torch.arange(
                    len(image), dtype=torch.long, device=device)

                cla_loss = (loss_img(logits_per_image, ground_truth) +
                            loss_txt(logits_per_text, ground_truth)) / 2

                loss = cla_loss + 0.5 * loss_m

                # if args.step > 0:
                #     w_diff = torch.tensor(1e-10, device=device)
                #     for w, w_t in zip(server_model.fea_attn.parameters(), model.fea_attn.parameters()):
                #         w_diff += torch.pow(torch.norm(w - w_t), 2).float()  # model difference
                #         # print(w_diff)
                #     w_diff = torch.sqrt(w_diff)
                #     prox_loss_clf.update((1e-2 / 2. * w_diff).item())
                #     loss += 1e-1 / 2. * w_diff  # dif loss

                train_loss_clf.update(cla_loss.item())
                train_loss_transfer.update(0.5 * loss_m.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print("cla loss: ", train_loss_clf.avg, 'trans loss:', train_loss_transfer.avg)

    if args.method == 'CocoOpCLIP':
        tokenized_prompts = mlp.tokenized_prompts #
        text_encoder = TextEncoderCocoOp(clip_model=model.model).to('cuda') # freezed
        for _ in tqdm(range(0, len(data_loader))):
            image, text, label = next(source_data)  # .next()
            if len(text) > 1:
                image = image.to(device)
                label = label.to(device)
                image_features = model.model.encode_image(image)
                image_features = image_features / \
                image_features.norm(dim=1, keepdim=True)

                prompts = mlp(image_features)
                logit_scale = model.model.logit_scale.exp()
                logits = [ ]
                for pts_i, imf_i in zip(prompts, image_features):
                    text_features = text_encoder(pts_i, tokenized_prompts)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    l_i = logit_scale * imf_i @ text_features.t()
                    logits.append(l_i)

                logits = torch.stack(logits)
                cla_loss = F.cross_entropy(logits, label)


                train_loss_clf.update(cla_loss.item())
                optimizer.zero_grad()
                cla_loss.backward()
                optimizer.step()
        print("cla (cocoop) loss: ", train_loss_clf.avg)

    if args.method == 'facmic':
        texts = model.labels
        t_features = clu.get_text_features_list(texts, model.model).float()  # update each round based on the model
        for _ in tqdm(range(0, len(data_loader))):
            image, text, label = next(source_data)
            image_t, _, _ = next(target_data)
            if len(text) > 1:
                image = image.to(device)
                text = text.to(device)
                image_t = image_t.to(device)
                image_features = model.model.encode_image(image).float()
                test_features = model.model.encode_image(image_t).float()
                text_features = model.model.encode_text(text).float()
                image_features_att = model.fea_attn(image_features)
                image_features = torch.mul(image_features_att, image_features)
                image_features_att_t = model.fea_attn(test_features)
                test_features = torch.mul(image_features_att_t, image_features)
                # test_features = torch.mul(i_attn, test_features)
                similarity = (100.0 * test_features @ t_features.T).softmax(dim=-1) # Logits based on similarity

                image_features = image_features / \
                image_features.norm(dim=1, keepdim=True)
                text_features = text_features / \
                text_features.norm(dim=1, keepdim=True)
                test_features = test_features / \
                test_features.norm(dim=1, keepdim=True)

                loss_m = mmd_loss(image_features, test_features, label, similarity)
                logit_scale = model.model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                ground_truth = torch.arange(
                len(image), dtype=torch.long, device=device)

                cla_loss = (loss_img(logits_per_image, ground_truth) +
                    loss_txt(logits_per_text, ground_truth))/2

                loss = cla_loss + 0.5 * loss_m

                train_loss_clf.update(cla_loss.item())
                train_loss_transfer.update(0.5 * loss_m.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print("cla loss: ", train_loss_clf.avg, 'trans loss:', train_loss_transfer.avg)

    # if args.method == 'facmic':
    #     texts = model.labels
    #     t_features = clu.get_text_features_list(texts, model.model).float()  # update each round based on the model
    #     for _ in tqdm(range(0, len(data_loader))):
    #         image, text, label = next(source_data)
    #         image_t, _, _ = next(target_data)
    #         if len(text) > 1:
    #             image = image.to(device)
    #             text = text.to(device)
    #             image_t = image_t.to(device)
    #             image_features = model.model.encode_image(image).float()
    #             test_features = model.model.encode_image(image_t).float()
    #             text_features = model.model.encode_text(text).float()
    #             image_features_att = model.fea_attn(image_features)
    #             image_features = torch.mul(image_features_att, image_features)
    #
    #             i_features = clu.get_image_features(
    #                 image_t, model.model, model.preprocess).float()
    #             i_attn = model.fea_attn(i_features)
    #             i_features = torch.mul(i_attn, i_features)
    #             # test_features = torch.mul(i_attn, test_features)
    #             similarity = clu.get_similarity(i_features, t_features)
    #
    #             image_features = image_features / \
    #                              image_features.norm(dim=1, keepdim=True)
    #             text_features = text_features / \
    #                             text_features.norm(dim=1, keepdim=True)
    #             test_features = test_features / \
    #                             test_features.norm(dim=1, keepdim=True)
    #
    #             loss_m = mmd_loss(image_features, test_features, label, similarity)
    #             logit_scale = model.model.logit_scale.exp()
    #             logits_per_image = logit_scale * image_features @ text_features.t()
    #             logits_per_text = logits_per_image.t()
    #
    #             ground_truth = torch.arange(
    #                 len(image), dtype=torch.long, device=device)
    #
    #             cla_loss = (loss_img(logits_per_image, ground_truth) +
    #                         loss_txt(logits_per_text, ground_truth)) / 2
    #
    #             loss = cla_loss + 0.5 * loss_m
    #
    #             train_loss_clf.update(cla_loss.item())
    #             train_loss_transfer.update(0.5 * loss_m.item())
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #     print("cla loss: ", train_loss_clf.avg, 'trans loss:', train_loss_transfer.avg)


def initlp(args, model, data_loader, device, mlp):
    totrain(model)
    source_data = iter(data_loader)
    text_weights = clip_classifier(model.labels, model.model)
    features, labels = [ ], [ ]
    with torch.no_grad():
        for _ in tqdm(range(0, len(data_loader))):
            image, text, label = next(source_data)
            images = image.to(device)
            text = text.to(device)
            label = label.to(device)
            image_features = model.model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features)
            labels.append(label)
    features, labels = torch.cat(features), torch.cat(labels)
    centroids = compute_centroids(features.unsqueeze(0), labels.unsqueeze(0),
                                  len(model.labels))  # [batch, num_class, d]
    mlp.weight.data = centroids[ 0 ]

    print('Running LP++')
    # lr_w
    lr_temp = calculate_lr_w(features)

    # init_alpha
    final_init_alpha_mean = calculate_init_alpha(features, labels,
                                                 text_weights,
                                                 num_classes=len(model.labels))  # feature shape : [batch, 1024]
    # print(text_weights)
    alpha_vec = Variable(
        final_init_alpha_mean * torch.ones(1, len(model.labels)).to(model.model.dtype).cuda(),
        # [1, num_class]
        requires_grad=True)

    # lr_alpha
    lr_alpha = calculate_lr_alpha(features, text_weights)
    print("final_init_alpha_mean: ", final_init_alpha_mean)
    print('Calculated lr_temp, lr_alpha: ', lr_temp, lr_alpha)

    return alpha_vec, lr_temp, lr_alpha, features, text_weights, labels

def trainlp(args, model, data_loader, optimizer, device, mlp, features, text_weights, alpha_vec,lr_temp, labels, lr_alpha):
    totrain(model)
    totrainMLP(mlp)
    source_data = iter(data_loader)
    if args.method == 'lp':
        # optimizer = torch.optim.SGD(mlp.parameters(), lr_temp, momentum=0.9)
        print('\nStart Training procedure!')
        mlp.train()
        vision_logits = mlp(features)
        text_logits = features @ text_weights
        logits = vision_logits + torch.ones(features.shape[ 0 ], 1).to(
            model.model.dtype).cuda() @ alpha_vec * text_logits  # inference
        # print(logits.shape)  # logits_image + torch.ones (logits_text)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mlp.eval()
        vision_logits_val = mlp(features)

        # print(vision_logits)
        # print(vision_logits_val)

        text_logits_val = features.detach() @ text_weights
        logits_val = vision_logits_val + torch.ones(features.shape[ 0 ], 1).to(
            model.model.dtype).cuda() @ alpha_vec * text_logits_val
        acc = np.mean(logits.argmax(dim=1).cpu().numpy() == labels.cpu().numpy()) * 100.0
        print('The accuracy for train data is ', acc)
        acc_val = np.mean(logits_val.argmax(dim=1).cpu().numpy() == labels.cpu().numpy()) * 100.0
        print('The accuracy for val data is ', acc_val)

        args.current_epoch += 1
        # # update for alpha
        if (args.current_epoch + 1) % 10 == 0:
            alpha_vec.data -= lr_alpha * alpha_vec.grad.data