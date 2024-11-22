import torch.nn as nn
import clip
from utils.clip_util import freeze_param, get_image_features
import torch
import torch.nn.functional as F
import loraclip



class ClipModelat(nn.Module):

    CLIP_MODELS = [
        'RN50',
        'RN101',
        'RN50x4',
        'RN50x16',
        'RN50x64',
        'ViT-B/32',
        'ViT-B/16',
        'ViT-L/14',
        'ViT-L/14@336px'
    ]

    def __init__(self, model_name='Vit-B/32', device='cuda', logger=None, attention=True, freezepy=True):
        super(ClipModelat, self).__init__()
        self.logger = logger
        if type(model_name) is int:
            model_name = self.index_to_model(model_name)
        self.model, self.preprocess = clip.load(
            model_name, device=device)
        self.model.eval()
        self.model.to(device)
        self.model_name = model_name
        self.attention = attention
        self.freezepy = freezepy
        self.device = device

    def initdgatal(self, dataloader):

        for batch in dataloader:
            with torch.no_grad():
                image, _, label = batch
                image = image.to(self.device)
                label = label.to(self.device)
                image_features = self.model.encode_image(image)
                break
        if self.freezepy:
            freeze_param(self.model)

        if self.attention:
            # pass
            # self.fea_attn = nn.Sequential(nn.Linear(image_features.shape[1], image_features.shape[1]), nn.BatchNorm1d(image_features.shape[1]),
            #                nn.ReLU6(),  nn.Linear(image_features.shape[1], image_features.shape[1]),  nn.Softmax(dim=1)).to(self.device)
            # self.fea_attn = Global_attention_block(image_features)
            # For FedClip
            self.fea_attn = nn.Sequential(nn.Linear(image_features.shape[1], image_features.shape[1]),
            nn.Tanh(), nn.Linear(image_features.shape[1], image_features.shape[1]), nn.Softmax(dim=1)).to(self.device)


    def index_to_model(self, index):
        return self.CLIP_MODELS[index]

    @staticmethod
    def get_model_name_by_index(index):
        name = ClipModelat.CLIP_MODELS[index]
        name = name.replace('/', '_')
        return name

    def setselflabel(self, labels):
        # print(labels)
        self.labels = labels


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        # self.linear = nn.Linear(input_size, hidden_size)
        # self.bn = nn.BatchNorm1d(hidden_size)
        # self.relu = nn.ReLU()
        # self.linear2 = nn.Linear(hidden_size, num_classes)
        # self.bn2 = nn.BatchNorm1d(num_classes)
        # self.relu2 = nn.ReLU()
        # self.linear3 = nn.Linear(num_classes, num_classes)
        # self.bn3 = nn.BatchNorm1d(num_classes)
        # self.relu3 = nn.ReLU()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU6(),
            # nn.Dropout(p=0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU6(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU6(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            # nn.ReLU6(),

            # nn.Linear(hidden_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            # nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            # nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
            # nn.BatchNorm1d(num_classes)
        )

    def forward(self, x):
        # x1 = self.linear(x)
        # x1 = self.bn(x1)
        # x1 = self.relu(x1)
        # x2 = self.linear(x)
        # x1 = x1 + x2
        # x1 = self.linear2(x1)
        # x1 = self.bn2(x1)
        # x1 = self.relu2(x1)
        # x1 = self.linear3(x1)
        # x1 = self.bn3(x1)
        # x1 = self.relu3(x1)
        return self.layers(x)


class ClipModelatLoRA(nn.Module):

    CLIP_MODELS = [
        'RN50',
        'RN101',
        'RN50x4',
        'RN50x16',
        'RN50x64',
        'ViT-B/32',
        'ViT-B/16',
        'ViT-L/14',
        'ViT-L/14@336px'
    ]

    def __init__(self, model_name='Vit-B/32', device='cuda', logger=None, rank=4, freezepy=True):
        super(ClipModelatLoRA, self).__init__()
        self.logger = logger
        if type(model_name) is int:
            model_name = self.index_to_model(model_name)
        self.model, self.preprocess = loraclip.load(
            model_name, device=device, r=rank, lora_mode="vision+text")
        self.model.eval()
        self.model.to(device)
        self.model_name = model_name
        self.attention = True
        self.freezepy = freezepy
        self.device = device

    def initdgatal(self, dataloader):

        for batch in dataloader:
            image, _, label = batch
            image = image.to(self.device)
            label = label.to(self.device)
            image_features = get_image_features(
                image, self.model, self.preprocess)
            break
        if self.freezepy:
            freeze_param(self.model)

        if self.attention:
            # pass
            self.fea_attn = nn.Sequential(nn.Linear(image_features.shape[1], image_features.shape[1]), nn.BatchNorm1d(image_features.shape[1]),
                           nn.LeakyReLU(),  nn.Linear(image_features.shape[1], image_features.shape[1]),  nn.Softmax(dim=1)).to(self.device)

    def index_to_model(self, index):
        return self.CLIP_MODELS[index]

    @staticmethod
    def get_model_name_by_index(index):
        name = ClipModelat.CLIP_MODELS[index]
        name = name.replace('/', '_')
        return name

    def setselflabel(self, labels):
        self.labels = labels