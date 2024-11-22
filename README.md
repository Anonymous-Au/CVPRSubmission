Codes for pFedCLIP and other baselines.


## Requirements

We suggest you to use the following packages:

clip==1.0

loraclip==0.1.0

numpy==1.22.0

opencv-python==4.9.0.80

openpyxl==3.1.2

Pillow==9.3.0

scikit-image==0.21.0

scikit-learn==1.1.3

scipy==1.10.0

tqdm==4.66.1

torch==1.13.1+cu117

torchvision=0.14.1+cu117

## How to use

```sh
python DN-1.py
```

### DN-1.py (example)

Run DN-1.py to simulate pFedCLIP algorithms. For convenience, we split all methodologies into 11 files, each file can be used for only one Method.

parser.add_argument('--test_envs', type=int, nargs='+', default=[3]) # default here is to set the global testing set, suppose there are 4 Clients, 3 here means it will treat Client 4 as the global while the rest as training clients.


### nets/models.py

models.py is the model backbone file.

### utils/clip_util.py

clip_util.py is the utils that CLIP will use. For FedAVG, MOON and FedProx, you have to do the following steps:
```sh
def freeze_param(model):
    for name, param in model.named_parameters():
        param.requires_grad = True
```
For pFedCLIP, CLIP-FC and FedCLIP, you have to set it as False.

### utils/prepare_data_dg_clip.py

prepare_data_dg_clip.py is the dataloader CLIP will use. You can define the percentage for training, val and test via:
```sh
l1, l2, l3 = int(l*0.6), int(l*0.2), int(l*0.2)
```

### utils/training.py

training.py is the training function for all methods.

## dataset

Here is a case study about the structure of our dataset as follows:

```sh
./data/ModernOffice31/
    a/
      bike/
        frame_0001.jpg
            ...
      back_pack/
      bottle/
      .../
    d/
    s/
    w/
```