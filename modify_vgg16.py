import torch
from os import path

if not path.exists("data/vgg16_places365.pth"):
    print("Please download vgg16_places365.pth from https://drive.google.com/drive/folders/1OGKfoIehp2MiJL2Iq_8VMTy76L6waGC8 and place it into data/ and then try running this scrpit again.")
    exit()

vgg19i = torch.load("data/vgg19_enc.model")
vgg16p = torch.load("data/vgg16_places365.pth")

convlist = {'0': '1_1', '2':'1_2', 
            '5': '2_1', '7':'2_2',
            '10': '3_1', '12': '3_2', '14': '3_3',
            '17': '4_1', '19': '4_2', '21': '4_3',
            '24': '5_1', '26': '5_2', '28': '5_3'}


c2 = {}
for k in vgg19i.keys():
   if not "features" in k:
      continue
   if '0_0' in k:
      continue
   k1 = k.replace("features", "")   
   k1 = k1[1:].replace("weight", "").replace("bias", "")[:-1].split(".")
   c2[k1[0]] = k1[1]

print(c2)

converted = {}
for l in vgg16p.keys():
    if "classifier" in l:
        continue
    lnum = l.replace('features', "").replace("weight", "").replace("bias","").replace(".", "")
    lname = convlist[lnum]
    l2 = l.replace(lnum, lname+"."+c2[lname])
    print(l, l2)
    converted[l2] = vgg16p[l]

converted['features.0_0.mean'] = vgg19i['features.0_0.mean']
converted['features.0_0.std'] = vgg19i['features.0_0.std']

torch.save(converted, "data/vgg16places_enc.model")
