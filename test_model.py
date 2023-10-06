import torch
import pickle
ckpt = '/Users/wanghui/Desktop/6-work/D24H-fulltime/work/14-age_woman/fpage/ibug/face_parsing/resnet/weights/resnet50-fcn-14.torch'

# state_dict = torch.load(ckpt)
with open(ckpt, 'rb') as f:
    model_weights = pickle.load(f)
print(model_weights.keys())