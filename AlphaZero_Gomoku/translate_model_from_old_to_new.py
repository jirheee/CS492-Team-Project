import torch

import json

from nn_architecture import PolicyValueNet
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-y","--force_yes",action="store_true",default = False)
args = parser.parse_args()

trained_path = r"D:\Dropbox\Workspace\03 Python\11 AI_Battle\CS492-Team-Project\Server\src\ml\models\1aaa41fa-526e-47c6-916c-07906127df3c\curr.model"

trained = torch.load(trained_path)

model_config = f"../Server/src/ml/models/1aaa41fa-526e-47c6-916c-07906127df3c/model.json"
with open(model_config, encoding='utf-8') as f:
    model_config = json.loads(f.read())

board_width = model_config["board"]["board_width"]
board_height = model_config["board"]["board_height"]
name = model_config["name"]

policy_value_net = PolicyValueNet(board_width, board_height, model_config["nn_type"], model_config["layers"])

new_format = policy_value_net.get_policy_param()


print("trained keys\n", trained.keys())
print("new keys\n", new_format.keys(),end="\n\n\n")


print("------------------------------trained--------------------------------")
for ii in [(ii,np.array2string(jj.numpy(),precision=2, separator=",",threshold=5)) for (ii,jj) in trained.items()]:
    print(ii[0],ii[1])
print("------------------------------trained--------------------------------\n")
print("------------------------------new_format--------------------------------")
for ii in [(ii,np.array2string(jj.to('cpu').numpy(),precision=2, separator=",",threshold=5)) for (ii,jj) in new_format.items()]:
    print(ii[0],ii[1])
print("------------------------------new_format--------------------------------\n\n")


print([trained[ii].shape for ii in trained.keys()])
print([new_format[ii].shape for ii in new_format.keys()])
num = len(trained.keys())

for ii in range(num):
    new_key = list(new_format.keys())[ii]
    old_key = list(trained.keys())[ii]
    print(f"mapping {old_key:20s} from trained ===> to {new_key:20s} in new_format")
    new_format[new_key]=trained[old_key]

print();print();
print([trained[ii].shape for ii in trained.keys()])
print([new_format[ii].shape for ii in new_format.keys()])
print();print()

print("------------------------------trained--------------------------------")
for ii in [(ii,np.array2string(jj.numpy(),precision=2, separator=",",threshold=5)) for (ii,jj) in trained.items()]:
    print(ii[0],ii[1])
print("------------------------------trained--------------------------------\n")
print("------------------------------new_format--------------------------------")
for ii in [(ii,np.array2string(jj.to('cpu').numpy(),precision=2, separator=",",threshold=5)) for (ii,jj) in new_format.items()]:
    print(ii[0],ii[1])
print("------------------------------new_format--------------------------------\n")

if args.force_yes or input("Save and overwrite the model with the new_format?").lower() in ["y","yes"]:
    torch.save(new_format,trained_path)
    print("saved")
else:
    print("exiting without saving")
