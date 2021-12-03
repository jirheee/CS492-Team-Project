import torch

import json

from nn_architecture import PolicyValueNet


trained_path = r"D:\Dropbox\Workspace\03 Python\11 AI_Battle\CS492-Team-Project\Server\src\ml\models\1aaa41fa-526e-47c6-916c-07906127df3c\curr.model"

trained = torch.load(trained_path)
print("trained keys\n", trained.keys())

model_config = f"../Server/src/ml/models/1aaa41fa-526e-47c6-916c-07906127df3c/model.json"
with open(model_config, encoding='utf-8') as f:
    model_config = json.loads(f.read())

board_width = model_config["board"]["board_width"]
board_height = model_config["board"]["board_height"]
name = model_config["name"]

policy_value_net = PolicyValueNet(board_width, board_height, model_config["nn_type"], model_config["layers"])

new_format = policy_value_net.get_policy_param()
print("new keys\n", new_format.keys())
print()
print()
print(trained)
print(new_format)
print()
print()
print([trained[ii].shape for ii in trained.keys()])
print([new_format[ii].shape for ii in new_format.keys()])
print()
print()
num = len(trained.keys())

for ii in range(num):
    new_key = list(new_format.keys())[ii]
    old_key = list(trained.keys())[ii]
    new_format[new_key]=trained[old_key]

print()
print()

print([trained[ii].shape for ii in trained.keys()])
print([new_format[ii].shape for ii in new_format.keys()])

print()
print()

print(trained)
print(new_format)

torch.save(new_format,trained_path)