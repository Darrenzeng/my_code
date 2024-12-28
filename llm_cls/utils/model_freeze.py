# 冻结参数

def freeze_parms(model):
    for name, param in model.named_parameters():
        if name.startswith("classifier"):
            param.requires_grad = True
        else:
            param.requires_grad = False
            
def parms_is_freeze_print(model):
    # 打印参数的冻结情况
    frozen_params = []
    trainable_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            frozen_params.append(name)
        else:
            trainable_params.append(name)

    print(f"共 {len(list(model.parameters()))} 个参数，其中：")
    print(f"  - 冻结了 {len(frozen_params)} 个参数：")
    # for fp in frozen_params:
    #     print(f"      {fp}")

    print(f"  - 可训练的 {len(trainable_params)} 个参数：")
    for tp in trainable_params:
        print(f"      {tp}")