import torch
import gpytorch
import pandas as pd
import numpy as np
import tqdm as tqdm
from torch.utils.data import TensorDataset, DataLoader

import GP_functions.Loss_function as Loss_function
import GP_functions.bound as bound
import GP_functions.Estimation as Estimation
import GP_functions.Training as Training
import GP_functions.Prediction as Prediction
import GP_functions.GP_models as GP_models
import GP_functions.Tools as Tools
import GP_functions.FeatureE as FeatureE

import itertools
from itertools import product

from sklearn.metrics import mean_squared_error


def evaluate_full_dataset_loss_dgp(model, x_data, y_data, mll, device='cuda', batch_size=1024):

    model.eval()
    total_loss = 0.0
    dataset = TensorDataset(x_data, y_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model.predict(x_batch)[0]
            loss = mll(output, y_batch)
            total_loss += loss.item() * x_batch.size(0)

    avg_loss = total_loss / len(dataset)
    model.train()
    return avg_loss


X_train = pd.read_csv('Data/X_train.csv', header=None, delimiter=',').values
X_test = pd.read_csv('Data/X_test.csv', header=None, delimiter=',').values


Y_train_21 = pd.read_csv('Data/Y_train_std_21.csv', header=None, delimiter=',').values
Y_test_21 = pd.read_csv('Data/Y_test_std_21.csv', header=None, delimiter=',').values

Y_train_std = pd.read_csv('Data/Y_train_std.csv', header=None, delimiter=',').values
Y_test_std = pd.read_csv('Data/Y_test_std.csv', header=None, delimiter=',').values

train_x = torch.tensor(X_train, dtype=torch.float32)
test_x = torch.tensor(X_test, dtype=torch.float32)


train_y_21 = torch.tensor(Y_train_21, dtype=torch.float32)
test_y_21 = torch.tensor(Y_test_21, dtype=torch.float32)

train_y = torch.tensor(Y_train_std, dtype=torch.float32)
test_y = torch.tensor(Y_test_std, dtype=torch.float32)




Device = 'cuda'


def evaluate_deepgp(
    model_cls, train_x, train_y, test_x, test_y,
    hidden_dim, num_inducing, covar_types,
    # 训练超参
    num_iterations, patience, batch_size, eval_every, eval_batch_size, lr, device
):
    """
    使用 minibatch 训练函数训练 DGP 模型，并在测试集上评价 MSE。
    返回训练好的模型和对应的 MSE。
    """
    # 训练模型
    model = Training.train_dgp_minibatch(
        DGP_model=model_cls,
        train_x=train_x,
        train_y=train_y,
        hidden_dim=hidden_dim,
        inducing_num=num_inducing,
        covar_types=covar_types,
        num_iterations=num_iterations,
        patience=patience,
        batch_size=batch_size,
        eval_every=eval_every,
        eval_batch_size=eval_batch_size,
        lr=lr,
        device=device
    )

    # 测试
    model.to(device).eval()
    test_x, test_y = test_x.to(device), test_y.to(device)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = model.predict(test_x)
        mean_pred = preds[0].cpu().numpy()
    mse = mean_squared_error(test_y.cpu().numpy(), mean_pred)
    return model, mse


def main():
    # 假设外部加载了以下变量：train_x, train_y_21, test_x, test_y_21
    # 例如：
    # train_x = torch.load('train_x.pt')
    # train_y_21 = torch.load('train_y_21.pt')
    # ...

    results = []
    failures = []
    best_mse = float('inf')
    best_model = None
    best_params = None

    model_setups = [
        ("DeepGP2", GP_models.DeepGP2, {"hidden_dim":10}),
        ("DeepGP3", GP_models.DeepGP3, {"hidden_dims":[10,10]}),
        ("DeepGP4", GP_models.DeepGP4, {"hidden_dims":[10,10,10]}),
    ]

    inducing_options = [100, 300, 500]
    covar_types_list = ["RBF", "RQ", "Matern5/2"]

    # 训练相关超参
    num_iterations = 5000
    patience = 100
    batch_size = 256
    eval_every = 200
    eval_batch_size = 1024
    lr = 0.05
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name, model_cls, extra_kwargs in model_setups:
        # 确定层数
        n_layers = {"DeepGP2":2, "DeepGP3":3, "DeepGP4":4}[name]
        for num_inducing in inducing_options:
            for covar_combo in product(covar_types_list, repeat=n_layers):
                covar_types = list(covar_combo)
                try:
                    model, mse = evaluate_deepgp(
                        model_cls,
                        train_x,
                        train_y_21,
                        test_x,
                        test_y_21,
                        # 模型结构参数
                        extra_kwargs.get("hidden_dim", extra_kwargs.get("hidden_dims")),
                        num_inducing,
                        covar_types,
                        # 训练超参
                        num_iterations,
                        patience,
                        batch_size,
                        eval_every,
                        eval_batch_size,
                        lr,
                        device
                    )
                    results.append({
                        "model": name,
                        "num_inducing": num_inducing,
                        "covar_types": covar_types,
                        "mse": mse,
                    })
                    print(f"{name} | inducing={num_inducing} | covar={covar_types} | MSE={mse:.4f}")

                    # 更新最佳模型
                    if mse < best_mse:
                        best_mse = mse
                        best_model = model
                        best_params = {
                            'model': name,
                            'num_hidden_dgp_dims': extra_kwargs.get('hidden_dim', extra_kwargs.get('hidden_dims')),
                            'inducing_num': num_inducing,
                            'covar_types': covar_types,
                        }
                except Exception as e:
                    failures.append({
                        "model": name,
                        "num_inducing": num_inducing,
                        "covar_types": covar_types,
                        "error": str(e),
                    })
                    print(f"FAILED: {name} | inducing={num_inducing} | covar={covar_types} | Error: {e}")

    # 输出最佳配置和前五
    top5 = sorted(results, key=lambda x: x["mse"])[:5]
    print("\nTop 5 configurations (lowest MSE):")
    for r in top5:
        print(r)
    print(f"\nBest overall: {best_params}, MSE={best_mse:.4f}")

    # 保存结果列表
    torch.save(results, "deepgp_results.pt")
    torch.save(failures, "deepgp_failures.pt")

    # 保存最佳模型检查点
    checkpoint = {
        'model_state_dict': best_model.state_dict(),
        'model_params': {
            'num_hidden_dgp_dims': best_params['num_hidden_dgp_dims'],
            'inducing_num': best_params['inducing_num'],
            'covar_types': best_params['covar_types'],
            'input_dim': train_x.size(1),
            'output_dim': train_y_21.size(1)
        }
    }
    save_path = 'best_dgp_checkpoint_21.pth'
    torch.save(checkpoint, save_path)
    print(f"Saved best model checkpoint to {save_path}")


if __name__ == "__main__":
    main()


# nohup python DGPSelection.py > DGPSelectionout.log 2>&1 &
