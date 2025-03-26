class DeepGP_2(gpytorch.models.deep_gps.DeepGP):
    def __init__(self, train_x_shape, train_y, num_hidden_dgp_dims = 4, inducing_num = 500):
        num_tasks = train_y.size(-1)

        hidden_layer_1 = DGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_hidden_dgp_dims,
            num_inducing=inducing_num, 
            linear_mean=True
        )


        last_layer = DGPHiddenLayer(
            input_dims=hidden_layer_1.output_dims,
            output_dims = num_tasks,
            num_inducing=inducing_num, 
            linear_mean=False
        )

        super().__init__()

        self.hidden_layer_1 = hidden_layer_1
        self.last_layer = last_layer

        # We're going to use a ultitask likelihood instead of the standard GaussianLikelihood
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer_1(inputs)
        output = self.last_layer(hidden_rep1)
        return output
    
    def predict(self, test_x):
        # with torch.no_grad():
        preds = self.likelihood(self(test_x)).to_data_independent_dist()

        return preds.mean.mean(0).squeeze(), preds.variance.mean(0).squeeze()
    



    
def evaluate_full_dataset_loss_dgp(model, x_data, y_data, mll, device='cuda', batch_size=1024):

    model.eval()
    total_loss = 0.0
    dataset = TensorDataset(x_data, y_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            loss = -mll(output, y_batch)
            total_loss += loss.item() * x_batch.size(0)

    avg_loss = total_loss / len(dataset)
    model.train()
    return avg_loss


def train_DGP_2_minibatch(
    full_train_x, 
    full_train_y, 
    num_hidden_dgp_dims=4, 
    inducing_num=500, 
    num_iterations=2000, 
    patience=50, 
    device='cuda',
    batch_size=32,
    eval_every=100,
    eval_batch_size=1024,
    lr=0.1
):
    """
    训练Deep GP (2层) 的完整流程，支持小批量训练、早停、全数据集评估和学习率调度。
    
    参数说明：
    - full_train_x, full_train_y: 训练数据
    - num_hidden_dgp_dims: Deep GP中隐藏层维度
    - inducing_num: 每层诱导点数量
    - num_iterations: 总迭代次数上限
    - patience: 早停耐心值 (评估损失连续多少次不下降就停止)
    - device: 'cpu' 或 'cuda'
    - batch_size: 小批量训练时的批量大小
    - eval_every: 每隔多少次迭代进行一次全数据评估
    - eval_batch_size: 进行全数据评估时的批量大小
    - lr: 初始学习率
    """

    full_train_x = full_train_x.to(device)
    full_train_y = full_train_y.to(device)


    model = GP_models.DeepGP_2(
        full_train_x.shape, 
        full_train_y, 
        num_hidden_dgp_dims, 
        inducing_num
    ).to(device)

    model.train()


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.DeepApproximateMLL(
        gpytorch.mlls.VariationalELBO(
            model.likelihood,
            model,
            num_data=full_train_y.size(0)
        )
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=25
    )


    best_loss = float('inf')
    best_state = model.state_dict()
    counter = 0


    data_loader = DataLoader(
        TensorDataset(full_train_x, full_train_y),
        batch_size=batch_size,
        shuffle=True
    )
    minibatch_iter = itertools.cycle(data_loader)


    with tqdm.tqdm(total=num_iterations, desc="Training DGP_2") as pbar:
        for step in range(num_iterations):
            x_batch, y_batch = next(minibatch_iter)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()

            if (step + 1) % eval_every == 0 or (step == num_iterations - 1):
                current_loss = evaluate_full_dataset_loss_dgp(
                    model=model,
                    x_data=full_train_x,
                    y_data=full_train_y,
                    mll=mll,
                    device=device,
                    batch_size=eval_batch_size
                )
                pbar.set_postfix(full_loss=current_loss)
                
                scheduler.step(current_loss)

                if current_loss < best_loss:
                    best_loss = current_loss
                    best_state = model.state_dict()
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        model.load_state_dict(best_state)
                        pbar.update(num_iterations - step - 1)
                        break

            pbar.update(1)

    return model
