# JSON 파일 format

## for training

### 1. Board - Board Information
- board_width: `int`, Board width
- board_height: `int`, Board height
- n_in_row: `int`, Minimum number of balls in row for winning, 0 <= n_in_row <= min(width, height)

### 2. Hyperparameters - HyperParameters for MCTS Training
- lr: `float`, Learning rate
- buffer_size: `int`, memory size
- batch_size: `int`, batch size for training
- epochs: `int`, epochs

### 3. NN_Information - Neural Network Architecture
- nn_type: [`CNN`, `GNN`]: type of Neural Architecture
- layers: Array of layers

    if layer_name == 'Conv':
        channels, kernel_size, stride, padding, bias information required(type: `int` except bias term: `str`)

    if layer_name == 'BatchNorm':
        no additional information required

    if layer_name == 'GCNConv': [논문 링크](https://arxiv.org/abs/1609.02907)
        channels, bias information required(channels: `int` bias: `str`)

    if layer_name == 'SGConv': [논문 링크](https://arxiv.org/abs/1902.07153)
        channels, bias information required(channels: `int` bias: `str`)
        
    if layer_name == 'GATConv': [논문_링크](https://arxiv.org/abs/1710.10903)
        channels, bias information required(channels: `int` bias: `str`) 
        
    if layer_name == 'GINConv': [논문_링크](https://arxiv.org/abs/1810.00826)
        channels, bias information required(channels: `int` bias: `str`)  
        
    if layer_name == 'SAGEConv': [논문_링크](https://arxiv.org/abs/1706.02216)
        channels, bias information required(channels: `int` bias: `str`) 
       
- activ_func: [`ReLU`, `Sigmoid`, `Tanh`, `LeakyReLU`], Activation Function 

-------

## for battle

### 1. Board - Board Information
- board_width: `int`, Board width
- board_height: `int`, Board height
- n_in_row: `int`, Minimum number of balls in row for winning, 0 <= n_in_row <= min(width, height)

### 2. Player1 - First Opponent
- model_path: path of the model
- information about nn_architecture

### 3. Player2 - Second Opponent
- model_path: path of the model
- information about nn_architecture
