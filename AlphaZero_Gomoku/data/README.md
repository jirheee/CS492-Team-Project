# JSON 파일 format

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
- n_layers: `int`, Number of layers
- layer_0, ..., layer_n: Information for layers
    if layer_name == 'Conv':
        channels, kernel_size, stride, padding information required(type: `int`)
- activ_func: [`ReLU`, `Sigmoid`, `Tanh`, `LeakyReLU`], Activation Function 