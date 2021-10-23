# RL Environment and Algorithm Implementation

### 1. game.py
- 오목 환경 관련 코드

### 2. human_play.py
- 학습시킨 AI와 사람과 대결할 수 있도록 만들어짐
- `python human_play.py` 라는 명령어로 대결 가능

### 3. mcts_pure.py
- MCTS 알고리즘 Implementation

### 4. nn_architecture.py
- 각종 Deep Learning Network 구현
- 현재는 pure CNN만 구현되어 있음

### 5. rl_algorithm.py
- 각종 RL Algorithm 구현
- 현재는 DQN만 구현되어 있음

### 6. train.py
- json파일을 입력으로 받아 User가 원하는 RL Algorithm과 Deep Learning Network를 가지고 Agent을 구성하고 학습시킴
- json파일 구조는 고쳐나가면 될듯
- 학습이 끝난 뒤에는 model 저장
- `python train.py` 라는 명령어로 실햄 가능