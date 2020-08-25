# Network Training

## File Structure
```
├── datasets.py
├── models.py
├── trainers.py
└── train.py
```

## Run
```
./train.py
```

## Arguments
```
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, default = 'SMPL')
parser.add_argument('--alpha', type = float, default = 0.75)
parser.add_argument('--latent_dim', type = int, default = 128)
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--learning_rate', type = float, default = 0.001)
parser.add_argument('--epochs', type = int, default = 600)
parser.add_argument('--workers', type = int, default = 8)
parser.add_argument('--save_path', type = str, default = './trained_weights/AAE_SMPL/')
opt = parser.parse_args()
```

## Others
[Training Data](https://drive.google.com/drive/folders/1dXSBUanoWKJSymLJ00edq__qSgUKwVJf?usp=sharing)