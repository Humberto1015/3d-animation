import argparse
import time
from trainers import DAETrainer

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--learning_rate', type = float, default = 0.001)
parser.add_argument('--epochs', type = int, default = 100)
parser.add_argument('--workers', type = int, default = 8)
parser.add_argument('--save_path', type = str, default = './trained_weights/DenoisingAutoEncoder/')
opt = parser.parse_args()

if __name__ == '__main__':

    trainer = DAETrainer(opt)
    trainer.build_dataset_train()
    trainer.build_dataset_valid()
    trainer.build_network()
    trainer.build_optimizer()
    trainer.build_losses()

    start_time = time.time()
    for epoch in range(opt.epochs):
        trainer.train_epoch()
        trainer.valid_epoch()
        trainer.save_network()
        trainer.increment_epoch()
    end_time = time.time()

    print ('It cost %f seconds to train the DAE.' % (end_time - start_time))