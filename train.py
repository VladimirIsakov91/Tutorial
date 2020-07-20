import numpy
from sklearn.datasets import make_classification
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from torch.optim.lr_scheduler import StepLR
import os
from ignite.contrib.handlers.neptune_logger import *
from model import Data, MLP


X, y = make_classification(n_samples=1000,
                           n_features=20,
                           n_informative=10)

X = X.astype(numpy.float32)
y = y.astype(numpy.int64)

lr = 0.001
weight_decay = 0.0001
batch_size = 64
step_size = 10
gamma = 0.1
epochs = 40

# def objective(trial):

# lr = trial.suggest_categorical('lr', [0.01, 0.001, 0.0001])
logger = NeptuneLogger(
    api_token=os.getenv('NEPTUNE_API_TOKEN'),
project_name = "vladimir.isakov/sandbox",
               name = 'Run',
                      params = {'batch_size': batch_size,
                                'epochs': epochs,
                                'lr': lr,
                                'step_size': step_size,
                                'gamma': gamma,
                                'weight_decay': weight_decay})

model = MLP(n_neurons=[(20, 100), (100, 60), (60, 2)],
            activation=nn.LeakyReLU(),
            batch_norm=True,
            dropout=0.2)

model.cuda()

optimizer = torch.optim.Adam(model.parameters(),
                             lr=lr,
                             weight_decay=weight_decay)

criterion = nn.CrossEntropyLoss()
trainer = create_supervised_trainer(model, optimizer, criterion, device='cuda')
val_metrics = {"accuracy": Accuracy(), "loss": Loss(criterion)}
evaluator = create_supervised_evaluator(model, metrics=val_metrics, device='cuda')
scheduler = StepLR(optimizer=optimizer,
                   step_size=step_size,
                   gamma=gamma)

data = Data(x=X,
            y=y)

loader = DataLoader(dataset=data,
                    shuffle=True,
                    batch_size=batch_size)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(data=loader)
    metrics = evaluator.state.metrics
    # print(optimizer.param_groups[0]['lr'])
    print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
          .format(trainer.state.epoch, metrics["accuracy"], metrics["loss"]))


@trainer.on(Events.EPOCH_COMPLETED)
def update_scheduler(trainer):
    scheduler.step()


logger.attach_output_handler(evaluator,
                             tag='evaluation',
                             metric_names=["loss", "accuracy"],
                             event_name=Events.EPOCH_COMPLETED,
                             global_step_transform=global_step_from_engine(trainer))

logger.attach(trainer,
              log_handler=OptimizerParamsHandler(tag='lr', optimizer=optimizer, param_name='lr'),
              event_name=Events.EPOCH_COMPLETED)


@trainer.on(Events.COMPLETED)
def end_logging(trainer):
    logger.close()


# return evaluator.state.metrics['accuracy']


if __name__ == '__main__':

    trainer.run(loader, max_epochs=epochs)

    model.cpu()

    #torch.save(model.state_dict(), "/artifacts")
    torch.save(model, "/artifacts")

#
#    #pruner = optuna.pruners.MedianPruner()
#
#    study = optuna.create_study(direction="maximize")
#    study.optimize(objective, n_trials=3, timeout=600)

