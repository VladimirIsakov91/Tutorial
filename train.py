import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.contrib.handlers.neptune_logger import *
from ignite.handlers import Checkpoint

from model import Data, MLP
from data import X, y
from parse import args


model = MLP(n_neurons=[(20, 100), (100, 60), (60, 2)],
            activation=nn.LeakyReLU(),
            batch_norm=True,
            dropout=0.2)

model.cuda()

logger = NeptuneLogger(api_token=os.getenv('NEPTUNE_API_TOKEN'),
                        project_name = "vladimir.isakov/sandbox",
                        experiment_name  = 'Run',
                        upload_source_files='./train.py',
                        #tags = 'v1',
                        params = {'batch_size': args.batch_size,
                                    'epochs': args.epochs,
                                    'lr': args.lr,
                                    'step_size': args.step_size,
                                    'gamma': args.gamma,
                                    'weight_decay': args.weight_decay,
                                    'model': repr(model)})

optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.lr,
                             weight_decay=args.weight_decay)

step_scheduler = StepLR(optimizer,
                        step_size=args.step_size,
                        gamma=args.gamma)

scheduler = LRScheduler(step_scheduler)

criterion = nn.CrossEntropyLoss()

trainer = create_supervised_trainer(model, optimizer, criterion, device='cuda')
val_metrics = {"accuracy": Accuracy(), "loss": Loss(criterion)}
evaluator = create_supervised_evaluator(model, metrics=val_metrics, device='cuda')

data = Data(x=X,
            y=y)

loader = DataLoader(dataset=data,
                    shuffle=True,
                    batch_size=args.batch_size)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(data=loader)
    metrics = evaluator.state.metrics
    print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
          .format(trainer.state.epoch, metrics["accuracy"], metrics["loss"]))


trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)


def score_function(engine):
    return engine.state.metrics["accuracy"]


evaluator.add_event_handler(Events.COMPLETED,
                            Checkpoint({"model": model},
                                        NeptuneSaver(logger),
                                        n_saved=1,
                                        filename_prefix="best",
                                        score_function=score_function,
                                        score_name="val_acc",
                                        global_step_transform=global_step_from_engine(trainer)))

logger.attach(evaluator,
              log_handler=OutputHandler(tag='evaluation',
                                        metric_names=["loss", "accuracy"],
                                        global_step_transform=global_step_from_engine(trainer)),
              event_name=Events.EPOCH_COMPLETED)

logger.attach(trainer,
              log_handler=OptimizerParamsHandler(tag='optimizer',
                                                 optimizer=optimizer,
                                                 param_name='lr'),
              event_name=Events.EPOCH_COMPLETED)


@trainer.on(Events.COMPLETED)
def end_logging(trainer):
    logger.close()


if __name__ == '__main__':

    trainer.run(loader, max_epochs=args.epochs)

    model.cpu()

    torch.save(model, "./model.pt")
    #print(repr(model))

