import argparse


parser = argparse.ArgumentParser()


parser.add_argument("--lr",
                    help="learning rate",
                    type=float,
                    default=0.001)

parser.add_argument("--weight_decay",
                    help="weight decay",
                    type=float,
                    default=0.0001)

parser.add_argument("--batch_size",
                    help="batch size",
                    type=int,
                    default=64)

parser.add_argument("--step_size",
                    help="step size",
                    type=int,
                    default=10)

parser.add_argument("--gamma",
                    help="gamma",
                    type=float,
                    default=0.1)

parser.add_argument("--epochs",
                    help="epochs",
                    type=int,
                    default=40)

args = parser.parse_args()
