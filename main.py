from argparse import ArgumentParser

from engine.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init_subclass__(cls, **kwargs):
        pass


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('-c', '--config', type=str, required=True, help='config file')
    params = parser.parse_args()
    trainer = Trainer(params.config)
    trainer.fit()
