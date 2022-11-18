from main import main

from omegaconf import OmegaConf
import optuna

import argparse
import os
if os.path.split(os.getcwd())[-1] != 'src':
    os.chdir('../src')


def objective(trial, sweep_nr):
    cfg = OmegaConf.load(f'config/notears_sweep_{sweep_nr}.yaml')
    cfg.causal_discovery.model.lambda1 = trial.suggest_float('lambda', 0, 1)
    cfg.causal_discovery.model.w_threshold = trial.suggest_float('w_thresh', 0, 1, log=True)
    shds = []
    for seed in range(50, 100):
        cfg.start_seed = seed
        cfg.end_seed = seed + 1
        shds.append(main(cfg))
        trial.report(shds[-1], step=seed)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return sum(shds) / len(shds)


def sweep(args):
    study = optuna.create_study(
        study_name='notears',
        storage=f'sqlite:///sweep_{args.sweep_nr}/optuna_hparam_search.db',
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    func = lambda trial: objective(trial, args.sweep_nr)
    study.optimize(func, n_trials=50, n_jobs=1)


if __name__ == '__main__':

    # collect cmd line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_nr', default=1, type=int)
    args: argparse.Namespace = parser.parse_args()

    os.environ['WANDB_MODE'] = 'offline'

    sweep(args)

