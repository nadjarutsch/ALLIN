from main import main

from omegaconf import OmegaConf, DictConfig
import optuna
from hydra import compose, initialize
import hydra

import argparse
import os
if os.path.split(os.getcwd())[-1] != 'src':
    os.chdir('../src')


def objective(trial, args):
    lambda1 = trial.suggest_float('lambda', 0, 1)
    w_threshold = trial.suggest_float('w_thresh', 1e-3, 1, log=True)
    initialize(version_base=None, config_path="config")
    cfg = compose(config_name=f"{args.model}_sweep_{args.sweep_nr}",
                  overrides=[f"causal_discovery.model.lambda1={lambda1}",
                             f"causal_discovery.model.w_threshold={w_threshold}"])

    shds = []
    for seed in range(50, 70):
        cfg.start_seed = seed
        cfg.end_seed = seed + 1
        shds.append(main(cfg))
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        trial.report(shds[-1], step=seed)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return sum(shds) / len(shds)


def sweep(args):
    os.makedirs(f'{args.model}_sweep_{args.sweep_nr}', exist_ok=True)
    study = optuna.create_study(
        study_name=f'{args.model}_{args.sweep_nr}',
        storage=f'sqlite:///{args.model}_sweep_{args.sweep_nr}/optuna_hparam_search.db',
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    func = lambda trial: objective(trial, args)
    study.optimize(func, n_trials=50, n_jobs=1)


if __name__ == '__main__':

    # collect cmd line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_nr', default=1, type=int)
    parser.add_argument('--model', type=str)
    args: argparse.Namespace = parser.parse_args()

    os.environ['WANDB_MODE'] = 'offline'

    sweep(args)

