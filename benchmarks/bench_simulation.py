"""Benchmark script for COMMOTIONSFramework straight-crossing simulations.

Usage:
    python benchmarks/bench_simulation.py [--repo <path>] [--fast-horizon]
"""
import argparse
import os
import sys
import time
import warnings


def main():
    parser = argparse.ArgumentParser(description='Benchmark SC simulations.')
    parser.add_argument('--repo', default=None,
                        help='Repository root path (defaults to parent of benchmarks/).')
    parser.add_argument('--fast-horizon', action='store_true',
                        help='Enable sc_scenario_helper.FAST_ANALYTIC_HORIZON.')
    args = parser.parse_args()

    repo = (os.path.abspath(args.repo) if args.repo
            else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path[:0] = [repo, os.path.join(repo, 'SCPaper')]
    os.chdir(os.path.join(repo, 'SCPaper'))

    warnings.filterwarnings('ignore')
    import numpy as np
    np.seterr(all='ignore')

    import sc_scenario_helper  # noqa: E402
    import sc_fitting  # noqa: E402
    import sc_run  # noqa: E402

    if args.fast_horizon and hasattr(sc_scenario_helper, 'FAST_ANALYTIC_HORIZON'):
        sc_scenario_helper.FAST_ANALYTIC_HORIZON = True

    mode_label = ('FAST_ANALYTIC_HORIZON=True' if args.fast_horizon
                  else 'default (bit-identical)')
    print(f'Benchmarking {repo} [{mode_label}]')

    # Warm-up
    det_model = sc_run.get_model_with_params('oVAoBEvoAI')
    sc_run.run_simulation(det_model, initial_cp_distances=(3, 40),
                          initial_speeds=sc_fitting.AGENT_FREE_SPEEDS,
                          end_time=2, ped_prio=False, idx_parameterisation=0)

    # 1. Deterministic oVAoBEvoAI across 5 ONE_AG_SCENARIOS x 5 parameterisations
    t0 = time.perf_counter()
    n_det = 0
    for idx in range(5):
        p = det_model.get_params_dict(idx)
        for scen in sc_fitting.ONE_AG_SCENARIOS.values():
            sc_fitting.construct_model_and_simulate_scenario(
                'oVAoBEvoAI', dict(p), scen, i_variation=1,
                apply_stop_criteria=False)
            n_det += 1
    dt_det = time.perf_counter() - t0
    print(f'  det oVAoBEvoAI ({n_det} sims):        {dt_det:.3f} s  '
          f'({dt_det / n_det * 1000:.2f} ms/sim)')

    # 2. Deterministic oVAoVAloBEvoAI (with looming)
    loom_model = sc_run.get_model_with_params('oVAoVAloBEvoAI')
    t0 = time.perf_counter()
    n_loom = 0
    for idx in range(2):
        p = loom_model.get_params_dict(idx)
        for scen in sc_fitting.ONE_AG_SCENARIOS.values():
            sc_fitting.construct_model_and_simulate_scenario(
                'oVAoVAloBEvoAI', dict(p), scen, i_variation=1,
                apply_stop_criteria=False)
            n_loom += 1
    dt_loom = time.perf_counter() - t0
    print(f'  det oVAoVAloBEvoAI ({n_loom} sims):    {dt_loom:.3f} s  '
          f'({dt_loom / n_loom * 1000:.2f} ms/sim)')

    # 3. Stochastic oVAoBEvoAIoEAoSNvoPF (10 sims, 10 s duration, dt=0.025 s)
    stoch_model = sc_run.get_model_with_params('oVAoBEvoAIoEAoSNvoPF')
    t0 = time.perf_counter()
    n_stoch = 10
    for idx in range(n_stoch):
        p = stoch_model.get_params_dict(idx)
        sc_fitting.construct_model_and_simulate_scenario(
            'oVAoBEvoAIoEAoSNvoPF', dict(p),
            sc_fitting.PROB_FIT_SCENARIOS['EncounterPedPrio'],
            noise_seeds=(idx + 1, idx + 101), apply_stop_criteria=False)
    dt_stoch = time.perf_counter() - t0
    print(f'  stoch oVAoBEvoAIoEAoSNvoPF ({n_stoch} sims): {dt_stoch:.3f} s  '
          f'({dt_stoch / n_stoch * 1000:.2f} ms/sim)')


if __name__ == '__main__':
    main()
