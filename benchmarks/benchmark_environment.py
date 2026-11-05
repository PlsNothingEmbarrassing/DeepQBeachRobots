"""Reproducible environment-only throughput benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from DeepQBeachRobots.environment import BeachCleaningEnv


ENVIRONMENT = {
    "n_obstacles": 8,
    "n_small_rubbish": 40,
    "width": 20,
    "height": 20,
    "max_steps": 500,
    "observation_radius": 5,
}


def _step(environment, observation, rng):
    actions = np.flatnonzero(observation["mask"])
    action = int(rng.choice(actions))
    next_observation, _, terminated, truncated, _ = environment.step(action)
    return environment.reset()[0] if terminated or truncated else next_observation


def benchmark(steps: int, warmup_steps: int, repeats: int, seed: int):
    results = []
    for repeat in range(repeats):
        environment = BeachCleaningEnv(
            seed=seed + repeat * 100_000, **ENVIRONMENT
        )
        rng = np.random.default_rng(seed + repeat)
        observation, _ = environment.reset()
        for _ in range(warmup_steps):
            observation = _step(environment, observation, rng)
        started_at = time.perf_counter()
        for _ in range(steps):
            observation = _step(environment, observation, rng)
        elapsed = time.perf_counter() - started_at
        results.append(steps / elapsed)
    return {
        "steps": steps,
        "warmup_steps": warmup_steps,
        "repeats": repeats,
        "steps_per_second": results,
        "median_steps_per_second": statistics.median(results),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--baseline-steps-per-second", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.steps, args.warmup_steps, args.repeats) <= 0:
        parser.error("steps, warmup steps, and repeats must be positive")
    if (
        args.baseline_steps_per_second is not None
        and args.baseline_steps_per_second <= 0
    ):
        parser.error("baseline steps per second must be positive")

    result = benchmark(args.steps, args.warmup_steps, args.repeats, args.seed)
    if args.baseline_steps_per_second is not None:
        result["baseline_steps_per_second"] = args.baseline_steps_per_second
        result["speedup"] = (
            result["median_steps_per_second"] / args.baseline_steps_per_second
        )
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered)
    if result.get("speedup", 2.0) < 2.0:
        raise SystemExit("Environment throughput did not reach the required 2x speedup")


if __name__ == "__main__":
    main()
