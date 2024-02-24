import argparse
from pathlib import Path
from typing import Any

from rlgear.utils import MetaWriter


def get_params() -> tuple[argparse.Namespace, dict[Any, Any], MetaWriter, Path]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address")
    parser.add_argument("--redis-password")

    import rlgear.utils
    import rlgear_starter

    args = rlgear.utils.add_rlgear_args(parser).parse_args()

    config_dir = Path(rlgear_starter.__file__).resolve().parent / "config"

    params, meta_writer, log_dir = rlgear.utils.from_yaml(
        args.yaml_file, config_dir, args.exp_name
    )[:3]
    print("done reading params")
    return args, params, meta_writer, log_dir


def adj_tune_kwargs(tune_kwargs: dict[str, Any]) -> None:
    import rlgear.utils
    from ray.tune.registry import ENV_CREATOR, _global_registry

    if tune_kwargs["num_samples"] == 1:
        tune_kwargs["trial_dirname_creator"] = rlgear.rllib_utils.dirname_creator

    tune_kwargs["progress_reporter"] = rlgear.utils.import_class(
        tune_kwargs["progress_reporter"]
    )
    tune_kwargs["checkpoint_config"] = rlgear.utils.import_class(
        tune_kwargs["checkpoint_config"]
    )


def register() -> None:
    import ray.tune.registry
    from ray.rllib.models import ModelCatalog
    from rlgear_starter.envs import env_creator_simple_env
    from rlgear_starter.models import FCNet

    ModelCatalog.register_custom_model("FCNet", FCNet)

    ray.tune.registry.register_env("SimpleEnv-v0", env_creator_simple_env)


def run_rlgear_starter() -> None:
    args, params, meta_writer, log_dir = get_params()

    # delay importing ray until after params are loaded so command line fails
    # can happen quickly
    import ray
    import rlgear.rllib_utils

    if args.redis_password is None:
        args.redis_password = rlgear.rllib_utils.gen_passwd(64)

    print("starting ray")
    ray.init(address=args.address, _redis_password=args.redis_password)
    print("done starting ray")

    tune_kwargs = rlgear.rllib_utils.make_tune_kwargs(
        params, meta_writer, log_dir, args.debug
    )
    register()
    adj_tune_kwargs(tune_kwargs)

    if args.debug:
        # this is useful if you use something like ipdb or lvdb
        # otherwise tune seems to disable command line completion/etc
        # when seting breakpoints
        trainer = rlgear.rllib_utils.get_trainer(tune_kwargs)

        ct = 0
        while True:
            trainer.train()
            ct += 1
            print(f"completed {ct} debugging iterations")
    else:
        ray.tune.run(**tune_kwargs)


if __name__ == "__main__":
    run_rlgear_starter()
