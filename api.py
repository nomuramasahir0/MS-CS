import json
from bedrock.parser.arg_parser import get_args
import sys


def main():
    args = get_args()
    dict_args = vars(args)
    if args.optimizer == "GPUCB":
        from bedrock.optimizers.gp_ucb.gp_ucb_runner import GPUCBRunner

        runner = GPUCBRunner(**dict_args)
    else:
        print("Optimizer does not exist.")
        sys.exit(0)
    runner.run()
    best_params = runner.best_params
    target_fval = args.obj_func.target_evaluate(best_params)
    try:
        final = {"best_params": best_params, "target_fval": target_fval}
        json.dump(
            final, open("%s/final.json" % args.path, "w"), indent=4, sort_keys=True
        )
    except TypeError:
        print("failed to save final.json")


if __name__ == "__main__":
    main()
