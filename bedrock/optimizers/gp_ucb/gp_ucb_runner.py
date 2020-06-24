import json
import time

from bedrock.optimizers.gp_ucb.gp_ucb import GPUCB
from bedrock.optimizers.runner_base import RunnerBase


class GPUCBRunner(RunnerBase):
    def __init__(self, **params):
        super().__init__(**params)
        self.gpucb = GPUCB(**params)

    def run(self):
        start = time.time()
        # optimize
        self.gpucb.optimize()
        # collect log
        self.log = self.gpucb.log
        # make csv
        self.best_params = self.gpucb.best_params
        self.best_fval = self.gpucb.best_fval
        # save
        super().save()
        elapsed_time = time.time() - start
        params_setting = {
            "estimator_type": self.params["estimator_type"],
            "problem": self.params["problem"],
            "elapsed_time": elapsed_time,
            "B": self.params["B"],
            "seed": self.params["seed"],
            "dim": self.params["dim"],
        }
        if self.params["dataset"] == "MSSynthetic1D":
            params_setting["num_sources"] = self.params["num_sources"]
            params_setting["source_mu_bound"] = self.params["source_mu_bound"]
            params_setting["target_mu_bound"] = self.params["target_mu_bound"]
            params_setting["num_data"] = self.params["num_data"]
        elif self.params["dataset"] == "ParkinsonSVM" or "GvHDLGBM":
            params_setting["model"] = self.params["model"]
            params_setting["target_name"] = self.params["target_name"]
            params_setting["ratio_validation"] = self.params["ratio_validation"]
            params_setting["ratio_dre"] = self.params["ratio_dre"]
            params_setting["is_source_concat_for_naive"] = self.params[
                "is_source_concat_for_naive"
            ]
            params_setting["is_separate_source_dens"] = self.params[
                "is_separate_source_dens"
            ]

        json.dump(
            params_setting,
            open("{}/params.json".format(self.params["path"]), "w"),
            indent=4,
            sort_keys=True,
        )
        print("best_params:{}, best_fval:{}".format(self.best_params, self.best_fval))
