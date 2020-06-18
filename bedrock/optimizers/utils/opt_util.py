
def basic_log_setup(space):
    log = dict()
    log['evals'] = []
    log['fval'] = []
    log['best_fval'] = []
    for s in space.continuous_space:
        log[s.label] = []
    return log
