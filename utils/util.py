import datetime
import random
import string
import sys


def get_path_with_time(func_name, estimator_type, optimizer, tuning_path):
    time_name = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    if tuning_path == '':
        path = 'log/' + func_name + '/' + estimator_type + '/' + func_name + '_' + optimizer + '_' + time_name
    else:
        path = 'log/' + func_name + '_' + tuning_path + '/' + estimator_type + '/' + func_name + '_' + optimizer + '_' + time_name
    return path


def flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]


def randomname(n):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=n))


def make_obj_fn(fn, continuous_space):
    def obj_fn(x):
        if isinstance(x, dict):
            config = x.copy()
        else:
            x = x.reshape(x.size)
            config = {}
            for i in range(len(x)):
                config[continuous_space[i].label] = continuous_space[i].convert(x[i])
        return fn(config)

    return obj_fn


def _resolve_name(name, package, level):
    """Return the absolute name of the module to be imported."""
    if not hasattr(package, 'rindex'):
        raise ValueError("'package' not set to a string")
    dot = len(package)
    for x in range(level, 1, -1):
        try:
            dot = package.rindex('.', 0, dot)
        except ValueError:
            raise ValueError("attempted relative import beyond top-level "
                             "package")
    return "%s.%s" % (package[:dot], name)


def import_module(name, package=None):
    """Import a module.

    The 'package' argument is required when performing a relative import. It
    specifies the package to use as the anchor point from which to resolve the
    relative import to an absolute import.

    """
    if name.startswith('.'):
        if not package:
            raise TypeError("relative imports require the 'package' argument")
        level = 0
        for character in name:
            if character != '.':
                break
            level += 1
        name = _resolve_name(name[level:], package, level)
    __import__(name)
    return sys.modules[name]


def load_class(name):
    dot = name.rindex('.')
    module, class_name = name[:dot], name[dot + 1:]
    mod = import_module(module)
    return getattr(mod, class_name)


def delete_last_empty_line(s):
    end_index = len(s) - 1
    while end_index >= 0 and (s[end_index] == "\n" or s[end_index] == "\r"):
        end_index -= 1
    s = s[:end_index + 1]
    return s


def read_file(file_name):
    with open(file_name, "r") as f:
        s = f.read()
        s = delete_last_empty_line(s)
        s_l = s.split("\n")
        for i, l in enumerate(s_l):
            if l.endswith("\r"):
                s_l[i] = s_l[i][:-1]
    return s_l
