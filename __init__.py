import numpy as np
import locale

locale.setlocale(locale.LC_ALL, '')

_params = {}
_param_aliases = {}

def params_with_name(name):
    return [p for n,p in _params.items() if name in n]

def delete_all_params():
    _params.clear()

def alias_params(replace_dict):
    for old,new in replace_dict.items():
        _param_aliases[old] = new

def delete_param_aliases():
    _param_aliases.clear()

def print_model_settings(locals_):
    all_vars = [(k,v) for (k,v) in locals_.items() if (k.isupper() and k!='T' and k!='SETTINGS' and k!='ALL_SETTINGS')]
    all_vars = sorted(all_vars, key=lambda x: x[0])
    for var_name, var_value in all_vars:
        print "\t{}: {}".format(var_name, var_value)


def print_model_settings_dict(settings):
    all_vars = [(k,v) for (k,v) in settings.items()]
    all_vars = sorted(all_vars, key=lambda x: x[0])
    for var_name, var_value in all_vars:
        print "\t{}: {}".format(var_name, var_value)
