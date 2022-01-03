import os
import importlib.util

"""
Questo metodo mi consente di caricare in maniera dinamica i vari moduli di riferimento per G1, G2, D, Syntetich.
Ad esempio: models/mono/G1.py
Ad esempio: dataset/Syntetich.py
"""
def _import_module(name_module, path):
    spec = importlib.util.spec_from_file_location(name_module, os.path.join(path, name_module + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module