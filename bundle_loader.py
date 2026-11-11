"""
bundle_loader.py -- serves the project's modules from a few bundle files.

The code of every module is kept EXACTLY as it was; only the files were merged.  Each *.pym bundle holds several modules, every one
between a `# @@MODULE name` line and a `# @@END name` line.  This loader imports each section as its own module, under its old name
(`import ai_trade_decision`, `from market_data import ...` keep working, each module keeps its own namespace, globals and logger).

app.py imports this file first.  Run the unit tests with:   python bundle_loader.py --tests [module_name ...]
"""
import importlib.abc
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUNDLES = ("core_infra", "data_feeds", "analytics", "strategies", "trade_core", "ai_brain", "market_intel", "multi_asset", "tests_all")
_MARK, _END = "# @@MODULE ", "# @@END "

_INDEX = {}   # module name -> (bundle path, first source line index, end line index)


def _scan():
    for b in _BUNDLES:
        path = os.path.join(_HERE, b + ".pym")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8", newline="") as fh:
            lines = fh.read().split("\n")
        name, start = None, 0
        for i, ln in enumerate(lines):
            if name is None and ln.startswith(_MARK):
                name, start = ln[len(_MARK):].strip(), i + 1
            elif name is not None and ln.startswith(_END) and ln[len(_END):].strip() == name:
                _INDEX[name] = (path, start, i)
                name = None


class _Loader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path, start, end = _INDEX[module.__name__]
        with open(path, "r", encoding="utf-8", newline="") as fh:
            lines = fh.read().split("\n")
        # padding keeps the line numbers equal to the line numbers in the bundle file (tracebacks point to the right line)
        source = "\n" * start + "\n".join(lines[start:end]) + "\n"
        module.__file__ = path
        exec(compile(source, path, "exec"), module.__dict__)


class _Finder(importlib.abc.MetaPathFinder):
    _loader = _Loader()

    def find_spec(self, fullname, path=None, target=None):
        if fullname in _INDEX:
            return importlib.util.spec_from_loader(fullname, self._loader, origin=_INDEX[fullname][0])
        return None


_scan()
if not any(isinstance(f, _Finder) for f in sys.meta_path):
    sys.meta_path.insert(0, _Finder())


def module_names():
    return sorted(_INDEX)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--tests":
        import unittest
        names = sys.argv[2:] or [n for n in module_names() if n.startswith("tests_")]
        suite = unittest.TestSuite()
        for n in names:
            suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(importlib.import_module(n)))
        res = unittest.TextTestRunner(verbosity=1).run(suite)
        sys.exit(0 if res.wasSuccessful() else 1)
    print("Modules served by the bundles:", ", ".join(module_names()))
