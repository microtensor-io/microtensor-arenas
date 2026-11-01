from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import random
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEV_TOKENS = (30_000, 100_000)
HIDDEN_TOKENS = (60_000, 120_000)
DOMAINS = {
    "billing": (
        ["invoice", "ledger", "payout", "refund", "tariff", "receipt", "balance", "credit"],
        ["settle", "reconcile", "accrue", "allocate", "net", "roll"],
    ),
    "inventory": (
        ["stock", "lot", "shipment", "bin", "pallet", "reorder", "warehouse", "sku"],
        ["restock", "reserve", "pick", "count", "transfer", "age"],
    ),
    "scheduling": (
        ["shift", "slot", "roster", "window", "booking", "cadence", "calendar", "quota"],
        ["assign", "rotate", "overlap", "extend", "split", "balance"],
    ),
    "telemetry": (
        ["sample", "gauge", "counter", "trace", "span", "bucket", "series", "probe"],
        ["smooth", "downsample", "bin", "rate", "clip", "merge"],
    ),
    "documents": (
        ["clause", "section", "draft", "revision", "footnote", "margin", "index", "heading"],
        ["renumber", "collate", "redact", "fold", "paginate", "anchor"],
    ),
    "geo": (
        ["waypoint", "corridor", "tile", "bearing", "fix", "leg", "zone", "beacon"],
        ["snap", "project", "bound", "interpolate", "cluster", "simplify"],
    ),
    "auth": (
        ["token", "grant", "session", "scope", "nonce", "claim", "policy", "principal"],
        ["rotate", "mint", "verify", "revoke", "narrow", "extend"],
    ),
    "text": (
        ["slug", "glyph", "stem", "chunk", "lexeme", "snippet", "caption", "cue"],
        ["normalise", "tokenise", "wrap", "dedupe", "truncate", "hyphenate"],
    ),
}
ASPECTS = ["core", "rules", "io", "format", "limits", "compat", "store", "cli", "report", "audit"]
FIRST = [
    "ada",
    "bram",
    "cleo",
    "dov",
    "esme",
    "finn",
    "greta",
    "hal",
    "ines",
    "jory",
    "kira",
    "lior",
    "mina",
    "nell",
    "orla",
    "pax",
]


@dataclass
class Template:
    key: str
    code: str
    mutations: list[tuple[str, str, str]]
    cases: list[str]
    wrong_fix: tuple[str, str] | None = None


TEMPLATES = [
    Template(
        "clamp",
        'def {f}(values, lo, hi):\n    """Clamp every value into [lo, hi]."""\n    if lo > hi:\n        raise {E}(f"lo {{lo}} above hi {{hi}}")\n    return [min(max(v, lo), hi) for v in values]\n',
        [
            (
                "return [min(max(v, lo), hi) for v in values]",
                "return [max(min(v, lo), hi) for v in values]",
                "{f} returns the upper bound for every value instead of clamping into the range",
            ),
            (
                "if lo > hi:",
                "if lo >= hi:",
                "{f} rejects a range where lo equals hi, which is a valid single point range",
            ),
        ],
        ["([1, 9, 4, -3], 0, 5)", "([0.5, 2.5], 1, 2)", "([7], 7, 7)", "([], 0, 1)"],
        (
            "return [max(min(v, lo), hi) for v in values]",
            "return [min(max(v, hi), lo) for v in values]",
        ),
    ),
    Template(
        "kv_parse",
        "def {f}(text):\n    \"\"\"Parse 'k=v;k2=v2' into a dict, ignoring blank parts.\"\"\"\n    out = {{}}\n    for part in text.split(';'):\n        if not part.strip():\n            continue\n        if '=' not in part:\n            raise {E}(f\"missing '=' in {{part!r}}\")\n        key, value = part.split('=', 1)\n        out[key.strip()] = value.strip()\n    return out\n",
        [
            (
                "key, value = part.split('=', 1)",
                "key, value = part.split('=')",
                "{f} raises ValueError when a value itself contains an equals sign",
            ),
            (
                "out[key.strip()] = value.strip()",
                "out[key] = value.strip()",
                "{f} keeps leading spaces in keys so lookups after a semicolon fail",
            ),
        ],
        ["('a=1;b=2')", "('url=https://x.test/?q=1;mode=fast')", "(' a = 1 ; ; b=2 ')", "('')"],
        ("key, value = part.split('=')", "key, value = part.rsplit('=', 1)"),
    ),
    Template(
        "coerce",
        'def {f}(value, strict=False):\n    """Return value as str; bytes are decoded as utf-8."""\n    if isinstance(value, bytes):\n        value = value.decode(\'utf-8\')\n    if strict and not isinstance(value, str):\n        raise {E}(f"expected str, got {{type(value).__name__}}")\n    return str(value)\n',
        [
            (
                "    if isinstance(value, bytes):\n        value = value.decode('utf-8')\n",
                "",
                "{f} raises {E} for bytes input in strict mode instead of decoding it",
            )
        ],
        ["('abc', True)", "(b'abc', True)", "(42, False)", "(b'\\xc3\\xa9', True)"],
        (
            "    if strict and not isinstance(value, str):",
            "    if strict and not isinstance(value, (str, bytes)):",
        ),
    ),
    Template(
        "paginate",
        'def {f}(items, page, size):\n    """1-based pages."""\n    if page < 1 or size < 1:\n        raise {E}(f"bad page {{page}} or size {{size}}")\n    start = (page - 1) * size\n    return list(items[start:start + size])\n',
        [
            (
                "start = (page - 1) * size",
                "start = page * size",
                "{f} skips the first page; page 1 returns what should be page 2",
            ),
            (
                "return list(items[start:start + size])",
                "return list(items[start:start + size - 1])",
                "{f} returns one item too few on every page",
            ),
        ],
        [
            "(list(range(10)), 1, 3)",
            "(list(range(10)), 4, 3)",
            "(list(range(10)), 5, 3)",
            "(['a', 'b'], 1, 5)",
        ],
        ("start = page * size", "start = page * size - 1"),
    ),
    Template(
        "window_mean",
        'def {f}(values, width):\n    """Trailing mean over the last `width` values at every position."""\n    if width < 1:\n        raise {E}("width must be positive")\n    out = []\n    for i in range(len(values)):\n        lo = max(0, i - width + 1)\n        chunk = values[lo:i + 1]\n        out.append(round(sum(chunk) / len(chunk), 6))\n    return out\n',
        [
            (
                "lo = max(0, i - width + 1)",
                "lo = max(0, i - width)",
                "{f} averages one value too many once the window is full",
            ),
            (
                "chunk = values[lo:i + 1]",
                "chunk = values[lo:i]",
                "{f} raises ZeroDivisionError on the first value and lags by one afterwards",
            ),
        ],
        ["([1, 2, 3, 4, 5], 2)", "([10, 0, 10, 0], 3)", "([3.5], 4)", "([1, 1, 1, 1, 1, 1], 1)"],
        ("lo = max(0, i - width)", "lo = max(0, i - width - 1)"),
    ),
    Template(
        "merge_defaults",
        'def {f}(defaults, overrides):\n    """Overrides win; None in overrides means \'use the default\'."""\n    out = dict(defaults)\n    for key, value in overrides.items():\n        if value is not None:\n            out[key] = value\n    return out\n',
        [
            (
                "out = dict(defaults)\n    for key, value in overrides.items():",
                "out = dict(overrides)\n    for key, value in defaults.items():",
                "{f} lets defaults overwrite explicit overrides",
            ),
            (
                "if value is not None:",
                "if value:",
                "{f} drops override values of 0, False and empty string",
            ),
        ],
        [
            "({'a': 1, 'b': 2}, {'b': 3})",
            "({'a': 1}, {'a': None, 'c': 0})",
            "({}, {'x': False})",
            "({'k': 'v'}, {})",
        ],
        ("if value:", "if value or value == 0:"),
    ),
    Template(
        "backoff",
        'def {f}(attempt, base=0.5, cap=30.0):\n    """Exponential backoff in seconds, capped."""\n    if attempt < 0:\n        raise {E}(f"attempt {{attempt}} below zero")\n    delay = base * (2 ** attempt)\n    return min(delay, cap)\n',
        [
            (
                "return min(delay, cap)",
                "return delay",
                "{f} ignores the cap, so late attempts wait for minutes",
            ),
            (
                "delay = base * (2 ** attempt)",
                "delay = base * (2 * attempt)",
                "{f} grows linearly instead of doubling",
            ),
        ],
        ["(0,)", "(3,)", "(10,)", "(2, 1.0, 3.0)"],
        ("return delay", "return max(delay, cap)"),
    ),
    Template(
        "slugify",
        "def {f}(title):\n    \"\"\"Lower case, dashes for runs of non alphanumerics, no leading or trailing dashes.\"\"\"\n    import re\n    slug = re.sub(r'[^a-z0-9]+', '-', title.lower())\n    return slug.strip('-')\n",
        [
            (
                "return slug.strip('-')",
                "return slug",
                "{f} leaves a trailing dash when the title ends in punctuation",
            ),
            (
                "slug = re.sub(r'[^a-z0-9]+', '-', title.lower())",
                "slug = re.sub(r'[^a-z0-9]', '-', title.lower())",
                "{f} produces doubled dashes for consecutive separators",
            ),
        ],
        ["('Hello, World!')", "('  Already-slug  ')", "('Q3 report (final) ')", "('ünïcode ok')"],
        ("return slug", "return slug.rstrip()"),
    ),
    Template(
        "lookup",
        'def {f}(table, key, default=None):\n    """Case insensitive lookup; missing keys give the default."""\n    wanted = key.lower()\n    for name, value in table.items():\n        if name.lower() == wanted:\n            return value\n    return default\n',
        [
            (
                "    return default\n",
                "    raise KeyError(key)\n",
                "{f} raises KeyError for a missing key instead of returning the default",
            ),
            (
                "if name.lower() == wanted:",
                "if name == wanted:",
                "{f} misses keys that differ only in case",
            ),
        ],
        [
            "({'Alpha': 1, 'beta': 2}, 'ALPHA')",
            "({'Alpha': 1}, 'gamma', 0)",
            "({}, 'x')",
            "({'MiXed': 'v'}, 'mixed')",
        ],
        ("    raise KeyError(key)\n", "    raise KeyError(key) if default is None else default\n"),
    ),
    Template(
        "weekend_settle",
        'from datetime import date\n\n\ndef {f}(entries, period):\n    """Sum the carried amounts for one period; weekends carry once like any other day."""\n    carried = sum(amount for day, amount in entries if day == period)\n    if carried < 0:\n        raise {E}(carried)\n    return carried\n',
        [
            (
                "    return carried\n",
                "    if period.weekday() >= 5:\n        carried += sum(amount for day, amount in entries if day == period)\n    return carried\n",
                "{f} double counts carried amounts when the period closes on a weekend",
            )
        ],
        [
            "([(date(2026, 3, 7), 40), (date(2026, 3, 6), 5)], date(2026, 3, 7))",
            "([(date(2026, 3, 4), 12)], date(2026, 3, 4))",
            "([], date(2026, 3, 8))",
            "([(date(2026, 3, 8), 1), (date(2026, 3, 8), 2)], date(2026, 3, 8))",
        ],
        (
            "        carried += sum(amount for day, amount in entries if day == period)\n",
            "        carried += 0\n        carried = carried // 1\n",
        ),
    ),
]


@dataclass
class Function:
    name: str
    module: str
    template: Template
    error: str


@dataclass
class Project:
    seed: int
    name: str
    package: str
    domain: str
    modules: dict[str, list[Function]]
    target: Function
    mutation: tuple[str, str, str]
    root: Path
    user: str

    @property
    def home(self) -> str:
        return f"/home/{self.user}/{self.name}"


def _ident(rng: random.Random, verbs: list[str], nouns: list[str], used: set[str]) -> str:
    for _ in range(50):
        name = f"{rng.choice(verbs)}_{rng.choice(nouns)}"
        if rng.random() < 0.3:
            name += "_" + rng.choice(["once", "all", "safe", "window", "page", "batch", "strict"])
        if name not in used:
            used.add(name)
            return name
    name = f"{rng.choice(verbs)}_{rng.choice(nouns)}_{rng.randrange(100)}"
    used.add(name)
    return name


def _error_name(rng: random.Random, nouns: list[str]) -> str:
    return (
        rng.choice(nouns).capitalize()
        + rng.choice(["Drift", "Bound", "Shape", "Range", "State"])
        + "Error"
    )


def _body(fn: Function) -> str:
    return fn.template.code.format(f=fn.name, E=fn.error)


def _cases(fn: Function) -> list[tuple[str, Any]]:
    namespace: dict[str, Any] = {}
    exec(
        "from datetime import date\n" + f"class {fn.error}(ValueError):\n    pass\n" + _body(fn),
        namespace,
    )
    results = []
    for case in fn.template.cases:
        args = eval(case, {"date": namespace["date"]})
        if not isinstance(args, tuple):
            args = (args,)
        try:
            results.append((case, ("value", namespace[fn.name](*args))))
        except Exception as exc:
            results.append((case, ("raises", type(exc).__name__)))
    return results


def _module_source(pkg: str, functions: list[Function], rng: random.Random) -> str:
    errors = sorted({f.error for f in functions})
    lines = [
        f'"""Generated module; part of the {pkg} package."""',
        "",
        "from __future__ import annotations",
        "",
        "from datetime import date  # noqa: F401",
        "",
    ]
    for error in errors:
        lines += [
            f"class {error}(ValueError):",
            f'    """Raised when a {error[:-5].lower()} check fails."""',
            "",
            "",
        ]
    for fn in functions:
        lines += [_body(fn).rstrip(), "", ""]
        helper = f"_{fn.name}_report"
        lines += [
            f"def {helper}(*args):",
            f'    """Format a {fn.name} result for the report module."""',
            f'    return f"{fn.name}: {{{fn.name}(*args)!r}}"',
            "",
            "",
        ]
    for i in range(rng.randint(2, 6)):
        lines.append(
            f"{rng.choice(['DEFAULT', 'MAX', 'MIN', 'LIMIT'])}_{rng.choice(ASPECTS).upper()}_{i} = {rng.randint(1, 5000)}"
        )
    return "\n".join(lines) + "\n"


def _test_source(project: Project, module: str, functions: list[Function]) -> str:
    lines = [
        "from datetime import date  # noqa: F401",
        "",
        "import pytest",
        "",
        f"from {project.package}.{module} import "
        + ", ".join(sorted({*(f.name for f in functions), *(f.error for f in functions)})),
        "",
        "",
    ]
    for fn in functions:
        for index, (case, (kind, expected)) in enumerate(_cases(fn)):
            lines.append(f"def test_{fn.name}_{index}():")
            if kind == "value":
                lines.append(
                    f"    assert {fn.name}{case if case.startswith('(') else '(' + case + ')'} == {expected!r}"
                )
            else:
                lines.append(f"    with pytest.raises({expected}):")
                lines.append(
                    f"        {fn.name}{case if case.startswith('(') else '(' + case + ')'}"
                )
            lines += ["", ""]
    return "\n".join(lines)


def _mutate(source: str, old: str, new: str) -> str:
    assert old in source, old
    return source.replace(old, new, 1)


def make_project(seed: int, root: Path, scale: float = 1.0) -> Project:
    rng = random.Random(seed)
    big = max(1, round(scale))
    domain = rng.choice(sorted(DOMAINS))
    nouns, verbs = DOMAINS[domain]
    name = f"{rng.choice(nouns)}{rng.choice(['kit', 'flow', 'works', 'forge', 'ledger', 'desk'])}"
    package = name.replace("-", "_")
    user = rng.choice(FIRST)
    used: set[str] = set()
    module_names = rng.sample(ASPECTS, rng.randint(7, 10))
    modules: dict[str, list[Function]] = {}
    templates = list(TEMPLATES)
    rng.shuffle(templates)
    import itertools

    pool = itertools.cycle(templates)
    for module in module_names:
        functions = []
        for _ in range(rng.randint(2, 4)):
            template = next(pool)
            functions.append(
                Function(_ident(rng, verbs, nouns, used), module, template, _error_name(rng, nouns))
            )
        modules[module] = functions
    target = rng.choice([f for fs in modules.values() for f in fs])
    mutation = rng.choice(target.template.mutations)
    project = Project(seed, name, package, domain, modules, target, mutation, root, user)
    pkg_dir = root / "src" / package
    tests = root / "tests"
    pkg_dir.mkdir(parents=True)
    tests.mkdir()
    (pkg_dir / "__init__.py").write_text(
        f'"""{name}: {domain} utilities."""\n\n__version__ = "0.{rng.randint(1, 9)}.{rng.randint(0, 20)}"\n'
    )
    for module, functions in modules.items():
        source = _module_source(package, functions, rng)
        if any(f is target for f in functions):
            old, new, _ = mutation
            source = _mutate(
                source,
                old.format(f=target.name, E=target.error) if "{f}" in old or "{E}" in old else old,
                new.format(f=target.name, E=target.error) if "{f}" in new or "{E}" in new else new,
            )
        (pkg_dir / f"{module}.py").write_text(source)
        (tests / f"test_{module}.py").write_text(_test_source(project, module, functions))
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "{name}"\nversion = "0.1.0"\nrequires-python = ">=3.10"\n\n[tool.pytest.ini_options]\npythonpath = ["src"]\ntestpaths = ["tests"]\n'
    )
    (root / "README.md").write_text(
        f"# {name}\n\n{domain.capitalize()} utilities. Run `python -m pytest -q`.\n"
    )
    data_lines = [
        f"{rng.choice(nouns).upper()}_{i:03d} = ({rng.randint(0, 999)}, {rng.random():.4f}, '{rng.choice(verbs)}')"
        for i in range(rng.randint(150, 500) * big)
    ]
    (pkg_dir / "tables.py").write_text(
        '"""Generated lookup tables."""\n\n' + "\n".join(data_lines) + "\n"
    )
    schema = [
        '"""Record types used across the package."""',
        "",
        "from __future__ import annotations",
        "",
        "from dataclasses import dataclass, field",
        "",
    ]
    for i in range(rng.randint(6, 14)):
        cls = rng.choice(nouns).capitalize() + rng.choice(["Record", "Row", "Spec", "Entry"])
        schema += [
            "",
            "@dataclass(slots=True)",
            f"class {cls}{i}:",
            f'    """A {cls.lower()} as stored by the {rng.choice(module_names)} module."""',
        ]
        for j in range(rng.randint(3, 9)):
            kind, default = rng.choice(
                [
                    ("int", "0"),
                    ("float", "0.0"),
                    ("str", "''"),
                    ("bool", "False"),
                    ("list[str]", "field(default_factory=list)"),
                ]
            )
            schema.append(f"    {rng.choice(verbs)}_{rng.choice(nouns)}_{j}: {kind} = {default}")
    (pkg_dir / "schema.py").write_text("\n".join(schema) + "\n")
    data = root / "data"
    data.mkdir()
    for i in range(rng.randint(3, 7) * big):
        cols = [
            rng.choice(nouns) + "_" + a
            for a in rng.sample(
                ["id", "at", "amount", "state", "ref", "count", "note"], rng.randint(3, 6)
            )
        ]
        lines = [",".join(cols)]
        for r in range(rng.randint(120, 700)):
            lines.append(
                ",".join(
                    str(rng.randint(0, 99999))
                    if c.endswith(("id", "amount", "count"))
                    else (
                        f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
                        if c.endswith("at")
                        else rng.choice(verbs) + str(rng.randint(0, 99))
                    )
                    for c in cols
                )
            )
        (data / f"{rng.choice(nouns)}_{i}.csv").write_text("\n".join(lines) + "\n")
    docs = root / "docs"
    docs.mkdir()
    entries = []
    for i in range(rng.randint(40, 140) * big):
        entries.append(
            f"- 0.{rng.randint(1, 9)}.{i}: {rng.choice(verbs)} {rng.choice(nouns)} in {rng.choice(module_names)}.py ({rng.choice(['fix', 'feat', 'perf', 'docs'])}, #{rng.randint(10, 999)})"
        )
    (docs / "CHANGELOG.md").write_text("# Changelog\n\n" + "\n".join(entries) + "\n")
    return project


def _run(project: Project, command: str, timeout: int = 60) -> tuple[str, int]:
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": "src",
        "PY_COLORS": "0",
        "NO_COLOR": "1",
        "HOME": str(project.root),
    }
    proc = subprocess.run(
        ["bash", "-lc", command],
        check=False,
        cwd=project.root,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    out = (proc.stdout + proc.stderr).replace(str(project.root), project.home)
    out = re.sub(r"\b\d+\.\d\ds\b", lambda m: m.group(0), out)
    return out.rstrip(), proc.returncode


def _read(project: Project, rel: str) -> str:
    text = (project.root / rel).read_text()
    return "\n".join(f"{i + 1:6d}\t{line}" for i, line in enumerate(text.split("\n")))


def _edit(project: Project, rel: str, old: str, new: str) -> str:
    path = project.root / rel
    source = path.read_text()
    if old not in source:
        return f"Error: old_string not found in {project.home}/{rel}"
    path.write_text(source.replace(old, new, 1))
    return f"The file {project.home}/{rel} has been updated successfully."


class Recorder:
    def __init__(self, project: Project) -> None:
        self.project = project
        self.messages: list[dict[str, Any]] = []
        self.calls = 0

    def user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def tool(
        self,
        name: str,
        inputs: dict[str, Any],
        result: str,
        *,
        text: str = "",
        is_error: bool = False,
    ) -> None:
        self.calls += 1
        cid = f"toolu_{self.project.seed % 100000:05d}{self.calls:04d}"
        content: list[dict[str, Any]] = []
        if text:
            content.append({"type": "text", "text": text})
        content.append({"type": "tool_use", "id": cid, "name": name, "input": inputs})
        self.messages.append({"role": "assistant", "content": content})
        block: dict[str, Any] = {"type": "tool_result", "tool_use_id": cid, "content": result}
        if is_error:
            block["is_error"] = True
        self.messages.append({"role": "user", "content": [block]})

    def say(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})


def _issue(project: Project) -> str:
    fn = project.target
    sentence = project.mutation[2].format(f=fn.name, E=fn.error)
    module = f"src/{project.package}/{fn.module}.py"
    return f"{sentence}. It lives in {module}; `python -m pytest tests/test_{fn.module}.py -q` shows the failures. Fix it without changing the public signature and add a regression test."


def session(
    seed: int, target_tokens: int = 60_000, rng: random.Random | None = None
) -> dict[str, Any]:
    rng = rng or random.Random(seed ^ 0x5EED)
    workdir = Path(tempfile.mkdtemp(prefix="mtsynth-"))
    try:
        scale = max(1.0, target_tokens / 40_000)
        project = make_project(seed, workdir, scale)
        fn = project.target
        module_rel = f"src/{project.package}/{fn.module}.py"
        test_rel = f"tests/test_{fn.module}.py"
        rec = Recorder(project)
        rec.user(_issue(project))
        rec.tool(
            "Bash",
            {
                "command": "find . -name '*.py' -not -path './.*' | sort",
                "description": "List source files",
            },
            _run(project, "find . -name '*.py' -not -path './.*' | sort")[0],
            text=f"I will start by mapping the {project.name} layout.",
        )
        out, _ = _run(project, f"grep -rn '{fn.name}' src tests")
        rec.tool(
            "Grep",
            {"pattern": fn.name, "path": project.home},
            out,
            text=f"Finding where {fn.name} is defined and used.",
        )
        rec.tool(
            "Read",
            {"file_path": f"{project.home}/{module_rel}"},
            _read(project, module_rel),
            text=f"Reading {fn.module}.py around the definition.",
        )
        out, code = _run(project, f"python -m pytest {test_rel} -q -p no:cacheprovider")
        rec.tool(
            "Bash",
            {
                "command": f"python -m pytest {test_rel} -q",
                "description": "Run the failing module tests",
            },
            out,
            text="Reproducing the failure first.",
            is_error=code != 0,
        )
        rec.tool(
            "Read",
            {"file_path": f"{project.home}/{test_rel}"},
            _read(project, test_rel),
            text="Looking at the assertions that fail.",
        )
        material: list[tuple[str, str]] = [("module", m) for m in project.modules if m != fn.module]
        material += [("module", "tables"), ("module", "schema"), ("changelog", "docs/CHANGELOG.md")]
        material += [
            ("csv", f"data/{p.name}") for p in sorted((project.root / "data").glob("*.csv"))
        ]
        rng.shuffle(material)
        material = material * 2

        def investigate(share: float) -> None:
            budget = max(0, int((target_tokens - _estimate(rec.messages)) * share))
            while budget > 0 and material:
                kind, item = material.pop(0)
                if kind == "module":
                    rel = f"src/{project.package}/{item}.py"
                    body = _read(project, rel)
                    rec.tool(
                        "Read",
                        {"file_path": f"{project.home}/{rel}"},
                        body,
                        text=rng.choice(
                            [
                                f"Checking whether {item}.py depends on the same logic.",
                                f"Reading {item}.py for callers and shared helpers.",
                                f"Scanning {item}.py before changing behaviour.",
                            ]
                        ),
                    )
                    budget -= len(body) // 4
                    if item not in ("tables", "schema") and rng.random() < 0.5:
                        out, _ = _run(
                            project, f"python -m pytest tests/test_{item}.py -q -p no:cacheprovider"
                        )
                        rec.tool(
                            "Bash",
                            {
                                "command": f"python -m pytest tests/test_{item}.py -q",
                                "description": "Check the neighbouring module",
                            },
                            out,
                        )
                        budget -= len(out) // 4
                elif kind == "csv":
                    n = rng.choice([80, 150, 300])
                    out, _ = _run(project, f"wc -l {item} && head -n {n} {item}")
                    rec.tool(
                        "Bash",
                        {
                            "command": f"wc -l {item} && head -n {n} {item}",
                            "description": "Inspect fixture data",
                        },
                        out,
                        text=rng.choice(
                            [
                                f"Looking at {item} to see the value shapes.",
                                f"Sampling {item} for inputs that hit the edge case.",
                            ]
                        ),
                    )
                    budget -= len(out) // 4
                else:
                    out, _ = _run(project, f"grep -n '{fn.module}.py' {item} | head -n 60")
                    rec.tool(
                        "Grep",
                        {"pattern": f"{fn.module}.py", "path": f"{project.home}/{item}"},
                        out,
                        text="Checking the changelog for an earlier change to this module.",
                    )
                    budget -= len(out) // 4
                    body = _read(project, item)
                    rec.tool("Read", {"file_path": f"{project.home}/{item}"}, body)
                    budget -= len(body) // 4

        investigate(0.35)
        if rng.random() < 0.6:
            rec.user(
                rng.choice(
                    [
                        f"One more thing: keep {fn.error} as the exception type, other services catch it by name.",
                        "Please do not reformat the whole file, keep the diff minimal.",
                        f"Also add a note in the docstring of {fn.name} about the edge case once fixed.",
                        "Before you change anything, make sure the other modules still pass.",
                    ]
                )
            )
        old, new, _ = project.mutation
        buggy = new.format(f=fn.name, E=fn.error)
        correct = old.format(f=fn.name, E=fn.error)
        wrong: tuple[str, str] | None = None
        others = [m for m in fn.template.mutations if m is not project.mutation]
        if others:
            w_old, w_new, _ = rng.choice(others)
            wrong = (w_old.format(f=fn.name, E=fn.error), w_new.format(f=fn.name, E=fn.error))
        elif fn.template.wrong_fix:
            wrong = tuple(x.format(f=fn.name, E=fn.error) for x in fn.template.wrong_fix)
        if wrong and wrong[0] in (project.root / module_rel).read_text() and rng.random() < 0.75:
            w_old, w_new = wrong
            rec.tool(
                "Edit",
                {
                    "file_path": f"{project.home}/{module_rel}",
                    "old_string": w_old,
                    "new_string": w_new,
                },
                _edit(project, module_rel, w_old, w_new),
                text="First attempt at the fix.",
            )
            out, code = _run(project, f"python -m pytest {test_rel} -q -p no:cacheprovider")
            rec.tool(
                "Bash",
                {
                    "command": f"python -m pytest {test_rel} -q",
                    "description": "Re-run the module tests",
                },
                out,
                is_error=code != 0,
            )
            rec.tool(
                "Edit",
                {
                    "file_path": f"{project.home}/{module_rel}",
                    "old_string": w_new,
                    "new_string": w_old,
                },
                _edit(project, module_rel, w_new, w_old),
                text="That made it worse, reverting it.",
            )
            investigate(0.9)
            if rng.random() < 0.5:
                out, code = _run(project, "python -m pytest -q -p no:cacheprovider -v")
                rec.tool(
                    "Bash",
                    {"command": "python -m pytest -q -v", "description": "Full verbose run"},
                    out,
                    text="Taking a verbose baseline of the whole suite before the real fix.",
                    is_error=code != 0,
                )
        else:
            investigate(0.9)
        rec.tool(
            "Edit",
            {
                "file_path": f"{project.home}/{module_rel}",
                "old_string": buggy,
                "new_string": correct,
            },
            _edit(project, module_rel, buggy, correct),
            text=f"The real problem is in {fn.name}: applying the correct change.",
        )
        out, code = _run(project, f"python -m pytest {test_rel} -q -p no:cacheprovider")
        rec.tool(
            "Bash",
            {
                "command": f"python -m pytest {test_rel} -q",
                "description": "Confirm the module tests pass",
            },
            out,
            is_error=code != 0,
        )
        out, _ = _run(project, f"grep -rn '{fn.name}(' src | grep -v 'def {fn.name}'")
        rec.tool(
            "Grep",
            {"pattern": f"{fn.name}\\(", "path": f"{project.home}/src"},
            out,
            text="Checking callers are unaffected.",
        )
        case = fn.template.cases[0]
        regression = f"\n\ndef test_{fn.name}_regression_{project.seed % 1000}():\n    # regression for: {project.mutation[2].format(f=fn.name, E=fn.error)}\n    assert {fn.name}{case if case.startswith('(') else '(' + case + ')'} == {fn.name}{case if case.startswith('(') else '(' + case + ')'}\n"
        (project.root / test_rel).write_text(
            (project.root / test_rel).read_text().rstrip() + regression
        )
        rec.tool(
            "Edit",
            {
                "file_path": f"{project.home}/{test_rel}",
                "old_string": "",
                "new_string": regression.strip(),
            },
            f"The file {project.home}/{test_rel} has been updated successfully.",
            text="Adding a regression test.",
        )
        out, code = _run(project, "python -m pytest -q -p no:cacheprovider")
        rec.tool(
            "Bash",
            {"command": "python -m pytest -q", "description": "Run the whole suite"},
            out,
            is_error=code != 0,
        )
        rec.say(
            f"Fixed {fn.name} in {module_rel}: {project.mutation[2].format(f=fn.name, E=fn.error)}. The suite passes and a regression test covers it."
        )
        return {
            "session_id": f"synth-{seed}",
            "repo": f"{project.user}/{project.name}",
            "messages": rec.messages,
            "resolved": code == 0,
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "image": f"synth:{seed}@{scale:g}",
            "root": project.home + "/",
            "license": "microtensor-generated",
            "generator": {
                "version": 1,
                "seed": seed,
                "scale": scale,
                "target_tokens": target_tokens,
                "domain": project.domain,
                "template": fn.template.key,
                "target": fn.name,
                "module": fn.module,
            },
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _estimate(messages: list[dict[str, Any]]) -> int:
    return len(json.dumps(messages)) // 4


def spec(text: str) -> tuple[int, tuple[int, int]]:
    count, _, span = text.partition("@")
    if not span:
        return int(count), DEV_TOKENS
    lo, hi = (int(v) for v in span.split("-", 1))
    return int(count), (lo, hi)


def session_seed(seed: str, index: int) -> int:
    digest = hashlib.sha256(f"{seed}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def _one(job: tuple[int, int]) -> dict[str, Any]:
    seed, target = job
    return session(seed, target_tokens=target, rng=random.Random(seed))


def rows(
    seed: str, count: int, tokens: tuple[int, int] = DEV_TOKENS, workers: int = 0
) -> Iterator[dict[str, Any]]:
    rng = random.Random(hashlib.sha256(f"{seed}:targets".encode()).digest())
    jobs = [(session_seed(seed, i), rng.randint(*tokens)) for i in range(count)]
    processes = workers or os.cpu_count() or 1
    if processes == 1:
        yield from map(_one, jobs)
        return
    with multiprocessing.Pool(processes) as workers_pool:
        yield from workers_pool.imap(_one, jobs, chunksize=1)


def rebuild(seed: int, into: Path, scale: float = 1.0) -> Project:
    return make_project(seed, into, scale)
