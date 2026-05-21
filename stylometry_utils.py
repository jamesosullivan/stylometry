"""Shared utilities for the stylometry scripts.

The functions in this file intentionally use only the Python standard library at
import time. Third-party packages are imported inside functions after the calling
script has checked or installed its dependencies.
"""

from __future__ import annotations

import importlib.util
import os
import re
import site
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence


WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


def _normalise_package_name(name: str) -> str:
    """Normalise package names according to common Python packaging rules."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _ensure_user_site_on_path() -> None:
    """Add Python's user-site directory to sys.path if pip installed there.

    On macOS, pip may install packages under ~/Library/Python/... during a
    running script. If that directory was not present when Python started,
    importlib may not see newly installed packages until the script restarts.
    This helper makes the current process as robust as possible before we fall
    back to a one-time restart.
    """
    try:
        user_site = site.getusersitepackages()
    except Exception:
        return
    if user_site and user_site not in sys.path:
        sys.path.append(user_site)


def _import_available(import_name: str) -> bool:
    """Return True if an import name is available in this Python environment."""
    _ensure_user_site_on_path()
    importlib.invalidate_caches()
    return importlib.util.find_spec(import_name) is not None


def ensure_dependencies(requirements: Mapping[str, str], auto_install: bool = True) -> None:
    """Check required import names and optionally pip-install missing packages.

    Args:
        requirements: Mapping of import name -> pip package specifier.
            Example: {"sklearn": "scikit-learn>=1.4"}.
        auto_install: If True, install missing packages with the current Python.

    Raises:
        RuntimeError: If packages are missing and auto_install is False, or if
        installation appears to fail after installation and one restart.
    """
    missing_imports = [
        import_name for import_name in requirements
        if not _import_available(import_name)
    ]

    if not missing_imports:
        return

    missing_specs = [requirements[import_name] for import_name in missing_imports]

    if not auto_install:
        joined = " ".join(missing_specs)
        raise RuntimeError(
            "Missing required packages: " + joined +
            "\nInstall them with: " + sys.executable + " -m pip install " + joined
        )

    print("Installing missing dependencies:", ", ".join(missing_specs))
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_specs])

    # Make newly installed user-site packages visible if possible.
    _ensure_user_site_on_path()
    importlib.invalidate_caches()

    still_missing = [
        import_name for import_name in requirements
        if not _import_available(import_name)
    ]

    if not still_missing:
        return

    # On some Python/macOS combinations, packages installed into the user site
    # are not visible until process startup. Restart once, then re-check.
    if os.environ.get("STYLOMETRY_DEPENDENCIES_INSTALLED") != "1":
        print("Restarting script so newly installed packages are available...")
        new_env = os.environ.copy()
        new_env["STYLOMETRY_DEPENDENCIES_INSTALLED"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], new_env)

    details = ", ".join(
        f"{import_name} ({requirements[import_name]})"
        for import_name in still_missing
    )
    raise RuntimeError(
        "The following packages are still missing after installation: " + details +
        "\nTry creating a virtual environment and running: python -m pip install -r requirements.txt"
    )


def read_requirements_file(requirements_path: str | Path = "requirements.txt") -> dict[str, str]:
    """Read requirements.txt and return {normalised_package_name: specifier_line}.

    Blank lines and comments are ignored. Inline comments are also removed for
    ordinary package specifiers. Editable installs, constraints and direct URL
    requirements are intentionally rejected because these teaching scripts only
    need simple PyPI package requirements.
    """
    path = Path(requirements_path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find requirements file: {path}\n"
            "Put requirements.txt in the same folder as stylometry_utils.py."
        )

    requirements: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if " #" in line:
            line = line.split(" #", 1)[0].strip()
        if line.startswith(("-", "--")) or " @ " in line:
            raise ValueError(
                f"Unsupported requirement on line {line_number} of {path}: {raw_line!r}. "
                "Use ordinary PyPI specifiers such as pandas>=2.0."
            )
        match = re.match(r"^([A-Za-z0-9_.-]+)", line)
        if not match:
            raise ValueError(f"Could not parse requirement on line {line_number} of {path}: {raw_line!r}")
        requirements[_normalise_package_name(match.group(1))] = line
    return requirements


def ensure_requirements(
    import_to_package: Mapping[str, str],
    requirements_path: str | Path = "requirements.txt",
    auto_install: bool = True,
) -> None:
    """Check/import dependencies using package versions from requirements.txt.

    Args:
        import_to_package: Mapping of import name -> package name as it appears
            in requirements.txt. Example: {"sklearn": "scikit-learn"}.
        requirements_path: Path to requirements.txt. Relative paths are resolved
            beside this utility module.
        auto_install: If True, install missing packages with the current Python.
    """
    requirements = read_requirements_file(requirements_path)
    resolved: dict[str, str] = {}
    missing_from_file: list[str] = []

    for import_name, package_name in import_to_package.items():
        normalised = _normalise_package_name(package_name)
        specifier = requirements.get(normalised)
        if specifier is None:
            missing_from_file.append(package_name)
        else:
            resolved[import_name] = specifier

    if missing_from_file:
        raise RuntimeError(
            "requirements.txt is missing entries for: " + ", ".join(missing_from_file)
        )

    ensure_dependencies(resolved, auto_install=auto_install)


def expand_path(path_like: str | Path) -> Path:
    """Expand ~ and return an absolute Path without requiring the path to exist."""
    return Path(path_like).expanduser().resolve()


def ensure_output_parent(path_like: str | Path) -> Path:
    """Create the parent folder for an output file and return the resolved path."""
    path = expand_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_texts_from_folder(folder_path: str | Path) -> dict[str, str]:
    """Load all .txt files in a folder as {filename: text}, sorted by filename."""
    folder = expand_path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"The folder '{folder}' does not exist.")
    if not folder.is_dir():
        raise NotADirectoryError(f"'{folder}' is not a folder.")

    texts: dict[str, str] = {}
    for file_path in sorted(folder.glob("*.txt")):
        texts[file_path.name] = file_path.read_text(encoding="utf-8", errors="replace")

    if not texts:
        raise FileNotFoundError(f"No .txt files were found in '{folder}'.")
    return texts


def tokenize_words(text: str) -> list[str]:
    """Tokenise text into Unicode alphanumeric word tokens and lowercase it.

    This replaces the old NLTK punkt dependency. For Burrows's Delta based on
    most frequent words, this avoids external tokenizer data while preserving the
    old script's core behaviour of lowercasing and removing punctuation.
    """
    return WORD_RE.findall(text.casefold())


def tokenise_corpus(texts: Mapping[str, str]) -> dict[str, list[str]]:
    """Tokenise a corpus, preserving filenames as keys."""
    return {name: tokenize_words(text) for name, text in texts.items()}


def compute_frequency_matrix(
    tokenised_texts: Mapping[str, Sequence[str]],
    mfw: int = 100,
    relative: bool = True,
    scale: float = 1000.0,
):
    """Compute a most-frequent-word matrix.

    Rows are words and columns are texts. By default, values are relative
    frequencies per 1,000 tokens, which is the safer default for unevenly sized
    texts. Set relative=False to reproduce the older raw-count behaviour.
    """
    import pandas as pd

    if mfw < 1:
        raise ValueError("mfw must be at least 1.")

    corpus_counts: Counter[str] = Counter()
    text_counts: dict[str, Counter[str]] = {}
    text_lengths: dict[str, int] = {}

    for name, tokens in tokenised_texts.items():
        counts = Counter(tokens)
        text_counts[name] = counts
        text_lengths[name] = len(tokens)
        corpus_counts.update(counts)

    if not corpus_counts:
        raise ValueError("The corpus contains no usable word tokens.")

    most_common_words = [word for word, _ in corpus_counts.most_common(mfw)]
    data: dict[str, list[float]] = {}
    for name, counts in text_counts.items():
        denominator = text_lengths[name]
        if relative:
            data[name] = [
                (counts[word] / denominator * scale) if denominator else 0.0
                for word in most_common_words
            ]
        else:
            data[name] = [float(counts[word]) for word in most_common_words]

    return pd.DataFrame(data, index=most_common_words).fillna(0.0)


def calculate_z_scores(frequency_matrix):
    """Standardise word frequencies across texts, safely handling zero variance."""
    import numpy as np

    means = frequency_matrix.mean(axis=1)
    stds = frequency_matrix.std(axis=1, ddof=0).replace(0, np.nan)
    return frequency_matrix.sub(means, axis=0).div(stds, axis=0).fillna(0.0)


def compute_burrows_delta(z_matrix):
    """Compute pairwise Burrows's Delta as mean absolute z-score difference."""
    import numpy as np
    import pandas as pd

    if z_matrix.shape[1] < 2:
        raise ValueError("At least two texts are required to compute pairwise distances.")

    vectors = z_matrix.to_numpy(dtype=float).T  # texts x features
    distances = np.mean(np.abs(vectors[:, None, :] - vectors[None, :, :]), axis=2)
    np.fill_diagonal(distances, 0.0)
    return pd.DataFrame(distances, index=z_matrix.columns, columns=z_matrix.columns)


def extract_groups(filenames: Iterable[str]) -> list[str]:
    """Extract group labels from filenames using the text before the first underscore."""
    groups: list[str] = []
    for filename in filenames:
        stem = Path(filename).stem
        groups.append(stem.split("_", 1)[0] if "_" in stem else stem)
    return groups


def make_group_colours(groups: Sequence[str], cmap_name: str = "tab10") -> dict[str, object]:
    """Return a deterministic group -> colour mapping using Matplotlib's new API."""
    import matplotlib

    unique_groups = sorted(set(groups))
    if not unique_groups:
        return {}

    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    denominator = max(len(unique_groups) - 1, 1)
    return {group: cmap(i / denominator) for i, group in enumerate(unique_groups)}


def save_dataframe_csv(dataframe, output_path: str | Path) -> Path:
    """Save a pandas DataFrame to CSV, creating parent directories as needed."""
    path = ensure_output_parent(output_path)
    dataframe.to_csv(path)
    return path


def chunk_text_by_words(text: str, chunk_words: int = 250) -> list[str]:
    """Split a document into rough word-count chunks for transformer embedding."""
    if chunk_words < 1:
        raise ValueError("chunk_words must be at least 1.")

    words = text.split()
    if not words:
        return []
    return [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), chunk_words)]
