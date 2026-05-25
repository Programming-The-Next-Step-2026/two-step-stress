"""CSV trial logging and the session-metadata JSON sidecar for the two-step task.

One CSV per participant, one row per trial, flushed after every row so an
interrupted session keeps all completed trials.  The companion
``..._session.json`` sidecar stores run-level metadata (RNG seed, key mapping,
refresh rate, …).  File naming never overwrites an existing participant file: a
numeric suffix is appended instead.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

# CSV schema — column order is the on-disk order.  Matches docs/psychopy_plan.md
# §6 and the data-schema section of CLAUDE.md verbatim.
TRIAL_COLUMNS: tuple[str, ...] = (
    "participant_id",
    "block",
    "block_type",
    "trial_in_block",
    "trial_global",
    "stage1_choice",
    "stage1_rt",
    "transition",
    "stage2_state",
    "stage2_choice",
    "stage2_rt",
    "reward",
    "reward_probs",
    "nback_letter",
    "nback_is_match",
    "nback_response",
    "nback_correct",
    "nback_rt",
    "timestamp",
)

# Sentinel written for any missing or null field.
NA: str = "NA"


def _resolve_unique_path(path: Path) -> Path:
    """Return ``path`` unchanged, or with a numeric suffix if it already exists.

    The first collision becomes ``stem_1``, the next ``stem_2``, and so on, so a
    participant's earlier data is never overwritten.

    Parameters
    ----------
    path : pathlib.Path
        The desired output path.

    Returns
    -------
    pathlib.Path
        A path that does not currently exist on disk.
    """
    if not path.exists():
        return path
    i = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def _format_value(column: str, value: Any) -> Any:
    """Coerce one cell value to its on-disk representation.

    ``None`` becomes the ``NA`` sentinel; ``reward_probs`` is JSON-encoded as a
    list of floats; everything else is passed through and stringified by the
    csv writer.

    Parameters
    ----------
    column : str
        The column name (only ``reward_probs`` gets special handling).
    value : Any
        The raw value from the trial dict.

    Returns
    -------
    Any
        A value safe to hand to ``csv.DictWriter``.
    """
    if value is None:
        return NA
    if column == "reward_probs":
        if isinstance(value, str):  # already serialised, or the NA sentinel
            return value
        return json.dumps([float(x) for x in value])
    return value


class TrialWriter:
    """Incremental, no-overwrite CSV writer for one participant's trials.

    Opens ``<output_dir>/sub-<participant_id>_<timestamp>.csv`` (with a numeric
    suffix if that name is taken), writes the header immediately, and flushes
    after every row.  Use :meth:`write_session_json` for the metadata sidecar.

    May be used as a context manager so the file is closed cleanly on exit
    (including on an Escape-triggered teardown).

    Parameters
    ----------
    participant_id : str
        Participant identifier; used verbatim in the filename (the caller is
        responsible for filename-safety).
    timestamp : str
        Session timestamp string, used verbatim in the filename.
    output_dir : str or pathlib.Path, optional
        Directory for the CSV and JSON files.  Created if absent.  Defaults to
        ``data/raw`` (gitignored).

    Attributes
    ----------
    path : pathlib.Path
        The resolved CSV path actually written to (may carry a numeric suffix).
    """

    def __init__(
        self,
        participant_id: str,
        timestamp: str,
        output_dir: str | Path = Path("data/raw"),
    ) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        desired = out / f"sub-{participant_id}_{timestamp}.csv"
        self.path: Path = _resolve_unique_path(desired)

        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=TRIAL_COLUMNS)
        self._writer.writeheader()
        self._file.flush()
        logger.info("Logging trials to %s", self.path)

    def write_row(self, row: Mapping[str, Any]) -> None:
        """Write one trial row and flush.

        Keys absent from ``row`` are logged as ``NA``; keys outside
        :data:`TRIAL_COLUMNS` are ignored.

        Parameters
        ----------
        row : Mapping[str, Any]
            Trial values keyed by column name.
        """
        formatted = {
            col: _format_value(col, row.get(col)) for col in TRIAL_COLUMNS
        }
        self._writer.writerow(formatted)
        self._file.flush()

    @property
    def session_json_path(self) -> Path:
        """Path of the session JSON sidecar (CSV stem + ``_session.json``)."""
        return self.path.with_name(f"{self.path.stem}_session.json")

    def write_session_json(self, metadata: Mapping[str, Any]) -> Path:
        """Write run-level metadata to the JSON sidecar.

        Parameters
        ----------
        metadata : Mapping[str, Any]
            Session metadata (RNG seed, key mapping, refresh rate, …).
            Non-JSON-native values are stringified via ``default=str``.

        Returns
        -------
        pathlib.Path
            The path written.
        """
        path = self.session_json_path
        with path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info("Wrote session metadata to %s", path)
        return path

    def close(self) -> None:
        """Close the CSV file if it is still open."""
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "TrialWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
