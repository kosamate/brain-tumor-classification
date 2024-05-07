from __future__ import annotations
import sys
from typing import Tuple, TextIO, List
from pathlib import Path


class _Tee:
    def __init__(self, files: Tuple[TextIO]) -> None:
        self.files = files

    def write(self, msg: str) -> None:
        for f in self.files:
            f.write(msg)
            f.flush()

    def flush(self) -> None:
        for f in self.files:
            f.flush()


class Redirect:
    _RESULTS_PATH = Path("D:/Project/Dipterv/Results")
    _NAME_TEMPLATE = "class_log_CNN{n}.txt"

    def __init__(self, bypass=False) -> None:
        if bypass:
            print("======== LOG BYPASS = TRUE ========")
        self._original_out = sys.stdout
        self._n = self._get_next_test_case_number()
        result_file_path = self._RESULTS_PATH / self._NAME_TEMPLATE.format(n=self._n)
        self._files = [result_file_path.open("wt")] if not bypass else []

    def __enter__(self) -> int:
        self._redirect_print_to_files(self._files)
        return self._n

    def __exit__(self, *args) -> None:
        sys.stdout.flush()
        sys.stdout = self._original_out
        for file in self._files:
            file.close()

    def _redirect_print_to_files(self, files: List[TextIO]) -> None:
        files.append(sys.stdout)
        sys.stdout = _Tee(tuple(files))

    def _get_next_test_case_number(self) -> int:
        n = 1
        for file in self._RESULTS_PATH.glob("class_log_CNN*.txt"):
            if file.is_file():
                if file.stat().st_size != 0:
                    n += 1
                else:
                    file.unlink()
        return n
