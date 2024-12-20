import sys
import pathlib
from typing import TextIO


class _Tee:
    def __init__(self, files: tuple[TextIO]) -> None:
        self.files = files

    def write(self, msg: str) -> None:
        for f in self.files:
            f.write(msg)
            f.flush()

    def flush(self) -> None:
        for f in self.files:
            f.flush()


class Redirect:
    _NAME_TEMPLATE = "class_log_CNN{n}.txt"

    def __init__(self, result_path: pathlib.Path, bypass=False) -> None:
        if bypass:
            print("======== LOG BYPASS = TRUE ========")
        self.result_path = result_path
        self._original_out = sys.stdout
        self._n = self._get_next_test_case_number()
        result_file_path = self.result_path / self._NAME_TEMPLATE.format(n=self._n)
        self._files = [result_file_path.open("wt", encoding="utf-8")] if not bypass else []

    def __enter__(self) -> int:
        self._redirect_print_to_files(self._files)
        return self._n

    def __exit__(self, *args) -> None:
        sys.stdout.flush()
        sys.stdout = self._original_out
        for file in self._files[0:-1]:
            file.close()

    def _redirect_print_to_files(self, files: list[TextIO]) -> None:
        files.append(sys.stdout)
        sys.stdout = _Tee(tuple(files))

    def _get_next_test_case_number(self) -> int:
        n = 1
        for file in self.result_path.glob("class_log_CNN*.txt"):
            if file.is_file():
                if file.stat().st_size != 0:
                    n += 1
                else:
                    file.unlink()
        return n
