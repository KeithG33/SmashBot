"""train_value trains a value net alone on the BC data and eval, checkpoints
it, and a resumed run continues exactly as an uninterrupted one."""
import dataclasses

from smashbot import train_value
from smashbot.tests.test_resume import _assert_same, _config, _latest


def test_value_only_run_evals_saves_and_resumes_exactly(tmp_path, capsys):
    whole = _config(str(tmp_path), "whole", 4)
    train_value.main(whole)
    out = capsys.readouterr().out
    assert out.count("eval_wide @") == 2 and "nan" not in out
    first = _config(str(tmp_path), "halves", 2)
    train_value.main(first)
    train_value.main(dataclasses.replace(first, runtime=dataclasses.replace(first.runtime, steps=4, restore="auto")))
    a, b = _latest(str(tmp_path), "whole"), _latest(str(tmp_path), "halves")
    assert a["step"] == b["step"] == 4 and "policy" not in a
    for key in ("value", "value_opt", "train_data", "value_hidden"):
        _assert_same(a[key], b[key], key)
