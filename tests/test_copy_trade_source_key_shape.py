import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _has_slice(node):
    for child in ast.walk(node):
        if isinstance(child, ast.Subscript) and isinstance(child.slice, ast.Slice):
            return True
    return False


def _bad_copy_trade_key_nodes(tree):
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            has_prefix = any(
                isinstance(value, ast.Constant)
                and isinstance(value.value, str)
                and "copy_trade:" in value.value
                for value in node.values
            )
            if has_prefix and any(
                isinstance(value, ast.FormattedValue) and _has_slice(value.value)
                for value in node.values
            ):
                offenders.append(node)
        elif isinstance(node, ast.BinOp):
            try:
                text = ast.unparse(node)
            except Exception:
                text = ""
            if "copy_trade:" in text and "[" in text and ":" in text:
                offenders.append(node)
    return offenders


def test_copy_trade_source_keys_do_not_slice_trader_addresses():
    offenders = []
    for path in [*(PROJECT_ROOT / "src").rglob("*.py"), *(PROJECT_ROOT / "scripts").rglob("*.py")]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in _bad_copy_trade_key_nodes(tree):
            offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")

    assert offenders == []

