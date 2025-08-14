from __future__ import annotations

from collections.abc import (
    Callable,
    Hashable,
)
import json
from typing import (
    TYPE_CHECKING,
    Any,
)

from pandas.core.dtypes.common import is_scalar

from pandas.core.series import Series

if TYPE_CHECKING:
    from pandas import DataFrame


OP_SYMBOLS = {
    "__add__": "+",
    "__radd__": "+",
    "__sub__": "-",
    "__rsub__": "-",
    "__mul__": "*",
    "__rmul__": "*",
    "__truediv__": "/",
    "__rtruediv__": "/",
    "__floordiv__": "//",
    "__rfloordiv__": "//",
    "__ge__": ">=",
    "__gt__": ">",
    "__le__": "<=",
    "__lt__": "<",
    "__eq__": "==",
    "__ne__": "!=",
    "__mod__": "%",
}


def parse_args(df: DataFrame, *args: Any) -> tuple[Series]:
    return tuple([x._func(df) if isinstance(x, Expr) else x for x in args])


def parse_kwargs(df: DataFrame, **kwargs: Any) -> dict[Hashable, Series]:
    return {
        key: val._func(df) if isinstance(val, Expr) else val
        for key, val in kwargs.items()
    }


class Expr:
    def __init__(
        self, func: Callable[[DataFrame], Any], _expr_data: dict[str, Any] | None = None
    ) -> None:
        self._func = func
        # Store expression structure for serialization
        self._expr_data = _expr_data or {}

    def __call__(self, df: DataFrame) -> Series:
        result = self._func(df)
        if not (isinstance(result, Series) or is_scalar(result)):
            msg = (
                "Expected function which returns Series or scalar, "
                f"got function which returns: {type(result)}"
            )
            raise TypeError(msg)
        return result

    def _with_binary_op(self, op: str, other: Any) -> Expr:
        if isinstance(other, Expr):
            expr_data = {
                "expr_type": "binary_op",
                "op": op,
                "left": self._expr_data,
                "left_repr": self._generate_repr(),
                "right": other._expr_data,
                "right_repr": other._generate_repr(),
            }

            return Expr(
                lambda df: getattr(self._func(df), op)(other._func(df)), expr_data
            )
        else:
            expr_data = {
                "expr_type": "binary_op",
                "op": op,
                "left": self._expr_data,
                "left_repr": self._generate_repr(),
                "right_value": other,
            }

            return Expr(lambda df: getattr(self._func(df), op)(other), expr_data)

    # Binary ops
    def __add__(self, other: Any) -> Expr:
        return self._with_binary_op("__add__", other)

    def __radd__(self, other: Any) -> Expr:
        return self._with_binary_op("__radd__", other)

    def __sub__(self, other: Any) -> Expr:
        return self._with_binary_op("__sub__", other)

    def __rsub__(self, other: Any) -> Expr:
        return self._with_binary_op("__rsub__", other)

    def __mul__(self, other: Any) -> Expr:
        return self._with_binary_op("__mul__", other)

    def __rmul__(self, other: Any) -> Expr:
        return self._with_binary_op("__rmul__", other)

    def __truediv__(self, other: Any) -> Expr:
        return self._with_binary_op("__truediv__", other)

    def __rtruediv__(self, other: Any) -> Expr:
        return self._with_binary_op("__rtruediv__", other)

    def __floordiv__(self, other: Any) -> Expr:
        return self._with_binary_op("__floordiv__", other)

    def __rfloordiv__(self, other: Any) -> Expr:
        return self._with_binary_op("__rfloordiv__", other)

    def __ge__(self, other: Any) -> Expr:
        return self._with_binary_op("__ge__", other)

    def __gt__(self, other: Any) -> Expr:
        return self._with_binary_op("__gt__", other)

    def __le__(self, other: Any) -> Expr:
        return self._with_binary_op("__le__", other)

    def __lt__(self, other: Any) -> Expr:
        return self._with_binary_op("__lt__", other)

    def __eq__(self, other: object) -> Expr:  # type: ignore[override]
        return self._with_binary_op("__eq__", other)

    def __ne__(self, other: object) -> Expr:  # type: ignore[override]
        return self._with_binary_op("__ne__", other)

    def __mod__(self, other: Any) -> Expr:
        return self._with_binary_op("__mod__", other)

    # Everything else
    def __getattr__(self, attr: str, /) -> Callable[..., Expr]:
        def func(df: DataFrame, *args: Any, **kwargs: Any) -> Any:
            parsed_args = parse_args(df, *args)
            parsed_kwargs = parse_kwargs(df, **kwargs)
            return getattr(self(df), attr)(*parsed_args, **parsed_kwargs)

        def wrapper(*args: Any, **kwargs: Any) -> Expr:
            # Serialize args and kwargs
            serialized_args = []
            for arg in args:
                if isinstance(arg, Expr):
                    serialized_args.append(
                        {
                            "is_expr": True,
                            "data": arg._expr_data,
                            "repr": arg._generate_repr(),
                        }
                    )
                else:
                    serialized_args.append(arg)

            serialized_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, Expr):
                    serialized_kwargs[k] = {
                        "is_expr": True,
                        "data": v._expr_data,
                        "repr": v._generate_repr(),
                    }
                else:
                    serialized_kwargs[k] = v

            expr_data = {
                "expr_type": "method_call",
                "base": self._expr_data,
                "base_repr": self._generate_repr(),
                "method": attr,
                "args": serialized_args,
                "kwargs": serialized_kwargs,
            }

            return Expr(lambda df: func(df, *args, **kwargs), expr_data)

        return wrapper

    def _generate_repr(self) -> str:
        """Generate a string representation from expr_data."""
        if not self._expr_data:
            return "Expr(...)"

        expr_type = self._expr_data.get("expr_type")

        if expr_type == "col":
            col_name = self._expr_data["col_name"]
            return f"col({col_name!r})"

        elif expr_type == "binary_op":
            op = self._expr_data["op"]
            op_symbol = OP_SYMBOLS.get(op, op)

            left_repr = self._expr_data["left_repr"]

            if "right_repr" in self._expr_data:
                # Right operand is an Expr
                right_repr = self._expr_data["right_repr"]
                if op.startswith("__r"):
                    return f"({right_repr} {op_symbol} {left_repr})"
                else:
                    return f"({left_repr} {op_symbol} {right_repr})"
            else:
                # Right operand is a scalar
                right_value = self._expr_data["right_value"]
                if op.startswith("__r"):
                    return f"({right_value!r} {op_symbol} {left_repr})"
                else:
                    return f"({left_repr} {op_symbol} {right_value!r})"

        elif expr_type == "method_call":
            base_repr = self._expr_data["base_repr"]
            method = self._expr_data["method"]
            args = self._expr_data.get("args", [])
            kwargs = self._expr_data.get("kwargs", {})

            # Generate args representation
            args_repr = []
            for arg in args:
                if isinstance(arg, dict) and arg.get("is_expr"):
                    args_repr.append(arg["repr"])
                else:
                    args_repr.append(repr(arg))

            kwargs_repr = []
            for k, v in kwargs.items():
                if isinstance(v, dict) and v.get("is_expr"):
                    kwargs_repr.append(f"{k}={v['repr']}")
                else:
                    kwargs_repr.append(f"{k}={v!r}")

            all_args = args_repr + kwargs_repr
            args_str = ", ".join(all_args)
            return f"{base_repr}.{method}({args_str})"

        elif expr_type == "namespace_method":
            base_repr = self._expr_data["base_repr"]
            namespace = self._expr_data["namespace"]
            method = self._expr_data["method"]
            args = self._expr_data.get("args", [])
            kwargs = self._expr_data.get("kwargs", {})

            # Generate args representation
            args_repr = []
            for arg in args:
                if isinstance(arg, dict) and arg.get("is_expr"):
                    args_repr.append(arg["repr"])
                else:
                    args_repr.append(repr(arg))

            kwargs_repr = []
            for k, v in kwargs.items():
                if isinstance(v, dict) and v.get("is_expr"):
                    kwargs_repr.append(f"{k}={v['repr']}")
                else:
                    kwargs_repr.append(f"{k}={v!r}")

            all_args = args_repr + kwargs_repr
            args_str = ", ".join(all_args)
            return f"{base_repr}.{namespace}.{method}({args_str})"

        elif expr_type == "namespace_property":
            base_repr = self._expr_data["base_repr"]
            namespace = self._expr_data["namespace"]
            property_name = self._expr_data["property"]
            return f"{base_repr}.{namespace}.{property_name}"

        else:
            return "Expr(...)"

    def __repr__(self) -> str:
        return self._generate_repr()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the expression to a dictionary."""
        return {
            "type": "expr",
            "expr_data": self._expr_data,
        }

    def to_json(self) -> str:
        """Serialize the expression to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Expr:
        """Deserialize an expression from a dictionary."""
        if data["type"] != "expr":
            raise ValueError(f"Expected type 'expr', got {data['type']}")

        expr_data = data["expr_data"]

        return cls._build_expr_from_data(expr_data)

    @classmethod
    def from_json(cls, json_str: str) -> Expr:
        """Deserialize an expression from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def _build_expr_from_data(cls, expr_data: dict[str, Any]) -> Expr:
        """Rebuild the expression function from serialized data."""
        if not expr_data:
            # This shouldn't happen for properly serialized expressions
            raise ValueError("Cannot rebuild expression from empty data")

        expr_type = expr_data.get("expr_type")

        if expr_type == "col":
            col_name = expr_data["col_name"]
            return col(col_name)

        elif expr_type == "binary_op":
            left_expr = cls._build_expr_from_data(expr_data["left"])
            op = expr_data["op"]

            if "right" in expr_data and "right_repr" in expr_data:
                # Right operand is an Expr
                right_expr = cls._build_expr_from_data(expr_data["right"])
                return getattr(left_expr, op)(right_expr)
            else:
                # Right operand is a scalar
                right_value = expr_data["right_value"]
                return getattr(left_expr, op)(right_value)

        elif expr_type == "method_call":
            base_expr = cls._build_expr_from_data(expr_data["base"])
            method = expr_data["method"]
            args = expr_data.get("args", [])
            kwargs = expr_data.get("kwargs", {})

            # Rebuild args that are Exprs
            rebuilt_args = []
            for arg in args:
                if isinstance(arg, dict) and arg.get("is_expr"):
                    rebuilt_args.append(cls._build_expr_from_data(arg["data"]))
                else:
                    rebuilt_args.append(arg)

            # Rebuild kwargs that are Exprs
            rebuilt_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, dict) and v.get("is_expr"):
                    rebuilt_kwargs[k] = cls._build_expr_from_data(v["data"])
                else:
                    rebuilt_kwargs[k] = v

            return getattr(base_expr, method)(*rebuilt_args, **rebuilt_kwargs)

        elif expr_type == "namespace_method":
            base_expr = cls._build_expr_from_data(expr_data["base"])
            namespace = expr_data["namespace"]
            method = expr_data["method"]
            args = expr_data.get("args", [])
            kwargs = expr_data.get("kwargs", {})

            # Rebuild args that are Exprs
            rebuilt_args = []
            for arg in args:
                if isinstance(arg, dict) and arg.get("is_expr"):
                    rebuilt_args.append(cls._build_expr_from_data(arg["data"]))
                else:
                    rebuilt_args.append(arg)

            # Rebuild kwargs that are Exprs
            rebuilt_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, dict) and v.get("is_expr"):
                    rebuilt_kwargs[k] = cls._build_expr_from_data(v["data"])
                else:
                    rebuilt_kwargs[k] = v

            namespace_expr = getattr(base_expr, namespace)
            return getattr(namespace_expr, method)(*rebuilt_args, **rebuilt_kwargs)

        elif expr_type == "namespace_property":
            base_expr = cls._build_expr_from_data(expr_data["base"])
            namespace = expr_data["namespace"]
            property_name = expr_data["property"]

            namespace_expr = getattr(base_expr, namespace)
            return getattr(namespace_expr, property_name)

        else:
            raise ValueError(f"Unknown expression type: {expr_type}")

    # Namespaces
    @property
    def dt(self) -> NamespaceExpr:
        return NamespaceExpr(self, "dt")

    @property
    def str(self) -> NamespaceExpr:
        return NamespaceExpr(self, "str")

    @property
    def cat(self) -> NamespaceExpr:
        return NamespaceExpr(self, "cat")

    @property
    def list(self) -> NamespaceExpr:
        return NamespaceExpr(self, "list")

    @property
    def sparse(self) -> NamespaceExpr:
        return NamespaceExpr(self, "sparse")

    @property
    def struct(self) -> NamespaceExpr:
        return NamespaceExpr(self, "struct")


class NamespaceExpr:
    def __init__(self, func: Expr, namespace: str) -> None:
        self._func = func
        self._namespace = namespace

    def __getattr__(self, attr: str) -> Any:
        if isinstance(getattr(getattr(Series, self._namespace), attr), property):
            expr_data = {
                "expr_type": "namespace_property",
                "base": self._func._expr_data,
                "base_repr": self._func._generate_repr(),
                "namespace": self._namespace,
                "property": attr,
            }

            return Expr(
                lambda df: getattr(getattr(self._func(df), self._namespace), attr),
                expr_data,
            )

        def func(df: DataFrame, *args: Any, **kwargs: Any) -> Any:
            parsed_args = parse_args(df, *args)
            parsed_kwargs = parse_kwargs(df, **kwargs)
            return getattr(getattr(self._func(df), self._namespace), attr)(
                *parsed_args, **parsed_kwargs
            )

        def wrapper(*args: Any, **kwargs: Any) -> Expr:
            # Serialize args and kwargs
            serialized_args = []
            for arg in args:
                if isinstance(arg, Expr):
                    serialized_args.append(
                        {
                            "is_expr": True,
                            "data": arg._expr_data,
                            "repr": arg._generate_repr(),
                        }
                    )
                else:
                    serialized_args.append(arg)

            serialized_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, Expr):
                    serialized_kwargs[k] = {
                        "is_expr": True,
                        "data": v._expr_data,
                        "repr": v._generate_repr(),
                    }
                else:
                    serialized_kwargs[k] = v

            expr_data = {
                "expr_type": "namespace_method",
                "base": self._func._expr_data,
                "base_repr": self._func._generate_repr(),
                "namespace": self._namespace,
                "method": attr,
                "args": serialized_args,
                "kwargs": serialized_kwargs,
            }

            return Expr(lambda df: func(df, *args, **kwargs), expr_data)

        return wrapper


def col(col_name: Hashable) -> Expr:
    if not isinstance(col_name, Hashable):
        msg = f"Expected Hashable, got: {type(col_name)}"
        raise TypeError(msg)

    expr_data = {
        "expr_type": "col",
        "col_name": col_name,
    }

    return Expr(lambda df: df[col_name], expr_data)
