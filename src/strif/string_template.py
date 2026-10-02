import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "StringTemplate",
    "MissingVariableError",
    "TemplateSyntaxError",
    "RecursionLimitError",
    "CompiledTemplate",
    "compile_template",
    "render_template",
    "replace_template",
    "template_hash",
    "DEFAULT_MAX_DEPTH",
]


class TemplateSyntaxError(ValueError):
    """Raised when a template cannot be parsed (bad token, unclosed block, ...)."""

    def __init__(self, message: str, pos: int | None = None) -> None:
        if pos is not None:
            message = f"{message} (at position {pos})"
        super().__init__(message)
        self.pos = pos


class MissingVariableError(KeyError):
    """Strict render mode: a referenced variable was not supplied."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name

    def __str__(self) -> str:
        return f"missing template variable: {self.name!r}"


class RecursionLimitError(RuntimeError):
    """Recursive expansion exceeded the depth limit (e.g. self reference)."""

    def __init__(self, name: str, depth: int) -> None:
        super().__init__(f"recursion limit of {depth} exceeded while expanding {name!r}")
        self.name = name
        self.depth = depth


_VARIABLE_OPEN = "${"
_ESCAPED_OPEN = "$${"
_IF_OPEN = "${?"
_ELSE_TOKEN = "${|}"
_END_TOKEN = "${/}"
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

DEFAULT_MAX_DEPTH = 32


@dataclass(frozen=True)
class _TextNode:
    text: str
    pos: int


@dataclass(frozen=True)
class _EscapeNode:
    """A ``$${`` sequence, decided before any block parsing: renders as ``${``."""

    pos: int


@dataclass(frozen=True)
class _VariableNode:
    name: str
    pos: int


@dataclass(frozen=True)
class _ConditionNode:
    name: str
    then_nodes: tuple[Any, ...]
    else_nodes: tuple[Any, ...]
    open_pos: int
    else_pos: int | None
    close_pos: int


@dataclass(frozen=True)
class CompiledTemplate:
    template: str
    hash: str
    nodes: tuple[Any, ...]


def template_hash(template: str) -> str:
    """Content hash. Changing the template text changes the hash (cache invalidation)."""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def compile_template(template: str) -> CompiledTemplate:
    """Parse a template into a reusable, hash-addressed compiled form."""
    if not isinstance(template, str):
        raise TemplateSyntaxError("template must be a string")
    nodes, pos, stop = _parse_block(template, 0, level=0)
    if stop == "pipe":
        raise TemplateSyntaxError(f"unexpected {_ELSE_TOKEN!r} without matching condition", pos)
    if stop == "slash":
        raise TemplateSyntaxError(f"unexpected {_END_TOKEN!r} without matching condition", pos)
    return CompiledTemplate(template=template, hash=template_hash(template), nodes=tuple(nodes))


def _parse_block(template: str, pos: int, level: int) -> tuple[list[Any], int, str]:
    """Parse one nesting level. Returns (nodes, end_pos, stop_reason)."""
    nodes: list[Any] = []
    text_start = pos
    total = len(template)

    def push_text(end: int) -> None:
        if end > text_start:
            nodes.append(_TextNode(template[text_start:end], text_start))

    while pos < total:
        # Escape is always decided first, ahead of variables and condition blocks.
        if template.startswith(_ESCAPED_OPEN, pos):
            push_text(pos)
            nodes.append(_EscapeNode(pos))
            pos += len(_ESCAPED_OPEN)
            text_start = pos
        elif template.startswith(_IF_OPEN, pos):
            push_text(pos)
            open_pos = pos
            match = _IDENT_RE.match(template, pos + len(_IF_OPEN))
            if not match:
                raise TemplateSyntaxError("expected a variable name after '${?'", pos)
            name = match.group(0)
            pos = match.end()
            if pos >= total or template[pos] != "}":
                raise TemplateSyntaxError(f"missing '}}' in condition for {name!r}", pos)
            pos += 1

            then_nodes, pos, stop = _parse_block(template, pos, level + 1)
            else_nodes: list[Any] = []
            else_pos: int | None = None
            if stop == "pipe":
                else_pos = pos
                pos += len(_ELSE_TOKEN)
                else_nodes, pos, stop = _parse_block(template, pos, level + 1)
            if stop != "slash":
                raise TemplateSyntaxError(f"unclosed condition block for {name!r}", open_pos)
            close_pos = pos
            pos += len(_END_TOKEN)
            nodes.append(
                _ConditionNode(
                    name=name,
                    then_nodes=tuple(then_nodes),
                    else_nodes=tuple(else_nodes),
                    open_pos=open_pos,
                    else_pos=else_pos,
                    close_pos=close_pos,
                )
            )
            text_start = pos
        elif template.startswith(_ELSE_TOKEN, pos):
            if level == 0:
                raise TemplateSyntaxError(
                    f"unexpected {_ELSE_TOKEN!r} without matching condition", pos
                )
            push_text(pos)
            return nodes, pos, "pipe"
        elif template.startswith(_END_TOKEN, pos):
            if level == 0:
                raise TemplateSyntaxError(
                    f"unexpected {_END_TOKEN!r} without matching condition", pos
                )
            push_text(pos)
            return nodes, pos, "slash"
        elif template.startswith(_VARIABLE_OPEN, pos):
            push_text(pos)
            var_pos = pos
            match = _IDENT_RE.match(template, pos + len(_VARIABLE_OPEN))
            if not match:
                raise TemplateSyntaxError("expected a variable name after '${'", pos)
            name = match.group(0)
            pos = match.end()
            if pos >= total or template[pos] != "}":
                raise TemplateSyntaxError(f"missing '}}' after variable {name!r}", pos)
            pos += 1
            nodes.append(_VariableNode(name, var_pos))
            text_start = pos
        else:
            pos += 1

    push_text(total)
    return nodes, pos, "end"


@dataclass(frozen=True)
class _RenderContext:
    variables: dict[str, Any]
    strict: bool
    recurse: bool
    max_depth: int
    emit: Callable[[dict[str, Any]], None]
    compile: Callable[[str], CompiledTemplate]
    depth: int = 0


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _is_truthy(present: bool, value: Any) -> bool:
    # True iff the variable has a value and that value is non-empty.
    return present and bool(value)


def _eval_nodes(nodes: tuple[Any, ...] | list[Any], ctx: _RenderContext, level: int) -> str:
    parts: list[str] = []
    for node in nodes:
        if isinstance(node, _TextNode):
            parts.append(node.text)
            ctx.emit({"type": "text", "text": node.text, "pos": node.pos, "level": level})
        elif isinstance(node, _EscapeNode):
            parts.append(_VARIABLE_OPEN)
            ctx.emit(
                {
                    "type": "escape",
                    "raw": _ESCAPED_OPEN,
                    "literal": _VARIABLE_OPEN,
                    "pos": node.pos,
                    "level": level,
                }
            )
        elif isinstance(node, _VariableNode):
            parts.append(_eval_variable(node, ctx, level))
        elif isinstance(node, _ConditionNode):
            parts.append(_eval_condition(node, ctx, level))
    return "".join(parts)


def _eval_variable(node: _VariableNode, ctx: _RenderContext, level: int) -> str:
    present = node.name in ctx.variables
    if not present:
        if ctx.strict:
            raise MissingVariableError(node.name)
        ctx.emit(
            {
                "type": "variable",
                "name": node.name,
                "value": "",
                "present": False,
                "blank": True,
                "expanded": False,
                "pos": node.pos,
                "level": level,
                "depth": ctx.depth,
            }
        )
        return ""

    value = _stringify(ctx.variables[node.name])
    expanded = False
    if ctx.recurse and _VARIABLE_OPEN in value:
        next_depth = ctx.depth + 1
        if next_depth > ctx.max_depth:
            raise RecursionLimitError(node.name, ctx.max_depth)
        sub_compiled = ctx.compile(value)
        ctx.emit(
            {
                "type": "recurse_enter",
                "name": node.name,
                "pos": node.pos,
                "level": level,
                "depth": next_depth,
            }
        )
        sub_ctx = _RenderContext(
            variables=ctx.variables,
            strict=ctx.strict,
            recurse=ctx.recurse,
            max_depth=ctx.max_depth,
            emit=ctx.emit,
            compile=ctx.compile,
            depth=next_depth,
        )
        value = _eval_nodes(sub_compiled.nodes, sub_ctx, 0)
        ctx.emit(
            {
                "type": "recurse_exit",
                "name": node.name,
                "pos": node.pos,
                "level": level,
                "depth": next_depth,
            }
        )
        expanded = True

    ctx.emit(
        {
            "type": "variable",
            "name": node.name,
            "value": value,
            "present": True,
            "blank": value == "",
            "expanded": expanded,
            "pos": node.pos,
            "level": level,
            "depth": ctx.depth,
        }
    )
    return value


def _eval_condition(node: _ConditionNode, ctx: _RenderContext, level: int) -> str:
    condition_level = level + 1
    present = node.name in ctx.variables
    truthy = _is_truthy(present, ctx.variables.get(node.name))
    ctx.emit(
        {
            "type": "condition_open",
            "name": node.name,
            "truthy": truthy,
            "present": present,
            "branch": "then" if truthy else ("else" if node.else_pos is not None else None),
            "pos": node.open_pos,
            "level": condition_level,
            "depth": ctx.depth,
        }
    )

    parts: list[str] = []
    if truthy:
        parts.append(_eval_nodes(node.then_nodes, ctx, condition_level))
        if node.else_pos is not None:
            ctx.emit(
                {
                    "type": "condition_else",
                    "taken": False,
                    "pos": node.else_pos,
                    "level": condition_level,
                    "depth": ctx.depth,
                }
            )
    elif node.else_pos is not None:
        ctx.emit(
            {
                "type": "condition_else",
                "taken": True,
                "pos": node.else_pos,
                "level": condition_level,
                "depth": ctx.depth,
            }
        )
        parts.append(_eval_nodes(node.else_nodes, ctx, condition_level))

    ctx.emit(
        {
            "type": "condition_close",
            "name": node.name,
            "pos": node.close_pos,
            "level": condition_level,
            "depth": ctx.depth,
        }
    )
    return "".join(parts)


def _render(
    compiled: CompiledTemplate,
    variables: dict[str, Any] | None,
    *,
    strict: bool,
    recurse: bool,
    max_depth: int,
    emit: Callable[[dict[str, Any]], None] | None,
    compile_fn: Callable[[str], CompiledTemplate] | None,
    depth: int,
) -> str:
    compile_fn = compile_fn or compile_template
    ctx = _RenderContext(
        variables=dict(variables or {}),
        strict=strict,
        recurse=recurse,
        max_depth=max_depth,
        emit=emit or (lambda _event: None),
        compile=compile_fn,
        depth=depth,
    )
    return _eval_nodes(compiled.nodes, ctx, 0)


def render_template(
    template: str | CompiledTemplate,
    variables: dict[str, Any] | None = None,
    *,
    recurse: bool = False,
    max_depth: int = DEFAULT_MAX_DEPTH,
    emit: Callable[[dict[str, Any]], None] | None = None,
    compile_fn: Callable[[str], CompiledTemplate] | None = None,
) -> str:
    """
    Render a compiled template (or template string).

    Strict mode: a missing variable raises `MissingVariableError`.
    Recursive expansion of variables containing template syntax only happens
    when ``recurse=True``; self references raise `RecursionLimitError`.
    ``emit``, if given, receives the ordered decision trace (one dict per token).
    """
    compile_fn = compile_fn or compile_template
    compiled = compile_fn(template) if isinstance(template, str) else template
    return _render(
        compiled,
        variables,
        strict=True,
        recurse=recurse,
        max_depth=max_depth,
        emit=emit,
        compile_fn=compile_fn,
        depth=0,
    )


def replace_template(
    template: str | CompiledTemplate,
    variables: dict[str, Any] | None = None,
    *,
    recurse: bool = False,
    max_depth: int = DEFAULT_MAX_DEPTH,
    emit: Callable[[dict[str, Any]], None] | None = None,
    compile_fn: Callable[[str], CompiledTemplate] | None = None,
) -> str:
    """Like `render_template`, but missing variables are replaced with empty strings."""
    compile_fn = compile_fn or compile_template
    compiled = compile_fn(template) if isinstance(template, str) else template
    return _render(
        compiled,
        variables,
        strict=False,
        recurse=recurse,
        max_depth=max_depth,
        emit=emit,
        compile_fn=compile_fn,
        depth=0,
    )


@dataclass(frozen=True)
class StringTemplate:
    """
    A validated template string that supports only specified fields.
    Can subclass to have a type with a given set of `allowed_fields`.
    Provide a type with a field name to allow validation of int/float format strings.

    Examples:
    >>> t = StringTemplate("{name} is {age} years old", ["name", "age"])
    >>> t.format(name="Alice", age=30)
    'Alice is 30 years old'

    >>> t = StringTemplate("{count:3d}@{price:.2f}", [("count", int), ("price", float)])
    >>> t.format(count=10, price=19.99)
    ' 10@19.99'
    """

    template: str

    allowed_fields: Sequence[str | tuple[str, type | None]]
    """List of allowed fields. If `d` or `f` formats are used, give tuple with the type."""

    strict: bool = False
    """If True, raise a ValueError if the template is missing an allowed field."""

    def __post_init__(self):
        if not isinstance(self.template, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError("Template must be a string")

        # Confirm only the allowed fields are in the template.
        field_types = self._field_types()
        try:
            placeholder_values = {field: (type or str)(123) for field, type in field_types.items()}
            self.template.format(**placeholder_values)
        except KeyError as e:
            raise ValueError(f"Template contains unsupported variable: {e}") from None
        except ValueError as e:
            raise ValueError(
                f"Invalid template (forgot to provide a type when using non-str format "
                f"strings?): {e}"
            ) from None

    def _field_types(self) -> dict[str, type | None]:
        return {
            field[0] if isinstance(field, tuple) else field: (
                field[1] if isinstance(field, tuple) else None
            )
            for field in self.allowed_fields
        }

    def format(self, **kwargs: Any) -> str:
        field_types = self._field_types()
        allowed_keys = field_types.keys()
        unexpected_keys = set(kwargs.keys()) - allowed_keys
        if self.strict and unexpected_keys:
            raise ValueError(f"Unexpected keyword arguments: {', '.join(unexpected_keys)}")

        # Type check the values, if types were provided.
        for f, expected_type in field_types.items():
            if f in kwargs and expected_type:
                if not isinstance(kwargs[f], expected_type):
                    raise ValueError(
                        f"Invalid type for '{f}': expected {expected_type.__name__} but got "
                        f"{repr(kwargs[f])} ({type(kwargs[f]).__name__})"
                    )

        return self.template.format(**kwargs)

    def __bool__(self) -> bool:
        return bool(self.template)
