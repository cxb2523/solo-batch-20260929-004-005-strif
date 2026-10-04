import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "StringTemplate",
    "MissingVariableError",
    "TemplateSyntaxError",
    "RecursionLimitError",
    "CompiledTemplate",
    "compile_template",
    "render",
    "replace",
    "template_hash",
    "MAX_RENDER_DEPTH",
]

# ---------------------------------------------------------------------------
# Template engine
#
# Syntax:
#   ${name}        variable substitution
#   $${            literal "${" (escape is resolved before conditionals)
#   $if{name}      conditional block: true iff variable is present and non-empty
#   $else          else branch (inside a conditional)
#   $end           end of conditional block
#
# Conditionals may be nested to arbitrary depth. When recurse=True is passed
# to render(), substituted values are themselves rendered as templates.
# ---------------------------------------------------------------------------

MAX_RENDER_DEPTH = 32


class MissingVariableError(KeyError):
    """Raised by render() (strict mode) when a variable is absent or None."""


class TemplateSyntaxError(ValueError):
    """Raised when a template has unbalanced or malformed tags."""


class RecursionLimitError(RuntimeError):
    """Raised when recursive expansion exceeds MAX_RENDER_DEPTH."""


@dataclass(frozen=True)
class _Text:
    text: str


@dataclass(frozen=True)
class _Escape:
    pass


@dataclass(frozen=True)
class _Var:
    name: str


@dataclass(frozen=True)
class _Cond:
    name: str
    when_true: tuple[Any, ...]
    when_false: tuple[Any, ...]


@dataclass(frozen=True)
class CompiledTemplate:
    template: str
    digest: str
    nodes: tuple[Any, ...] = field(default=())


def template_hash(template: str) -> str:
    """Stable content hash used as the compile-cache key."""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def compile_template(template: str) -> CompiledTemplate:
    """Parse a template string into a reusable compiled form."""
    if not isinstance(template, str):
        raise TemplateSyntaxError("Template must be a string")
    nodes = tuple(_parse_sequence(template, 0, 0)[0])
    return CompiledTemplate(template=template, digest=template_hash(template), nodes=nodes)


def _parse_sequence(template: str, start: int, level: int, open_pos: int | None = None):
    """Parse one block (the top level or one conditional branch pair).

    Returns (true-nodes, true-and-else-nodes-or-None, next-index). Returns a
    tuple of nodes plus the index just past the block terminator ($end at
    level 0 is unclosed; at deeper levels it belongs to the parent).
    """
    nodes_true: list[Any] = []
    nodes_false: list[Any] | None = None
    current = nodes_true
    out: list[str] = []
    text_start = start
    length = len(template)

    def flush(end: int) -> None:
        if end > text_start:
            out.append(template[text_start:end])

    i = start
    while i < length:
        ch = template[i]
        if ch != "$":
            i += 1
            continue
        nxt = template[i + 1] if i + 1 < length else ""

        # Escape beats everything else, including conditionals.
        if template.startswith("$${", i):
            flush(i)
            current.append(_Text("".join(out)))
            current.append(_Escape())
            out.clear()
            i += 3
            text_start = i
            continue

        if nxt == "{":
            name, j = _read_braced_name(template, i + 2)
            flush(i)
            current.append(_Text("".join(out)))
            current.append(_Var(name))
            out.clear()
            i = j
            text_start = i
            continue

        if nxt == "i" and template.startswith("$if{", i):
            name, j = _read_braced_name(template, i + 4)
            flush(i)
            current.append(_Text("".join(out)))
            out.clear()
            child_true, child_false, j = _parse_sequence(template, j, level + 1, i)
            current.append(_Cond(name, tuple(child_true), tuple(child_false or ())))
            i = j
            text_start = i
            continue

        if template.startswith("$else", i) and _at_tag_boundary(template, i + 5):
            if level == 0 or nodes_false is not None:
                raise TemplateSyntaxError(
                    f"Unexpected $else at position {i} (no matching open $if)"
                )
            flush(i)
            nodes_true.append(_Text("".join(out)))
            out.clear()
            nodes_false = []
            current = nodes_false
            i += 5
            text_start = i
            continue

        if template.startswith("$end", i) and _at_tag_boundary(template, i + 4):
            if level == 0:
                raise TemplateSyntaxError(f"Unexpected $end at position {i} (no matching open $if)")
            flush(i)
            current.append(_Text("".join(out)))
            out.clear()
            return nodes_true, nodes_false, i + 4

        # Unrecognized "$..." sequence: literal text.
        i += 1

    flush(length)
    current.append(_Text("".join(out)))
    if level > 0:
        pos = open_pos if open_pos is not None else 0
        raise TemplateSyntaxError(f"Unclosed $if block opened at position {pos}")
    return nodes_true, None, length


def _at_tag_boundary(template: str, i: int) -> bool:
    """Tags like $else/$end must not be followed by an identifier character."""
    if i >= len(template):
        return True
    ch = template[i]
    return not (ch.isalnum() or ch == "_")


def _read_braced_name(template: str, i: int) -> tuple[str, int]:
    """Read a `{name}` token starting at the char after the opening brace."""
    length = len(template)
    start = i
    while i < length and template[i] != "}":
        ch = template[i]
        if not (ch.isalnum() or ch == "_"):
            raise TemplateSyntaxError(f"Invalid variable name near position {start - 1}")
        i += 1
    if i >= length:
        raise TemplateSyntaxError(f"Unclosed brace opened at position {start - 1}")
    name = template[start:i]
    if not name:
        raise TemplateSyntaxError(f"Empty variable name at position {start - 1}")
    return name, i + 1


TraceSink = Callable[[dict[str, Any]], None]


class _Renderer:
    variables: Mapping[str, Any]
    strict: bool
    recurse: bool
    sink: TraceSink | None
    compiler: Callable[[str], CompiledTemplate]
    depth: int
    root_digest: str | None

    def __init__(
        self,
        variables: Mapping[str, Any],
        *,
        strict: bool,
        recurse: bool,
        sink: TraceSink | None,
        compiler: Callable[[str], CompiledTemplate] | None,
        depth: int = 0,
        root_digest: str | None = None,
    ) -> None:
        self.variables = variables
        self.strict = strict
        self.recurse = recurse
        self.sink = sink
        self.compiler = compiler or compile_template
        self.depth = depth
        self.root_digest = root_digest

    def emit(self, event: dict[str, Any]) -> None:
        if self.sink is not None:
            self.sink(event)

    def value_of(self, name: str) -> tuple[bool, Any]:
        if name not in self.variables:
            return False, None
        value = self.variables[name]
        if value is None:
            return False, None
        return True, value

    def render_nodes(self, nodes: Sequence[Any], level: int) -> str:
        parts: list[str] = []
        for node in nodes:
            if isinstance(node, _Text):
                parts.append(node.text)
            elif isinstance(node, _Escape):
                self.emit({"type": "escape", "literal": "${", "level": level})
                parts.append("${")
            elif isinstance(node, _Var):
                parts.append(self.render_var(node, level))
            elif isinstance(node, _Cond):
                parts.append(self.render_cond(node, level))
        return "".join(parts)

    def render_var(self, node: _Var, level: int) -> str:
        present, value = self.value_of(node.name)
        if not present:
            self.emit(
                {
                    "type": "variable",
                    "name": node.name,
                    "level": level,
                    "present": False,
                    "action": "error" if self.strict else "empty",
                }
            )
            if self.strict:
                raise MissingVariableError(node.name)
            return ""
        text = "" if value is None else str(value)
        self.emit(
            {
                "type": "variable",
                "name": node.name,
                "level": level,
                "present": True,
                "value": text,
            }
        )
        if self.recurse and text:
            return self.expand(text, node.name, level)
        return text

    def expand(self, text: str, via: str, level: int) -> str:
        next_depth = self.depth + 1
        if next_depth > MAX_RENDER_DEPTH:
            self.emit(
                {
                    "type": "recursion_limit",
                    "name": via,
                    "level": level,
                    "depth": next_depth,
                }
            )
            raise RecursionLimitError(
                f"recursive expansion exceeded depth {MAX_RENDER_DEPTH} via {via!r}"
            )
        compiled = self.compiler(text)
        return _Renderer(
            self.variables,
            strict=self.strict,
            recurse=True,
            sink=self.sink,
            compiler=self.compiler,
            depth=next_depth,
            root_digest=self.root_digest,
        ).render_nodes(compiled.nodes, level)

    def render_cond(self, node: _Cond, level: int) -> str:
        present, value = self.value_of(node.name)
        truthy = present and bool(value)
        self.emit(
            {
                "type": "condition",
                "name": node.name,
                "level": level,
                "present": present,
                "truthy": truthy,
                "decision": "then" if truthy else "else",
            }
        )
        chosen = node.when_true if truthy else node.when_false
        result = self.render_nodes(chosen, level + 1)
        self.emit(
            {
                "type": "condition_end",
                "name": node.name,
                "level": level,
                "decision": "then" if truthy else "else",
            }
        )
        return result


def render(
    template: str | CompiledTemplate,
    variables: Mapping[str, Any] | None = None,
    *,
    strict: bool = True,
    recurse: bool = False,
    sink: TraceSink | None = None,
    compiler: Callable[[str], CompiledTemplate] | None = None,
) -> str:
    """Render a template.

    strict=True (default): raise MissingVariableError on absent/None values.
    strict=False (i.e. replace() semantics): missing values become "".
    recurse=True: substituted values are themselves rendered as templates.
    sink: optional callback receiving ordered decision-trace events.
    compiler: optional compile hook (lets callers inject a cached compiler).
    """
    compiled = template if isinstance(template, CompiledTemplate) else compile_template(template)
    variables = dict(variables or {})
    renderer = _Renderer(
        variables,
        strict=strict,
        recurse=recurse,
        sink=sink,
        compiler=compiler,
        depth=0,
        root_digest=compiled.digest,
    )

    def emit(event: dict[str, Any]) -> None:
        if sink is not None:
            sink(event)

    emit({"type": "render_start", "digest": compiled.digest, "recurse": recurse})
    try:
        result = renderer.render_nodes(compiled.nodes, 0)
    except RecursionLimitError:
        emit({"type": "render_aborted", "digest": compiled.digest, "reason": "recursion_limit"})
        raise
    emit({"type": "render_end", "digest": compiled.digest})
    return result


def replace(
    template: str | CompiledTemplate,
    variables: Mapping[str, Any] | None = None,
    *,
    recurse: bool = False,
    sink: TraceSink | None = None,
    compiler: Callable[[str], CompiledTemplate] | None = None,
) -> str:
    """Like render() but missing variables are substituted with empty strings."""
    return render(
        template,
        variables,
        strict=False,
        recurse=recurse,
        sink=sink,
        compiler=compiler,
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
    """List of allowed field names. If `d` or `f` formats are used, give tuple with the type."""
    # Sequence is covariant so compatible with List[str]

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
                f"Invalid template (forgot to provide a type when using non-str format strings?): {e}"
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
                        f"Invalid type for '{f}': expected {expected_type.__name__} but got {repr(kwargs[f])} ({type(kwargs[f]).__name__})"
                    )

        return self.template.format(**kwargs)

    def __bool__(self) -> bool:
        return bool(self.template)
