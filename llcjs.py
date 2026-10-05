#!/usr/bin/env python3
"""LLCJS -- LL-to-JavaScript demo emitter.

Compiles an LL source file to a JavaScript file in the form expected by
the companion website's runtime (``LLP.mount(container, spec)``).

This emitter handles the LLP patterns used by the chapter-1 to chapter-4
demos:

* arbitrary number of input parameters of types ``int``, ``int[]``, or
  ``int[][]``;
* one optional state variable (``int`` scalar or ``int[]`` array);
* in-place algorithms with no state variable (the input array is used as
  the rendered vector);
* forbidden predicates that are either a plain Boolean expression or an
  ``exists`` quantifier (with optional ``pick``);
* ``advance`` bodies built from ``=``, ``swap``, ``if``/``else``, and
  block sequences;
* an optional sidecar ``<name>.demo.json`` file controlling UI metadata
  (container id, input labels/defaults, custom schedulers, describe
  text, etc.);
* ``setMode(PriorityQueue)`` / ``setPriority(expr, min|max)`` scheduling --
  compiled to a ``nextIndex(G, ctx)`` that scans for the extremal-priority
  forbidden index, since there is no real decrease-key heap at runtime.

Constructs we don't yet handle (par, other ``setMode`` targets such as
``LinkedList``/``Deque``, ensure clauses, nested ``exists`` over both
quantifier variables) raise ``NotImplementedError`` so the build script
falls back gracefully.

Usage:
    python3 llcjs.py FILE.llp [--out FILE.js]
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llc import tokenize, Parser, Program, Decl, Stmt, Expr, Type  # noqa: E402


# ============================================================
#  Expression translation
# ============================================================

JS_BIN = {
    '+': '+', '-': '-', '*': '*', '/': '/', '%': '%',
    '<': '<', '<=': '<=', '>': '>', '>=': '>=',
    '=': '===', '==': '===', '!=': '!==',
    'and': '&&', 'or': '||',
}


def _js_str(s: str) -> str:
    """Escape a Python string so it can sit inside a single-quoted JS literal."""
    return (s.replace('\\', '\\\\')
             .replace("'", "\\'")
             .replace('\n', '\\n'))


def _render_new_array_dims(sizes: List[str], fill: str) -> str:
    """Build nested `Array.from(...)` construction for an N-dimensional
    `new T[s0][s1]...[sN]` allocation (sizes already translated to JS text,
    N >= 2). A 4D array is just a 2D array of 2D arrays: each additional
    dimension is one more level of Array.from nesting around the same
    innermost `new Array(...).fill(fill)`."""
    if len(sizes) == 2:
        return f'Array.from({{ length: {sizes[0]} }}, () => new Array({sizes[1]}).fill({fill}))'
    inner = _render_new_array_dims(sizes[1:], fill)
    return f'Array.from({{ length: {sizes[0]} }}, () => {inner})'


def expr_to_js(e: Expr, ctx: 'EmitContext') -> str:
    """Translate an LL expression to a JS expression. ``ctx`` carries the
    name-to-JS mapping for input/var references and indexing offsets."""
    k = e.kind
    if k == 'int':
        return str(e.value)
    if k == 'real':
        return repr(e.value)
    if k == 'bool':
        return 'true' if e.value else 'false'
    if k == 'infinity':
        return 'Infinity'
    if k == 'id':
        return ctx.id_to_js(e.name)
    if k == 'index':
        # The LL DSL uses 1-based array indices like A[1..n]. In ordinary
        # (0-indexed) emit mode we shift literal indices >= 1 down by one;
        # quantifier vars are already emitted as 0-indexed (their range
        # [1..n] becomes [0..n-1] at the loop level). In 1-indexed mode the
        # input/G arrays are padded with a sentinel at slot 0, so no shift.
        idx = e.idx
        if (not ctx.one_indexed) and idx.kind == 'int' and idx.value >= 1:
            shifted = str(idx.value - 1)
            return f'{ctx.id_to_js(e.name)}[{shifted}]'
        return f'{ctx.id_to_js(e.name)}[{expr_to_js(idx, ctx)}]'
    if k == 'index2':
        def _emit_idx(x):
            if (not ctx.one_indexed) and x.kind == 'int' and x.value >= 1:
                return str(x.value - 1)
            return expr_to_js(x, ctx)
        chain = ''.join(f'[{_emit_idx(x)}]' for x in [e.idx, e.rhs, *e.idx_extra])
        return f'{ctx.id_to_js(e.name)}{chain}'
    if k == 'method_call':
        if e.name == 'clone':
            return ''  # consumed at the decl-init site (see _emit_buildState)
        raise NotImplementedError(f'method_call {e.name!r} not supported in JS emit')
    if k == 'call':
        args = [expr_to_js(a, ctx) for a in e.args]
        if e.name == 'min' and len(args) == 2:
            return f'Math.min({args[0]}, {args[1]})'
        if e.name == 'max' and len(args) == 2:
            return f'Math.max({args[0]}, {args[1]})'
        if e.name == 'abs' and len(args) == 1:
            return f'Math.abs({args[0]})'
        # A call to one of the class's own auxiliary methods (see
        # JSEmitter._render_aux_methods, which emits each as a top-level JS
        # function so it's in scope here). Every such call carries an
        # implicit trailing `ctx` -- see EmitContext.aux_names.
        if e.name in ctx.aux_names:
            args = args + ['ctx']
        return f'{e.name}({", ".join(args)})'
    if k == 'field_access':
        if e.name == 'length':
            return f'{expr_to_js(e.lhs, ctx)}.length'
        raise NotImplementedError(f'field_access {e.name!r} not supported in JS emit')
    if k == 'bin':
        if e.op == 'in':
            # X in Y: set/array membership -- Y is always a plain JS array
            # here (a `set<int>` param, or a quantifier range), so this is
            # just Array.includes, not an infix operator.
            return f'{expr_to_js(e.rhs, ctx)}.includes({expr_to_js(e.lhs, ctx)})'
        op = JS_BIN.get(e.op)
        if op is None:
            raise NotImplementedError(f'binary operator {e.op!r}')
        return f'({expr_to_js(e.lhs, ctx)} {op} {expr_to_js(e.rhs, ctx)})'
    if k == 'un':
        if e.op in ('not', '!'):
            return f'!({expr_to_js(e.lhs, ctx)})'
        if e.op in ('neg', '-'):
            return f'-({expr_to_js(e.lhs, ctx)})'
    if k == 'quant':
        if e.quant in ('min', 'max', 'sum', 'count'):
            return _render_reduction_quant(e, ctx)
        if e.quant in ('exists', 'forall'):
            return _render_bool_quant(e, ctx)
        raise NotImplementedError(
            f'quantifier {e.quant!r} as sub-expression')
    if k == 'new_array':
        # Inside an auxiliary method (unlike JSEmitter._classify_init,
        # which handles the top-level state-var decl's own `new T[n]`
        # specially for sizing purposes) -- e.g.
        # LLPManOptimalStronglyStableMarriage's `int[][] adj = new int[n][];`,
        # or LLPwithRejection's 4D `new int[n][m][n][m]`. JS has no native
        # multi-dim arrays, just nested arrays-of-arrays -- a 4D allocation
        # is exactly a 2D array of 2D arrays, no different in kind from a
        # plain 2D one, just one more level of Array.from nesting.
        fill = 'false' if e.name == 'boolean' else '0'
        if e.idx2 is None:
            return f'new Array({expr_to_js(e.idx, ctx)}).fill({fill})'
        sizes = [expr_to_js(x, ctx) for x in [e.idx, e.idx2, *e.idx_extra]]
        return _render_new_array_dims(sizes, fill)
    raise NotImplementedError(f'expr kind {k!r}')


def _render_bool_quant(e: Expr, ctx: EmitContext) -> str:
    """Translate an exists/forall quantifier appearing as a sub-expression
    (e.g. `!G[j] && (exists i in pre[j] : G[i])`) to a JS IIFE. This is
    distinct from an exists used as the *whole* forbidden predicate, which
    render_forbidden_body/_render_exists render as a plain for-loop instead
    of an IIFE (no sub-expression context to satisfy there)."""
    header = _quant_header(e.q_var, e.q_lo, e.q_hi, e.q_set, ctx)
    body_js = expr_to_js(e.q_body, ctx)
    if e.quant == 'exists':
        return f'(() => {{ {header} {{ if ({body_js}) return true; }} return false; }})()'
    return f'(() => {{ {header} {{ if (!({body_js})) return false; }} return true; }})()'


def _is_bool_expr(e: Expr) -> bool:
    """True if e is a boolean *condition*, not a value -- distinguishes
    `min r in [1..n] : !(r in R[j]) && ...` (a witness search: the
    smallest/largest r satisfying a predicate) from `max i in pre[j] :
    G[i] + t[i]` (a value reduction: the min/max of a computed number).
    Both use the same min/max keyword: the difference is only in what
    q_body computes."""
    if e.kind == 'bool':
        return True
    if e.kind == 'un' and e.op in ('not', '!'):
        return True
    if e.kind == 'bin' and e.op in ('<', '<=', '>', '>=', '=', '!=', 'and', 'or', 'in'):
        return True
    if e.kind == 'quant' and e.quant in ('exists', 'forall'):
        return True
    return False


def _render_reduction_quant(e: Expr, ctx: EmitContext) -> str:
    """Translate a min/max/sum/count quantifier appearing as a sub-expression
    (e.g. `min i in pre[k] : G[i] + w[i][k]`) to a JS IIFE that loops and
    accumulates -- JS has no such reduction-expression syntax of its own."""
    body_js = expr_to_js(e.q_body, ctx)
    if e.quant in ('min', 'max') and _is_bool_expr(e.q_body):
        # Witness search, not a value reduction (see _is_bool_expr). qvar
        # is bound to a plain value here (e.g. a room number), never used
        # to index an array, so the usual 1-based-to-0-based index shift
        # must not apply -- no_shift=True keeps both bounds as written.
        # Ascending iteration order makes the first match the minimum;
        # for max, keep scanning and remember the last match instead of
        # returning on the first one.
        header = _quant_header(e.q_var, e.q_lo, e.q_hi, e.q_set, ctx, no_shift=True)
        qvar = e.q_var
        if e.quant == 'min':
            return (f'(() => {{ {header} {{ if ({body_js}) return {qvar}; }} '
                     f'throw new Error(\'no {qvar} satisfies the min predicate\'); }})()')
        return (f'(() => {{ let __found; {header} {{ if ({body_js}) __found = {qvar}; }} '
                 f'if (__found === undefined) throw new Error('
                 f'\'no {qvar} satisfies the max predicate\'); return __found; }})()')
    header = _quant_header(e.q_var, e.q_lo, e.q_hi, e.q_set, ctx)
    if e.quant == 'min':
        init, update = 'Infinity', 'if (__v < __acc) __acc = __v;'
    elif e.quant == 'max':
        init, update = '-Infinity', 'if (__v > __acc) __acc = __v;'
    elif e.quant == 'sum':
        init, update = '0', '__acc += __v;'
    else:  # 'count'
        init, update = '0', 'if (__v) __acc += 1;'
    return (
        f'(() => {{ let __acc = {init}; {header} {{ '
        f'const __v = {body_js}; {update} }} return __acc; }})()'
    )


def stmt_to_js(s: Stmt, ctx: 'EmitContext', indent: str = '    ',
               terminate_return: bool = False) -> List[str]:
    """Translate an LL statement to JavaScript source lines.

    ``terminate_return`` distinguishes the two places a `return` can occur:
    inside an auxiliary method it's a normal JS return; inside the guard's
    *advance* body (LLPKruskal-style aux methods never set this flag) it
    means "stop the algorithm now", which mount()'s widget expects as
    ``advance()`` returning ``{ step, terminate: true }``, not a bare
    `return` -- see JSEmitter._render_advance, the only caller that passes
    terminate_return=True.
    """
    k = s.kind
    if k == 'assign':
        if s.lhs_index is not None:
            lhs = f'{ctx.id_to_js(s.lhs_name)}[{expr_to_js(s.lhs_index, ctx)}]'
        else:
            # Scalar assignment to a state var. If the var is wrapped (G),
            # we write G[0] = ...; otherwise plain.
            lhs = ctx.scalar_lhs_to_js(s.lhs_name)
        return [f'{indent}{lhs} = {expr_to_js(s.rhs, ctx)};']
    if k == 'swap':
        a = expr_to_js(s.swap_a, ctx)
        b = expr_to_js(s.swap_b, ctx)
        return [
            f'{indent}{{ const __t = {a}; {a} = {b}; {b} = __t; }}',
        ]
    if k == 'if':
        out = [f'{indent}if ({expr_to_js(s.cond, ctx)}) {{']
        for t in s.then_stmts:
            out.extend(stmt_to_js(t, ctx, indent + '  ', terminate_return))
        if s.else_stmts:
            out.append(f'{indent}}} else {{')
            for t in s.else_stmts:
                out.extend(stmt_to_js(t, ctx, indent + '  ', terminate_return))
        out.append(f'{indent}}}')
        return out
    if k == 'block':
        out = []
        for t in s.body:
            out.extend(stmt_to_js(t, ctx, indent, terminate_return))
        return out
    if k == 'decl':
        # A local var inside an auxiliary method body (e.g. `int c = 0;`);
        # not to be confused with the class-level state-vector decl, which
        # JSEmitter._classify_state handles separately.
        init = expr_to_js(s.decl_init, ctx) if s.decl_init is not None else _zero_js(s.decl_type)
        return [f'{indent}let {s.decl_name} = {init};']
    if k == 'while':
        out = [f'{indent}while ({expr_to_js(s.cond, ctx)}) {{']
        for t in s.body:
            out.extend(stmt_to_js(t, ctx, indent + '  ', terminate_return))
        out.append(f'{indent}}}')
        return out
    if k == 'forall':
        header = _quant_header(s.loop_var, s.loop_lo, s.loop_hi, s.loop_set, ctx)
        out = [f'{indent}{header} {{']
        for t in s.body:
            out.extend(stmt_to_js(t, ctx, indent + '  ', terminate_return))
        out.append(f'{indent}}}')
        return out
    if k == 'return':
        if terminate_return:
            val_js = expr_to_js(s.return_value, ctx) if s.return_value is not None else 'G'
            return [
                f'{indent}return {{ step: `terminate: G = '
                f'${{JSON.stringify({val_js})}}`, terminate: true }};'
            ]
        if s.return_value is None:
            return [f'{indent}return;']
        return [f'{indent}return {expr_to_js(s.return_value, ctx)};']
    if k == 'call':
        args = [expr_to_js(a, ctx) for a in s.call_args]
        if s.call_name in ctx.aux_names:
            args = args + ['ctx']
        return [f'{indent}{s.call_name}({", ".join(args)});']
    raise NotImplementedError(f'stmt kind {k!r}')


def _zero_js(t: 'Type') -> str:
    if t.is_2d or t.is_array:
        return '[]'
    if t.base == 'boolean':
        return 'false'
    return '0'


# ============================================================
#  EmitContext — holds the name->JS mappings during translation
# ============================================================

class EmitContext:
    """Tracks how each LL identifier should be rendered in JS."""

    def __init__(self, one_indexed: bool = False,
                 aux_names: Optional[set] = None):
        # name -> 'input' | 'scalar_local' | 'array_local' | 'inplace_array'
        # plus the JS expression to use for that name when it appears as an
        # rvalue id (e.g., 'inputs.A', 'G', 'G[0]').
        self.bindings: Dict[str, Tuple[str, str]] = {}
        self.one_indexed = one_indexed
        # Names of the class's own auxiliary methods (see
        # JSEmitter._render_aux_methods): a call to one of these always
        # carries an implicit trailing `ctx` argument, since an aux method
        # may reference entry-method params/`n` it never declared as its
        # own parameters (mirroring how these read as shared class fields
        # in the book's Java rendering) -- see expr_to_js/stmt_to_js 'call'.
        self.aux_names = aux_names or set()

    def bind(self, name: str, kind: str, js_expr: str) -> None:
        self.bindings[name] = (kind, js_expr)

    def id_to_js(self, name: str) -> str:
        # The implicit `n` is the iteration domain length, available in
        # the outer JS scope as `n` or `G.length`.
        if name == 'n':
            return 'n'
        if name in self.bindings:
            return self.bindings[name][1]
        return name  # quantifier variable, etc.

    def scalar_lhs_to_js(self, name: str) -> str:
        # For scalar-state assignments we always store at G[0] in JS.
        if name in self.bindings and self.bindings[name][0] == 'scalar_local':
            return self.bindings[name][1]    # 'G[0]'
        if name in self.bindings:
            return self.bindings[name][1]
        return name


# ============================================================
#  Predicate translation: forbidden / pick witness
# ============================================================

def render_forbidden_body(pred: Expr, ctx: EmitContext, indent: str = '    ') -> List[str]:
    """Translate the guard predicate to the body of forbidden(j, G, ctx)."""
    if pred.kind == 'quant':
        out = _render_exists(pred, ctx, indent, return_on_match='return true;')
        out.append(f'{indent}return false;')
        return out
    # plain expression
    return [f'{indent}return {expr_to_js(pred, ctx)};']


def _loop_bounds(qvar: str, lo_expr: Optional[Expr], hi_expr: Optional[Expr],
                  ctx: EmitContext, no_shift: bool = False) -> Tuple[str, str, str]:
    """Return (init, cond, incr) for a `[lo..hi]`-ranged loop (a quantifier
    or a forall statement -- both share this range syntax). In 0-indexed
    mode we loop ``i = lo; i < <hi>``; in 1-indexed mode ``i = lo; i <= <hi>``.
    Constants use 1-based-to-0-based shifting; non-constant lower bounds
    (e.g. ``j+1``) pass through verbatim.

    In 0-indexed mode, an upper bound written as ``X - 1`` (the extremely
    common ``[0..n-1]`` idiom) is recognized specially: ``i <= X - 1`` and
    ``i < X`` iterate the same range, but rendering the exclusive ``<``
    test against the *unadjusted* expression (``i < (X - 1)``) silently
    drops the range's last index -- ``[0..n-1]`` would only reach ``n-2``.
    Confirmed live: a `GCD1`-shaped program on input ``[4, 8, 3]`` returned
    ``[4, 4, 3]`` instead of the correct ``[1, 1, 1]``, because two of the
    three indices never got to compare themselves against the array's
    last slot. This mirrors the identical optimization already applied to
    the Java backend's ``_for_loop_test`` in lang/llc.py.

    `no_shift` disables the 1-based-to-0-based lower-bound shift entirely:
    it's only correct when qvar is used to *index an array*, but a min/max
    predicate search (e.g. `min r in [1..n] : !(r in R[j]) && ...`) binds
    qvar to a plain *value* (a room number, here) that gets returned/used
    directly -- shifting it would silently return the wrong value."""
    if hi_expr is not None:
        if (not ctx.one_indexed and not no_shift and hi_expr.kind == 'bin'
                and hi_expr.op == '-' and hi_expr.rhs.kind == 'int'
                and hi_expr.rhs.value == 1):
            hi = expr_to_js(hi_expr.lhs, ctx)
        else:
            hi = expr_to_js(hi_expr, ctx)
    else:
        hi = 'n'
    if lo_expr is not None:
        if ctx.one_indexed or no_shift:
            lo = expr_to_js(lo_expr, ctx)
        elif lo_expr.kind == 'int' and lo_expr.value >= 1:
            lo = str(lo_expr.value - 1)
        else:
            lo = expr_to_js(lo_expr, ctx)
    else:
        lo = '1' if ctx.one_indexed else '0'
    # no_shift keeps both bounds exactly as written, so the range is
    # inclusive on both ends (like one_indexed mode's `<=`), regardless of
    # whether this program otherwise runs 0- or 1-indexed.
    if ctx.one_indexed or no_shift:
        return (f'let {qvar} = {lo}', f'{qvar} <= {hi}', f'{qvar}++')
    return (f'let {qvar} = {lo}', f'{qvar} < {hi}', f'{qvar}++')


def _quant_header(qvar: str, lo_expr: Optional[Expr], hi_expr: Optional[Expr],
                   set_expr: Optional[Expr], ctx: EmitContext,
                   bare_var: bool = False, no_shift: bool = False) -> str:
    """Return the JS `for (...)` header (no trailing ` {`) for a range- or
    set-valued binding (a quantifier's `Q in [lo..hi]`/`Q in someSet`, or a
    `forall` statement's identical range syntax). A set-valued binding
    (`q_set`/`loop_set`, e.g. `exists i in pre[j]`) iterates the set's
    *values* directly via for-of, not its indices. `bare_var` omits the
    let/const keyword, for reusing an outer-scope-declared loop variable
    (see render_pick_witness). `no_shift`: see _loop_bounds."""
    if set_expr is not None:
        set_js = expr_to_js(set_expr, ctx)
        kw = '' if bare_var else 'const '
        return f'for ({kw}{qvar} of {set_js})'
    init, cond, incr = _loop_bounds(qvar, lo_expr, hi_expr, ctx, no_shift=no_shift)
    if bare_var and init.startswith('let '):
        init = init[len('let '):]
    return f'for ({init}; {cond}; {incr})'


def _render_exists(pred: Expr, ctx: EmitContext, indent: str,
                   return_on_match: str) -> List[str]:
    """Translate `exists Q in [LO..HI] [pick Q] : BODY` to a JS for-loop
    that emits ``return_on_match`` when BODY is true. Supports nested
    ``exists ... : exists ... :`` predicates. The caller appends any
    trailing ``return false;``."""
    if pred.quant != 'exists':
        raise NotImplementedError(f'quantifier {pred.quant!r} as forbidden predicate')
    header = _quant_header(pred.q_var, pred.q_lo, pred.q_hi, pred.q_set, ctx)
    body = pred.q_body
    if body.kind == 'quant':
        inner = _render_exists(body, ctx, indent + '  ', return_on_match)
        return [
            f'{indent}{header} {{',
            *inner,
            f'{indent}}}',
        ]
    body_js = expr_to_js(body, ctx)
    return [
        f'{indent}{header} {{',
        f'{indent}  if ({body_js}) {return_on_match}',
        f'{indent}}}',
    ]


def render_pick_witness(pred: Expr, ctx: EmitContext, indent: str = '    ') -> List[str]:
    """Re-find the witness inside advance() so the advance body can use it.
    Declares the qvar in outer scope so it survives the loop."""
    if pred.kind != 'quant' or pred.pick_var is None:
        return []
    qvar = pred.q_var
    body = pred.q_body
    body_js = expr_to_js(body, ctx)
    header = _quant_header(pred.q_var, pred.q_lo, pred.q_hi, pred.q_set, ctx,
                            bare_var=True)
    return [
        f'{indent}let {qvar};',
        f'{indent}{header} {{',
        f'{indent}  if ({body_js}) break;',
        f'{indent}}}',
    ]


# ============================================================
#  Pretty-printer for the demo `program:` field (plain text)
# ============================================================

PRETTY_BIN = {
    '+': '+', '-': '-', '*': '*', '/': '/', '%': 'mod',
    '<': '<', '<=': '<=', '>': '>', '>=': '>=',
    '=': '=', '==': '=', '!=': '!=',
    'and': 'and', 'or': 'or',
}


def expr_to_pretty(e: Expr) -> str:
    k = e.kind
    if k == 'int':
        return str(e.value)
    if k == 'id':
        return e.name
    if k == 'bool':
        return 'true' if e.value else 'false'
    if k == 'index':
        return f'{e.name}[{expr_to_pretty(e.idx)}]'
    if k == 'index2':
        return f'{e.name}[{expr_to_pretty(e.idx)}][{expr_to_pretty(e.rhs)}]'
    if k == 'bin':
        op = PRETTY_BIN.get(e.op, e.op)
        return f'{expr_to_pretty(e.lhs)} {op} {expr_to_pretty(e.rhs)}'
    if k == 'un':
        if e.op in ('not', '!'): return f'not {expr_to_pretty(e.lhs)}'
        if e.op in ('neg', '-'): return f'-{expr_to_pretty(e.lhs)}'
    if k == 'quant':
        rng = ''
        if e.q_lo is not None and e.q_hi is not None:
            rng = f' in [{expr_to_pretty(e.q_lo)}..{expr_to_pretty(e.q_hi)}]'
        elif e.q_set is not None:
            rng = f' in {expr_to_pretty(e.q_set)}'
        pick = f' pick {e.pick_var}' if e.pick_var else ''
        return f'{e.quant} {e.q_var}{rng}{pick} : {expr_to_pretty(e.q_body)}'
    if k == 'method_call':
        return f'.{e.name}()'
    return f'<{k}>'


def stmt_to_pretty(s: Stmt) -> str:
    if s.kind == 'assign':
        lhs = (f'{s.lhs_name}[{expr_to_pretty(s.lhs_index)}]'
               if s.lhs_index is not None else s.lhs_name)
        return f'{lhs} := {expr_to_pretty(s.rhs)}'
    if s.kind == 'swap':
        return f'swap({expr_to_pretty(s.swap_a)}, {expr_to_pretty(s.swap_b)})'
    if s.kind == 'if':
        return ('if ' + expr_to_pretty(s.cond) + ' then '
                + '; '.join(stmt_to_pretty(t) for t in s.then_stmts)
                + (' else ' + '; '.join(stmt_to_pretty(t) for t in s.else_stmts)
                   if s.else_stmts else ''))
    return f'<{s.kind}>'


def render_program_text(prog: Program, guard: Stmt) -> str:
    name = prog.name
    in_lines = []
    for p in prog.params:
        ty = _ty_pretty(p.type)
        in_lines.append(f'  input:  {p.name} : {ty}')
    out_lines = []
    # The receiver of a `.clone()` init is normally the first int[] param.
    first_arr = next((p for p in prog.params
                      if p.type.is_array and not p.type.is_2d), None)
    receiver = first_arr.name if first_arr else (
        prog.params[0].name if prog.params else '')
    for d in prog.decls:
        ty = _ty_pretty(d.type)
        init_text = ''
        if d.init is not None:
            if d.init.kind == 'method_call' and d.init.name == 'clone':
                init_text = ' := ' + (receiver + '.clone()' if receiver else 'clone()')
            else:
                init_text = ' := ' + expr_to_pretty(d.init)
        out_lines.append(f'  output: {d.name} : {ty}, initially {d.name}{init_text}')
    pred_text = expr_to_pretty(guard.guard_predicate)
    adv_text = stmt_to_pretty(guard.guard_advance)
    out = [name]
    out.extend(in_lines)
    out.extend(out_lines)
    out.append('')
    out.append(f'  forbidden(j) : {pred_text}')
    out.append(f'  advance(j)   : {adv_text}')
    return '\n'.join(out)


def _ty_pretty(t: Type) -> str:
    base = t.base
    if t.is_2d: return f'matrix of {base}'
    if t.is_array: return f'array of {base}'
    return base


def clean_llp_for_display(src: str) -> str:
    """Slice a .llp source string into the form shown in the demo's
    `program:` field: drop the leading file-level `// ...` comments and
    any wrapping `class X { ... }`, leaving only the method(s) in
    canonical LL syntax."""
    lines = src.splitlines()
    # Drop leading comment block (file-level docs) and blank lines.
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith('//') or s == '':
            i += 1
            continue
        break
    text = '\n'.join(lines[i:]).rstrip()
    # Peel a wrapping `class X { ... }`.  We don't try to parse; we look
    # for a class-opener line and a matching closing brace at column 0.
    m = re.match(r'^class\s+\w+\s*\{\s*\n(.*?)\n\s*\}\s*$', text, re.DOTALL)
    if m:
        inner = m.group(1)
        # De-indent one level (each line lost its leading 2 spaces).
        deindented = []
        for line in inner.splitlines():
            if line.startswith('  '):
                deindented.append(line[2:])
            elif line == '':
                deindented.append('')
            else:
                deindented.append(line)
        text = '\n'.join(deindented).rstrip()
    return text


# ============================================================
#  Core emitter
# ============================================================

class JSEmitter:
    def __init__(self, prog: Program, sidecar: dict, src_basename: str,
                 src_text: str = ''):
        self.prog = prog
        self.sc = sidecar
        self.src_basename = src_basename
        self.src_text = src_text
        self.one_indexed = bool(sidecar.get('one_indexed', False))
        self.aux_names = {m.name for m in prog.aux_methods}
        self.guards = [s for s in prog.stmts if s.kind == 'guard']
        if len(self.guards) != 1:
            raise NotImplementedError('JS emitter requires exactly one guard')
        self.guard = self.guards[0]
        self._classify_state()

    # ----- classify the state vector G -----

    def _classify_state(self):
        decls = list(self.prog.decls)
        params = self.prog.params

        # Any number of leading scalar decls before the real state
        # declaration (e.g. `int n = v[0].length; int m = v.length;`) are
        # plain locals, not additional state candidates -- peel them off
        # before the ambiguity check below. Each is emitted into buildState
        # as an ordinary `const` statement (see _render_pre_array_decls),
        # in scope for whatever follows -- including the state array's own
        # size expression, if it references one of them (see
        # _classify_init's 'new_array' case and _array_size_js).
        self.pre_array_decls: List[Decl] = []
        while len(decls) > 1 and not decls[0].type.is_array:
            self.pre_array_decls.append(decls[0])
            decls = decls[1:]
        # A *lone* remaining scalar decl is normally scalar_local -- a
        # genuinely evolving single value (e.g. `int G = A[1];`). But one
        # whose init is specifically `<expr>.length` (e.g. `int n =
        # G.length;`, where G is itself an in-place state *param*, per
        # ExcludedRooms) is unambiguously just a length alias, not real
        # state -- peel it too, so the in_place branch below picks the
        # param up correctly instead of this misidentifying `n` as G.
        if (len(decls) == 1 and not decls[0].type.is_array
                and decls[0].init is not None
                and decls[0].init.kind == 'field_access'
                and decls[0].init.name == 'length'):
            self.pre_array_decls.append(decls[0])
            decls = []

        # extra_array_specs: additional *real* mutable state arrays beyond
        # the primary G -- e.g. LLPJobSchedulingFixed's `G` (start times)
        # plus a `fixed` boolean-tracking array, both declared and both
        # mutated in advance. Each is bound into ctx exactly like a normal
        # entry param (see _build_ctx/_render_extra_arrays), which is
        # already how a mutated *input* like LLPKruskal's `parent` works:
        # ctx is one stable object reused for the algorithm's whole run,
        # so an array living at ctx.<name> persists in-place across steps
        # with no change needed to mount()'s widget loop.
        self.extra_array_specs: List[Tuple[str, str, tuple]] = []
        if len(decls) > 1:
            if all(d.type.is_array for d in decls):
                preferred = [d for d in decls if d.name == self.prog.result_name]
                if not preferred:
                    preferred = [d for d in decls if d.name == 'G']
                if len(preferred) != 1:
                    names = ', '.join(d.name for d in decls)
                    raise NotImplementedError(
                        f'ambiguous state vector: {len(decls)} local array '
                        f'declarations ({names}) and none is named `G` or '
                        f'matches the return value.')
                primary = preferred[0]
                extras = [d for d in decls if d is not primary]
                decls = [primary]
                for d in extras:
                    self.extra_array_specs.append(
                        (d.name, d.type.base, self._classify_init(d.init)))
            else:
                raise NotImplementedError('multiple var declarations')

        # No declarations: in-place algorithm. The sole int[]/boolean[]
        # input is used as G. This only makes sense when there is exactly
        # one array-typed parameter to guess from -- with two or more (e.g.
        # `void ReleaseTimes(int[] r, int[] G)`), silently picking "the
        # first" is as likely to be wrong as right, and if the *other* one
        # happens to be named `G` the emitted buildState() ends up with two
        # `const G = ...` declarations, a JS SyntaxError at run time rather
        # than a compile-time one. Ask for an explicit `var`/`output`
        # declaration instead so the state vector is never guessed.
        if not decls:
            arr_params = [p for p in params
                          if p.type.is_array and not p.type.is_2d]
            if not arr_params:
                raise NotImplementedError('no state var and no array input')
            if len(arr_params) > 1:
                # Not actually ambiguous if one array-typed param is either
                # (a) what the entry method returns (prog.result_name, set
                # when the return expression is a bare identifier), or (b)
                # literally named G -- this codebase's universal naming
                # convention for the state vector, relevant for void
                # methods that have no return value to go by.
                preferred = [p for p in arr_params if p.name == self.prog.result_name]
                if not preferred:
                    preferred = [p for p in arr_params if p.name == 'G']
                if len(preferred) == 1:
                    arr_params = preferred
                else:
                    names = ', '.join(p.name for p in arr_params)
                    raise NotImplementedError(
                        f'ambiguous state vector: no `var`/`output` declaration, '
                        f'and {len(arr_params)} array-typed inputs ({names}) to '
                        f'choose from. Add an explicit declaration, e.g. '
                        f'`int[] G = 0;` or `int[] G = clone({arr_params[0].name});`, '
                        f'to say which one is the mutated state.')
            arr_param = arr_params[0]
            self.G_kind = 'in_place'
            self.G_name = arr_param.name           # e.g. 'A'
            self.G_init = ('clone', arr_param.name)
            self.iter_param = arr_param.name
            return

        d = decls[0]
        if d.type.is_array:
            self.G_kind = 'array_local'
            self.G_name = d.name                    # e.g. 'G'
            self.G_base = d.type.base                # 'int' | 'boolean'
            # iter_param is only a fallback sizing source (see
            # _array_size_js) for when there are no pre_array_decls; a
            # program with one may have no 1D/2D array param at all (e.g.
            # every array input is 3D), so don't require one.
            try:
                self.iter_param = self._first_array_like_param().name
            except NotImplementedError:
                if not self.pre_array_decls:
                    raise
                self.iter_param = None
            self.G_init = self._classify_init(d.init)
        elif not d.type.is_array and d.type.base == 'int':
            self.G_kind = 'scalar_local'
            self.G_name = d.name
            self.iter_param = self._first_array_param().name
            self.G_init = self._classify_init(d.init)
        else:
            raise NotImplementedError(f'state var type {d.type.base} {d.type.is_array}')

    def _first_array_param(self) -> Decl:
        for p in self.prog.params:
            if p.type.is_array and not p.type.is_2d:
                return p
        raise NotImplementedError('no int[] input array')

    def _first_array_like_param(self) -> Decl:
        """Like _first_array_param, but also accepts a 2D param (using its
        outer-dimension .length) when no 1D int[]/bool[] param exists --
        e.g. LLPLexicallyFirstSetCover(int[][] S), whose only array-shaped
        input is the 2D S. Only used as a sizing source (any array's outer
        length), never as a `.clone()` receiver -- _first_array_param stays
        1D-only for that."""
        try:
            return self._first_array_param()
        except NotImplementedError:
            pass
        for p in self.prog.params:
            if p.type.is_array and p.type.is_2d:
                return p
        raise NotImplementedError('no array-typed input')

    def _classify_init(self, init: Optional[Expr]):
        if init is None:
            return ('zero', None)
        if init.kind == 'method_call' and init.name == 'clone':
            return ('clone', self._first_array_param().name)
        if init.kind == 'index':
            # e.g., int G = A[1]
            return ('index_first', init.name, init.idx)
        if init.kind == 'int':
            return ('int', init.value)
        if init.kind == 'un' and init.op in ('neg', '-') and init.lhs.kind == 'int':
            return ('int', -init.lhs.value)
        if init.kind == 'bool':
            return ('bool', init.value)
        if init.kind == 'new_array':
            return ('zero', init.idx)
        if init.kind == 'infinity':
            return ('int', 'Infinity')
        return ('expr', init)

    def _render_array_construction(self, name: str, base: str, init: tuple,
                                    pre_ctx: 'EmitContext') -> List[str]:
        """Emit `const <name> = ...;` for an array-typed local -- G itself,
        or one of extra_array_specs. `pre_ctx` is buildState's own scope
        (params + any pre_array_decls bound bare), used to translate a
        'zero' init's actual size expression when it has one. For an extra
        array (name != 'G'), size from G.length instead: G is already
        constructed by the time extras run (see _render_buildState), so
        it's always safe, whereas the source's own size expression is
        often just the bare, not-yet-computed `n` (e.g. LLPJobSchedulingFixed's
        `boolean[] fixed = new boolean[n];`) -- translating that literally
        here would reference `n` before its own `const n = G.length;`."""
        if init[0] == 'clone':
            return [f'    const {name} = {init[1]}.slice();']
        if init[0] in ('zero', 'bool', 'int'):
            # 'zero': e.g. `int[] G = new int[n];` -- init[1] is the actual
            # size expression from the source (translated via pre_ctx, so
            # it can reference a pre_array_decl like `n`), when there was
            # one to read (a bare `int[] G;` with no initializer has none,
            # hence the fallback).
            # 'bool': e.g. `boolean[] C = false;` (fill every slot).
            # 'int':  e.g. `int[] G = infinity;` or `int[] G = 0;` (ditto).
            if name != 'G':
                size_expr = 'G.length'
            elif init[0] == 'zero' and init[1] is not None:
                size_expr = expr_to_js(init[1], pre_ctx)
            elif self.pre_array_decls:
                size_expr = self.pre_array_decls[-1].name
            else:
                size_expr = f'{self.iter_param}.length'
            if init[0] == 'bool':
                fill_val = str(init[1]).lower()
            elif init[0] == 'int':
                fill_val = str(init[1])
            else:
                fill_val = 'false' if base == 'boolean' else '0'
            return [f'    const {name} = new Array({size_expr}).fill({fill_val});']
        raise NotImplementedError(f'array {name} with init {init}')

    # ----- emit -----

    def _render_aux_methods(self) -> str:
        """Emit each of the class's auxiliary methods (e.g. `happenedBefore`,
        `find`/`union`, `minCrossCut`) as a top-level JS function, so a
        `call` expression or statement inside the guard predicate / advance
        body can reach them. Each gets its own EmitContext binding its own
        params to their bare JS names -- no `ctx.<name>` indirection, since
        these are plain functions, not the forbidden/advance closures.

        An aux method's body may reference an entry-method param (or `n`)
        it never declared as its own parameter -- LLPPrim's minCrossCut(j)
        reads `fixed`/`W`/`n` this way, exactly as those read as shared
        class fields in the book's Java rendering. Since these are separate
        top-level JS functions, not nested closures, that sharing has to be
        wired explicitly: every aux method takes an implicit trailing `ctx`
        parameter (see EmitContext.aux_names for the call-site half of
        this) and destructures whichever entry-params/n aren't already
        among its own declared parameters."""
        out: List[str] = []
        entry_names = ([p.name for p in self.prog.params]
                       + [name for name, _, _ in self.extra_array_specs])
        # The state array (see the matching ctx_fields.append('G') in
        # _render_buildState) -- unless an in-place entry param is already
        # literally named `G` (e.g. LLPBFS's `int[] G`), which would
        # otherwise destructure as `const { G, G } = ctx;`, a SyntaxError.
        if 'G' not in entry_names:
            entry_names.append('G')
        for m in self.prog.aux_methods:
            aux_ctx = EmitContext(one_indexed=self.one_indexed, aux_names=self.aux_names)
            own_names = {p.name for p in m.params}
            # A name the body declares itself at its own top level (e.g.
            # `int n = G.length;`, common enough that it collides with the
            # auto-injected `n`) would double-declare if also destructured
            # from ctx -- both are lexical (const/let) in the same function
            # scope, a hard JS SyntaxError, not just shadowing. Only *own*
            # top-level decls count: one nested in an if/while is
            # block-scoped and shadows a function-level const harmlessly.
            own_top_level_decls = {s.decl_name for s in m.body if s.kind == 'decl'}
            for p in m.params:
                aux_ctx.bind(p.name, 'param', p.name)
            implicit = [name for name in entry_names
                        if name not in own_names and name not in own_top_level_decls]
            if 'n' not in own_top_level_decls:
                implicit.append('n')
            params_js = ', '.join([p.name for p in m.params] + ['ctx'])
            out.append(f'function {m.name}({params_js}) {{')
            if implicit:
                out.append(f'  const {{ {", ".join(implicit)} }} = ctx;')
            for s in m.body:
                out.extend(stmt_to_js(s, aux_ctx, '  '))
            out.append('}')
            out.append('')
        return '\n'.join(out)

    def emit(self) -> str:
        ctx = self._build_ctx()

        if self.G_kind == 'scalar_local':
            # Scalar G: there is a single component to advance, but the LL
            # predicate quantifies over the array index. Emit forbidden as
            # an exists-loop and a custom nextIndex that returns the witness.
            ctx_fb = self._build_ctx(loop_var_for_j='__jj')
            forbidden_body = '\n'.join(self._render_scalar_forbidden(ctx_fb))
            next_block = self._render_scalar_next_index(ctx)
        else:
            body_lines = render_forbidden_body(self.guard.guard_predicate, ctx)
            if self.one_indexed:
                # In 1-indexed mode the renderer also asks about cell 0
                # (the sentinel slot G[0]); short-circuit it as not-forbidden.
                body_lines = ['    if (j === 0) return false;'] + body_lines
            forbidden_body = '\n'.join(body_lines)
            if self.prog.mode == 'heap':
                next_block = self._render_priority_next_index(ctx)
            elif self.prog.mode != 'array':
                raise NotImplementedError(
                    f'setMode {self.prog.mode!r} not supported by the JS emitter '
                    f'(only the default array mode and PriorityQueue are)')
            else:
                next_block = self._render_scheduler()

        advance_body = self._render_advance(ctx)
        buildState = self._render_buildState()
        aux_methods = self._render_aux_methods()
        program_text = (clean_llp_for_display(self.src_text)
                        if self.src_text else
                        render_program_text(self.prog, self.guard))
        inputs_js = self._render_inputs()

        title = self.sc.get('title') or self._default_title()
        describe = self.sc.get('describe') or ''
        on_terminate = self.sc.get('on_terminate') or self._default_on_terminate()
        container_id = self.sc.get('container_id') or f'demo-{self.prog.name.lower()}'

        return JS_TEMPLATE.format(
            src_basename=self.src_basename,
            aux_methods=aux_methods,
            container_id=container_id,
            title=title,
            program_text=program_text,
            inputs_js=inputs_js,
            buildState=buildState,
            forbidden_body=forbidden_body,
            advance_body=advance_body,
            describe=describe,
            on_terminate=on_terminate,
            scheduler=next_block,
        )

    # ----- helpers -----

    def _build_ctx(self, loop_var_for_j: Optional[str] = None) -> EmitContext:
        ctx = EmitContext(one_indexed=self.one_indexed, aux_names=self.aux_names)
        # Inputs are accessible from inside forbidden/advance via ctx.<name>;
        # the scaffolded forbidden/advance signature destructures them.
        for p in self.prog.params:
            ctx.bind(p.name, 'input', f'ctx.{p.name}')
        # State var.
        if self.G_kind == 'array_local':
            ctx.bind(self.G_name, 'array_local', 'G')
        elif self.G_kind == 'scalar_local':
            ctx.bind(self.G_name, 'scalar_local', 'G[0]')
        elif self.G_kind == 'in_place':
            # The input array is rendered as G; references to its name use G.
            ctx.bind(self.G_name, 'inplace_array', 'G')
        # Extra state arrays (see extra_array_specs) -- bound the same way
        # a normal input is, since they live at ctx.<name> just like one.
        for name, _, _ in self.extra_array_specs:
            ctx.bind(name, 'input', f'ctx.{name}')
        # Optional override: rebind LL's `j` to a different JS name. Used
        # when the emitted forbidden() loops over j internally and must
        # not clash with the function's own j parameter.
        if loop_var_for_j:
            ctx.bind('j', 'qvar', loop_var_for_j)
        return ctx

    def _render_scalar_forbidden(self, ctx: EmitContext) -> List[str]:
        """Emit forbidden(j, G, ctx) for a scalar-G program: scan the j-domain
        (ctx.n) and return true if any index satisfies the LL predicate. The
        cell-render highlight thus marks the (single) G cell as forbidden iff
        the algorithm has not yet terminated."""
        pred = self.guard.guard_predicate
        if self.one_indexed:
            init = 'let __jj = 1'
            cond = '__jj <= n'
        else:
            init = 'let __jj = 0'
            cond = '__jj < n'
        if pred.kind == 'quant':
            # Should be rare for scalar G; treat as a normal exists.
            body = render_forbidden_body(pred, ctx)
            return body
        body_js = expr_to_js(pred, ctx)
        return [
            f'    for ({init}; {cond}; __jj++) {{',
            f'      if ({body_js}) return true;',
            f'    }}',
            f'    return false;',
        ]

    def _render_scalar_next_index(self, ctx: EmitContext) -> str:
        """Emit a nextIndex(G, ctx) that returns the first j in the j-domain
        satisfying the LL predicate, else null. The j returned is then passed
        to advance() so the LL advance can reference j as its picked witness."""
        pred = self.guard.guard_predicate
        if self.one_indexed:
            init = 'let j = 1'
            cond = 'j <= n'
        else:
            init = 'let j = 0'
            cond = 'j < n'
        body_js = expr_to_js(pred, ctx)
        return (
            ',\n  nextIndex(G, ctx) {\n'
            '    const n = ctx.n;\n'
            f'    for ({init}; {cond}; j++) {{\n'
            f'      if ({body_js}) return j;\n'
            '    }\n'
            '    return null;\n'
            '  }'
        )

    def _render_priority_next_index(self, ctx: EmitContext) -> str:
        """Emit a nextIndex(G, ctx) for a `setMode(PriorityQueue)` program:
        scan the j-domain, keep only indices the LL predicate still marks
        forbidden, and return whichever has the extremal value of the
        `setPriority(expr, min|max)` expression (ties broken by lowest j).

        This is a plain linear scan, not a real min-heap, but it is called
        fresh before every step (mount()'s findForbidden()), so it always
        returns the same vertex a decrease-key heap would have -- including
        when other vertices' priorities changed since the last step, which
        a real heap would need an explicit decrease-key for anyway. Fine
        for the small instances these demos run; see the book's own Java
        backend (llc.py's 'heap' mode) for the real-heap version used when
        asymptotic efficiency matters."""
        pred = self.guard.guard_predicate
        if self.one_indexed:
            init = 'let j = 1'
            cond = 'j <= n'
        else:
            init = 'let j = 0'
            cond = 'j < n'
        pred_js = expr_to_js(pred, ctx)
        prio_js = expr_to_js(self.prog.priority_expr, ctx)
        better = '>' if self.prog.priority_order == 'max' else '<'
        return (
            ',\n  nextIndex(G, ctx) {\n'
            '    const n = ctx.n;\n'
            '    let best = null, bestKey = null;\n'
            f'    for ({init}; {cond}; j++) {{\n'
            f'      if (!({pred_js})) continue;\n'
            f'      const key = {prio_js};\n'
            f'      if (best === null || key {better} bestKey) {{ best = j; bestKey = key; }}\n'
            '    }\n'
            '    return best;\n'
            '  }'
        )

    def _render_advance(self, ctx: EmitContext) -> str:
        out: List[str] = []
        out.append('    const __snapshot = JSON.stringify(G);')
        if self.guard.guard_pick_var:
            out.extend(render_pick_witness(self.guard.guard_predicate, ctx))
        out.extend(stmt_to_js(self.guard.guard_advance, ctx,
                               terminate_return=True))
        out.append('    return { step: `advance j=${j}: G ' +
                   '${__snapshot}' + ' \u2192 ${JSON.stringify(G)}` };')
        return '\n'.join(out)

    def _render_buildState(self) -> str:
        lines: List[str] = []
        # Read each input from the parsed inputs object. Skip the param
        # chosen as the in-place state var when it's literally named `G`
        # (e.g. `int[] ReleaseTimes(int[] r, int[] G)`) -- the G-init block
        # right below declares `const G = ...` itself, and JS doesn't allow
        # `const G = G.slice()` (self-reference in its own initializer).
        skip_param = (self.G_name if self.G_kind == 'in_place'
                      and self.G_name == 'G' else None)
        for p in self.prog.params:
            if p.name == skip_param:
                continue
            raw = f'inputs.{p.name}'
            if self.one_indexed and p.type.is_array and not p.type.is_2d:
                lines.append(f'    const {p.name} = [null, ...{raw}];')
            elif self.one_indexed and p.type.is_2d:
                lines.append(f'    const {p.name} = [null];')
                lines.append(f'    for (const __row of {raw}) '
                             f'{p.name}.push([null, ...__row]);')
            else:
                lines.append(f'    const {p.name} = {raw};')

        # in_place G is constructed here, before pre_array_decls: unlike
        # array_local (below), it never depends on one for sizing, and a
        # pre_array_decl might depend on *it* instead (ExcludedRooms' `int
        # n = G.length;`, where G is the in-place param) -- constructing G
        # first covers both directions without needing to detect which one
        # a given file uses.
        if self.G_kind == 'in_place':
            # inputs.<name> directly: see skip_param above -- there's no
            # bare `<name>` local binding to read from when it's `G` itself.
            src = f'inputs.{self.G_name}' if skip_param else self.G_name
            if self.one_indexed:
                # G is the input array, padded.
                lines.append(f'    const G = {src}.slice();')
            else:
                lines.append(f'    const G = {src}.slice();')

        # Any scalar local declared before the state array (e.g.
        # `int n = v[0].length; int m = v.length;`) is emitted here, in
        # order, as an ordinary `const` -- and stays bound in pre_ctx so
        # later ones, or the array's own size expression right below, can
        # reference it.
        pre_ctx = EmitContext(one_indexed=self.one_indexed, aux_names=self.aux_names)
        for p in self.prog.params:
            pre_ctx.bind(p.name, 'param', p.name)
        if self.G_kind == 'in_place' and skip_param:
            # The in-place source array is bound to its own name above
            # (e.g. `const R = inputs.R;`), but *skip_param* itself (G's
            # own original param name, when literally `G`) was skipped
            # there -- the JS variable actually holding it is `G` (see the
            # in_place construction above), so bind that here too, in case
            # a pre_array_decl references it (ExcludedRooms' `int n =
            # G.length;`).
            pre_ctx.bind(skip_param, 'param', 'G')
        for d in self.pre_array_decls:
            lines.extend(stmt_to_js(
                Stmt(kind='decl', decl_type=d.type, decl_name=d.name, decl_init=d.init),
                pre_ctx, '    '))
            pre_ctx.bind(d.name, 'param', d.name)

        # Initialize G (array_local / scalar_local only -- in_place already
        # constructed above).
        if self.G_kind == 'array_local':
            lines.extend(self._render_array_construction(
                'G', self.G_base, self.G_init, pre_ctx))
            if self.G_init[0] == 'clone' and self.one_indexed:
                lines.append("    G[0] = '\u00b7';")
        elif self.G_kind == 'scalar_local':
            init = self.G_init
            if init[0] == 'index_first':
                # e.g., int G = A[1]; in JS use 0-indexed unless one_indexed.
                slot = '1' if self.one_indexed else '0'
                lines.append(f'    const G = [{init[1]}[{slot}]];')
            elif init[0] == 'int':
                lines.append(f'    const G = [{init[1]}];')
            elif init[0] == 'bool':
                lines.append(f'    const G = [{str(init[1]).lower()}];')
            elif init[0] == 'zero':
                lines.append('    const G = [0];')
            else:
                raise NotImplementedError(f'scalar G with init {init}')

        # Extra state arrays beyond G (see extra_array_specs) -- e.g. a
        # `fixed` boolean-tracking array alongside the primary G.
        for name, base, init in self.extra_array_specs:
            lines.extend(self._render_array_construction(name, base, init, pre_ctx))

        # ctx.n = the iteration domain size (LL's `n` semantically).
        # For 1-indexed: n is the original input length (= padded length - 1).
        # For 0-indexed: n is the array's actual length. Computed here,
        # before the state-init statements below, since those may reference
        # `n` (e.g. `forall k in [0..n-1] : G[k] = 1;`).
        # ... unless a pre_array_decl already declared a local `n` (e.g.
        # `int n = v[0].length;`) -- then that's the authoritative value
        # (it's what G itself was sized from), and a second `const n =`
        # here would be a duplicate-declaration SyntaxError.
        if any(d.name == 'n' for d in self.pre_array_decls):
            pass
        elif self.G_kind == 'in_place':
            lines.append(f'    const n = {"G.length - 1" if self.one_indexed else "G.length"};')
        elif self.G_kind == 'array_local':
            lines.append(f'    const n = {"G.length - 1" if self.one_indexed else "G.length"};')
        else:  # scalar_local
            ctx_n = (f'{self.iter_param}.length - 1' if self.one_indexed
                     else f'{self.iter_param}.length')
            lines.append(f'    const n = {ctx_n};')

        # Any top-level statement between the state-var decl and the guard
        # (e.g. `G[0] = 0;` pinning a source vertex, or a `forall` filling
        # G's initial values) runs here, in buildState's own scope.
        lines.extend(self._render_state_init_stmts())

        # If the scheduler is odd-even, persist phase across steps via ctx.
        if self.sc.get('scheduler') == 'odd-even':
            lines.append('    const __phase = { phase: 0 };')

        # ctx contents: every input by name + any extra state arrays + n +
        # indexLabels. ctx is one stable object reused for the algorithm's
        # whole run (mount() only rebuilds it on reset()), so an extra
        # array living at ctx.<name> persists its in-place mutations across
        # steps exactly like any mutated input (e.g. LLPKruskal's parent).
        ctx_fields: List[str] = []
        for p in self.prog.params:
            ctx_fields.append(p.name)
        for name, _, _ in self.extra_array_specs:
            ctx_fields.append(name)
        # The state array itself, so a top-level aux method that references
        # it by its bare LL name (translated to the JS identifier `G`
        # everywhere -- see ctx.bind(self.G_name, ..., 'G') in _build_ctx)
        # can reach it via ctx too -- aux methods are separate top-level JS
        # functions, not closures, so they otherwise can't see buildState's
        # own `const G`. Safe for every G_kind: the local variable in scope
        # here is always literally named `G` (see the decl-emitting branches
        # above), and this is the same array object advance() mutates
        # in place, so ctx.G always reflects the live, current values.
        # (Skipped when an in-place entry param is itself literally named
        # `G` -- e.g. LLPBFS's `int[] G` -- since it is already on this
        # list; appending it again would emit a duplicate object-literal
        # key here and, worse, `const { G, G } = ctx;` -- a SyntaxError --
        # in _render_aux_methods below.)
        if 'G' not in ctx_fields:
            ctx_fields.append('G')
        ctx_fields.append('n')
        if self.sc.get('scheduler') == 'odd-even':
            ctx_fields.append('__phase')

        # Top-level n controls renderer cell count and default findForbidden.
        # Scalar G: render a single cell; algorithm scans ctx.n via nextIndex.
        if self.G_kind == 'scalar_local':
            top_n = '1'
        else:
            top_n = 'G.length'

        labels_js = self._labels_js()
        lines.append(f'    return {{')
        lines.append(f'      G,')
        lines.append(f'      ctx: {{ {", ".join(ctx_fields)}, '
                     f'indexLabels: {labels_js}, valueLabel: \'G\' }},')
        lines.append(f'      n: {top_n}')
        lines.append(f'    }};')
        return '\n'.join(lines)

    def _render_state_init_stmts(self) -> List[str]:
        """Translate any top-level statement between the state-var decl and
        the guard -- e.g. `G[0] = 0;` (pinning a source vertex) or a
        `forall k in [0..n-1] : G[k] = 1;` (a non-zero fill) -- into
        buildState. These run in buildState's own scope, where params are
        bare identifiers (`const pre = inputs.pre;`) and the state var is
        the bare `G`, not the `ctx.<name>` indirection forbidden/advance
        use. Silently dropping these (as earlier versions of this emitter
        did) is a real correctness bug, not just a cosmetic gap: it changes
        the algorithm's actual initial state."""
        pre_ctx = EmitContext(one_indexed=self.one_indexed, aux_names=self.aux_names)
        for p in self.prog.params:
            pre_ctx.bind(p.name, 'param', p.name)
        if self.G_kind == 'scalar_local':
            pre_ctx.bind(self.G_name, 'scalar_local', 'G[0]')
        else:
            pre_ctx.bind(self.G_name, 'array_local', 'G')
        out: List[str] = []
        for s in self.prog.stmts:
            if s.kind == 'guard':
                break
            out.extend(stmt_to_js(s, pre_ctx, '    '))
        return out

    def _labels_js(self) -> str:
        if self.G_kind == 'scalar_local':
            return "['G']"
        if self.one_indexed:
            return ("Array.from({length: G.length}, "
                    "(_, i) => i === 0 ? '\u00b7' : 'j=' + i)")
        return "Array.from({length: G.length}, (_, i) => 'j=' + i)"

    def _render_inputs(self) -> str:
        # Use sidecar overrides where present; fall back to heuristic defaults.
        sc_inputs = self.sc.get('inputs', {})
        items = []
        for p in self.prog.params:
            override = sc_inputs.get(p.name, {})
            if p.type.is_2d:
                kind = 'matrix'
                default = override.get('default', '1, 2, 3\n2, 3, 1\n3, 1, 2')
            elif p.type.is_array:
                kind = 'int-array'
                default = override.get('default', '1, 2, 3')
            else:
                kind = 'int'
                default = override.get('default', '0')
            label = override.get('label', f'{p.name}')
            extra = ''
            if kind == 'matrix':
                extra = f', rows: {override.get("rows", 4)}'
            esc_default = _js_str(default)
            esc_label = _js_str(label)
            items.append(
                "    { name: '" + p.name + "', label: '" + esc_label
                + "', kind: '" + kind + "', default: '" + esc_default
                + "'" + extra + " }"
            )
        return ',\n'.join(items)

    def _render_scheduler(self) -> str:
        sched = self.sc.get('scheduler')
        if not sched or sched == 'default':
            return ''
        if sched == 'odd-even':
            return _ODD_EVEN_SCHEDULER
        raise NotImplementedError(f'scheduler {sched!r}')

    def _default_title(self) -> str:
        return f'{self.prog.name}'

    def _default_on_terminate(self) -> str:
        if self.G_kind == 'scalar_local':
            return 'final value: G[0] = ${G[0]}'
        return 'final state: G = ${JSON.stringify(G)}'


# ============================================================
#  Templates
# ============================================================

JS_TEMPLATE = """\
// Auto-generated from {src_basename} by lang/llcjs.py.
// Do not edit by hand; edit the .llp source and rerun the build.

{aux_methods}LLP.mount(document.getElementById('{container_id}'), {{
  title: '{title}',
  program: `\n{program_text}\n`,
  inputs: [
{inputs_js}
  ],
  buildState(inputs) {{
{buildState}
  }},
  forbidden(j, G, ctx) {{
    const n = ctx.n;
{forbidden_body}
  }},
  advance(j, G, ctx) {{
    const n = ctx.n;
{advance_body}
  }},
  describe(G, ctx) {{
    return `{describe}`;
  }},
  onTerminate(G, ctx) {{
    return `{on_terminate}`;
  }}{scheduler}
}});
"""


_ODD_EVEN_SCHEDULER = """,
  nextIndex(G, ctx) {
    const n = ctx.n;
    const phaseObj = ctx.__phase;
    for (let pass = 0; pass < 2; pass++) {
      for (let j = 0; j < n; j++) {
        if ((j % 2) === phaseObj.phase) {
          if (j < n - 1 && G[j] > G[j + 1]) return j;
        }
      }
      phaseObj.phase = 1 - phaseObj.phase;
    }
    return null;
  }"""


# ============================================================
#  Imperative emitter (guard-free programs, e.g. Chapter 1)
# ============================================================

class JSImperativeEmitter:
    """Emits a `LLP.mountFunction(...)` spec for guard-free LL programs.

    The emitted JS instruments every assignment / return / loop iteration
    with a `__rec(label, frame)` call so the website runtime can step
    through execution one mutation at a time.
    """

    def __init__(self, prog: Program, sidecar: dict, src_basename: str,
                 src_text: str = ''):
        self.prog = prog
        self.sc = sidecar
        self.src_basename = src_basename
        self.src_text = src_text

    def emit(self) -> str:
        prog = self.prog
        title = self.sc.get('title', f'{prog.name}: demo')
        container_id = self.sc.get('container', f'demo-{prog.name.lower()}')
        signature = self._signature(prog.entry_name, prog.params, prog.return_type)
        program_text = clean_llp_for_display(self.src_text) if self.src_text else ''
        inputs_decl = self._inputs_decl()
        # Emit entry + every aux method as nested JS functions.  All share
        # the closure over `__rec`, so any frame can record a snapshot.
        method_lines: List[str] = []
        method_lines.extend(self._emit_method(prog.entry_name, prog.params,
                                              prog.return_type, prog.stmts))
        for m in prog.aux_methods:
            method_lines.extend(self._emit_method(m.name, m.params,
                                                  m.return_type, m.body))
        body_arg_names = [p.name for p in prog.params]
        call_args = ', '.join(f'inputs.{n}' for n in body_arg_names)
        out: List[str] = []
        out.append(f'// Auto-generated from {self.src_basename} by lang/llcjs.py.')
        out.append('// Do not edit by hand; edit the .llp source and rerun the build.')
        out.append('')
        out.append(f"LLP.mountFunction(document.getElementById('{container_id}'), {{")
        out.append(f'  title: {json.dumps(title)},')
        out.append(f'  signature: {json.dumps(signature)},')
        if program_text:
            out.append(f'  program: {json.dumps(program_text)},')
        out.append('  inputs: ' + inputs_decl + ',')
        out.append('  trace(inputs) {')
        out.append('    const steps = [];')
        out.append('    function __clone(v) {')
        out.append('      if (Array.isArray(v)) return v.map(__clone);')
        out.append('      return v;')
        out.append('    }')
        out.append('    function __rec(label, frame) {')
        out.append('      // trace() runs synchronously and to completion before any')
        out.append('      // Step/Run-all button exists (see llp.js mountFunction reset()),')
        out.append('      // so a non-terminating input has no yield point to be')
        out.append('      // interrupted at from outside -- this cap is the only thing')
        out.append('      // that can stop it short of freezing the tab.')
        out.append('      if (steps.length >= 100000) {')
        out.append('        throw new Error("trace exceeded 100000 steps without finishing -- '
                   'this looks like a non-terminating input rather than a slow one");')
        out.append('      }')
        out.append('      const snap = {};')
        out.append('      for (const k in frame) snap[k] = __clone(frame[k]);')
        out.append('      steps.push({ label, vars: snap });')
        out.append('    }')
        out.append('    let __opCount = 0;')
        out.append('    function __tick() {')
        out.append('      // Unconditional loop-iteration cap, independent of __rec above:')
        out.append('      // __rec only fires from a decl/assign/swap/return statement, so a')
        out.append('      // loop whose body is built only from call/if/while/forall (no leaf')
        out.append('      // statement of those four kinds anywhere inside it) would never')
        out.append('      // increment steps.length at all and could still spin forever')
        out.append('      // uncaught -- this ticks on every loop iteration itself instead,')
        out.append('      // so it catches that shape too.')
        out.append('      if (++__opCount > 100000) {')
        out.append('        throw new Error("trace exceeded 100000 loop iterations without '
                   'finishing -- this looks like a non-terminating input rather than a slow one");')
        out.append('      }')
        out.append('    }')
        for line in method_lines:
            out.append('    ' + line)
        out.append(f'    const result = {prog.entry_name}({call_args});')
        out.append('    return { steps, result };')
        out.append('  }')
        out.append('});')
        return '\n'.join(out) + '\n'

    # --- helpers ---

    def _signature(self, name, params, ret) -> str:
        ret_s = self._ts(ret) if ret is not None else 'void'
        ps = ', '.join(f'{self._ts(p.type)} {p.name}' for p in params)
        return f'{ret_s} {name}({ps})'

    def _ts(self, t: 'Type') -> str:
        s = t.base
        if t.is_2d:
            s += '[][]'
        elif t.is_array:
            s += '[]'
        return s

    def _inputs_decl(self) -> str:
        # One descriptor per entry-method param; defaults from sidecar.
        defaults = self.sc.get('defaults', {})
        labels = self.sc.get('labels', {})
        items = []
        for p in self.prog.params:
            kind = ('int-array' if p.type.is_array and not p.type.is_2d
                    else 'matrix' if p.type.is_2d
                    else p.type.base)
            d = defaults.get(p.name)
            if d is None:
                d = '' if p.type.is_array or p.type.is_2d else '0'
            label = labels.get(p.name, p.name)
            items.append('{ ' + (
                f'name: {json.dumps(p.name)}, label: {json.dumps(label)}, '
                f'kind: {json.dumps(kind)}, default: {json.dumps(str(d))}'
            ) + ' }')
        return '[' + ', '.join(items) + ']'

    def _emit_method(self, name, params, ret, body) -> List[str]:
        sig_params = ', '.join(p.name for p in params)
        out: List[str] = []
        out.append(f'function {name}({sig_params}) {{')
        # Record entry: snapshot all params.
        out.append(f'  __rec({json.dumps(f"enter {name}")}, {{ {sig_params} }});')
        locals_set = set(p.name for p in params)
        # The parser pulls the entry method's own *leading* declarations
        # out into prog.decls (JSEmitter's guard-based state classifier
        # wants them separated from prog.stmts) -- but a guard-free
        # program has no such split conceptually, it's all one flat
        # statement sequence, and this emitter never reads prog.decls
        # anywhere else. Re-synthesize them as ordinary decl statements
        # here, or they're silently dropped entirely (confirmed live:
        # LLPwithRejection's `int[][] valid = ...; int i = 0;` never made
        # it into the output, so the very next `while (i < n)` referenced
        # an undeclared `i`). Aux methods (this same function, called per
        # method) don't get this treatment -- their own decls already stay
        # in their own m.body, never split out.
        if name == self.prog.entry_name:
            for d in self.prog.decls:
                out.extend(self._emit_stmt(
                    Stmt(kind='decl', decl_type=d.type, decl_name=d.name, decl_init=d.init),
                    '  ', locals_set, name))
        for s in body:
            out.extend(self._emit_stmt(s, '  ', locals_set, name))
        # Re-add the trailing `return result_name;` that the parser stripped
        # (only for the entry method, since `_scan_result` only walks
        # `prog.stmts`).  Aux methods keep their explicit returns intact.
        is_entry = (name == self.prog.entry_name)
        if is_entry and ret is not None and self.prog.result_name is not None:
            out.append(f'  return {self.prog.result_name};')
        out.append('}')
        return out

    def _emit_stmt(self, s, indent, locals_set, fname) -> List[str]:
        """Translate one LL statement; mutates locals_set when introducing a decl."""
        k = s.kind
        if k == 'decl':
            init = self._jx(s.decl_init) if s.decl_init is not None else self._zero(s.decl_type)
            locals_set.add(s.decl_name)
            frame = '{ ' + ', '.join(sorted(locals_set)) + ' }'
            return [
                f'{indent}let {s.decl_name} = {init};',
                f'{indent}__rec("{s.decl_name} = " + JSON.stringify({s.decl_name}), {frame});',
            ]
        if k == 'assign':
            if s.lhs_index is not None:
                lhs = f'{s.lhs_name}[{self._jx(s.lhs_index)}]'
                if s.lhs_index2 is not None:
                    lhs = f'{lhs}[{self._jx(s.lhs_index2)}]'
                    for extra in s.lhs_index_extra:
                        lhs = f'{lhs}[{self._jx(extra)}]'
            else:
                lhs = s.lhs_name
            frame = '{ ' + ', '.join(sorted(locals_set)) + ' }'
            return [
                f'{indent}{lhs} = {self._jx(s.rhs)};',
                f'{indent}__rec({json.dumps(f"{lhs} = ...")}, {frame});',
            ]
        if k == 'if':
            out = [f'{indent}if ({self._jx(s.cond)}) {{']
            for t in s.then_stmts:
                out.extend(self._emit_stmt(t, indent + '  ', set(locals_set), fname))
            if s.else_stmts:
                out.append(f'{indent}}} else {{')
                for t in s.else_stmts:
                    out.extend(self._emit_stmt(t, indent + '  ', set(locals_set), fname))
            out.append(f'{indent}}}')
            return out
        if k == 'while':
            out = [f'{indent}while ({self._jx(s.cond)}) {{', f'{indent}  __tick();']
            for t in s.body:
                out.extend(self._emit_stmt(t, indent + '  ', set(locals_set), fname))
            out.append(f'{indent}}}')
            return out
        if k == 'forall':
            lo = self._jx(s.loop_lo)
            hi = self._jx(s.loop_hi)
            inner_locals = set(locals_set); inner_locals.add(s.loop_var)
            out = [f'{indent}for (let {s.loop_var} = {lo}; {s.loop_var} <= {hi}; {s.loop_var}++) {{',
                   f'{indent}  __tick();']
            for t in s.body:
                out.extend(self._emit_stmt(t, indent + '  ', inner_locals, fname))
            out.append(f'{indent}}}')
            return out
        if k == 'return':
            if s.return_value is None:
                return [f'{indent}__rec({json.dumps(f"return from {fname}")}, {{}});',
                        f'{indent}return;']
            v = self._jx(s.return_value)
            return [
                f'{indent}{{ const __r = {v}; '
                f'__rec({json.dumps(f"return from {fname} = ")} + JSON.stringify(__r), {{}});'
                f' return __r; }}'
            ]
        if k == 'call':
            args = ', '.join(self._jx(a) for a in s.call_args)
            return [f'{indent}{s.call_name}({args});']
        if k == 'swap':
            a = self._jx(s.swap_a); b = self._jx(s.swap_b)
            frame = '{ ' + ', '.join(sorted(locals_set)) + ' }'
            return [f'{indent}{{ const __t = {a}; {a} = {b}; {b} = __t; }}',
                    f'{indent}__rec({json.dumps(f"swap")}, {frame});']
        if k == 'block':
            out: List[str] = []
            for t in s.body:
                out.extend(self._emit_stmt(t, indent, set(locals_set), fname))
            return out
        raise NotImplementedError(f'imperative emitter: stmt kind {k!r}')

    def _jx(self, e: 'Expr') -> str:
        """Translate an LL expression to a JS expression (no LL context tricks)."""
        k = e.kind
        if k == 'int': return str(e.value)
        if k == 'real': return repr(e.value)
        if k == 'bool': return 'true' if e.value else 'false'
        if k == 'infinity': return 'Infinity'
        if k == 'id': return e.name
        if k == 'index':
            return f'{e.name}[{self._jx(e.idx)}]'
        if k == 'index2':
            chain = ''.join(f'[{self._jx(x)}]' for x in [e.idx, e.rhs, *e.idx_extra])
            return f'{e.name}{chain}'
        if k == 'bin':
            # Integer division: emit Math.trunc so that LL's `int / int`
            # behaves like Java rather than JS float division.
            if e.op == '/':
                return f'Math.trunc({self._jx(e.lhs)} / {self._jx(e.rhs)})'
            if e.op == 'in':
                return f'{self._jx(e.rhs)}.includes({self._jx(e.lhs)})'
            op = JS_BIN.get(e.op)
            if op is None: raise NotImplementedError(f'op {e.op!r}')
            return f'({self._jx(e.lhs)} {op} {self._jx(e.rhs)})'
        if k == 'un':
            if e.op in ('not', '!'): return f'!({self._jx(e.lhs)})'
            if e.op in ('neg', '-'): return f'-({self._jx(e.lhs)})'
        if k == 'call':
            args = ', '.join(self._jx(a) for a in e.args)
            if e.name == 'min' and len(e.args) == 2:
                return f'Math.min({args})'
            if e.name == 'max' and len(e.args) == 2:
                return f'Math.max({args})'
            if e.name == 'abs':
                return f'Math.abs({args})'
            return f'{e.name}({args})'
        if k == 'field_access':
            return f'{self._jx(e.lhs)}.{e.name}'
        if k == 'method_call':
            args = ', '.join(self._jx(a) for a in e.args)
            return f'{self._jx(e.lhs)}.{e.name}({args})'
        if k == 'new_array':
            fill = 'false' if e.name == 'boolean' else '0'
            if e.idx2 is None:
                return f'new Array({self._jx(e.idx)}).fill({fill})'
            sizes = [self._jx(x) for x in [e.idx, e.idx2, *e.idx_extra]]
            return _render_new_array_dims(sizes, fill)
        raise NotImplementedError(f'imperative emitter: expr kind {k!r}')

    def _zero(self, t: 'Type') -> str:
        if t.is_2d or t.is_array: return '[]'
        if t.base == 'boolean': return 'false'
        return '0'


# ============================================================
#  Entry point
# ============================================================

def _has_guards(prog: Program) -> bool:
    def walk(stmts):
        for s in stmts:
            if s.kind == 'guard': return True
            if s.kind == 'if' and (walk(s.then_stmts) or walk(s.else_stmts)): return True
            if s.kind in ('forall', 'while', 'block') and walk(s.body): return True
            if s.kind == 'par' and walk(s.par_branches): return True
        return False
    return walk(prog.stmts)


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', help='LL source file (.llp)')
    ap.add_argument('--out', help='output JS file (default: <stem>.js)')
    args = ap.parse_args(argv)

    src = open(args.source).read()
    stem = os.path.splitext(os.path.basename(args.source))[0]
    prog = Parser(tokenize(src), default_name=stem).parse_program()

    sidecar_path = os.path.splitext(args.source)[0] + '.demo.json'
    sidecar = {}
    if os.path.isfile(sidecar_path):
        with open(sidecar_path) as f:
            sidecar = json.load(f)

    if _has_guards(prog):
        em = JSEmitter(prog, sidecar, os.path.basename(args.source), src)
    else:
        em = JSImperativeEmitter(prog, sidecar, os.path.basename(args.source), src)
    js = em.emit()

    out = args.out or os.path.splitext(args.source)[0] + '.js'
    with open(out, 'w') as f:
        f.write(js)
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
