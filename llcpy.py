#!/usr/bin/env python3
"""LLCPY -- LL-to-Python emitter.

Compiles an LL source file to a readable Python module, in the same spirit
as llc.py's Java output: one class per program, the entry method named
after the class, auxiliary methods alongside it, and a `__main__` block
carrying the demo the .llp's own main shows.

Why this exists
---------------
The Java port of every program is generated from the .llp and has never
drifted from it. The hand-written Python and Rust ports have: on
2026-10-23, 56 of 139 programs had a .py or .rs older than their .llp, and
the differences were not cosmetic -- Prim was running three different
algorithms across its four language ports, and LLPGCD.rs held an algorithm
from a different chapter entirely. Output comparison does not find these,
because every one of them returned correct results; only the complexity
differed. Generating Python removes the whole class of problem.

Architecture
------------
The parser is llc.py's -- this module imports tokenize/Parser/Program/etc.
and only replaces the emitter, exactly as llcjs.py does. Where a lowering
DECISION has to be made (is this guard a plain rescan, a worklist, a
topological sweep, a keyed container?), the decision is taken from a
CodeGen instance built over the same Program, so the Python and the Java
cannot disagree about which shape a program gets. Only the printing
differs.

Usage:
  python3 llcpy.py FILE.llp [--out DIR]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llc import (tokenize, Parser, Program, Decl, Stmt, Expr, Type,  # noqa: E402
                 CodeGen, extract_top_comment, scan_signatures,
                 _is_wholesale_init)

import keyword
import re

INT_INF = '2147483647'        # Java's Integer.MAX_VALUE, so checksums agree
DBL_INF = "float('inf')"


class _DoubleCtx:
    """Minimal stand-in for a double type, so `infinity` compared against a
    double array picks the real infinity rather than the int sentinel --
    llc.py knows the array's type at that point and emits
    Double.POSITIVE_INFINITY."""
    base = 'double'


_DBL = _DoubleCtx()


# Builtins this emitter calls by name. A program is free to use any of
# them as a variable -- OptimalBinarySearchTree.llp loops over `len` -- and
# Python then treats the name as local for the WHOLE function, so the
# `len(...)` call above the loop raises UnboundLocalError.
_PY_SHADOW = {'len', 'min', 'max', 'abs', 'sum', 'range', 'list', 'int',
              'float', 'bool', 'str', 'any', 'all', 'next', 'id', 'type',
              'dict', 'set', 'sorted', 'iter', 'filter', 'map', 'object'}


def pyname(n: str) -> str:
    """LL has its own reserved words, so a program may legitimately use a
    name Python reserves -- TwoSAT.llp takes a parameter called `from`.
    Suffix those rather than rejecting the program."""
    return n + '_' if (keyword.iskeyword(n) or n in _PY_SHADOW) else n


class PyGen:
    def __init__(self, prog: Program, signatures=None, src_text: str = ''):
        self.p = prog
        self.src_text = src_text
        # The Java generator, used only to answer lowering questions so the
        # two backends cannot pick different shapes for the same program.
        self.cg = CodeGen(prog, signatures=signatures, src_text=src_text)
        self.syms = {d.name: d for d in prog.decls}
        for pm in prog.params:
            self.syms[pm.name] = pm
        self.locals: dict = {}
        self._tmp = 0
        self._picks: list = []
        self._cur_idx: list = []
        self._pick_idx: dict = {}
        # Set only while a container sweep's advance is being
        # emitted: the initialisation before the guard assigns
        # the key array too, and must not push (the Java guards
        # those with `if (_container != null)`).
        self._container_name = None
        self._siblings_used: set = set()

    # ---------- small helpers ----------

    def fresh(self, p='_t'):
        self._tmp += 1
        return f'{p}{self._tmp}'

    def _type_of(self, name):
        d = self.locals.get(name) or self.syms.get(name)
        if d is None:
            return None
        return getattr(d, 'type', None) or getattr(d, 'decl_type', None)

    def _infty_for(self, t: Type | None) -> str:
        """Match llc.py's choice: int contexts use Integer.MAX_VALUE, double
        contexts use an actual infinity. Keeping the same sentinel is what
        lets the generated Python and Java agree on checksums."""
        if t is not None and t.base == 'double':
            return DBL_INF
        return INT_INF

    def _zero(self, t: Type) -> str:
        return self._zero_for_base(getattr(t, 'base', None))

    def _zero_for_base(self, base) -> str:
        if base == 'double':
            return '0.0'
        if base == 'boolean':
            return 'False'
        return '0'


    # ---------- block syntax hooks ----------
    # Python closes a block by dedenting; a brace language needs a closing
    # line. Everything structural goes through these four so a sibling
    # backend can reuse the whole emitter and change only the surface.

    SELF = 'self.'
    COMMENT = '# '
    EQ, NE, AND = '==', '!=', 'and'
    MEMBER_IND = '    '        # indent of a method inside the class
    BODY_IND = '        '      # indent of a statement inside a method

    def _blk(self, ind, header, body):
        return [f'{ind}{header}:'] + body

    def _close(self, ind):
        return []

    def _if(self, ind, cond, body, els=None):
        out = self._blk(ind, f'if {cond}', body) + self._close(ind)
        if els:
            out += self._blk(ind, 'else', els) + self._close(ind)
        return out

    def _while(self, ind, cond, body):
        return self._blk(ind, f'while {cond}', body) + self._close(ind)

    def _forrange(self, ind, v, lo, hi, body):
        return (self._blk(ind, f'for {v} in range({lo}, ({hi}) + 1)', body)
                + self._close(ind))

    def _forin(self, ind, v, it, body):
        return self._blk(ind, f'for {v} in {it}', body) + self._close(ind)

    def _def(self, ind, name, params, body):
        ps = ', '.join(['self'] + list(params))
        return self._blk(ind, f'def {name}({ps})', body) + self._close(ind)

    def _lit(self, v):
        return {'true': 'True', 'false': 'False', 'null': 'None'}.get(v, v)

    # Leaf statements and container primitives. The guard lowerings below are
    # written entirely in terms of these, so a sibling backend inherits the
    # lowering itself -- the part that decides what algorithm runs -- and
    # supplies only how a line is spelled.

    def _break(self, ind):
        return [f'{ind}break']

    def _decl(self, ind, name, rhs):
        return [f'{ind}{name} = {rhs}']

    def _let(self, ind, name, rhs):
        return [f'{ind}{name} = {rhs}']

    def _set(self, ind, lhs, rhs):
        return [f'{ind}{lhs} = {rhs}']

    def _incr(self, ind, lhs, delta):
        if str(delta).startswith('-'):
            return [f'{ind}{lhs} -= {str(delta)[1:]}']
        return [f'{ind}{lhs} += {delta}']

    def _swap(self, ind, a, b):
        return [f'{ind}{a}, {b} = {b}, {a}']

    def _append(self, ind, arr, v):
        return [f'{ind}{arr}.append({v})']

    def _raise(self, ind, msg):
        return [f'{ind}raise ValueError({msg!r})']

    def _fill(self, fill, size):
        return f'[{fill}] * ({size})'

    def _zeros(self, size):
        return f'[0] * ({size})'

    def _lists(self, size):
        return f'[[] for _ in range({size})]'

    def _emptylist(self):
        return '[]'

    def _len(self, x):
        return f'len({x})'

    def _pass(self, ind):
        return [ind + 'pass']

    def _continue(self, ind):
        return [ind + 'continue']

    def _nonempty(self, name):
        return name

    def _heappush(self, ind, name, key, val):
        return [f'{ind}heapq.heappush({name}, ({key}, {val}))']

    def _heappop(self, name):
        """The index at the top of the container; the key is discarded."""
        return f'heapq.heappop({name})[1]'

    def _reduce_identity(self, q, base):
        """What min/max returns over an empty selection. It has to follow the
        element type: a reduction over doubles that fell back to the int
        sentinel made LLP-BellmanFord report 2147483653 where the Java,
        which knows the array is double[], reports infinity."""
        if base == 'double':
            return DBL_INF if q == 'min' else '-' + DBL_INF
        return INT_INF if q == 'min' else '-' + INT_INF

    def _index_fill(self, ind, lhs, size):
        """`int[] G = _i;` fills each cell with its own index -- llc.py emits
        `for (int i = 0; i < n; i++) this.G[i] = i;` for it."""
        v = self._tmpname('_ix')
        return (self._set(ind, lhs, self._zeros(size))
                + self._forrange(ind, v, '0', f'{size} - 1',
                                 self._set(ind + '    ', f'{lhs}[{v}]', v)))

    @staticmethod
    def _is_index_fill(init):
        return (init is not None and init.kind == 'id' and init.name == '_i')

    def _new_instance(self, name):
        return f'{name}()'

    def _class_decl(self, name, body):
        return [f'class {name}:'] + body

    # ---------- the runnable demo ----------

    _MAIN_DECL = re.compile(r'^\s*([A-Za-z0-9_]+(?:\s*\[\s*\])*)\s+(\w+)\s*=\s*(.*);\s*$')

    def _main_literal(self, v):
        return re.sub(r'\btrue\b', 'True', re.sub(r'\bfalse\b', 'False', v))

    def _main_block(self, body):
        return ["if __name__ == '__main__':"] + body

    def _main_decl(self, ind, name, value):
        return [f'{ind}{name} = {value}']

    def _main_print(self, ind, expr):
        return [f'{ind}print({expr})']

    def _main_argexpr(self, a):
        # pyname here too: LLPAntichain takes a parameter called `len`, which
        # the declaration renames -- passing the un-renamed name handed the
        # program Python's builtin instead of the array.
        m = re.match(r'^(\w+)\.length\s*(.*)$', a)
        if m:
            tail = (' ' + m.group(2)) if m.group(2) else ''
            return f'len({pyname(m.group(1))}){tail}'
        return pyname(a) if re.match(r'^\w+$', a) and not a.isdigit() else a

    def emit_main(self):
        """A runnable demo, built from the same synthesizer llc.py's Java
        main uses. Reusing it rather than re-deriving inputs keeps the three
        languages showing the same example, and keeps the tuning recorded in
        _synthesize_main_inputs -- several of those defaults exist to avoid
        confirmed hangs -- in one place."""
        try:
            synth = self.cg._synthesize_main_inputs()
        except Exception:
            synth = None
        if synth is None:
            return []
        decls, args = synth
        ind = '    '
        body = []
        for d in decls:
            m = self._MAIN_DECL.match(d)
            if m is None:
                return []
            jtype, name, val = m.group(1), m.group(2), m.group(3)
            val = re.sub(r'^new\s+[A-Za-z0-9_]+(?:\s*\[\s*\])+\s*', '', val).strip()
            val = val.replace('{', '[').replace('}', ']')
            body += self._main_decl(ind, pyname(name), self._main_literal(val))
        call_args = ', '.join(self._main_argexpr(a) for a in args)
        entry = self.p.entry_name or self.p.name
        body += self._main_decl(ind, 'prog', self._new_instance(self.p.name))
        rt = getattr(self.p, 'return_type', None)
        is_void = (rt is None) or (getattr(rt, 'base', None) == 'void'
                                   and not getattr(rt, 'is_array', False)
                                   and not getattr(rt, 'is_2d', False))
        call = f'prog.{entry}({call_args})'
        if is_void:
            body.append(f'{ind}{call}' + (';' if self.SELF == 'this.' else ''))
            names = {self._MAIN_DECL.match(d).group(2) for d in decls
                     if self._MAIN_DECL.match(d)}
            for a, prm in zip(args, self.p.params):
                if (prm.type.is_array or prm.type.is_2d) and a in names:
                    body += self._main_print(ind, pyname(a))
        else:
            body += self._main_decl(ind, 'result', call)
            body += self._main_print(ind, 'result')
        return [''] + self._main_block(body)

    def _tmpname(self, base):
        """Scratch name for a variable the lowering introduces. Python lets a
        second guard in the same function rebind `_changed`; a `let` does not,
        so a brace backend overrides this to keep them distinct."""
        return base

    # ---------- expressions ----------

    # LL writes equality as `=`, the way the book's pseudocode does.
    OPS = {'=': '==', '&&': 'and', '||': 'or', 'and': 'and', 'or': 'or', '!=': '!=',
           '==': '==', '<': '<', '>': '>', '<=': '<=', '>=': '>=',
           '+': '+', '-': '-', '*': '*', '%': '%'}

    def expr(self, e: Expr, ctx: Type | None = None) -> str:
        k = e.kind
        if k == 'int':
            return str(e.value)
        if k == 'real':
            return repr(float(e.value))
        if k == 'bool':
            return 'True' if e.value else 'False'
        if k == 'infinity':
            return self._infty_for(ctx)
        if k == 'id':
            if e.name == 'infinity':
                return self._infty_for(ctx)
            if e.name in ('true', 'false', 'null'):
                return self._lit(e.name)
            return self._ref(e.name)
        if k == 'index':
            return f'{self.expr(e.lhs) if e.lhs is not None else self._ref(e.name)}[{self.expr(e.idx)}]'
        if k == 'index2':
            # The second index lives in `rhs`, not `idx2` -- llc.py's Java
            # emitter reads [e.idx, e.rhs, *e.idx_extra] and this has to
            # agree with it. Reading idx2 here silently dropped the second
            # subscript: aw[j][t] came out as aw[j].
            base = self.expr(e.lhs) if e.lhs is not None else self._ref(e.name)
            out = base
            for ix in [e.idx, e.rhs] + list(e.idx_extra or []):
                if ix is not None:
                    out += f'[{self.expr(ix)}]'
            return out
        if k == 'field_access':
            obj = self.expr(e.lhs)
            if e.name == 'length':
                return f'len({obj})'
            if e.idx is not None:
                return f'{obj}.{e.name}[{self.expr(e.idx)}]'
            return f'{obj}.{e.name}'
        if k == 'method_call':
            return self._method_call(e)
        if k == 'new_heap':
            return '[]'
        if k == 'bin':
            op = self.OPS.get(e.op, e.op)
            if e.op == '/':
                lt = self._static_base(e.lhs)
                rt = self._static_base(e.rhs)
                # LL's `/` on ints is Java integer division, which truncates
                # toward zero; Python's // floors. They differ on negatives,
                # so emit int(a / b) there rather than a // b.
                if lt == 'int' and rt == 'int':
                    return f'int({self.expr(e.lhs)} / {self.expr(e.rhs)})'
                return f'{self.expr(e.lhs)} / {self.expr(e.rhs)}'
            if e.op in ('<', '>', '<=', '>=', '=', '==', '!='):
                # A comparison gives `infinity` its context: the sentinel has
                # to match the type of what it is compared against.
                lb, rb = self._static_base(e.lhs), self._static_base(e.rhs)
                if 'double' in (lb, rb):
                    ctx = _DBL
            return f'({self.expr(e.lhs, ctx)} {op} {self.expr(e.rhs, ctx)})'
        if k == 'un':
            if e.op in ('not', '!'):
                return f'(not {self.expr(e.lhs)})'
            if e.op == 'neg':
                return f'(-{self.expr(e.lhs)})'
            return f'({e.op}{self.expr(e.lhs)})'
        if k == 'new_array':
            return self._new_array(e)
        if k == 'slice':
            return f'{self._ref(e.name)}[{self.expr(e.slice_lo)}:({self.expr(e.slice_hi)}) + 1]'
        if k == 'call':
            return self._call(e)
        if k == 'quant' or e.quant:
            return self.quant(e)
        raise NotImplementedError(f'expression kind {k!r}')

    def _static_base(self, e: Expr):
        if e.kind == 'int':
            return 'int'
        if e.kind == 'double':
            return 'double'
        if e.kind in ('id', 'index', 'index2'):
            t = self._type_of(e.name)
            return t.base if t is not None else None
        if e.kind == 'bin':
            lt = self._static_base(e.lhs)
            rt = self._static_base(e.rhs)
            return 'double' if 'double' in (lt, rt) else lt
        return None

    IMPLICIT_FIELDS = {'n'}


    def _ref(self, name):
        if name in self.locals:
            return pyname(name)
        if name in self.syms or name in self.IMPLICIT_FIELDS:
            return f'self.{pyname(name)}'
        return pyname(name)

    def _new_array(self, e: Expr) -> str:
        # The element type is in `name` as a plain string (`new boolean[n]`
        # parses with name='boolean'), not in `value`. Reading only `value`
        # filled every array with 0, so a boolean[] printed as 0 where the
        # Java printed false -- 11 programs.
        t = e.value if isinstance(e.value, Type) else None
        fill = (self._zero(t) if t is not None
                else self._zero_for_base(getattr(e, 'name', None)))
        dims = [self.expr(x) for x in
                ([e.idx, e.idx2] + list(e.idx_extra or [])) if x is not None]
        if e.jagged or not dims:
            return f'[None] * ({dims[0]})' if dims else '[]'
        return self._nested_array(dims, fill)

    def _nested_array(self, dims, fill):
        """`new int[n][m][n][m]` is four dimensions, not two: idx and idx2
        hold the first pair and idx_extra the rest. Building only the first
        two left LLPwithRejection assigning into an int where it wanted a
        row."""
        out = f'[{fill}] * ({dims[-1]})'
        for d in reversed(dims[:-1]):
            out = f'[{out} for _ in range({d})]'
        return out

    def _method_call(self, e: Expr) -> str:
        """heap<int> builtin methods map onto heapq over (key, value) tuples,
        which orders exactly as the Java PriorityQueue<int[]> comparator does
        (key first, then value)."""
        # `UnionFind.union(...)` names a sibling PROGRAM, not a local object.
        # llc.py compiles it to `new UnionFind().union(...)`; the sibling's
        # class is inlined above this one so the file stays standalone.
        if (e.lhs is not None and getattr(e.lhs, 'kind', None) == 'id'
                and self._sibling(e.lhs.name) is not None):
            self._siblings_used.add(e.lhs.name)
            args = ', '.join(self.expr(a) for a in e.args)
            return f'{self._new_instance(e.lhs.name)}.{e.name}({args})'
        r = self.expr(e.lhs)
        meth = e.name
        args = [self.expr(a) for a in e.args]
        if meth == 'push':
            return f'heapq.heappush({r}, ({args[0]}, {args[1]}))'
        if meth == 'removeMin':
            return f'heapq.heappop({r})'
        if meth == 'isEmpty':
            return f'(not {r})'
        if meth == 'peekKey':
            return f'{r}[0][0]'
        if meth == 'peekValue':
            return f'{r}[0][1]'
        if meth in ('length', 'size'):
            return f'len({r})'
        if meth == 'clone':
            return f'list({r})'
        # list<T>: LL spells these the way the book does; a Python list
        # has the same operations under different names.  `add` is the
        # one-argument list append -- a heap uses push(key, value).
        if meth == 'add' and len(args) == 1:
            return f'{r}.append({args[0]})'
        if meth == 'removeFirst' and not args:
            return f'{r}.pop(0)'
        if meth == 'removeLast' and not args:
            return f'{r}.pop()'
        if meth == 'first' and not args:
            return f'{r}[0]'
        return f'{r}.{meth}({", ".join(args)})'

    def _call(self, e: Expr) -> str:
        name = e.name
        args = [self.expr(a) for a in e.args]
        if name == 'min':
            return f'min({", ".join(args)})'
        if name == 'max':
            return f'max({", ".join(args)})'
        if name == 'abs':
            return f'abs({args[0]})'
        sib = self._sibling(name)
        if sib is not None:
            # `Dijkstra(adj, wPrime, u)` names a sibling PROGRAM, not a method
            # of this one -- llc.py compiles it to
            # `new Dijkstra().shortestPath(...)` and ships the two .java
            # files together. The companion serves one file per language, so
            # the sibling's class is inlined above this one instead.
            self._siblings_used.add(name)
            return (f'{self._new_instance(name)}.'
                    f'{sib.entry_name or sib.name}({", ".join(args)})')
        return f'{self.SELF}{name}({", ".join(args)})'

    def _sibling(self, name):
        if name == self.p.name or name == (self.p.entry_name or self.p.name):
            return None
        if any(m.name == name for m in self.p.aux_methods):
            return None
        sigs = getattr(self.cg, 'signatures', None) or {}
        return sigs.get(name)

    # ---------- quantifiers ----------

    def quant(self, e: Expr) -> str:
        """min/max/sum/count/exists/forall over a range or a set.

        Emitted as a generator expression where possible. `pick` needs the
        witness afterwards, so those become a helper loop instead and are
        handled by the statement layer.
        """
        v = e.q_var
        if e.q_set is not None:
            it = f'for {v} in {self.expr(e.q_set)}'
        else:
            it = f'for {v} in range({self.expr(e.q_lo)}, ({self.expr(e.q_hi)}) + 1)'
        where = f' if {self.expr(e.q_where)}' if e.q_where is not None else ''
        prev = dict(self.locals)
        self.locals[v] = None
        try:
            body = self.expr(e.q_body) if e.q_body is not None else v
            q = e.quant
            if q == 'exists':
                if e.pick_var:
                    if e not in self._picks:
                        self._picks.append(e)
                        self._pick_idx[id(e)] = list(self._cur_idx)
                    # The witness is needed by the advance, so this becomes a
                    # helper that records it -- the Java stashes it in a field
                    # for the same reason.
                    return f'self._pick_{e.pick_var}({self._pick_args(e)})'
                return f'any({body} {it}{where})'
            if q == 'forall':
                return f'all({body} {it}{where})'
            if q == 'sum':
                return f'sum({body} {it}{where})'
            if q == 'count':
                return f'sum(1 {it}{where} if {body})' if e.q_body is not None \
                       else f'sum(1 {it}{where})'
            if q in ('min', 'max'):
                ident = self._reduce_identity(
                    q, self._static_base(e.q_body) if e.q_body is not None else None)
                return f'{q}(({body} {it}{where}), default={ident})'
        finally:
            self.locals = prev
        raise NotImplementedError(f'quantifier {e.quant!r}')

    def _pick_args(self, e):
        """The witness search runs inside a guard, so it needs that guard's
        index variables -- the predicate mentions them."""
        return ', '.join(pyname(v) for v in self._pick_idx.get(id(e), []))

    def pick_helpers(self):
        """One helper per `exists ... pick v`: scan the range, record the
        first witness in self.<v>, and report whether one was found."""
        out = []
        for e in self._picks:
            v = e.pick_var
            gidx = self._pick_idx.get(id(e), [])
            prev = dict(self.locals)
            self.locals[v] = None
            for gv in gidx:
                self.locals[gv] = None
            try:
                if e.q_set is not None:
                    it = f'for {v} in {self.expr(e.q_set)}'
                else:
                    it = (f'for {v} in range({self.expr(e.q_lo)}, '
                          f'({self.expr(e.q_hi)}) + 1)')
                cond = self.expr(e.q_body) if e.q_body is not None else 'True'
                sig = ''.join(', ' + pyname(g) for g in gidx)
                out += ['', f'    def _pick_{v}(self{sig}):',
                        f'        {it}:',
                        f'            if {cond}:',
                        f'                self.{v} = {v}',
                        '                return True',
                        '        return False']
            finally:
                self.locals = prev
        return out

    # ---------- statements ----------

    def stmts(self, ss, ind):
        out = []
        for s in ss:
            out += self.stmt(s, ind)
        return out or self._pass(ind)

    def stmt(self, s: Stmt, ind: str):
        k = s.kind
        if k == 'exprstmt':
            return [f'{ind}{self.expr(s.rhs if s.rhs is not None else s.cond)}']
        if k in ('decl', 'var'):
            name = s.decl_name
            self.locals[name] = s
            t = s.decl_type
            if t is not None and (t.is_heap or t.is_list):
                # A heap is a Python list used with heapq; a list<T> with no
                # initialiser is an empty sequence, the same as Java's
                # `new LinkedList<>()`.  `[]` is the empty list in the JS
                # emitter too, which inherits this branch.
                rhs = '[]'
            elif (t is not None and (t.is_array or t.is_2d or t.is_set)
                    and s.decl_init is not None
                    and not _is_wholesale_init(s.decl_init)):
                # `boolean[] G = false;` means allocate n cells and fill them,
                # exactly as the Java lowering reads it.
                size = self.SELF + 'n + 1' if self.p.base == 1 else self.SELF + 'n'
                if self._is_index_fill(s.decl_init):
                    return (self._decl(ind, pyname(name), self._zeros(size))
                            + self._index_fill(ind, pyname(name), size)[1:])
                rhs = self._fill(self.expr(s.decl_init, t), size)
            elif s.decl_init is not None:
                rhs = self.expr(s.decl_init, t)
            else:
                rhs = self._zero(t) if t is not None else self._lit('null')
            return self._decl(ind, pyname(name), rhs)
        if k == 'assign':
            out = self._set(ind, self._lhs(s), self.expr(s.rhs, self._lhs_type(s)))
            # Lowering the key of an index already in the container re-pushes
            # it at its new priority; the stale entry is discarded when it
            # surfaces and the guard rejects it.
            pa = self._priority_array()
            if (self._container_name and pa and s.lhs_name == pa
                    and s.lhs_index is not None):
                ix = self.expr(s.lhs_index)
                out += self._heappush(ind, self._container_name,
                                      f'{self._ref(pa)}[{ix}]', ix)
            return out
        if k == 'if':
            return self._if(ind, self.expr(s.cond),
                            self.stmts(s.then_stmts, ind + '    '),
                            self.stmts(s.else_stmts, ind + '    ') if s.else_stmts else None)
        if k == 'break':
            return self._break(ind)
        if k == 'while':
            return self._while(ind, self.expr(s.cond), self.stmts(s.body, ind + '    '))
        if k == 'forall':
            v = s.loop_var
            prev = dict(self.locals)
            self.locals[v] = None
            try:
                body = self.stmts(s.body, ind + '    ')
                if s.loop_set is not None:
                    return self._forin(ind, pyname(v), self.expr(s.loop_set), body)
                return self._forrange(ind, pyname(v), self.expr(s.loop_lo),
                                      self.expr(s.loop_hi), body)
            finally:
                self.locals = prev
        if k == 'swap':
            return self._swap(ind, self.expr(s.swap_a), self.expr(s.swap_b))
        if k == 'return':
            v = getattr(s, 'return_value', None)
            if v is None:
                return [f'{ind}return']
            if v.kind == 'id' and v.name == 'null':
                return [f'{ind}return {self._lit("null")}']
            return [f'{ind}return {self.expr(v)}']
        if k == 'block':
            return self.stmts(s.body, ind)
        if k == 'call':
            if s.call_name in ('setPriority', 'setContainer', 'setAcyclic',
                               'setBase', 'setMode'):
                return []          # configuration, consumed by the lowering
            args = ', '.join(self.expr(a) for a in s.call_args)
            return [f'{ind}{self.SELF}{s.call_name}({args})']
        if k == 'guard':
            return self.guard(s, ind)
        raise NotImplementedError(f'statement kind {k!r}')

    def _lhs(self, s: Stmt) -> str:
        out = self._ref(s.lhs_name)
        if s.lhs_index is not None:
            out += f'[{self.expr(s.lhs_index)}]'
        if s.lhs_index2 is not None:
            out += f'[{self.expr(s.lhs_index2)}]'
        for x in (s.lhs_index_extra or []):
            out += f'[{self.expr(x)}]'
        if s.lhs_field:
            out += f'.{s.lhs_field}'
            if s.lhs_field_index is not None:
                out += f'[{self.expr(s.lhs_field_index)}]'
        return out

    def _lhs_type(self, s: Stmt):
        return self._type_of(s.lhs_name)

    # ---------- the guard lowerings ----------

    def guard(self, g: Stmt, ind: str):
        """forbidden(j): PRED => advance: STMT

        The shape is taken from the Java generator so the two cannot differ:
        a topological sweep when setAcyclic promises one, otherwise the
        rescan fixpoint.
        """
        idx = g.guard_index or ['j']
        acy = self.p.acyclic_array
        # `forbidden ... => advance ... && forbidden ... => advance ...` is
        # ONE fixpoint over several rules, not several fixpoints: every
        # sweep tests every index against every rule. Driving rule 1 to its
        # own fixpoint first can leave it forbidden again once rule 2 moves
        # G, which is llc.py's reason for _emit_fixpoint_loop_multi.
        group = list(g.guard_group) if g.guard_group else [g]
        out = []
        prev = dict(self.locals)
        prev_idx = list(self._cur_idx)
        self._cur_idx = list(idx)
        for v in idx:
            self.locals[v] = None
        # The pick witness is a field, as it is in the Java (which stashes it
        # in picked_<v>), so the advance can read it after the guard ran.
        for cl in group:
            pv = getattr(cl, 'guard_pick_var', None)
            if pv:
                self.syms.setdefault(pv, None)
                self.locals.pop(pv, None)
        try:
            if self.p.mode == 'heap' and len(idx) == 1 and len(group) == 1:
                out += self._container_sweep(g, idx[0], *self._bounds(g, 0), ind)
            elif acy and len(idx) == 1:
                if len(group) > 1:
                    # llc.py falls back to separate loops outside the plain
                    # array sweep; rather than guess which, refuse. No
                    # program currently combines the two.
                    raise NotImplementedError(
                        'a &&-joined guard group under setAcyclic')
                out += self._topological(g, idx[0], *self._bounds(g, 0), acy, ind)
            else:
                # Rescan to a fixpoint: sweep every index, advance the
                # forbidden ones, repeat until a sweep changes nothing.
                # Multi-index guards nest one loop per index, as the Java
                # lowering does.
                dep = len(idx)
                ch = self._tmpname('_changed')
                Ii = ind + '    ' * (dep + 1)        # body of the innermost loop
                inner = self._always(g, Ii)
                for cl in group:
                    inner += self._if(Ii, self.expr(cl.guard_predicate),
                                      self._advance(cl, Ii + '    ')
                                      + self._set(Ii + '    ', ch,
                                                  self._lit('true')))
                for d in range(dep - 1, -1, -1):
                    lo, hi = self._bounds(g, d)
                    inner = self._forrange(ind + '    ' * (d + 1), pyname(idx[d]),
                                           lo, hi, inner)
                out += self._let(ind, ch, self._lit('true'))
                out += self._while(ind, ch,
                                   self._set(ind + '    ', ch,
                                             self._lit('false')) + inner)
        finally:
            self.locals = prev
            self._cur_idx = prev_idx
        return out

    def _always(self, g, ind):
        """`always : T v = expr;` bindings attached to a guard.

        llc.py emits these at the top of BOTH forbidden() and advance().
        Here the two are inlined into one sweep body, so one binding ahead
        of the predicate serves both, and an advance that re-reads the
        definition (the book's Always is re-read wherever it appears)
        assigns to the same name.

        Emitting nothing, as this did until 2026-10-24, leaves the guard
        reading whatever the PREVIOUS index's advance left in that name --
        a wrong guard that happens not to crash while the predicate's
        first disjunct short-circuits.
        """
        out = []
        for atype, aname, aexpr in (getattr(g, 'guard_always', None) or []):
            self.locals[aname] = None
            out += self._decl(ind, pyname(aname), self.expr(aexpr, atype))
        return out

    def _priority_array(self):
        """The array setContainer keys on, when the key is `A[j]` -- an
        assignment to it has to re-push, the way llc.py's
        `_container.add(new double[]{priority(k), k})` does."""
        pe = getattr(self.p, 'priority_expr', None)
        if pe is not None and pe.kind == 'index' and pe.name:
            return pe.name
        return None

    def _container_sweep(self, g, v, lo, hi, ind):
        """setContainer(PriorityQueue, key[j], min): a min-heap of
        (key, index) with lazy deletion, as llc.py emits it.

        This is not an optimisation that a rescan would merely do slowly.
        A program under setContainer is written knowing the index it is
        handed already carries the smallest key, so its predicate does not
        re-establish global minimality -- LLP-Prim's says only "unfixed and
        reachable". Sweeping such a predicate in index order fixes vertices
        in the wrong order and returns a different cost vector.
        """
        if getattr(self.p, 'priority_order', 'min') not in ('min', None):
            raise NotImplementedError('setContainer order other than min')
        pv = pyname(v)
        I1, I2 = ind + '    ', ind + '        '
        cn = self._tmpname('_container')
        pa = self._priority_array()
        out = [f'{ind}{self.COMMENT}setContainer: indices are offered '
               f'smallest-key-first, so the',
               f'{ind}{self.COMMENT}predicate below need not establish '
               f'global minimality itself.']
        out += self._let(ind, cn, self._emptylist())
        key_of = (lambda e: f'{self._ref(pa)}[{e}]') if pa else (lambda e: '0')
        out += self._forrange(ind, pv, lo, hi,
                              self._heappush(I1, cn, key_of(pv), pv))
        prev_c = getattr(self, '_container_name', None)
        self._container_name = cn
        try:
            body = self._let(I1, pv, self._heappop(cn))
            body += self._always(g, I1)
            # Lazy deletion: a stale entry surfaces, fails the guard and is
            # dropped, exactly as the Java's `if (!_forbidden0(j)) continue;`
            body += self._if(I1, f'(not ({self.expr(g.guard_predicate)}))'
                             if self.AND == 'and'
                             else f'(!({self.expr(g.guard_predicate)}))',
                             self._continue(I2))
            body += self._advance(g, I1)
            out += self._while(ind, self._nonempty(cn), body)
        finally:
            self._container_name = prev_c
        return out

    def _topological(self, g, v, lo, hi, acy, ind):
        """setAcyclic(A): settle each index once, in topological order.

        Mirrors llc.py's lowering -- Kahn's algorithm with the same cycle
        check, so a caller that breaks the promise is told so rather than
        quietly getting a wrong answer.
        """
        a = self._ref(acy)
        I = ind
        I1, I2, I3 = I + '    ', I + '        ', I + '            '
        n = self.SELF + 'n'
        EQ, NE, AND = self.EQ, self.NE, self.AND
        T = {b: self._tmpname(b) for b in
             ('_indeg', '_succ', '_q', '_seen', '_h', '_total')}
        out = [f'{I}{self.COMMENT}topological order over {acy}, declared acyclic']
        out += self._let(I, T['_indeg'], self._zeros(n))
        out += self._let(I, T['_succ'], self._lists(n))
        # in-degree over the declared-acyclic predecessor lists
        out += self._forrange(
            I, '_k', lo, hi,
            self._forin(I1, '_i', f'{a}[_k]',
                        self._if(I2, f'_i >= 0 {AND} _i < {n}',
                                 self._incr(I3, f'{T["_indeg"]}[_k]', '1')
                                 + self._append(I3, f'{T["_succ"]}[_i]', '_k'))))
        out += self._let(I, T['_q'], self._emptylist())
        out += self._forrange(I, '_k', lo, hi,
                              self._if(I1, f'{T["_indeg"]}[_k] {EQ} 0',
                                       self._append(I2, T['_q'], '_k')))
        out += self._let(I, T['_seen'], '0')
        out += self._let(I, T['_h'], '0')
        # the settled index is assigned inside the loop, so it has to be
        # introduced outside it -- a brace language will not accept a bare
        # assignment to an undeclared name under strict mode.
        out += self._let(I, pyname(v), '0')
        body = (self._set(I1, pyname(v), f'{T["_q"]}[{T["_h"]}]')
                + self._incr(I1, T['_h'], '1')
                + self._incr(I1, T['_seen'], '1')
                + self._always(g, I1)
                + self._if(I1, self.expr(g.guard_predicate), self._advance(g, I2))
                + self._forin(I1, '_x', f'{T["_succ"]}[{pyname(v)}]',
                              self._incr(I2, f'{T["_indeg"]}[_x]', '-1')
                              + self._if(I2, f'{T["_indeg"]}[_x] {EQ} 0',
                                         self._append(I3, T['_q'], '_x'))))
        out += self._while(I, f'{T["_h"]} < {self._len(T["_q"])}', body)
        out += self._let(I, T['_total'], f'({hi}) - ({lo}) + 1')
        out += self._if(I, f'{T["_seen"]} {NE} {T["_total"]}',
                        self._raise(I1, 'pre has a cycle; setAcyclic '
                                        'promises it does not'))
        return out

    def _advance(self, g: Stmt, ind: str):
        """The advance is a single statement, often a block."""
        adv = g.guard_advance
        if adv is None:
            return self._pass(ind)
        return self.stmt(adv, ind)

    def _bounds(self, g: Stmt, d: int = 0):
        rs = g.guard_ranges or []
        rng = rs[d] if d < len(rs) else None
        if rng is not None:
            return self.expr(rng[0]), self.expr(rng[1])
        base = self.p.base
        n = self.SELF + 'n'
        return (str(base), n if base == 1 else n + ' - 1')

    # ---------- module assembly ----------

    def emit(self) -> str:
        p = self.p
        L = []
        top = extract_top_comment(self.src_text) if self.src_text else ''
        L.append('"""' + (top.strip() if top else p.name) + '\n')
        L.append('Auto-generated from %s.llp by lang/llcpy.py.' % p.name)
        L.append('Do not edit by hand; edit the .llp source and rerun the build.')
        L.append('"""')
        if self._needs_heapq():
            L.append('import heapq')
        L.append('')
        L.append('')
        body = self._class_body()          # populates _siblings_used
        sibs = self._sibling_classes()
        for sib in sibs:
            L += sib
            L.append('')
            L.append('')
        L += self._class_decl(p.name, body)
        L.append('')
        L += self.emit_main()
        L.append('')
        out = '\n'.join(L) + '\n'
        if getattr(self, '_sibling_needs_heap', False) and 'import heapq' not in out:
            out = out.replace('"""\n', '"""\nimport heapq\n', 1)
        return out

    def _sibling_classes(self, _seen=None):
        """Each sibling program a call referenced, emitted as its own class
        in this same file so the result stays a single standalone module --
        validate-site.py's check_python_standalone requires exactly that."""
        seen = _seen if _seen is not None else set()
        out = []
        sigs = getattr(self.cg, 'signatures', None) or {}
        for name in sorted(self._siblings_used):
            if name in seen or name not in sigs:
                continue
            seen.add(name)
            g = type(self)(sigs[name], signatures=sigs, src_text='')
            inner = g._class_body()
            if g._needs_heapq():
                inner += getattr(g, '_heap_methods', list)()
                self._sibling_needs_heap = True
            out += g._sibling_classes(seen)
            out.append(self._class_decl(name, inner))
        return out

    def _class_body(self):
        out = self._entry()
        for m in self.p.aux_methods:
            out.append('')
            out += self._aux(m)
        out += self.pick_helpers()
        return out

    def _needs_heapq(self):
        """Structural, not textual: an inlined sibling is constructed from its
        parsed Program with no source text of its own, and Johnson's inlined
        Dijkstra declares a heap<int> that a text search could not see."""
        if self.p.mode == 'heap':
            return True
        if any(getattr(d.type, 'is_heap', False) for d in self.p.decls):
            return True
        seen = set()

        def walk(o):
            if id(o) in seen:
                return False
            seen.add(id(o))
            if isinstance(o, (list, tuple)):
                return any(walk(x) for x in o)
            if not hasattr(o, '__dict__'):
                return False
            if getattr(o, 'kind', None) == 'new_heap':
                return True
            t = getattr(o, 'decl_type', None)
            if t is not None and getattr(t, 'is_heap', False):
                return True
            return any(walk(v) for v in vars(o).values())

        return (walk(self.p.stmts) or walk(self.p.aux_methods)
                or 'heap<' in (self.src_text or ''))

    def _is_recursive(self):
        """Does the entry method call itself?

        It matters because the entry normally copies its parameters into
        fields, so the guards and the aux methods can read them without being
        handed them. A recursive call would then overwrite the fields the
        caller's own frame is still reading -- MergeSortIntro returned its
        input unsorted that way. Such a program keeps its parameters local.
        """
        entry = self.p.entry_name or self.p.name
        seen = set()

        def walk(o):
            if id(o) in seen:
                return False
            seen.add(id(o))
            if isinstance(o, (list, tuple)):
                return any(walk(x) for x in o)
            if not hasattr(o, '__dict__'):
                return False
            if getattr(o, 'call_name', None) == entry:
                return True
            if getattr(o, 'kind', None) == 'call' and getattr(o, 'name', None) == entry:
                return True
            return any(walk(v) for v in vars(o).values())

        return walk(self.p.stmts) or walk(self.p.trailing_return)

    def _entry(self):
        p = self.p
        I = self.BODY_IND
        body = []
        rec = self._is_recursive()
        if rec:
            for d in p.params:
                self.locals[d.name] = d
        else:
            for d in p.params:
                body += self._set(I, self.SELF + pyname(d.name), pyname(d.name))
        # n, as llc.py infers it: the first array parameter's length. Skipped
        # when the program declares its own n -- the declaration is emitted
        # just below and would otherwise assign the field a second time.
        if (not any(d.name == 'n' for d in p.params)
                and not any(d.name == 'n' for d in p.decls)
                and any(d.type.is_array or d.type.is_set for d in p.params)):
            first = next(d for d in p.params if d.type.is_array or d.type.is_set)
            n = self._len(self._ref(first.name))
            if p.base == 1:
                n += ' - 1'
            if rec:
                self.locals['n'] = None
                body += self._let(I, 'n', n)
            else:
                body += self._set(I, self.SELF + 'n', n)
        for d in p.decls:
            lhs = self.SELF + pyname(d.name)
            if d.type.is_heap or d.type.is_list:
                # Both start empty: a heap is a list used with heapq, and a
                # `list<T>` with no initialiser is Java's new LinkedList<>().
                # Without this a list field is never created, because the
                # `d.init is None` arm below skips it.
                body += self._set(I, lhs, self._emptylist())
            elif d.init is None:
                continue
            elif ((d.type.is_array or d.type.is_2d or d.type.is_set)
                    and not _is_wholesale_init(d.init)):
                size = self.SELF + ('n + 1' if p.base == 1 else 'n')
                if self._is_index_fill(d.init):
                    body += self._index_fill(I, lhs, size)
                else:
                    body += self._set(I, lhs,
                                      self._fill(self.expr(d.init, d.type), size))
            else:
                body += self._set(I, lhs, self.expr(d.init, d.type))
        body += self.stmts(p.stmts, I)
        if p.trailing_return is not None:
            body += self.stmt(p.trailing_return, I)
        return self._def(self.MEMBER_IND, p.entry_name or p.name,
                         [pyname(d.name) for d in p.params],
                         body or self._pass(I))

    def _aux(self, m):
        prev = dict(self.locals)
        self.locals = {d.name: d for d in m.params}
        try:
            return self._def(self.MEMBER_IND, m.name,
                             [pyname(d.name) for d in m.params],
                             self.stmts(m.body, self.BODY_IND))
        finally:
            self.locals = prev


def main():
    ap = argparse.ArgumentParser(description='LL to Python emitter.')
    ap.add_argument('source')
    ap.add_argument('--out', default=None,
                    help='directory to write <Name>.py into (default: beside the source)')
    a = ap.parse_args()
    src = open(a.source).read()
    stem = os.path.splitext(os.path.basename(a.source))[0]
    prog = Parser(tokenize(src), default_name=stem).parse_program()
    try:
        sigs = scan_signatures(a.source)
    except Exception:
        sigs = None
    code = PyGen(prog, signatures=sigs, src_text=src).emit()
    out_dir = a.out or os.path.dirname(os.path.abspath(a.source))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, prog.name + '.py')
    with open(path, 'w') as f:
        f.write(code)
    print('wrote ' + path)


if __name__ == '__main__':
    main()
