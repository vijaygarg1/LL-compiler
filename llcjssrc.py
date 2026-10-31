#!/usr/bin/env python3
"""LLCJSSRC -- LL-to-JavaScript *source* emitter.

Not to be confused with llcjs.py, which emits the interactive demo modules
the website mounts (LLP.mountFunction with a trace() that records animation
steps). This emits the algorithm on its own, as a plain JavaScript class,
for the "JS" source link beside LL / Java / C++ / Python.

It is a subclass of llcpy.py's PyGen. Everything that decides WHAT runs --
expression and statement dispatch, the guard lowerings, which shape a
program gets -- lives in PyGen and is inherited unchanged; this file
supplies only HOW a line is spelled. That split is the point: a Prim that
was a heap in one language and a linear scan in another is exactly the
defect this line of work exists to remove, so the two backends are not
allowed to hold separate copies of the lowering.

Usage:
  python3 llcjssrc.py FILE.llp [--out DIR]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llc import (tokenize, Parser, Type,  # noqa: E402
                 extract_top_comment, scan_signatures, _is_wholesale_init)
from llcpy import PyGen, pyname, INT_INF  # noqa: E402


class JsGen(PyGen):

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._scratch: dict = {}

    def _tmpname(self, base):
        """`let` forbids redeclaration, so a second guard in the same method
        cannot reuse `_changed`. The first use keeps the plain name, which is
        all most programs need; later ones are numbered."""
        k = self._scratch.get(base, 0)
        self._scratch[base] = k + 1
        return base if k == 0 else f'{base}{k}'

    # ---------- surface constants ----------

    SELF = 'this.'
    COMMENT = '// '
    EQ, NE, AND = '===', '!==', '&&'
    MEMBER_IND = '  '
    BODY_IND = '    '

    # LL spells equality `=`, as the book's pseudocode does. JavaScript's
    # `==` coerces, so every equality becomes strict.
    OPS = {'=': '===', '==': '===', '!=': '!==', '&&': '&&', '||': '||',
           'and': '&&', 'or': '||', '<': '<', '>': '>', '<=': '<=', '>=': '>=',
           '+': '+', '-': '-', '*': '*', '%': '%'}

    # ---------- block syntax ----------

    def _blk(self, ind, header, body):
        return [f'{ind}{header} {{'] + body

    def _close(self, ind):
        return [f'{ind}}}']

    def _if(self, ind, cond, body, els=None):
        out = self._blk(ind, f'if ({cond})', body)
        if els:
            out += [f'{ind}}} else {{'] + els
        return out + self._close(ind)

    def _while(self, ind, cond, body):
        return self._blk(ind, f'while ({cond})', body) + self._close(ind)

    def _forrange(self, ind, v, lo, hi, body):
        return (self._blk(ind, f'for (let {v} = {lo}; {v} <= ({hi}); {v}++)', body)
                + self._close(ind))

    def _forin(self, ind, v, it, body):
        return self._blk(ind, f'for (const {v} of {it})', body) + self._close(ind)

    def _def(self, ind, name, params, body):
        return (self._blk(ind, f'{name}({", ".join(params)})', body)
                + self._close(ind))

    def _lit(self, v):
        return v                      # true / false / null are already JS

    # ---------- leaf statements and containers ----------

    def _decl(self, ind, name, rhs):
        return [f'{ind}let {name} = {rhs};']

    def _let(self, ind, name, rhs):
        return [f'{ind}let {name} = {rhs};']

    def _set(self, ind, lhs, rhs):
        return [f'{ind}{lhs} = {rhs};']

    def _incr(self, ind, lhs, delta):
        if str(delta).startswith('-'):
            return [f'{ind}{lhs} -= {str(delta)[1:]};']
        return [f'{ind}{lhs} += {delta};']

    def _swap(self, ind, a, b):
        # Not `[a, b] = [b, a]`: a statement opening with `[` is parsed as a
        # continuation of the expression on the line above.
        t = self.fresh('_sw')
        return [f'{ind}const {t} = {a}; {a} = {b}; {b} = {t};']

    def _append(self, ind, arr, v):
        return [f'{ind}{arr}.push({v});']

    def _raise(self, ind, msg):
        return [f'{ind}throw new Error("{msg}");']

    def _fill(self, fill, size):
        return f'new Array({size}).fill({fill})'

    def _zeros(self, size):
        return f'new Array({size}).fill(0)'

    def _lists(self, size):
        return f'Array.from({{length: {size}}}, () => [])'

    def _emptylist(self):
        return '[]'

    def _len(self, x):
        return f'{x}.length'

    def _pass(self, ind):
        return []

    def _continue(self, ind):
        return [ind + 'continue;']

    def _nonempty(self, name):
        return f'{name}.length > 0'

    def _heappush(self, ind, name, key, val):
        return [f'{ind}this._heapPush({name}, {key}, {val});']

    def _heappop(self, name):
        return f'this._heapPop({name})[1]'

    def _reduce_identity(self, q, base):
        if base == 'double':
            return 'Infinity' if q == 'min' else '-Infinity'
        return INT_INF if q == 'min' else '-' + INT_INF

    def _new_instance(self, name):
        return f'new {name}()'

    def _class_decl(self, name, body):
        return [f'class {name} {{'] + body + ['}']

    def _main_literal(self, v):
        return v                       # true/false are already JS

    def _main_block(self, body):
        return ["if (typeof require !== 'undefined' && require.main === module) {"] \
            + body + ['}']

    def _main_decl(self, ind, name, value):
        return [f'{ind}const {name} = {value};']

    def _main_print(self, ind, expr):
        return [f'{ind}console.log(JSON.stringify({expr}));']

    def _main_argexpr(self, a):
        import re as _re
        m = _re.match(r'^(\w+)\.length\s*(.*)$', a)
        if m:
            tail = (' ' + m.group(2)) if m.group(2) else ''
            return f'{pyname(m.group(1))}.length{tail}'
        return pyname(a) if _re.match(r'^\w+$', a) and not a.isdigit() else a

    # ---------- expressions ----------

    def expr(self, e, ctx=None):
        k = e.kind
        if k == 'bool':
            return 'true' if e.value else 'false'
        if k == 'id' and e.name == 'null':
            return 'null'
        if k == 'field_access' and e.name == 'length' and e.idx is None:
            return f'{self.expr(e.lhs)}.length'
        if k == 'un' and e.op in ('not', '!'):
            return f'(!{self.expr(e.lhs)})'
        if k == 'slice':
            return (f'{self._ref(e.name)}.slice({self.expr(e.slice_lo)}, '
                    f'({self.expr(e.slice_hi)}) + 1)')
        if k == 'bin' and e.op == '/':
            lt, rt = self._static_base(e.lhs), self._static_base(e.rhs)
            # LL's `/` on ints is Java integer division, truncating toward
            # zero. JS `/` is real division, so int contexts need the trunc.
            if lt == 'int' and rt == 'int':
                return f'Math.trunc({self.expr(e.lhs)} / {self.expr(e.rhs)})'
            return f'({self.expr(e.lhs)} / {self.expr(e.rhs)})'
        return super().expr(e, ctx)

    def _infty_for(self, t):
        return 'Infinity' if (t is not None and t.base == 'double') else INT_INF

    def _zero_for_base(self, base):
        if base == 'double':
            return '0.0'
        if base == 'boolean':
            return 'false'
        return '0'

    def _ref(self, name):
        if name in self.locals:
            return pyname(name)
        if name in self.syms or name in self.IMPLICIT_FIELDS:
            return f'this.{pyname(name)}'
        return pyname(name)

    def _new_array(self, e):
        t = e.value if isinstance(e.value, Type) else None
        fill = (self._zero(t) if t is not None
                else self._zero_for_base(getattr(e, 'name', None)))
        dims = [self.expr(x) for x in
                ([e.idx, e.idx2] + list(e.idx_extra or [])) if x is not None]
        if e.jagged or not dims:
            return f'new Array({dims[0]}).fill(null)' if dims else '[]'
        return self._nested_array(dims, fill)

    def _nested_array(self, dims, fill):
        # Array.from with a factory, not .fill(row): .fill would store one
        # shared row object at every index.
        out = f'new Array({dims[-1]}).fill({fill})'
        for d in reversed(dims[:-1]):
            out = f'Array.from({{length: {d}}}, () => {out})'
        return out

    def _break(self, ind):
        return [f'{ind}break;']

    def _method_call(self, e):
        """heap<int> maps onto the binary min-heap emitted at the bottom of
        the class, over [key, value] pairs ordered key-then-value -- the same
        order the Java PriorityQueue<int[]> comparator gives."""
        if (e.lhs is not None and getattr(e.lhs, 'kind', None) == 'id'
                and self._sibling(e.lhs.name) is not None):
            return super()._method_call(e)
        r = self.expr(e.lhs)
        meth, args = e.name, [self.expr(a) for a in e.args]
        if meth == 'push':
            return f'this._heapPush({r}, {args[0]}, {args[1]})'
        if meth == 'removeMin':
            return f'this._heapPop({r})'
        if meth == 'isEmpty':
            return f'({r}.length === 0)'
        if meth == 'peekKey':
            return f'{r}[0][0]'
        if meth == 'peekValue':
            return f'{r}[0][1]'
        if meth in ('length', 'size'):
            return f'{r}.length'
        if meth == 'clone':
            return f'{r}.slice()'
        # list<T> onto a JS array -- same split as the Python emitter:
        # one-argument `add` is the list append, push(key, value) is the heap.
        if meth == 'add' and len(args) == 1:
            return f'{r}.push({args[0]})'
        if meth == 'removeFirst' and not args:
            return f'{r}.shift()'
        if meth == 'removeLast' and not args:
            return f'{r}.pop()'
        if meth == 'first' and not args:
            return f'{r}[0]'
        return f'{r}.{meth}({", ".join(args)})'

    def _call(self, e):
        # Only the three builtins differ; everything else -- including a call
        # that names a sibling PROGRAM rather than a method -- is the base's
        # decision, so the two backends cannot resolve a name differently.
        name, args = e.name, [self.expr(a) for a in e.args]
        if name == 'min':
            return f'Math.min({", ".join(args)})'
        if name == 'max':
            return f'Math.max({", ".join(args)})'
        if name == 'abs':
            return f'Math.abs({args[0]})'
        return super()._call(e)

    # ---------- quantifiers ----------

    def quant(self, e):
        """JavaScript has no comprehension, so a quantifier becomes an arrow
        function invoked on the spot. Inline rather than hoisted to a method:
        the body may mention any local in scope, not just the guard indices,
        and an arrow closes over all of them -- including `this`.

        The empty-selection identities match llcpy.py's, so the two backends
        reduce an empty range the same way.
        """
        v = e.q_var
        prev = dict(self.locals)
        self.locals[v] = None
        try:
            if e.q_set is not None:
                head = f'for (const {v} of {self.expr(e.q_set)})'
            else:
                head = (f'for (let {v} = {self.expr(e.q_lo)}; '
                        f'{v} <= ({self.expr(e.q_hi)}); {v}++)')
            body = self.expr(e.q_body) if e.q_body is not None else v
            skip = (f'if (!({self.expr(e.q_where)})) continue; '
                    if e.q_where is not None else '')
            q = e.quant
            acc = ''
            if q == 'exists':
                if e.pick_var:
                    if e not in self._picks:
                        self._picks.append(e)
                        self._pick_idx[id(e)] = list(self._cur_idx)
                    # The advance reads the witness afterwards, so it has to
                    # outlive the search -- a helper storing it on `this`,
                    # exactly as the Java and Python backends do.
                    return f'this._pick_{e.pick_var}({self._pick_args(e)})'
                inner, tail = f'if ({body}) return true;', 'return false;'
            elif q == 'forall':
                inner, tail = f'if (!({body})) return false;', 'return true;'
            elif q == 'sum':
                acc, inner, tail = 'let _a = 0; ', f'_a += {body};', 'return _a;'
            elif q == 'count':
                acc, tail = 'let _a = 0; ', 'return _a;'
                inner = f'if ({body}) _a += 1;' if e.q_body is not None else '_a += 1;'
            elif q in ('min', 'max'):
                ident = self._reduce_identity(
                    q, self._static_base(e.q_body) if e.q_body is not None
                    else None)
                cmp = '<' if q == 'min' else '>'
                # Python's min(..., default=IDENT) uses IDENT only when the
                # range is empty. Seeding the accumulator with it instead
                # lets it compete as an extra element, which caps every
                # result at IDENT: LLP-BellmanFord relaxing from a vertex
                # still at the sentinel returned the sentinel rather than
                # sentinel + w. _s records whether anything was seen.
                acc, tail = f'let _a = {ident}, _s = false; ', 'return _a;'
                inner = (f'const _v = {body}; '
                         f'if (!_s || _v {cmp} _a) {{ _a = _v; _s = true; }}')
            else:
                raise NotImplementedError(f'quantifier {q!r}')
            return f'(() => {{ {acc}{head} {{ {skip}{inner} }} {tail} }})()'
        finally:
            self.locals = prev

    def pick_helpers(self):
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
                    head = f'for (const {v} of {self.expr(e.q_set)})'
                else:
                    head = (f'for (let {v} = {self.expr(e.q_lo)}; '
                            f'{v} <= ({self.expr(e.q_hi)}); {v}++)')
                cond = self.expr(e.q_body) if e.q_body is not None else 'true'
                sig = ', '.join(pyname(g) for g in gidx)
                out += ['',
                        f'  _pick_{v}({sig}) {{',
                        f'    {head} {{',
                        f'      if ({cond}) {{ this.{v} = {v}; return true; }}',
                        '    }',
                        '    return false;',
                        '  }']
            finally:
                self.locals = prev
        return out

    # ---------- statements ----------

    def stmt(self, s, ind):
        lines = super().stmt(s, ind)
        # These three are printed directly by the base class rather than
        # through a hook, because Python needs nothing added to them.
        if s.kind in ('exprstmt', 'return', 'call'):
            lines = [ln + ';' if ln.strip() else ln for ln in lines]
        return lines

    # ---------- module assembly ----------

    def emit(self):
        p = self.p
        body = self._class_body()          # populates _siblings_used
        if self._needs_heapq():
            body += self._heap_methods()
        sibs = []
        for sib in self._sibling_classes():
            sibs += sib + ['']
        top = extract_top_comment(self.src_text) if self.src_text else ''
        L = ['/*']
        for line in (top.strip() if top else p.name).split('\n'):
            L.append((' * ' + line).rstrip())
        L += [' *',
              f' * Auto-generated from {p.name}.llp by lang/llcjssrc.py.',
              ' * Do not edit by hand; edit the .llp source and rerun the build.',
              ' */',
              '',
              '']
        L += sibs
        L.append(f'class {p.name} {{')
        L += body
        L += ['}', '',
              f"if (typeof module !== 'undefined') module.exports = {p.name};"]
        L += self.emit_main()
        return '\n'.join(L) + '\n'

    def _heap_methods(self):
        return ['',
                '  // Binary min-heap of [key, value] pairs, ordered by key and',
                '  // then by value -- the order the Java PriorityQueue<int[]>',
                '  // comparator gives. JavaScript has no heap in its standard',
                '  // library, so the three operations the program uses are',
                '  // carried with it.',
                '  _heapLess(a, b) {',
                '    return a[0] !== b[0] ? a[0] < b[0] : a[1] < b[1];',
                '  }',
                '',
                '  _heapPush(h, k, v) {',
                '    h.push([k, v]);',
                '    let i = h.length - 1;',
                '    while (i > 0) {',
                '      const p = Math.trunc((i - 1) / 2);',
                '      if (!this._heapLess(h[i], h[p])) break;',
                '      const t = h[i]; h[i] = h[p]; h[p] = t;',
                '      i = p;',
                '    }',
                '  }',
                '',
                '  _heapPop(h) {',
                '    const top = h[0];',
                '    const last = h.pop();',
                '    if (h.length > 0) {',
                '      h[0] = last;',
                '      let i = 0;',
                '      while (true) {',
                '        const l = 2 * i + 1, r = l + 1;',
                '        let m = i;',
                '        if (l < h.length && this._heapLess(h[l], h[m])) m = l;',
                '        if (r < h.length && this._heapLess(h[r], h[m])) m = r;',
                '        if (m === i) break;',
                '        const t = h[i]; h[i] = h[m]; h[m] = t;',
                '        i = m;',
                '      }',
                '    }',
                '    return top;',
                '  }']


def main():
    ap = argparse.ArgumentParser(description='LL to JavaScript source emitter.')
    ap.add_argument('source')
    ap.add_argument('--out', default=None,
                    help='directory to write <Name>.js into (default: beside the source)')
    a = ap.parse_args()
    src = open(a.source).read()
    stem = os.path.splitext(os.path.basename(a.source))[0]
    prog = Parser(tokenize(src), default_name=stem).parse_program()
    try:
        sigs = scan_signatures(a.source)
    except Exception:
        sigs = None
    code = JsGen(prog, signatures=sigs, src_text=src).emit()
    out_dir = a.out or os.path.dirname(os.path.abspath(a.source))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, prog.name + '.js')
    with open(path, 'w') as f:
        f.write(code)
    print('wrote ' + path)


if __name__ == '__main__':
    main()
