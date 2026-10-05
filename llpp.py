#!/usr/bin/env python3
"""LLPP -- LL Pretty Printer.

Translates an LL source file to LaTeX in the algorithm2e style used
throughout the book. Output is a complete ``\\begin{algorithm} ... \\end{algorithm}``
fragment that can be pasted into a chapter.

Usage:  python3 llpp.py FILE.llp [--out FILE.tex]
"""

from __future__ import annotations
import sys
import os
import dataclasses
import re
from typing import Optional

# Reuse the lexer, parser, and AST from the compiler.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llc import (tokenize, Parser, Program, Decl, Stmt, Expr, Type,  # noqa: E402
                 scan_signatures, merge_observed_program, merge_union_programs)


# ------------------------------------------------------------
#  Expression formatter (AST -> LaTeX math)
# ------------------------------------------------------------

# LaTeX output for binary operators.
BIN = {
    '+': '+', '-': '-', '*': r' \cdot ', '/': '/', '%': r' \bmod ',
    '<': '<', '<=': r' \leq ', '>': '>', '>=': r' \geq ',
    '=': '=', '!=': r' \neq ',
    'and': r' \wedge ', 'or': r' \vee ',
}


# Precedence levels used to decide when to add parentheses around a
# sub-expression. Higher = binds tighter. The tree built by the parser
# does not record the original parenthesisation, so we re-insert
# parentheses only where they are needed for unambiguous reading.
PREC = {
    # A quantifier extends as far right as it can, so as an operand of a
    # connective it binds most loosely of all and must be bracketed --
    # otherwise `(forall k : P) and Q` prints as `forall k : P and Q`.
    'quant': 0,
    'or':  1, 'and': 2, 'not': 3,
    '<': 4, '<=': 4, '>': 4, '>=': 4, '=': 4, '!=': 4,
    '+': 5, '-': 5,
    '*': 6, '/': 6, '%': 6,
    'neg': 7,
    'call': 9, 'index': 9, 'atom': 10,
}


def prec(e: Expr) -> int:
    if e.kind == 'bin':
        return PREC.get(e.op, 0)
    if e.kind == 'un':
        return PREC.get(e.op, 0)
    if e.kind in ('call', 'index', 'index2', 'id', 'int', 'real', 'bool',
                  'infinity', 'method_call', 'field_access', 'new_array'):
        return PREC['atom']
    if e.kind == 'quant':
        return PREC['quant']
    return 0


def paren(child: Expr, parent_prec: int) -> str:
    s = fmt_expr(child)
    if prec(child) < parent_prec:
        return '(' + s + ')'
    return s


# Translation of identifiers into LaTeX math. The book mostly uses
# plain letters; multi-letter names stay as-is but get wrapped in
# ``\textit{}`` so they render in italic without kerning glitches.
def fmt_id(name: str) -> str:
    # A `// display:' marker in the source may set how a name is to be typeset.
    # A chapter sometimes writes a quantity in a notation that is not a legal
    # identifier -- the sender of a message as `w.p', say, where w is the
    # message and p its process -- and the printed algorithm should read the
    # way the chapter argues about it.
    if name in _DISPLAY:
        return _DISPLAY[name]
    # Compiler convention: `_i` is the per-element index inside an array
    # initialiser. Render it as a plain math `i` in LaTeX.
    if name == '_i':
        return 'i'
    if name == '_j':
        return 'j'
    if len(name) == 1:
        return name
    # Common names that appear in the book as plain math italic
    known_italic = {'pre', 'fixed', 'active', 'prev', 'last', 'finish',
                    'lateness', 'rem', 'numRooms', 'dist', 'room',
                    'partner', 'changed', 'req', 'priority', 'parent',
                    'key', 'dIn'}
    if name in known_italic:
        return r'\mathit{' + name + '}'
    # Default: wrap in mathit for multi-letter variable names
    return r'\mathit{' + name + '}'


def fmt_call(name: str, args):
    # Built-in quantifier-less functions: render with standard LaTeX commands.
    if name == 'min' and len(args) == 2:
        return r'\min(' + ', '.join(fmt_expr(a) for a in args) + ')'
    if name == 'max' and len(args) == 2:
        return r'\max(' + ', '.join(fmt_expr(a) for a in args) + ')'
    if name == 'abs' and len(args) == 1:
        return '|' + fmt_expr(args[0]) + '|'
    return fmt_id(name) + '(' + ', '.join(fmt_expr(a) for a in args) + ')'


def fmt_range_sub(q_var: str, lo: Expr, hi: Expr) -> str:
    """Format the subscript used under \\min, \\sum, etc."""
    return q_var + '=' + fmt_expr(lo)


def fmt_set_range(e: Expr) -> str:
    """Render what a quantifier or loop ranges over: a set-valued element
    like `pre[j]` as `pre(j)`, and a list named outright as its name."""
    if e.kind == 'index':
        return fmt_id(e.name) + '(' + fmt_expr(e.idx) + ')'
    return fmt_expr(e)


def fmt_quant(e: Expr) -> str:
    v = e.q_var
    body = fmt_expr(e.q_body)
    use_set = (e.q_set is not None)
    if use_set:
        rng = fmt_set_range(e.q_set)
    else:
        lo = fmt_expr(e.q_lo)
        hi = fmt_expr(e.q_hi)
    if e.quant == 'min':
        if use_set:
            return (r'\min \{' + body + ' \mid ' + v + r' \in ' + rng + r'\}')
        return r'\min_{' + v + '=' + lo + '}^{' + hi + r'} ' + paren(e.q_body, PREC['atom'])
    if e.quant == 'max':
        if use_set:
            return (r'\max \{' + body + ' \mid ' + v + r' \in ' + rng + r'\}')
        return r'\max_{' + v + '=' + lo + '}^{' + hi + r'} ' + paren(e.q_body, PREC['atom'])
    if e.quant == 'sum':
        if use_set:
            return (r'\sum_{' + v + r' \in ' + rng + r'} ' + paren(e.q_body, PREC['atom']))
        return r'\sum_{' + v + '=' + lo + '}^{' + hi + '} ' + paren(e.q_body, PREC['atom'])
    if e.quant == 'count':
        if use_set:
            return r'|\{\, ' + v + r' \in ' + rng + r' : ' + body + r' \,\}|'
        return (r'|\{\, ' + v + r' \in [' + lo + r'..' + hi + r'] : ' +
                body + r' \,\}|')
    if e.quant == 'exists':
        if use_set:
            return r'\exists ' + v + r' \in ' + rng + r' : ' + body
        return (r'\exists ' + v + r' \in [' + lo + r'..' + hi + r'] : ' + body)
    if e.quant in ('forall', 'forall_pred'):
        if use_set:
            return r'\forall ' + v + r' \in ' + rng + r' : ' + body
        return (r'\forall ' + v + r' \in [' + lo + r'..' + hi + r'] : ' + body)
    if e.quant == 'map':
        if use_set:
            return r'\{\, ' + body + r' \mid ' + v + r' \in ' + rng + r' \,\}'
        return (r'\{\, ' + body + r' \mid ' + v + r' \in [' + lo + r'..' +
                hi + r'] \,\}')
    if e.quant == 'filter':
        if use_set:
            return r'\{\, ' + v + r' \in ' + rng + r' \mid ' + body + r' \,\}'
        return (r'\{\, ' + v + r' \in [' + lo + r'..' + hi + r'] \mid ' +
                body + r' \,\}')
    raise ValueError(f'unknown quantifier {e.quant}')


def _cmp_operand(e: Expr, parent_prec: int, parent_op: str = '') -> str:
    """An operand of a connective, bracketed if it is a comparison, or if it
    is a conjunction under a disjunction: precedence makes those parentheses
    redundant, but the grouping of mixed connectives is worth showing rather
    than leaving the reader to recall that AND binds tighter than OR."""
    if e.kind == 'bin' and e.op in ('<', '<=', '>', '>=', '=', '!='):
        return '(' + fmt_expr(e) + ')'
    if parent_op == 'or' and e.kind == 'bin' and e.op == 'and':
        return '(' + fmt_expr(e) + ')'
    return paren(e, parent_prec)


def fmt_expr(e: Expr) -> str:
    k = e.kind
    if k == 'int':
        return str(e.value)
    if k == 'real':
        return f'{e.value}'
    if k == 'bool':
        return r'\mathit{true}' if e.value else r'\mathit{false}'
    if k == 'infinity':
        return r'\infty'
    if k == 'id':
        return fmt_id(e.name)
    if k == 'index':
        return fmt_id(e.name) + '[' + fmt_expr(e.idx) + ']'
    if k == 'index2':
        return (fmt_id(e.name)
                + '[' + fmt_expr(e.idx) + ']'
                + '[' + fmt_expr(e.rhs) + ']')
    if k == 'call':
        return fmt_call(e.name, e.args)
    if k == 'method_call':
        if e.name == 'clone' and e.lhs.kind == 'id':
            return fmt_id(e.lhs.name) + '[i]'
        # `q[i].nonEmpty()` reads as what it is -- an operation on that
        # queue -- so it is printed the way it is written, not turned
        # inside out into `nonEmpty(q[i])`.
        args = ', '.join(fmt_expr(a) for a in e.args)
        return fmt_expr(e.lhs) + '.' + fmt_id(e.name) + '(' + args + ')'
    if k == 'field_access':
        # `X.length` -> `|X|`-style; default to `X.length` literally.
        base = fmt_expr(e.lhs) + '.' + fmt_id(e.name)
        if e.idx is not None:
            base = base + '[' + fmt_expr(e.idx) + ']'
        return base
    if k == 'new_record':
        return (r'\mathbf{new}\,\mathit{' + e.name + '}('
                + ', '.join(fmt_expr(a) for a in e.args) + ')')
    if k == 'new_array':
        # `new int[expr]` -> `new \, T[expr]`
        return r'\mathbf{new}\,\mathit{' + e.name + '}[' + fmt_expr(e.idx) + ']'
    if k == 'un':
        p = PREC[e.op]
        s = paren(e.lhs, p)
        if e.op == 'not':
            return r'\neg ' + s
        if e.op == 'neg':
            return '-' + s
    if k == 'bin':
        p = PREC[e.op]
        if e.op in ('and', 'or'):
            # A comparison under a connective is bracketed even though
            # precedence would not require it, so the conjuncts of a guard
            # can be picked out at a glance.
            return (_cmp_operand(e.lhs, p, e.op) + BIN[e.op]
                    + _cmp_operand(e.rhs, p, e.op))
        # For comparison ops, don't force parens around arithmetic children.
        l = paren(e.lhs, p)
        r = paren(e.rhs, p + (1 if e.op in ('-', '/', '%') else 0))
        return l + BIN[e.op] + r
    if k == 'quant':
        return fmt_quant(e)
    raise ValueError(f'cannot format expr kind {k}')


# ------------------------------------------------------------
#  Statement formatter
# ------------------------------------------------------------

def _is_recv(s) -> bool:
    """True for a guard written as `upon receive(...)`."""
    return bool(getattr(s, 'recv_tag', None) or getattr(s, 'recv_binders', None)
                or getattr(s, 'recv_from', None))


def _fmt_recv(s) -> str:
    """`receive TAG(b1, ..., bk) from P_k`, in the book's phrasing."""
    parts = ''
    if s.recv_tag is not None:
        parts = r'\mathit{' + s.recv_tag.replace('_', r'\_') + '}'
    if s.recv_binders:
        parts += '(' + ', '.join(fmt_id(b) for b in s.recv_binders) + ')'
    out = r'{\bf receive} $' + (parts if parts else 'm') + '$'
    src = (str(s.recv_from_const) if getattr(s, 'recv_from_const', None)
           is not None else s.recv_from)
    if src is not None:
        out += (r' {\bf from} $' + _PROC_NAME + '_{'
                + (src if src.isdigit() else fmt_id(src)) + '}$')
    if getattr(s, 'guard_predicate', None) is not None:
        out += r' {\bf when} $' + fmt_expr(s.guard_predicate) + '$'
    return out

def stmt_lines(s: Stmt, indent: str = '') -> list:
    """Emit a statement using algorithm2e's own block macros.

    The package draws the vertical rule and the closing `end` for a block
    given as `\While{cond}{body}`; writing the nesting by hand with `\quad`
    would suppress both.  This is the style book.tex and parbook.tex use."""

    def body_of(stmts, ind):
        return body_lines(stmts, ind)

    inner = indent + '  '
    if s.kind == 'block':
        return body_of(s.body, indent)
    if s.kind == 'forall':
        if s.loop_set is not None:
            rng = fmt_set_range(s.loop_set)
        else:
            rng = ('[' + fmt_expr(s.loop_lo) + '..' + fmt_expr(s.loop_hi) + ']')
        hdr = '$' + s.loop_var + r' \in ' + rng + '$'
        one = _loop_inline_body(s.body)
        if one is not None:
            # A loop over a single statement is set on one line, with no rule
            # and no `end` -- the `\l...` form, as a short `if` already is.
            return [indent + r'\lForAll{' + hdr + '}{' + one + '}']
        return ([indent + r'\ForAll{' + hdr + '}{']
                + body_of(s.body, inner) + [indent + '}'])
    if s.kind == 'while':
        return ([indent + r'\While{$' + fmt_expr(s.cond) + '$}{']
                + body_of(s.body, inner) + [indent + '}'])
    if s.kind == 'if':
        return _if_lines(s, indent, body_of)
    return [indent + fmt_stmt(s) + r'\;']


def _if_chain(s):
    """Flatten `if / else if / ... / else` into [(cond, body), ...] with a
    trailing (None, body) for a final else."""
    chain = []
    cur = s
    while True:
        chain.append((cur.cond, cur.then_stmts))
        nxt = _lone_if(cur.else_stmts)
        if nxt is None:
            if cur.else_stmts:
                chain.append((None, cur.else_stmts))
            return chain
        cur = nxt


def _if_lines(s, indent, body_of):
    chain = _if_chain(s)
    inline = [_inline_body(b) for _, b in chain]
    if all(x is not None for x in inline):
        # Every branch fits on its header line: the `\l...` forms, which the
        # book's figures use for short conditionals.
        out = []
        for k, ((cond, _), text) in enumerate(zip(chain, inline)):
            if cond is None:
                out.append(indent + r'\lElse{' + text + '}')
            elif k == 0:
                out.append(indent + r'\lIf{$' + fmt_expr(cond)
                           + '$}{' + text + '}')
            else:
                out.append(indent + r'\lElseIf{$' + fmt_expr(cond) + '$}{' + text + '}')
        return out
    # Some branch needs lines of its own, so every branch takes block form.
    # Only the last one closes the chain, hence the `u` variants before it.
    out = []
    inner = indent + '  '
    for k, (cond, body) in enumerate(chain):
        last = (k == len(chain) - 1)
        if cond is None:
            head = r'\Else{'
        elif k == 0:
            head = (r'\If{$' if last else r'\uIf{$') + fmt_expr(cond) + '$}{'
        else:
            head = (r'\ElseIf{$' if last else r'\uElseIf{$') + fmt_expr(cond) + '$}{'
        out.append(indent + head)
        out += body_of(body, inner)
        out.append(indent + '}')
    return out


def _lone_if(stmts) -> 'Stmt':
    """The single `if` an else-branch consists of, unwrapping one block, or
    None if the branch is anything else."""
    if not stmts:
        return None
    while len(stmts) == 1 and stmts[0].kind == 'block':
        stmts = stmts[0].body
    if len(stmts) == 1 and stmts[0].kind == 'if':
        return stmts[0]
    return None


def _unwrap(stmts) -> list:
    """See through the braces of a body that is a single block."""
    while len(stmts) == 1 and stmts[0].kind == 'block':
        stmts = stmts[0].body
    return list(stmts)


def _has_nested_block(s) -> bool:
    """Does this statement contain a loop or a block anywhere inside it?
    Such a statement cannot be set on one line: fmt_stmt would flatten the
    inner loop onto the header, and a nested quantifier run together with its
    enclosing one is unreadable."""
    for fl in dataclasses.fields(s):
        v = getattr(s, fl.name, None)
        for x in (v if isinstance(v, list) else [v]):
            if isinstance(x, Stmt):
                if x.kind in ('block', 'forall', 'while') or _has_nested_block(x):
                    return True
    return False






def _loop_inline_body(stmts):
    """The body of a loop rendered on the header line, or None.  Unlike
    `_inline_body` this allows a single `if`, which renders on one line and
    is by far the commonest thing a one-statement loop contains -- but only
    when that `if` is itself flat and short; a loop inside it, or a chain of
    them long enough to run off the line, needs lines of its own."""
    st = _unwrap(stmts)
    if (len(st) == 1 and st[0].kind not in ('block', 'forall', 'while')
            and not _has_nested_block(st[0])):
        t = fmt_stmt(st[0])
        if _visual_len(t) <= 110:
            return t
    if len(st) > 1 and all(s.kind in _SIMPLE_STMT_KINDS for s in st):
        t = '; '.join(fmt_stmt(s) for s in st)
        if _visual_len(t) <= 110:
            return t
    return None


def _visual_len(t: str) -> int:
    """Roughly how wide a snippet sets, rather than how long its source is:
    \\mathit{sentWhite} is eighteen characters of LaTeX and about eight of
    type, and a decision about line length has to be about the latter."""
    t = re.sub(r'\\[a-zA-Z]+', 'x', t)      # a control word sets as a glyph or two
    return len(t.replace('$', '').replace('{', '').replace('}', ''))




def _inline_body(stmts):
    """The body rendered as one short snippet to sit on the header line, or
    None if it needs lines of its own."""
    st = _unwrap(stmts)
    if len(st) == 1 and st[0].kind not in ('block', 'forall', 'while', 'if'):
        t = fmt_stmt(st[0])
        if _visual_len(t) <= 90:
            return t
    if len(st) > 1 and all(s.kind in _SIMPLE_STMT_KINDS for s in st):
        t = '; '.join(fmt_stmt(s) for s in st)
        if _visual_len(t) <= 90:
            return t
    return None


# A statement short and self-contained enough that a run of them reads fine
# packed onto one printed line -- the same judgement _loop_inline_body and
# _inline_body already make for a single such statement filling a loop or
# `if` header, extended here to a whole run inside any body.  `send` and a
# local `decl` are just as self-contained and short as an assignment or a
# call, so they group the same way.
_SIMPLE_STMT_KINDS = {'assign', 'swap', 'call', 'exprstmt', 'send', 'decl'}


def body_lines(stmts: list, indent: str) -> list:
    """Render a body (the children of a rule, procedure, loop, or branch),
    packing each maximal run of simple statements onto one printed line when
    the joined line fits, to keep long algorithms compact.  A statement with
    nested structure of its own (if/forall/while/block) still gets whatever
    lines stmt_lines gives it, and a run that would run off the line falls
    back to one statement per line, as before."""
    out: list = []
    ss = _unwrap(stmts)
    i = 0
    while i < len(ss):
        if ss[i].kind in _SIMPLE_STMT_KINDS:
            run = [ss[i]]
            j = i + 1
            while j < len(ss) and ss[j].kind in _SIMPLE_STMT_KINDS:
                run.append(ss[j])
                j += 1
            texts = [fmt_stmt(x) for x in run]
            joined = '; '.join(texts)
            if len(run) == 1 or _visual_len(joined) <= 110:
                out.append(indent + joined + r'\;')
            else:
                out.extend(indent + t + r'\;' for t in texts)
            i = j
        else:
            out.extend(stmt_lines(ss[i], indent))
            i += 1
    return out


def fmt_stmt(s: Stmt) -> str:
    """Return an inline LaTeX snippet (no trailing \\; or newline)."""
    if s.kind == 'exprstmt':
        return '$' + fmt_expr(s.rhs) + '$'
    if s.kind == 'swap':
        return ('$\\mathit{swap}(' + fmt_expr(s.swap_a) + ', '
                + fmt_expr(s.swap_b) + ')$')
    if s.kind == 'assign':
        lhs = fmt_id(s.lhs_name)
        if s.lhs_index is not None:
            lhs = fmt_id(s.lhs_name) + '[' + fmt_expr(s.lhs_index) + ']'
            if s.lhs_index2 is not None:
                lhs = lhs + '[' + fmt_expr(s.lhs_index2) + ']'
        if getattr(s, 'lhs_field', None) is not None:
            lhs = lhs + '.' + fmt_id(s.lhs_field)
            if s.lhs_field_index is not None:
                lhs = lhs + '[' + fmt_expr(s.lhs_field_index) + ']'
        return f'${lhs} := {fmt_expr(s.rhs)}$'
    if s.kind == 'if':
        parts = [r'{\bf if} $', fmt_expr(s.cond), r'$ {\bf then} ']
        parts.append('; '.join(fmt_stmt(x) for x in s.then_stmts))
        if s.else_stmts:
            parts.append(r' {\bf else} ')
            parts.append('; '.join(fmt_stmt(x) for x in s.else_stmts))
        return ''.join(parts)
    if s.kind == 'forall':
        inner = '; '.join(fmt_stmt(x) for x in s.body)
        if s.loop_set is not None:
            rng = fmt_set_range(s.loop_set)
            return (r'{\bf forall} $' + s.loop_var + r' \in ' + rng
                    + r'$ : ' + inner)
        return (r'{\bf forall} $' + s.loop_var + r' \in ['
                + fmt_expr(s.loop_lo) + r'..' + fmt_expr(s.loop_hi)
                + r']$ : ' + inner)
    if s.kind == 'while':
        inner = '; '.join(fmt_stmt(x) for x in s.body)
        return (r'{\bf while} $(' + fmt_expr(s.cond) + r')$ \{ ' + inner + r' \}')
    if s.kind == 'decl':
        t = fmt_type(s.decl_type)
        s_init = ''
        if s.decl_init is not None:
            s_init = ' := $' + fmt_expr(s.decl_init) + '$'
        return f'{t} ${fmt_id(s.decl_name)}${s_init}'
    if s.kind == 'par':
        op = getattr(s, 'par_op', '[]')
        sep = {
            '[]': r' \parallel ',
            '&&': r' \wedge ',
            'or': r' \;\textbf{or}\; ',
            ';':  r' \mathbin{;} ',
            '|':  r' \mid ',
            '<>':   r' \mathbin{\langle\rangle} ',
            'sync': r' \;\textbf{sync}\; ',
        }.get(op, r' \parallel ')
        inner = sep.join(fmt_stmt(b) for b in s.par_branches)
        return r'[ ' + inner + r' ]'
    if s.kind == 'call':
        args = ', '.join(fmt_expr(a) for a in s.call_args)
        return f'${fmt_id(s.call_name)}({args})$'
    if s.kind == 'block':
        inner = '; '.join(fmt_stmt(x) for x in s.body)
        return r'\{ ' + inner + r' \}'
    if s.kind == 'return':
        if s.return_value is not None:
            return r'{\bf return} $' + fmt_expr(s.return_value) + r'$'
        return r'{\bf return}'
    if s.kind == 'exit':
        return r'{\bf exit}($' + fmt_expr(s.return_value) + r'$)'
    if s.kind == 'send':
        dest = fmt_expr(s.send_dest)
        payload = ', '.join(fmt_expr(e) for e in s.send_payload)
        if s.send_tag is not None:
            body = r'\mathit{' + s.send_tag.replace('_', r'\_') + '}(' + payload + ')'
        else:
            body = payload
        # The destination names a process, and is written that way -- the
        # same `P_k` phrasing `receive ... from P_k` already uses.
        return (r'{\bf send} $' + body + r'$ {\bf to} $'
                + _PROC_NAME + '_{' + dest + '}$')
    if s.kind == 'delete':
        return (r'{\bf delete}($' + fmt_expr(s.delete_target) + r'$)')
    if s.kind == 'guard' and _is_recv(s):
        idx_str = ', '.join(s.guard_index)
        return (r'{\bf upon}$(' + idx_str + r')$: ' + _fmt_recv(s)
                + ': ' + fmt_stmt(s.guard_advance))
    if s.kind == 'guard':
        pred = fmt_expr(s.guard_predicate)
        adv  = fmt_stmt(s.guard_advance)
        idx_str = ', '.join(s.guard_index)
        if s.is_upon:
            return (r'{\bf upon}$(' + idx_str + r')$: $' + pred + r'$: ' + adv)
        return (r'{\bf forbidden}$(' + idx_str + r')$: $' + pred + r'$ '
                r'\Rightarrow {\bf advance}: ' + adv)
    raise ValueError(f'unknown stmt kind {s.kind}')


# ------------------------------------------------------------
#  Top-level emitter
# ------------------------------------------------------------

def _fmt_range(r) -> str:
    if r is None:
        return ''
    lo, hi = r
    return f'$[{fmt_expr(lo)}..{fmt_expr(hi)}]$'


# Enum tables of the program being printed, so an enum-typed variable can be
# shown as its value set -- `color: array[1..n] of {red, green}` -- which is
# how the book's figures write it (`color in {red, green}`).
_ENUMS: dict = {}

# Struct tables, so a struct-typed variable prints under its own name
# (`cut: array[0..n] of Cut`) with the field list stated once above.
_STRUCTS: dict = {}

# Initial values of struct fields, keyed by (record, field).
_RECORD_INITS: dict = {}

# What a peer is called in this figure -- `P` unless the program says
# otherwise with setProcessName.
_PROC_NAME: str = 'P'


def fmt_field(t: Type, name: str, init=None) -> str:
    """One struct field.  An enum-typed field is written as membership
    (`color in {red, green}`), which is how the book's figures state it."""
    if t.base in _ENUMS:
        out = (r'$' + fmt_id(name) + r' \in \{'
               + ', '.join(fmt_id(v) for v in _ENUMS[t.base]) + r'\}$')
    else:
        out = f'${fmt_id(name)}$: {fmt_type(t)}'
    if init is not None:
        # Tied together so the closing `endstruct` cannot be stranded on a
        # line of its own, nor the value split from the word `initially`.
        out += ',~initially~$' + fmt_expr(init) + '$'
    return out


def fmt_type(t: Type) -> str:
    if getattr(t, 'is_list', False):
        if t.base in _ENUMS:
            inner = (r'$\{' + ', '.join(fmt_id(v) for v in _ENUMS[t.base])
                     + r'\}$')
        elif t.base in _STRUCTS:
            inner = r'$\mathit{' + t.base + '}$'
        else:
            inner = {'int': 'int', 'double': 'real',
                     'boolean': 'boolean'}.get(t.base, t.base)
        # An arrayed list keeps its dimensions: the figure's
        # `arrays[1..n,1..n] of lists of messages'.
        if t.is_2d:
            lo1, hi1 = t.range1 or (None, None)
            lo2, hi2 = t.range2 or (None, None)
            r1 = f'[{fmt_expr(lo1)}..{fmt_expr(hi1)}]' if t.range1 else '[]'
            r2 = f'[{fmt_expr(lo2)}..{fmt_expr(hi2)}]' if t.range2 else '[]'
            return f'array${r1}{r2}$ of lists of ' + inner
        if t.is_array:
            r = _fmt_range(t.range1)
            return (f'array{r} of lists of ' + inner if r
                    else 'array of lists of ' + inner)
        return 'list of ' + inner
    if t.base in _ENUMS:
        base_name = (r'$\{' + ', '.join(fmt_id(v) for v in _ENUMS[t.base]) + r'\}$')
    elif t.base in _STRUCTS:
        # The struct is anonymous, so the figure shows its fields where a
        # type name would otherwise go, as in the book's
        # `array[1..n] of struct v: ...; color in {red,green} endstruct`.
        flds = '; '.join(
            fmt_field(ft, fn, _RECORD_INITS.get((t.base, fn)))
            for ft, fn in _STRUCTS[t.base])
        base_name = r'\kw{struct} ' + flds + r'~\kw{endstruct}'
    else:
        # Neither a primitive nor a type declared in this program: a field
        # composing in another program (`DiningPhilosophers dining;`).
        # fmt_id renders its name -- honoring a `// display:` override, if
        # the figure wants a math name for it -- rather than a raw dict
        # lookup, which would otherwise raise KeyError.  fmt_id's output is
        # math (\mathit{...} et al.), which -- unlike \textit -- is only
        # legal inside math mode, so it must be wrapped in $...$ here, the
        # same way the enum branch above wraps its own $\{...\}$.
        base_name = {'int': 'int', 'double': 'real',
                     'boolean': 'boolean'}.get(t.base, '$' + fmt_id(t.base) + '$')
    if t.is_set and t.is_array:
        r = _fmt_range(t.range1)
        return f'array{r} of sets of ' + base_name if r else 'array of sets of ' + base_name
    if t.is_set:
        return 'set of ' + base_name
    if t.is_2d:
        if t.range1 or t.range2:
            lo1, hi1 = t.range1 or (None, None)
            lo2, hi2 = t.range2 or (None, None)
            r1 = f'[{fmt_expr(lo1)}..{fmt_expr(hi1)}]' if t.range1 else '[]'
            r2 = f'[{fmt_expr(lo2)}..{fmt_expr(hi2)}]' if t.range2 else '[]'
            return f'array${r1}{r2}$ of ' + base_name
        return '2D array of ' + base_name
    if t.is_array:
        r = _fmt_range(t.range1)
        if r:
            return f'array{r} of ' + base_name
        return 'array of ' + base_name
    return base_name


def fmt_decl(d: Decl) -> str:
    return f'${fmt_id(d.name)}$: {fmt_type(d.type)}'


# A little air between consecutive rules.  The block rules already show
# where each rule ends, so this is small.
_RULE_GAP = r'\vspace{1mm}'


# `// title:` and `// process:` markers, read from the raw source.  LL has
# no string literals, so the caption and the `P_i ::` header come from
# comments rather than from the language.
_TITLE: str = ''
_PROCESS: str = ''
# `// initially:` lines, one per condition the initial configuration must
# satisfy.  These are what the figures state under `initially'.
_INITIALLY: list = []
# Float placement for the generated `algorithm' environment.  `tbp' rather
# than `H': a program near a full page tall cannot share a page with text
# under `H', so LaTeX breaks the page and strands the remainder of the
# previous one as white space.  As a float it settles at the top or bottom
# of a page and the text flows around it.  A `// placement:' marker overrides
# this for a program that really must sit where it is written.
_DEFAULT_PLACEMENT: str = 'tbp'
_PLACEMENT: str = _DEFAULT_PLACEMENT
# `// display: name = latex' markers: how a name is to be typeset, when the
# chapter's notation for it is not a legal identifier.  See fmt_id.
_DISPLAY: dict = {}
# `// hide: name, name' marker: declarations the figure does not make.  The
# number of processes is a parameter of the program rather than state of a
# process, and the book's own figures never declare it.
_HIDE: set = set()
# `// channel: fifo' marker: a standing assumption the algorithm needs from
# its channels, beyond the book's default (reliable, arbitrary delay, no
# ordering promised).  Purely descriptive -- like `initially', it documents
# a precondition the correctness proof relies on, and does not affect
# codegen.  Absent means the default, so most figures print nothing extra.
_CHANNEL: str = ''
# `// style: reactive|fixpoint|stateful' marker: which of LL's extension
# groups (see the KEYWORDS comment in llc.py) this program actually uses --
# `reactive` for message passing (upon/send/receive/message), `fixpoint`
# for guarded commands (forbidden/advance/ensure), `stateful` for a plain
# object whose entry-body decls persist as fields across separately
# callable aux methods (setStateful()).  Purely documentary: read here so
# a maintainer skimming the header sees at a glance which LL dialect the
# rest of the file is written in, same as `title`/`process`; absent is
# fine, and any value is accepted (a typo prints a note, nothing more) so
# this can never be the reason a program fails to print.
_STYLE: str = ''
_STYLE_NAMES = ('reactive', 'fixpoint', 'stateful')
# `// caption: bare` marker: print just "Algorithm Name", with no
# ": <title>" after it -- for a figure whose name already says everything
# (e.g. AlphaSynchronizer) and whose `// title:` is documentation for the
# .ll source, not something the printed book needs repeated in the caption.
# Absent means the default: "Algorithm Name: <title>", as everywhere else.
_CAPTION: str = ''
# `// fontsize: small` (or `footnotesize`) marker: shrink the whole
# algorithm box's text size, for a program whose box is otherwise too tall
# to fit on one page even after compressing statement grouping. Absent
# means the default: whatever size is in effect where the box is placed.
_FONTSIZE: str = ''
_FONTSIZE_NAMES = ('small', 'footnotesize', 'scriptsize')

def read_markers(src: str) -> None:
    global _TITLE, _PROCESS, _INITIALLY, _PLACEMENT, _DISPLAY, _HIDE, _CHANNEL, _STYLE, _CAPTION, _FONTSIZE
    _TITLE = ''
    _PROCESS = ''
    _INITIALLY = []
    _PLACEMENT = _DEFAULT_PLACEMENT
    _DISPLAY = {}
    _HIDE = set()
    _CHANNEL = ''
    _STYLE = ''
    _CAPTION = ''
    _FONTSIZE = ''
    for line in src.splitlines():
        t = line.strip()
        if not t.startswith('//'):
            if t and not t.startswith('/*'):
                break          # markers are a header, not scattered about
            continue
        body = t[2:].strip()
        low = body.lower()
        if low.startswith('title:'):
            _TITLE = body[6:].strip()
        elif low.startswith('process:'):
            _PROCESS = body[8:].strip()
        elif low.startswith('initially:'):
            _INITIALLY.append(body[10:].strip())
        elif low.startswith('placement:'):
            _PLACEMENT = body[10:].strip()
        elif low.startswith('display:'):
            spec = body[8:]
            if '=' in spec:
                nm, tex = spec.split('=', 1)
                _DISPLAY[nm.strip()] = tex.strip()
        elif low.startswith('hide:'):
            for nm in body[5:].split(','):
                if nm.strip():
                    _HIDE.add(nm.strip())
        elif low.startswith('channel:'):
            _CHANNEL = body[8:].strip().lower()
        elif low.startswith('style:'):
            _STYLE = body[6:].strip().lower()
            if _STYLE not in _STYLE_NAMES:
                print(f"llpp: note: unrecognized '// style: {_STYLE}' "
                      f"(expected one of {_STYLE_NAMES}); ignoring",
                      file=sys.stderr)
        elif low.startswith('caption:'):
            _CAPTION = body[8:].strip().lower()
            if _CAPTION != 'bare':
                print(f"llpp: note: unrecognized '// caption: {_CAPTION}' "
                      f"(expected 'bare'); ignoring", file=sys.stderr)
                _CAPTION = ''
        elif low.startswith('fontsize:'):
            _FONTSIZE = body[9:].strip().lower()
            if _FONTSIZE not in _FONTSIZE_NAMES:
                print(f"llpp: note: unrecognized '// fontsize: {_FONTSIZE}' "
                      f"(expected one of {_FONTSIZE_NAMES}); ignoring",
                      file=sys.stderr)
                _FONTSIZE = ''


def _scan_initially(src_text: str) -> list:
    """Just the `// initially:' lines from a source's header, independent
    of read_markers (which overwrites _TITLE/_PROCESS/etc. as a side
    effect and is meant to be called once, on the top-level file). Used to
    pull a composed program's members' own `initially' conditions into the
    merged figure -- `observes ... with ...` and `union` both resolve to a
    stub file with no header comments of its own, so without this the
    merged figure would silently print no `initially' clause at all, even
    when every member declares one."""
    out = []
    for line in src_text.splitlines():
        t = line.strip()
        if not t.startswith('//'):
            if t and not t.startswith('/*'):
                break
            continue
        body = t[2:].strip()
        if body.lower().startswith('initially:'):
            out.append(body[10:].strip())
    return out


def _find_member_source(member_name: str, dirs: list) -> Optional[str]:
    """The raw source text of the .ll/.llp file that defines `member_name`,
    searched for the same way scan_signatures() finds it -- by file stem,
    across the same directories -- so its header markers can be read even
    though scan_signatures() itself keeps only the parsed Program."""
    for dir_ in dirs:
        for ext in ('.ll', '.llp'):
            path = os.path.join(dir_, member_name + ext)
            if os.path.isfile(path):
                with open(path) as fh:
                    return fh.read()
    return None


def _emit_procs(lines, prog, emitted) -> None:
    """Emit the program's procedures, in source order."""
    for m in prog.aux_methods:
        if m.name in emitted:
            continue
        emitted.add(m.name)
        lines.append(r'\BlankLine')
        params = ', '.join(fmt_id(d.name) for d in m.params)
        # An aux method that returns a value is a function, not a procedure.
        kw = r'\Fun{' if getattr(m, 'return_type', None) is not None \
            else r'\Proc{'
        lines.append(kw + '$' + fmt_id(m.name) + '(' + params + ')$}{')
        lines.extend(body_lines(m.body, '  '))
        lines.append('}')


def emit_latex(prog: Program) -> str:
    global _ENUMS, _STRUCTS, _RECORD_INITS, _PROC_NAME
    _ENUMS = prog.enums
    _STRUCTS = {n: f for n, f in prog.records.items()
                if n not in prog.messages}
    _RECORD_INITS = prog.record_inits
    _PROC_NAME = prog.process_name
    lines = []
    lines.append(r'\begin{algorithm}[' + _PLACEMENT + ']')
    if _FONTSIZE:
        # Scoped to the algorithm environment itself, so it never leaks to
        # surrounding text; algorithm2e's own spacing scales with it too.
        lines.append('\\' + _FONTSIZE)
    display_name = prog.name.replace('_', '-')
    lines.append(r'\SetAlgoRefName{' + display_name + '}')
    # One `var:` block holds the header params and the declarations alike --
    # the split into `input:`/`output:` said less than it cost -- with one
    # variable per line so a long state vector stays readable.
    emitted_procs = set()
    if _PROCESS:
        # `setProcess(i)` is required by codegen (it names the receive-group
        # merging helpers' own index variable) whenever the program receives
        # from several distinct senders, whether or not the *algorithm*
        # itself is a family of n processes -- a single centralized checker
        # that receives from P_1..P_n also needs it.  So `process_index`
        # being set is not by itself evidence of a `P_i :: 1 <= i <= n`
        # header: that header is only right when the declared process name
        # (`// process: P_i`) is itself subscripted by that same index,
        # e.g. `P_i`.  A fixed name like `P_0` prints bare, with no range.
        proc_subscript = _PROCESS.split('_', 1)[1] if '_' in _PROCESS else None
        if prog.process_index and proc_subscript == prog.process_index:
            # Naming the index's range here -- `1 <= i <= n` -- is what lets
            # the var block leave `n` out entirely: its declaration is this
            # line, not a separate one, and the scope of the index variable
            # is stated once instead of left implicit.
            idx = fmt_id(prog.process_index)
            n = fmt_id('n')
            rng = (f'{idx} < {n}' if prog.base == 0 else f'{idx} \\leq {n}')
            lo = '0' if prog.base == 0 else '1'
            lines.append(r'\AlgHead{$' + _PROCESS + ' :: ' + lo
                         + r' \leq ' + rng + '$}')
        else:
            # A single, fixed process (a centralized checker, say), not one
            # of a family P_i for 1 <= i <= n, so there is no range to state.
            lines.append(r'\AlgHead{$' + _PROCESS + '$}')
    if _CHANNEL == 'fifo' or prog.channel == 'fifo':
        lines.append(r'\AlgHead{\kw{assumes} FIFO channels}')
    # A distributed program takes `n`, the number of processes.  Its scope
    # is now declared on the process-header line above, so the var block
    # never states it separately -- whether or not the rules mention it,
    # unlike the figure's other state, which appears only if used.
    hide_n = prog.process_index is not None
    var_parts = [f'${fmt_id(p.name)}$: {fmt_type(p.type)}' for p in prog.params
                 if not (hide_n and p.name == 'n') and p.name not in _HIDE]
    # Declarations the source wrote together are set on one line, as
    # `left, right: int, initially 0;'.  Grouping follows the source rather
    # than being inferred, so which variables belong together stays the
    # author's judgement.
    decls = [d for d in prog.decls if d.kind in ('output', 'var')]
    k = 0
    while k < len(decls):
        d = decls[k]
        run = [d]
        if d.group is not None:
            while (k + len(run) < len(decls)
                   and decls[k + len(run)].group == d.group):
                run.append(decls[k + len(run)])
        names = ', '.join('$' + fmt_id(x.name) + '$' for x in run)
        part = names + ': ' + fmt_type(d.type)
        if d.init is not None:
            part += ', initially $' + fmt_expr(d.init) + '$'
        var_parts.append(part)
        k += len(run)
    if var_parts:
        # `var' stands alone and the declarations are indented under it, each
        # ended by a semicolon, as the book's own figures set them.
        lines.append(r'\AlgHead{\kw{var}}')
        for part in var_parts:
            lines.append(r'\AlgHead{\quad ' + part + ';}')
    if _INITIALLY:
        # Conditions on the initial configuration, numbered when there is
        # more than one, as the book's figures set them.
        lines.append(r'\AlgHead{\kw{initially}}')
        for k, cond in enumerate(_INITIALLY, 1):
            num = f'{k}. ' if len(_INITIALLY) > 1 else ''
            lines.append(r'\AlgHead{\quad ' + num + cond + '}')
    if prog.priority_expr is not None:
        lines.append(r'\textbf{priority}$(j)$: $'
                     + fmt_expr(prog.priority_expr) + r'$\;')
    lines.append(r'\BlankLine')
    # Emit the program body stmts.  Guard stmts at the top level use the
    # algorithm2e \Forbidden / \Advance markup for continuity with the book;
    # everything else uses plain fmt_stmt().
    for s in prog.stmts:
        if s.kind == 'guard':
            idx_str = ', '.join(s.guard_index)
            # Emit always clauses before the forbidden/ensure line.
            for atype, aname, aexpr in s.guard_always:
                lines.append(r'\Always{$(' + idx_str + r')$: $'
                             + fmt_id(aname) + ' = ' + fmt_expr(aexpr) + r'$}')
            # Ensure-sugar guards get the compact {\bf ensure} rendering.
            if s.ensure_op is not None:
                lhs = fmt_id(s.ensure_lhs_name)
                if s.ensure_lhs_index is not None:
                    lhs = lhs + '[' + fmt_expr(s.ensure_lhs_index) + ']'
                op_tex = {'<=': r' \leq ', '>=': r' \geq ',
                          '<': '<', '>': '>', '=': '='}[s.ensure_op]
                rhs = fmt_expr(s.ensure_rhs)
                lines.append(r'{\bf ensure}$(' + idx_str + ')$: $'
                             + lhs + op_tex + rhs + r'$\;')
                continue
            adv = s.guard_advance
            if _is_recv(s):
                adv_stmts0 = adv.body if adv.kind == 'block' else [adv]
                lines.append(r'\Upon{' + _fmt_recv(s) + '}{')
                lines.extend(body_lines(adv_stmts0, '  '))
                lines.append('}')
                lines.append(_RULE_GAP)
                continue
            pred = fmt_expr(s.guard_predicate)
            adv_stmts = adv.body if adv.kind == 'block' else [adv]
            body = '; '.join(fmt_stmt(x) for x in adv_stmts)
            # `upon` guards render with their own compact spelling.
            if s.is_upon:
                lines.append(r'\Upon{$' + pred + '$}{')
                lines.extend(body_lines(adv_stmts, '  '))
                lines.append('}')
                lines.append(_RULE_GAP)
                continue
            if s.guard_pick_var is not None:
                lines.append(r'\ForbiddenC{$(' + idx_str + r')$: $' + pred + '$}'
                             + r'{pick such $' + s.guard_pick_var + '$}')
            else:
                lines.append(r'\Forbidden{$(' + idx_str + r')$: $' + pred + '$}')
            lines.append(r'\Advance{: ' + body + '}')
        else:
            lines.append(fmt_stmt(s) + r'\;')
    # Render the trailing `return X` that _scan_result stripped from the body.
    if prog.trailing_return is not None:
        lines.append(fmt_stmt(prog.trailing_return) + r'\;')
    # Auxiliary methods are shown as procedures, the way the book's figures
    # present them (e.g. "Procedure paintState(i)" in the WCP checker).
    _emit_procs(lines, prog, emitted_procs)
    # Drop into the caption directly (no more rule/action iteration below).
    label = prog.name
    if _CAPTION == 'bare':
        # `\SetAlgoCaptionSeparator{}` is consumed by the very next
        # `\caption`, so this does not affect any algorithm after this one.
        lines.append(r'\SetAlgoCaptionSeparator{}')
        lines.append(r'\caption[]{\label{fig:' + label + '}}')
    else:
        cap = _TITLE if _TITLE else ('LLP program for ' + display_name)
        lines.append(r'\caption[]{' + cap + r' \label{fig:' + label + '}}')
    lines.append(r'\end{algorithm}')
    return '\n'.join(lines) + '\n'


# ------------------------------------------------------------
#  Driver
# ------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print('Usage: llpp.py FILE.llp [--out FILE.tex] [--also DIR ...]',
              file=sys.stderr)
        sys.exit(2)
    src_path = sys.argv[1]
    out_path = None
    if '--out' in sys.argv:
        i = sys.argv.index('--out')
        out_path = sys.argv[i + 1]
    also_dirs = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == '--also']
    with open(src_path) as f:
        src = f.read()
    read_markers(src)
    toks = tokenize(src)
    stem = os.path.splitext(os.path.basename(src_path))[0]

    prog = Parser(toks, default_name=stem).parse_program()
    # Names of the member programs actually folded into `prog` below, if
    # any -- used only to pull their `// initially:' header comments into
    # this figure afterward, since a composed program's own stub file (all
    # its raw source has is the `observes ... with ...`/`union' line) has
    # no header comments of its own for read_markers() to have found.
    member_names: list = []
    if prog.observes_subject is not None:
        sigs = scan_signatures(src_path, also_dirs)
        subject = sigs.get(prog.observes_subject)
        observer = sigs.get(prog.observes_observer)
        if subject is None:
            print(f"llpp: {src_path}: subject '{prog.observes_subject}' of "
                  f"`observes' not found (pass its directory with --also)",
                  file=sys.stderr)
            sys.exit(1)
        if observer is None:
            print(f"llpp: {src_path}: observer '{prog.observes_observer}' of "
                  f"`observes' not found (pass its directory with --also)",
                  file=sys.stderr)
            sys.exit(1)
        prog = merge_observed_program(subject, observer, prog.name)
        member_names = [subject.name, observer.name]
    if prog.union_members is not None:
        sigs = scan_signatures(src_path, also_dirs)
        members = []
        for member_name in prog.union_members:
            member = sigs.get(member_name)
            if member is None:
                print(f"llpp: {src_path}: union member '{member_name}' "
                      f"not found (pass its directory with --also)",
                      file=sys.stderr)
                sys.exit(1)
            members.append(member)
        prog = merge_union_programs(members, prog.name)
        member_names = [m.name for m in members]
    if member_names:
        dirs = [os.path.dirname(os.path.abspath(src_path)) or '.']
        dirs.extend(also_dirs)
        for member_name in member_names:
            member_src = _find_member_source(member_name, dirs)
            if member_src is not None:
                _INITIALLY.extend(_scan_initially(member_src))
    tex = emit_latex(prog)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path, 'w') as f:
            f.write(tex)
        print(f'wrote {out_path}')
    else:
        sys.stdout.write(tex)


if __name__ == '__main__':
    main()
