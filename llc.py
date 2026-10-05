#!/usr/bin/env python3
"""LLC -- LL Compiler.

Compiles an LL (Lattice Language) source file to a self-contained Java
class. See syntax.md for the grammar.

Usage:  python3 llc.py FILE.llp [--out DIR]
Output: DIR/<Name>.java (default DIR is the current directory).
"""

from __future__ import annotations
import re
import sys
import os
import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ============================================================
#  Lexer
# ============================================================

# LL is meant to be a generalization of Java: a program built entirely
# from the "Java core" keywords below is just Java, re-parsed and
# re-emitted; what makes it LL is the "LL extensions" layer on top,
# with no Java equivalent, that this compiler alone understands.
# Keeping the two lists apart here -- rather than one alphabetized set
# -- is itself the documentation: anywhere one of the extension words
# or constructs appears in a `.ll` file, that is unmistakably LL, not
# Java passed through.  (See also the `// style:` marker read by
# llpp.py's read_markers(), which lets a file say up front which of
# these extension groups it actually uses.)
#
# Java core (ordinary sequential Java; LL just reuses the syntax):
#   class void new if else while true false int double boolean
#   return exit enum struct infinity
#   -- plus the text-matched (not reserved) `swap(a, b)`, and the
#   generic-looking `list<T>` / `set<T>` collection types.
#
# LL extensions (no Java equivalent -- this is the generalization):
#   * Guarded commands / lattice-linear fixpoints:
#       forbidden  advance  ensure  always
#   * Message passing between distributed processes:
#       upon  send  receive  message
#   * Quantifiers over a process range or a set:
#       forall  exists  pick  (and the `in` of both)
#   * Composing one program's process into another's, or weaving an
#     observer's hooks around every send/receive of a subject program
#     (text-matched keywords, not reserved words):
#       observes ... with ...
#   * Per-program configuration, read once from the preamble (also
#     text-matched, not reserved -- see _CONFIG_NAMES):
#       setProcess  setBase  setPriority  setMode  setProcessName
#       setChannel  setStateful
KEYWORDS = {
    'class', 'void', 'new',
    'forbidden', 'advance', 'ensure', 'always', 'upon', 'send',
    'receive', 'message', 'enum', 'struct',
    'if', 'else', 'forall', 'exists', 'while',
    'true', 'false', 'infinity', 'in', 'pick',
    'int', 'double', 'boolean',
    'return', 'exit',
}

TOKEN_SPEC = [
    ('BLOCK_CMT', r'/\*[\s\S]*?\*/'),
    ('COMMENT', r'//[^\n]*'),
    ('WS',      r'[ \t\r\n]+'),
    ('REAL',    r'\d+\.\d+'),
    ('INT',     r'\d+'),
    ('DOTDOT',  r'\.\.'),
    ('ARROW',   r'=>'),
    ('ANDAND',  r'&&'),
    ('OROR',    r'\|\|'),
    ('PIPE',    r'\|'),
    ('NEQ',     r'!='),
    ('LEQ',     r'<='),
    ('GEQ',     r'>='),
    ('LT',      r'<'),
    ('GT',      r'>'),
    ('EQEQ',    r'=='),
    ('ASSIGN',  r'='),
    ('BANG',    r'!'),
    ('PLUS',    r'\+'),
    ('MINUS',   r'-'),
    ('STAR',    r'\*'),
    ('SLASH',   r'/'),
    ('PERCENT', r'%'),
    ('LPAREN',  r'\('),
    ('RPAREN',  r'\)'),
    ('LBRACK',  r'\['),
    ('RBRACK',  r'\]'),
    ('LBRACE',  r'\{'),
    ('RBRACE',  r'\}'),
    ('SEMI',    r';'),
    ('COLON',   r':'),
    ('COMMA',   r','),
    ('DOT',     r'\.'),
    ('ID',      r'[A-Za-z_][A-Za-z_0-9]*'),
]

MASTER_RE = re.compile('|'.join(f'(?P<{n}>{p})' for n, p in TOKEN_SPEC))


@dataclass
class Tok:
    kind: str
    text: str
    line: int


def tokenize(src: str) -> List[Tok]:
    toks: List[Tok] = []
    line = 1
    pos = 0
    while pos < len(src):
        m = MASTER_RE.match(src, pos)
        if not m:
            raise SyntaxError(f'line {line}: unexpected character {src[pos]!r}')
        kind = m.lastgroup
        text = m.group()
        if kind == 'WS':
            line += text.count('\n')
        elif kind == 'COMMENT':
            pass
        elif kind == 'BLOCK_CMT':
            line += text.count('\n')
        else:
            if kind == 'ID' and text in KEYWORDS:
                kind = 'KW_' + text.upper()
            toks.append(Tok(kind, text, line))
        pos = m.end()
    toks.append(Tok('EOF', '', line))
    return toks


# ============================================================
#  AST
# ============================================================

@dataclass
class Type:
    base: str             # 'int' | 'double' | 'boolean'
    is_array: bool = False
    is_2d: bool = False   # True for int[][] etc. (2 or more -- see dims)
    is_set: bool = False   # element is set<base>, e.g. set<int>[] for pre(j)
    is_list: bool = False  # list<base>: an ordered, growable sequence
    range1: Optional[tuple] = None  # (lo_expr, hi_expr) for first dimension
    range2: Optional[tuple] = None  # (lo_expr, hi_expr) for second dimension
    dims: int = 0
        # The *exact* bracket count, e.g. 4 for `int[][][][]`. is_array/is_2d
        # only ever distinguish "0"/"1"/"2 or more" and stay that way (many
        # checks throughout this file rely on that coarser boolean); dims is
        # purely additive, populated alongside them, and is 0 for anything
        # 0D/1D/2D exactly -- the only consumer that needs it is Java type
        # rendering (java_base_type) for 3D+ arrays, e.g. LLPwithRejection's
        # `int[][][][] R`, which a bare is_2d check would wrongly render
        # as `int[][]`, two dimensions short.


@dataclass
class Decl:
    kind: str             # 'var' | 'param'
    type: Type
    name: str
    init: Optional['Expr'] = None
    # Declarations written together -- `int left = 0, right = 0;` -- share a
    # group id, and the printer puts them on one line.  Grouping is the
    # author's choice, so it is recorded here rather than inferred.
    group: Optional[int] = None


# Rule/Action dataclasses were removed in the v2 refactor.  A guarded
# command (forbidden/ensure + advance) is now a kind of Stmt, parsed at
# the top level of the program body alongside other statements.


@dataclass
class Stmt:
    kind: str  # 'assign'|'if'|'forall'|'swap'|'par'|'call'|'decl'|'block'
               # |'return'|'guard'|'delete'
    # 'assign'
    lhs_name: Optional[str] = None
    lhs_index: Optional['Expr'] = None
    lhs_index2: Optional['Expr'] = None   # second index for 2D LHS, e.g., G[i][j] = ...
    lhs_index_extra: List['Expr'] = field(default_factory=list)
        # 3rd+ chained index on the LHS, e.g. RT[a][b][c][d] = ... holds [c, d]
        # (lhs_index/lhs_index2 keep a, b). Additive: empty for 1D/2D LHS.
    lhs_field: Optional[str] = None       # record field on the LHS, e.g. cut[i].color = ...
    lhs_field_index: Optional['Expr'] = None  # index into that field, e.g. cut[i].v[t] = ...
    rhs: Optional['Expr'] = None
    # 'if'
    cond: Optional['Expr'] = None
    then_stmts: List['Stmt'] = field(default_factory=list)
    else_stmts: List['Stmt'] = field(default_factory=list)
    # 'forall'
    loop_var: Optional[str] = None
    loop_lo: Optional['Expr'] = None
    loop_hi: Optional['Expr'] = None
    loop_set: Optional['Expr'] = None
    body: List['Stmt'] = field(default_factory=list)
    # 'swap'
    swap_a: Optional['Expr'] = None
    swap_b: Optional['Expr'] = None
    # 'par' -- bracket composition `[ B1 OP B2 (OP B3)* ]`
    # par_op selects the composition flavor: '[]' (parallel, default),
    # '&&' (predicate conjunction), 'or' (nondeterministic choice),
    # ';' (sequential), '|' (pipelining).
    par_branches: List['Stmt'] = field(default_factory=list)
    par_op: str = '[]'
    # 'call'
    call_name: Optional[str] = None
    call_args: List['Expr'] = field(default_factory=list)
    # 'decl'
    decl_type: Optional[Type] = None
    decl_name: Optional[str] = None
    decl_init: Optional['Expr'] = None
    # 'guard' -- forbidden(j): PRED => advance: STMT
    #            or forbidden(i, j): PRED => advance: STMT  (multi-index)
    guard_index: Optional[List[str]] = None
    guard_predicate: Optional['Expr'] = None
    guard_advance: Optional['Stmt'] = None
    guard_pick_var: Optional[str] = None
    # Source line the guard's predicate begins on, so a generated rule can
    # quote the `upon' it came from in its doc comment.
    guard_line: Optional[int] = None
    # Set when the guard was written with the `upon` spelling, so the
    # LaTeX printer can render `upon PRED : ACTION` instead of the
    # `forbidden ... => advance ...` form.  Purely presentational: the
    # node is otherwise identical to a `forbidden`/`advance` guard.
    is_upon: bool = False
    # If the guard was desugared from an `ensure` clause we keep the
    # original parts so the LaTeX printer can render the compact form.
    ensure_lhs_name: Optional[str] = None
    ensure_lhs_index: Optional['Expr'] = None
    ensure_op: Optional[str] = None
    ensure_rhs: Optional['Expr'] = None
    # 'always' clauses attached to a guard -- list of (name, expr) pairs.
    # Parsed from `always : var = expr` preceding a forbidden/ensure.
    guard_always: List[tuple] = field(default_factory=list)
    # 'return' -- optional return value (None = bare `return;`)
    return_value: Optional['Expr'] = None
    # 'send' -- send(DEST, PAYLOAD) or send(DEST, TAG, PAYLOAD).
    # The tag, when present, is a symbolic message name written as an
    # identifier; it is emitted as a string constant.
    send_dest: Optional['Expr'] = None
    send_tag: Optional[str] = None
    send_payload: List['Expr'] = field(default_factory=list)
    # A guard written as `upon receive(TAG, b1, ..., bk) from k`.  The
    # predicate is then implicit -- "some sender has such a message queued
    # for me" -- so guard_predicate is None and the emitter builds the
    # forbidden/advance pair directly.
    recv_tag: Optional[str] = None
    recv_binders: List[str] = field(default_factory=list)
    recv_from: Optional[str] = None
    recv_from_const: Optional[int] = None   # `from 0`: a fixed sender


@dataclass
class Expr:
    kind: str
    # variants (only relevant fields used per kind):
    value: Optional[object] = None    # for 'int', 'double', 'boolean', 'infinity'
    name: Optional[str] = None        # for 'id', 'index', 'call'
    idx: Optional['Expr'] = None      # for 'index'
    idx2: Optional['Expr'] = None     # for 'new_array' second dimension
    idx_extra: List['Expr'] = field(default_factory=list)
        # 3rd+ chained index, e.g. RT[a][b][c][d] (kind stays 'index2'; the
        # first two indices are still idx/idx2, this holds [c, d]) -- or,
        # for 'new_array', the 3rd+ *size* expression, e.g. `new int[n][m]
        # [n][m]` (idx/idx2 hold the first two sizes, this holds [n, m]).
        # Additive: empty for every 1D/2D use, which is untouched.
    jagged: bool = False
        # for 'new_array': True iff written `new T[n][]` (an empty second
        # bracket -- only the outer dimension sized). See the Java 'new_array'
        # codegen, the only consumer; the JS emitter doesn't need it.
    args: List['Expr'] = field(default_factory=list)  # for 'call'
    op: Optional[str] = None          # for 'bin', 'un'
    lhs: Optional['Expr'] = None
    rhs: Optional['Expr'] = None
    # quantifier
    quant: Optional[str] = None       # 'min'|'max'|'sum'|'count'|'exists'|'forall'
    q_var: Optional[str] = None
    q_lo: Optional['Expr'] = None
    q_hi: Optional['Expr'] = None
    q_body: Optional['Expr'] = None
    q_set: Optional['Expr'] = None    # if set-valued range: iterate over this int[]
    pick_var: Optional[str] = None    # "pick X" after the range; witness capture


@dataclass
class Method:
    """An auxiliary method declared inside a class, after the main
    constructor.  Distinguished from the main member by having an explicit
    return type."""
    name: str
    return_type: Optional['Type']        # None for void
    params: List[Decl] = field(default_factory=list)
    body: List['Stmt'] = field(default_factory=list)


@dataclass
class Program:
    name: str
    params: List[Decl] = field(default_factory=list)  # header: Name(int lo, int[] A)
    decls: List[Decl] = field(default_factory=list)   # var fields only
    mode: str = 'array'
    priority_expr: Optional[Expr] = None
    priority_order: str = 'min'
    base: int = 0  # index base: 0 for [0..n-1], 1 for [1..n]
    # What a peer process is called in the printed figure: `P` by default,
    # `M` for the monitor processes of the distributed WCP checker.
    process_name: str = 'P'
    # The index this program's rules run at, declared once by `setProcess(i)`
    # instead of being repeated on every rule.  It is the default binder of
    # an `upon` written without an index list, which is how the figures read:
    # the header says `P_i ::` and the rules below it say only `upon PRED`.
    process_index: Optional[str] = None
    # A standing assumption the algorithm needs from its channels, beyond
    # the default (reliable, arbitrary delay, no ordering promised), set by
    # `setChannel(fifo)`.  Purely descriptive: it documents a precondition
    # the correctness proof relies on, the same role `initially` plays for
    # the starting global state, and does not affect codegen.
    channel: Optional[str] = None
    # Set by `setStateful()`: this program is an object with several
    # independently-callable public operations (aux methods) that share
    # state across separate calls -- a register, a lock, a monitor -- as
    # opposed to a pure algorithm (sorting, divide-and-conquer) whose aux
    # methods are recursive helpers and whose entry-body variables are
    # therefore method-local by default. Explicit and opt-in, rather than
    # inferred from "has aux methods", because a recursive helper program
    # already has aux methods too and must keep its current, correct,
    # locals-only behavior.
    stateful: bool = False
    stmts: List['Stmt'] = field(default_factory=list)  # the program body
    # Name of the var that `return X` designates as the program's result.
    # If None the program has no shareable output (no subroutine ctor).
    result_name: Optional[str] = None
    # The trailing `return X` stripped from the body by _scan_result, kept so
    # the LaTeX pretty-printer can still render it.  Codegen ignores this and
    # re-emits the return from result_name.
    trailing_return: Optional['Stmt'] = None
    # Auxiliary methods declared inside the same class, after the main.
    aux_methods: List[Method] = field(default_factory=list)
    # The entry method's declared return type.  None means `void`.
    return_type: Optional['Type'] = None
    # Name of the entry method.  May differ from the class name.
    entry_name: str = ''
    # `message TAG(type name, ...)` declarations, keyed by tag.  Each value
    # is the list of (Type, name) fields of that message's payload.  Used to
    # type the binders of `upon receive(TAG, b1, ..., bk)`.
    messages: dict = field(default_factory=dict)
    # `enum Name { a, b, ... }` declarations, keyed by type name; each value
    # is the ordered list of constant names.  Enum constants are values, not
    # fields, so they never appear in the emitted var/output list.
    enums: dict = field(default_factory=dict)
    # Record types, keyed by name, each a list of (Type, fieldname).  Both
    # `struct` and `message` declare records; the message table above is the
    # subset that may be sent, so field lookup and Java class emission are
    # shared rather than duplicated.
    records: dict = field(default_factory=dict)
    record_inits: dict = field(default_factory=dict)
    anon_records: set = field(default_factory=set)
    # `class NAME observes SUBJECT with OBSERVER;` -- a composed program,
    # built by weaving OBSERVER's hook methods (onActivate, onReceive,
    # beforeSend, onDeactivate -- each optional) around every message
    # SUBJECT sends and receives, without editing SUBJECT itself. Resolved
    # by merge_observed_program() once every file's signature is known;
    # until then this Program is just the two names below, with an empty
    # body.
    observes_subject: Optional[str] = None
    observes_observer: Optional[str] = None
    # `class NAME union A, B, ...;` -- a composed program, the symmetric
    # counterpart of `observes ... with ...`: every member's rules
    # (top-level guards) are simply pooled, so a rule from any member fires
    # whenever it is enabled, with no hook-weaving and no asymmetry between
    # the members. Resolved by merge_union_programs() once every file's
    # signature is known; until then this Program is just the member names
    # below, with an empty body. See merge_union_programs() for the exact
    # rules on sharing a field/message/enum by name across members.
    union_members: Optional[List[str]] = None


# ============================================================
#  Parser (recursive descent)
# ============================================================

class Parser:
    def __init__(self, toks: List[Tok], default_name: Optional[str] = None):
        self.toks = toks
        self.pos = 0
        # Used as the class name when the source has no `class` wrapper
        # (i.e., a top-level function-only program for `--noclass` mode).
        self.default_name = default_name
        # `message TAG(...)` declarations seen so far, keyed by tag; consulted
        # when parsing `send` and `upon receive` to check and type payloads.
        self.messages: dict = {}
        # `enum Name { ... }` declarations seen so far, keyed by type name.
        self.enums: dict = {}
        # Record types (structs and messages), keyed by name.
        self.records: dict = {}
        # Initial value of a record field, keyed by (record, field).
        self.record_inits: dict = {}
        self._cur_record: Optional[str] = None
        # Of those, the ones from anonymous `struct` declarations, whose
        # names are generated and so must never be printed.
        self.anon_records: set = set()
        # Set by a `setProcess(i)` in the preamble; the default binder for an
        # `upon` rule written without an index list.  It is read straight off
        # the token stream rather than waited for, because the class-less form
        # peels its config calls only after the body has been parsed -- by
        # which time the rules would already have taken the fallback binder.
        self.process_index: Optional[str] = self._scan_process_index(toks)
        self._decl_group = 0

    @staticmethod
    def _scan_process_index(toks: List[Tok]) -> Optional[str]:
        """The argument of the first `setProcess(i)` in the token stream."""
        for k in range(len(toks) - 3):
            if (toks[k].kind == 'ID' and toks[k].text == 'setProcess'
                    and toks[k + 1].kind == 'LPAREN'
                    and toks[k + 2].kind == 'ID'
                    and toks[k + 3].kind == 'RPAREN'):
                return toks[k + 2].text
        return None

    def peek(self, off: int = 0) -> Tok:
        return self.toks[self.pos + off]

    def eat(self, *kinds: str) -> Tok:
        t = self.toks[self.pos]
        if t.kind not in kinds:
            raise SyntaxError(
                f'line {t.line}: expected {"/".join(kinds)}, got {t.kind} {t.text!r}')
        self.pos += 1
        return t

    def accept(self, *kinds: str) -> Optional[Tok]:
        if self.toks[self.pos].kind in kinds:
            return self.eat(*kinds)
        return None

    # program := "class" ID "{" return_type ID params? "{" decls config* stmt_list "}"
    #             aux_method* "}"
    #          |  method+
    # The first form is a Java class whose first member is the entry method
    # and any number of auxiliary methods follow.  The second form (the
    # `--noclass` style) is just one or more bare method definitions; the
    # class name is taken from `default_name` (set by the driver from the
    # filename).
    def _parse_enum_decls(self) -> dict:
        """enum Name { a, b, ... };  -- zero or more, before the program.

        Declares a finite value type, the LL spelling of the book's
        `color in {red, green}`.  The constants are values rather than
        variables, so they are not fields and do not appear in the
        program's var list."""
        enums: dict = {}
        while self.peek().kind == 'KW_ENUM':
            self.eat('KW_ENUM')
            name = self.eat('ID').text
            if name in enums:
                raise SyntaxError(f'enum `{name}` declared twice')
            vals: list = []
            self.eat('LBRACE')
            if self.peek().kind != 'RBRACE':
                while True:
                    vals.append(self.eat('ID').text)
                    if not self.accept('COMMA'):
                        break
            self.eat('RBRACE')
            self.accept('SEMI')
            if not vals:
                raise SyntaxError(f'enum `{name}` has no constants')
            enums[name] = vals
            self.enums[name] = vals
        return enums

    def _parse_record_fields(self, opener: str, closer: str) -> list:
        """The typed field list shared by `struct { ... }` and
        `message ( ... )`."""
        fields: list = []
        self.eat(opener)
        if self.peek().kind != closer:
            while True:
                ftype = self.parse_type()
                fname = self.eat('ID').text
                if self.accept('ASSIGN'):
                    # `Color color = red;` -- the field's initial value, which
                    # is how the figures state `initially red`.
                    self.record_inits[(self._cur_record, fname)] = \
                        self.parse_expr()
                fields.append((ftype, fname))
                self.accept('SEMI')
                if not self.accept('COMMA') and self.peek().kind == closer:
                    break
                if self.peek().kind == closer:
                    break
        self.eat(closer)
        return fields

    def _parse_named_struct_decls(self) -> None:
        """`struct Name { type field; ... };` before the class -- a named
        record type, usable wherever a type is: as a variable's type, and as
        the element type of a list.  Inside the class body the same keyword
        declares a variable with an anonymous record type instead; the two
        are told apart by where they appear, as in C."""
        while self.peek().kind == 'KW_STRUCT':
            self.eat('KW_STRUCT')
            name = self.eat('ID').text
            if name in self.records:
                raise SyntaxError(f'record type `{name}` declared twice')
            self._cur_record = name
            try:
                fields = self._parse_record_fields('LBRACE', 'RBRACE')
            finally:
                self._cur_record = None
            self.accept('SEMI')
            if not fields:
                raise SyntaxError(f'struct `{name}` has no fields')
            self.records[name] = fields

    def _parse_struct_decl(self) -> Stmt:
        """struct [lo..hi] name { type field; ... };  -- a declaration, not a
        type.  This is the LL spelling of the book's

            var cut: array[1..n] of struct
                  v: array[1..n] of integer;
                  color in {red, green};
                endstruct

        so there is no type name to invent or repeat: the struct is anonymous
        and belongs to the variable.  A Java class still has to be called
        something, so one is derived from the variable name and never appears
        in LL or in the printed figure.

        It shares its machinery with `message`: both declare a record, and a
        message is simply a record that may be sent."""
        self.eat('KW_STRUCT')
        rng = None
        if self.accept('LBRACK'):
            lo = self.parse_expr()
            self.eat('DOTDOT')
            hi = self.parse_expr()
            self.eat('RBRACK')
            rng = (lo, hi)
        name = self.eat('ID').text
        tname = self._anon_record_name(name)
        self._cur_record = tname
        try:
            fields = self._parse_record_fields('LBRACE', 'RBRACE')
        finally:
            self._cur_record = None
        if not fields:
            raise SyntaxError(f'struct `{name}` has no fields')
        self.records[tname] = fields
        self.anon_records.add(tname)
        self.accept('SEMI')
        return Stmt(kind='decl', decl_name=name,
                    decl_type=Type(base=tname, is_array=rng is not None,
                                   range1=rng))

    def _anon_record_name(self, var: str) -> str:
        """Java class name for an anonymous struct, derived from the variable
        it belongs to (`cut` -> `Cut`), with `_` appended on collision."""
        t = var[:1].upper() + var[1:]
        while t in self.records:
            t += '_'
        return t

    def _parse_message_decls(self) -> dict:
        """message TAG(type name, ...);  -- zero or more, before the program.

        Declares the payload shape of a message tag, which is what types the
        binders of `upon receive(TAG, b1, ..., bk)` and lets `send` carry a
        tuple rather than a single value."""
        msgs: dict = {}
        while self.peek().kind == 'KW_MESSAGE':
            self.eat('KW_MESSAGE')
            tag = self.eat('ID').text
            if tag in msgs:
                raise SyntaxError(f'message tag `{tag}` declared twice')
            fields: list = []
            self.eat('LPAREN')
            if self.peek().kind != 'RPAREN':
                while True:
                    ftype = self.parse_type()
                    fname = self.eat('ID').text
                    fields.append((ftype, fname))
                    if not self.accept('COMMA'):
                        break
            self.eat('RPAREN')
            self.accept('SEMI')
            msgs[tag] = fields
            self.records[tag] = fields
        return msgs

    def parse_program(self) -> Program:
        # `enum` and `message` declarations may be interleaved, so keep
        # taking whichever comes next until neither does.
        enums: dict = {}
        msgs: dict = {}
        while self.peek().kind in ('KW_ENUM', 'KW_MESSAGE', 'KW_STRUCT'):
            enums.update(self._parse_enum_decls())
            self._parse_named_struct_decls()
            msgs.update(self._parse_message_decls())
        self.messages = msgs
        if self.peek().kind != 'KW_CLASS':
            p = self._parse_program_noclass()
            p.messages = msgs
            p.enums = enums
            p.records = self.records
            p.anon_records = self.anon_records
            p.record_inits = self.record_inits
            return p
        self.eat('KW_CLASS')
        name = self.eat('ID').text
        # `class NAME observes SUBJECT with OBSERVER;` -- composed program,
        # resolved later by merge_observed_program() once both names' Programs
        # are available. 'observes'/'with' are not reserved words elsewhere,
        # so this is recognized by text, not token kind, and only in the one
        # position ordinary programs would otherwise have `{`.
        if self.peek().kind == 'ID' and self.peek().text == 'observes':
            self.eat('ID')
            subject_tok = self.eat('ID')
            with_tok = self.eat('ID')
            if with_tok.text != 'with':
                raise SyntaxError(
                    f"line {with_tok.line}: expected 'with' after "
                    f"'observes {subject_tok.text}'")
            observer_tok = self.eat('ID')
            self.accept('SEMI')
            self.eat('EOF')
            return Program(name=name, entry_name='init',
                           observes_subject=subject_tok.text,
                           observes_observer=observer_tok.text)
        # `class NAME union A, B, ...;` -- a symmetric composed program,
        # resolved later by merge_union_programs() once every member's
        # Program is available. 'union' is not a reserved word elsewhere
        # (same trick as 'observes'/'with' above), so it is recognized by
        # text, and only in the one position an ordinary program would
        # otherwise have `{`.
        if self.peek().kind == 'ID' and self.peek().text == 'union':
            self.eat('ID')
            members = [self.eat('ID').text]
            while self.accept('COMMA'):
                members.append(self.eat('ID').text)
            if len(members) < 2:
                raise SyntaxError(
                    f"line {self.peek().line}: 'union' needs at least two "
                    f"members, e.g. `class {name} union A, B;`")
            self.accept('SEMI')
            self.eat('EOF')
            return Program(name=name, entry_name='init',
                           union_members=members)
        self.eat('LBRACE')
        # entry method header: `<return_type> EntryName(params) { ... }`
        if self.accept('KW_VOID'):
            ret_type: Optional[Type] = None
        else:
            ret_type = self.parse_type()
        method_name_tok = self.eat('ID')
        prog = Program(name=name, return_type=ret_type,
                       entry_name=method_name_tok.text, messages=msgs,
                       enums=enums, records=self.records,
                       anon_records=self.anon_records,
                       record_inits=self.record_inits)
        if self.accept('LPAREN'):
            if self.peek().kind != 'RPAREN':
                prog.params.append(self._parse_param())
                while self.accept('COMMA'):
                    prog.params.append(self._parse_param())
            self.eat('RPAREN')
        self.eat('LBRACE')
        # field decls at the top of the constructor body.
        while self._is_field_start():
            prog.decls.extend(self.parse_decl())
        # Preamble: optional setMode(...) and setPriority(...) calls.
        while self._is_config_call():
            self._parse_config(prog)
        # The program body is a statement list terminating at `}`.
        if self.peek().kind != 'RBRACE':
            prog.stmts = self.parse_stmt_list()
        self.accept('SEMI')
        self.eat('RBRACE')   # closes constructor body
        # Members after the entry: auxiliary methods, and rules.  A rule is a
        # member of the process, not a statement of its initialiser -- it is
        # not executed in sequence with the declarations, it fires whenever
        # its guard holds.  Writing it at this level says so, and matches the
        # printed algorithm, where the rules and the procedures sit at one
        # level under `P_i ::`.  Rules written inside the entry body are
        # still accepted and land in the same list.
        while self._is_aux_method_start() or self.peek().kind == 'KW_UPON':
            if self.peek().kind == 'KW_UPON':
                self.eat('KW_UPON')
                prog.stmts.append(self._parse_upon())
                self.accept('SEMI')
            else:
                prog.aux_methods.append(self._parse_aux_method())
        self.eat('RBRACE')   # closes class
        self.eat('EOF')
        # A setStateful() program's real behavior lives in its aux methods
        # (getValue, setValue, ...); the entry method may legitimately be
        # nothing but field initialization, with no statement of its own.
        if not prog.stmts and not (prog.stateful and prog.aux_methods):
            raise SyntaxError('program has no body statements')
        self._hoist_init_forall(prog)
        self._scan_result(prog)
        return prog

    def _parse_program_noclass(self) -> Program:
        """Parse a class-less program (one or more bare method defs)."""
        if self.default_name is None:
            t = self.peek()
            raise SyntaxError(
                f'line {t.line}: expected `class` declaration; '
                f'got {t.kind} {t.text!r}')
        # First method becomes the entry; the rest are aux methods.
        first = self._parse_aux_method()
        prog = Program(name=self.default_name,
                       return_type=first.return_type,
                       entry_name=first.name,
                       params=list(first.params),
                       stmts=list(first.body))
        # Mirror the class-form behaviour: leading top-of-body Decl
        # statements that look like LLP *state-vector* declarations
        # (no init, or a literal / `_i` / scalar fill) are promoted to
        # `prog.decls` so the codegen treats them as instance fields,
        # making them visible to forbidden/advance helpers. Decls whose
        # init is a real computation (call, quantifier, expression) stay
        # as local statements.
        def _is_field_like(init) -> bool:
            if init is None:
                return True
            if init.kind in ('int', 'real', 'bool', 'infinity'):
                return True
            if init.kind == 'id' and init.name == '_i':
                return True
            return False
        leading = []
        while (prog.stmts and prog.stmts[0].kind == 'decl'
                and _is_field_like(prog.stmts[0].decl_init)):
            s = prog.stmts.pop(0)
            leading.append(Decl(kind='var', type=s.decl_type,
                                name=s.decl_name, init=s.decl_init))
        prog.decls.extend(leading)
        # Peel any leading `setMode(...)` / `setPriority(...)` config-style
        # calls and apply them as program-level settings (mirrors the
        # class-form `_parse_config` path).
        while (prog.stmts and prog.stmts[0].kind == 'call'
               and prog.stmts[0].call_name in self._CONFIG_NAMES):
            s = prog.stmts.pop(0)
            if s.call_name == 'setMode':
                if len(s.call_args) != 1 or s.call_args[0].kind != 'id':
                    raise SyntaxError('setMode argument must be an identifier')
                arg = s.call_args[0].name
                if arg not in self._MODE_NAMES:
                    raise SyntaxError(
                        f'unknown setMode argument {arg!r}; '
                        f'expected {sorted(self._MODE_NAMES)}')
                prog.mode = self._MODE_NAMES[arg]
            elif s.call_name == 'setPriority':
                if (len(s.call_args) != 2
                        or s.call_args[1].kind != 'id'
                        or s.call_args[1].name not in ('min', 'max')):
                    raise SyntaxError('setPriority requires (expr, min|max)')
                prog.priority_expr = s.call_args[0]
                prog.priority_order = s.call_args[1].name
            elif s.call_name == 'setProcessName':
                if len(s.call_args) != 1 or s.call_args[0].kind != 'id':
                    raise SyntaxError('setProcessName takes one name')
                prog.process_name = s.call_args[0].name
            elif s.call_name == 'setProcess':
                if len(s.call_args) != 1 or s.call_args[0].kind != 'id':
                    raise SyntaxError('setProcess takes one index name')
                prog.process_index = s.call_args[0].name
            elif s.call_name == 'setChannel':
                if (len(s.call_args) != 1 or s.call_args[0].kind != 'id'
                        or s.call_args[0].name != 'fifo'):
                    raise SyntaxError(
                        "unknown setChannel argument; expected 'fifo' "
                        "(omit setChannel for the default)")
                prog.channel = s.call_args[0].name
            else:  # setBase
                if (len(s.call_args) != 1
                        or s.call_args[0].kind != 'int'
                        or s.call_args[0].value not in (0, 1)):
                    raise SyntaxError('setBase argument must be 0 or 1')
                prog.base = s.call_args[0].value
        self._hoist_init_forall(prog)
        while self._is_aux_method_start():
            prog.aux_methods.append(self._parse_aux_method())
        self.eat('EOF')
        if not prog.stmts:
            raise SyntaxError('program has no body statements')
        self._scan_result(prog)
        return prog

    def _hoist_init_forall(self, prog: 'Program') -> None:
        """If `prog.stmts` starts with a `forall k in [a..b] : G[k] = X[k]`
        whose target G is a hoisted array field and whose source X is an
        input array param, replace G's init with `X.clone()` and drop the
        forall.  This makes guard-bearing programs render with the copy
        in the constructor (matching the seqCode style) instead of in the
        entry method body."""
        if not prog.stmts or prog.stmts[0].kind != 'forall':
            return
        s = prog.stmts[0]
        if s.loop_set is not None:
            return
        if s.loop_lo is None or s.loop_hi is None:
            return
        if len(s.body) != 1:
            return
        a = s.body[0]
        if (a.kind != 'assign' or a.lhs_index is None
                or a.lhs_index2 is not None):
            return
        if a.lhs_index.kind != 'id' or a.lhs_index.name != s.loop_var:
            return
        if a.rhs.kind != 'index':
            return
        if a.rhs.idx.kind != 'id' or a.rhs.idx.name != s.loop_var:
            return
        target = next((d for d in prog.decls if d.name == a.lhs_name), None)
        if target is None or not target.type.is_array:
            return
        source = a.rhs.name
        if not any(p.name == source for p in prog.params):
            return
        target.init = Expr(kind='method_call',
                           lhs=Expr(kind='id', name=source),
                           name='clone', args=[])
        prog.stmts.pop(0)

    def _is_aux_method_start(self) -> bool:
        k = self.peek().kind
        if k in ('KW_VOID', 'KW_INT', 'KW_DOUBLE', 'KW_BOOLEAN'):
            return True
        if k == 'ID' and (self.peek().text in self.enums
                          or self.peek().text in self.records):
            return True
        if (k == 'ID' and self.peek().text in ('set', 'list')
                and self.peek(1).kind == 'LT'):
            return True
        # A composed-class return type -- `SRSW[] readArray()` -- matching
        # the same naming-convention test _is_field_start uses for a
        # composed-class field: capitalised and not a known enum/record.
        if k == 'ID' and self.peek().text[:1].isupper():
            return True
        return False

    def _parse_aux_method(self) -> 'Method':
        """Parse `<return_type> <name>(<params>) { <body> }`."""
        if self.accept('KW_VOID'):
            ret_type: Optional[Type] = None
        else:
            ret_type = self.parse_type()
        name = self.eat('ID').text
        params: List[Decl] = []
        self.eat('LPAREN')
        if self.peek().kind != 'RPAREN':
            params.append(self._parse_param())
            while self.accept('COMMA'):
                params.append(self._parse_param())
        self.eat('RPAREN')
        self.eat('LBRACE')
        body: List[Stmt] = []
        if self.peek().kind != 'RBRACE':
            body = self.parse_stmt_list()
        self.accept('SEMI')
        self.eat('RBRACE')
        return Method(name=name, return_type=ret_type,
                      params=params, body=body)

    def _scan_result(self, prog: 'Program') -> None:
        """Walk every `return E` (with operand) in the entry-method body.
        If every return is a simple identifier and they all agree, that
        identifier becomes the program's `result_name` (used by the LLP
        subroutine-ctor pattern to know which field to forward).  If the
        returns are expressions (literals, computations, calls), result_name
        stays None and no trailing return is stripped.
        """
        names: List[str] = []
        all_ids = True
        def walk(stmts: List['Stmt']) -> None:
            nonlocal all_ids
            for s in stmts:
                if s.kind == 'return' and s.return_value is not None:
                    if s.return_value.kind == 'id' and s.return_value.name == 'null':
                        pass  # an early-exit sentinel, not a result-name candidate
                    elif s.return_value.kind == 'id':
                        names.append(s.return_value.name)
                    else:
                        all_ids = False
                if s.kind == 'if':
                    walk(s.then_stmts); walk(s.else_stmts)
                elif s.kind == 'forall':
                    walk(s.body)
                elif s.kind == 'block':
                    walk(s.body)
                elif s.kind == 'par':
                    walk(s.par_branches)
                elif s.kind == 'guard' and s.guard_advance is not None:
                    walk([s.guard_advance])
        walk(prog.stmts)
        if not all_ids or not names:
            return
        if len(set(names)) > 1:
            raise SyntaxError(
                f'inconsistent `return` operands: {sorted(set(names))}')
        if names:
            prog.result_name = names[0]
        # Strip a trailing `return X` so it does not show up as a body stmt.
        # Preserve the actual final return so the LaTeX emitter can render it.
        while prog.stmts and prog.stmts[-1].kind == 'return' \
                and prog.stmts[-1].return_value is not None:
            popped = prog.stmts.pop()
            if prog.trailing_return is None:
                prog.trailing_return = popped

    def _parse_param(self) -> Decl:
        t = self.parse_type()
        name = self.eat('ID').text
        return Decl(kind='param', type=t, name=name)

    # The configuration calls recognised in the preamble.  Like `setMode`,
    # they are matched by text rather than reserved, so none of these names
    # is taken away from the programmer.
    _CONFIG_NAMES = ('setMode', 'setPriority', 'setBase', 'setProcessName',
                     'setProcess', 'setChannel', 'setStateful')

    def _is_config_call(self) -> bool:
        """True if the next tokens look like `setMode(...)` or
        `setPriority(...)` -- i.e., a configuration call in the preamble."""
        return (self.peek().kind == 'ID'
                and self.peek().text in self._CONFIG_NAMES
                and self.peek(1).kind == 'LPAREN')

    # Mapping from the Java class names accepted by setMode(...) to the
    # internal mode key used by codegen.
    _MODE_NAMES = {
        'LinkedList': 'linkedlist',
        'PriorityQueue': 'heap',
        'Deque': 'deque',
    }

    def _parse_config(self, prog: 'Program') -> None:
        """Consume a setMode(...) or setPriority(...) preamble call."""
        name = self.eat('ID').text
        self.eat('LPAREN')
        if name == 'setMode':
            arg_tok = self.eat('ID')
            if arg_tok.text not in self._MODE_NAMES:
                raise SyntaxError(
                    f'line {arg_tok.line}: unknown setMode argument '
                    f'{arg_tok.text!r}; expected '
                    f'{sorted(self._MODE_NAMES)} (omit setMode for the default)')
            prog.mode = self._MODE_NAMES[arg_tok.text]
            self.eat('RPAREN')
            self.accept('SEMI')
        elif name == 'setPriority':
            prog.priority_expr = self.parse_expr()
            self.eat('COMMA')
            o_tok = self.eat('ID')
            if o_tok.text not in ('min', 'max'):
                raise SyntaxError(
                    f'line {o_tok.line}: setPriority order must be '
                    f'`min` or `max`, got {o_tok.text!r}')
            prog.priority_order = o_tok.text
            self.eat('RPAREN')
            self.accept('SEMI')
        elif name == 'setProcessName':
            # Purely for the printed figure: what a peer is called.  The
            # monitors of the distributed WCP checker are M_1..M_n, not
            # P_1..P_n, and the figure says so.
            n_tok = self.eat('ID')
            prog.process_name = n_tok.text
            self.eat('RPAREN')
            self.accept('SEMI')
        elif name == 'setProcess':
            # The index the rules of this program run at, stated once here
            # rather than on every rule, matching the figures: the header
            # says `P_i ::` and each rule below it says only `upon PRED`.
            # A rule that quantifies over something else -- a fork, a
            # monitored process -- still writes its own `upon (f) : ...`.
            i_tok = self.eat('ID')
            prog.process_index = i_tok.text
            self.process_index = i_tok.text
            self.eat('RPAREN')
            self.accept('SEMI')
        elif name == 'setBase':
            b_tok = self.eat('INT')
            if b_tok.text not in ('0', '1'):
                raise SyntaxError(
                    f'line {b_tok.line}: setBase argument must be '
                    f'0 or 1, got {b_tok.text!r}')
            prog.base = int(b_tok.text)
            self.eat('RPAREN')
            self.accept('SEMI')
        elif name == 'setChannel':
            # A standing assumption on the algorithm's channels, beyond the
            # default of reliable delivery with arbitrary delay and no
            # ordering promised.  Purely descriptive, like setProcessName:
            # it changes how the figure is printed, not the generated Java.
            arg_tok = self.eat('ID')
            if arg_tok.text != 'fifo':
                raise SyntaxError(
                    f"line {arg_tok.line}: unknown setChannel argument "
                    f"{arg_tok.text!r}; expected 'fifo' "
                    f"(omit setChannel for the default)")
            prog.channel = arg_tok.text
            self.eat('RPAREN')
            self.accept('SEMI')
        elif name == 'setStateful':
            # No arguments: opts this program into field-per-declaration
            # codegen (see Program.stateful) for a class whose aux methods
            # are independent public operations on shared state, not
            # recursive helpers -- a register, a lock, a monitor.
            prog.stateful = True
            self.eat('RPAREN')
            self.accept('SEMI')
        else:
            raise AssertionError('unreachable')

    def _looks_like_type_bracket(self, offset: int = 1) -> bool:
        """The token at `offset` is expected to be '[': report whether
        this bracket opens a type's `[lo..hi]` range rather than an index
        expression's `[k]`, by scanning to the matching ']' for a '..'
        along the way. Used to tell `ClassName[0..n-1] field;` (a
        composed-class array field declaration) apart from `existingArray
        [k]...` (indexing an already-declared, capitalised field/local) --
        both start with an uppercase ID followed by '[', but only the
        first is a type."""
        depth = 0
        k = offset
        while True:
            tok = self.peek(k)
            if tok.kind == 'EOF':
                return False
            if tok.kind == 'LBRACK':
                depth += 1
            elif tok.kind == 'RBRACK':
                depth -= 1
                if depth == 0:
                    return False
            elif tok.kind == 'DOTDOT' and depth == 1:
                return True
            k += 1

    def _is_field_start(self) -> bool:
        """True if the next tokens look like the start of a field decl
        (`TYPE ID ...`).  Used at program-body top level to peel off fields
        before the first non-decl statement."""
        k = self.peek().kind
        if k in ('KW_INT', 'KW_DOUBLE', 'KW_BOOLEAN', 'KW_STRUCT'):
            return True
        if k == 'ID' and (self.peek().text in self.enums
                          or self.peek().text in self.records):
            return True
        if (k == 'ID' and self.peek().text in ('set', 'list')
                and self.peek(1).kind == 'LT'):
            return True
        # A field whose type names another program -- `DiningPhilosophers
        # dining;` -- composing that program's process into this one.  The
        # Parser has no cross-file knowledge (that's scan_signatures, run by
        # the driver), so any capitalised, not-yet-known identifier is
        # accepted here on the strength of this codebase's naming convention
        # (types are capitalised, fields and procedures are not); CodeGen
        # confirms the name is a real sibling program once signatures are
        # available. A bracket here must look like a type's range, not an
        # index expression -- at the top of the entry body nothing is in
        # scope to index yet, but this keeps the check honest regardless.
        if k == 'ID' and self.peek().text[:1].isupper():
            if self.peek(1).kind == 'ID':
                return True
            if self.peek(1).kind == 'LBRACK' and self._looks_like_type_bracket():
                return True
        return False

    def parse_decl(self) -> List[Decl]:
        """One declaration statement, which may declare several names:
        `int left = 0, right = 0;`.  Returns one Decl per name; names from
        the same statement share a group id."""
        if self.peek().kind == 'KW_STRUCT':
            st = self._parse_struct_decl()
            return [Decl(kind='var', type=st.decl_type, name=st.decl_name)]
        t = self.parse_type()
        out: List[Decl] = []
        while True:
            name = self.eat('ID').text
            init = None
            if self.accept('ASSIGN'):
                init = self.parse_expr()
            out.append(Decl(kind='var', type=t, name=name, init=init))
            if not self.accept('COMMA'):
                break
        self.eat('SEMI')
        if len(out) > 1:
            self._decl_group += 1
            for d in out:
                d.group = self._decl_group
        return out

    def parse_type(self) -> Type:
        # "list" "<" base_type ">" -- an ordered, growable sequence.  The
        # element may be a scalar, an enum, or a named record.
        if (self.peek().kind == 'ID' and self.peek().text == 'list'
                and self.peek(1).kind == 'LT'):
            self.eat('ID')
            self.eat('LT')
            if self.peek().kind == 'ID':
                base_tok = self.eat('ID')
                if (base_tok.text not in self.records
                        and base_tok.text not in self.enums):
                    raise SyntaxError(
                        f'line {base_tok.line}: `list<{base_tok.text}>` needs '
                        f'a `struct {base_tok.text} {{...}}` or '
                        f'`enum {base_tok.text} {{...}}` declaration')
            else:
                base_tok = self.eat('KW_INT', 'KW_DOUBLE', 'KW_BOOLEAN')
            self.eat('GT')
            # A list type may itself be arrayed: `list<M>[1..n][1..n]` is the
            # figure's "arrays[1..n,1..n] of lists of messages" -- one list
            # per ordered pair of processes.
            dims = 0
            lranges = [None, None]
            while self.accept('LBRACK'):
                if self.peek().kind != 'RBRACK':
                    lo = self.parse_expr()
                    self.eat('DOTDOT')
                    hi = self.parse_expr()
                    lranges[dims] = (lo, hi)
                self.eat('RBRACK')
                dims += 1
            return Type(base=base_tok.text, is_list=True,
                        is_array=dims >= 1, is_2d=dims >= 2,
                        range1=lranges[0], range2=lranges[1])
        # "set" "<" base_type ">" ("[" "]")? | base_type ("[" "]")?
        if self.peek().kind == 'ID' and self.peek().text == 'set':
            self.eat('ID')
            self.eat('LT')
            base_tok = self.eat('KW_INT', 'KW_DOUBLE', 'KW_BOOLEAN')
            self.eat('GT')
            is_arr = bool(self.accept('LBRACK'))
            r1 = None
            if is_arr:
                if self.peek().kind != 'RBRACK':
                    lo = self.parse_expr()
                    self.eat('DOTDOT')
                    hi = self.parse_expr()
                    r1 = (lo, hi)
                self.eat('RBRACK')
            return Type(base=base_tok.text, is_array=is_arr, is_set=True,
                        range1=r1)
        if (self.peek().kind == 'ID'
                and (self.peek().text in self.enums
                     or self.peek().text in self.records)):
            base_tok = self.eat('ID')
        elif self.peek().kind == 'ID' and self.peek().text[:1].isupper():
            # Not a known enum or record: a reference to another program,
            # composed in as a field (see _is_field_start).  CodeGen
            # resolves and validates the name against scan_signatures.
            base_tok = self.eat('ID')
        else:
            base_tok = self.eat('KW_INT', 'KW_DOUBLE', 'KW_BOOLEAN')
        dims = 0
        ranges = [None, None]
        while self.accept('LBRACK'):
            if self.peek().kind != 'RBRACK':
                lo = self.parse_expr()
                self.eat('DOTDOT')
                hi = self.parse_expr()
                ranges[dims] = (lo, hi)
            self.eat('RBRACK')
            dims += 1
        return Type(base=base_tok.text, is_array=dims >= 1, is_2d=dims >= 2,
                    dims=dims, range1=ranges[0], range2=ranges[1])

    def _parse_guard(self) -> Stmt:
        """forbidden (j) : PRED => advance : STMT
        or forbidden (i, j) : PRED => advance : STMT  (multi-index)
        STMT is a single stmt; multiple must be wrapped in { ... }."""
        self.eat('LPAREN')
        indices = [self.eat('ID').text]
        while self.accept('COMMA'):
            indices.append(self.eat('ID').text)
        self.eat('RPAREN')
        self.eat('COLON')
        pred = self.parse_expr()
        self.eat('ARROW')
        self.eat('KW_ADVANCE')
        self.eat('COLON')
        adv = self.parse_stmt()
        pick_var = None
        if (pred.kind == 'quant' and pred.quant == 'exists'
                and pred.pick_var is not None):
            pick_var = pred.pick_var
        return Stmt(kind='guard', guard_index=indices, guard_predicate=pred,
                    guard_advance=adv, guard_pick_var=pick_var)

    def _parse_ensure(self) -> Stmt:
        """ensure (j) : LHS op RHS -- desugars to a guard stmt.
        or ensure (i, j) : LHS op RHS  (multi-index)"""
        self.eat('LPAREN')
        indices = [self.eat('ID').text]
        while self.accept('COMMA'):
            indices.append(self.eat('ID').text)
        self.eat('RPAREN')
        self.eat('COLON')
        name = self.eat('ID').text
        lhs_idx: Optional[Expr] = None
        if self.accept('LBRACK'):
            lhs_idx = self.parse_expr()
            self.eat('RBRACK')
        op_tok = self.eat('LEQ', 'GEQ', 'LT', 'GT', 'EQEQ')
        op = '=' if op_tok.text == '==' else op_tok.text
        rhs = self.parse_expr()
        if lhs_idx is not None:
            lhs_expr = Expr(kind='index', name=name, idx=lhs_idx)
        else:
            lhs_expr = Expr(kind='id', name=name)
        pred = Expr(kind='un', op='not',
                    lhs=Expr(kind='bin', op=op,
                             lhs=lhs_expr, rhs=rhs))
        adv = Stmt(kind='assign', lhs_name=name, lhs_index=lhs_idx, rhs=rhs)
        return Stmt(kind='guard', guard_index=indices, guard_predicate=pred,
                    guard_advance=adv, guard_pick_var=None,
                    ensure_lhs_name=name, ensure_lhs_index=lhs_idx,
                    ensure_op=op, ensure_rhs=rhs)

    def _parse_send(self) -> Stmt:
        """send(DEST, PAYLOAD...)  or  send(DEST, TAG, PAYLOAD...)

        The destination is an expression naming the receiving process.  In
        the tagged form the second argument is a symbolic message tag, a
        plain identifier (conventionally upper case) that names the message;
        when the tag has a `message` declaration the remaining arguments are
        its payload fields, so a tuple can be sent."""
        self.eat('LPAREN')
        dest = self.parse_expr()
        self.eat('COMMA')
        tag: Optional[str] = None
        if self.peek().kind == 'ID' and self.peek(1).kind == 'COMMA':
            tag = self.eat('ID').text
            self.eat('COMMA')
        elif self.peek().kind == 'ID' and self.peek(1).kind == 'RPAREN' \
                and self.peek().text in self.messages:
            # `send(dest, TAG)` -- a tag with an empty payload.
            tag = self.eat('ID').text
            self.eat('RPAREN')
            return Stmt(kind='send', send_dest=dest, send_tag=tag,
                        send_payload=[])
        elif self.peek(1).kind == 'COMMA':
            t = self.peek()
            raise SyntaxError(
                f'line {t.line}: the message tag of a tagged `send` must be '
                f'a plain identifier, e.g. `send(k, REQUEST, v)`; '
                f'got `{t.text}`')
        payload = [self.parse_expr()]
        while self.accept('COMMA'):
            payload.append(self.parse_expr())
        self.eat('RPAREN')
        if tag is None and len(payload) > 1:
            raise SyntaxError(
                'an untagged `send` carries a single payload; use '
                '`send(dest, TAG, ...)` with a `message` declaration to '
                'send several values')
        if tag is not None and tag in self.messages:
            want = len(self.messages[tag])
            if want != len(payload):
                raise SyntaxError(
                    f'message `{tag}` is declared with {want} payload '
                    f'field(s) but {len(payload)} were sent')
        return Stmt(kind='send', send_dest=dest, send_tag=tag,
                    send_payload=payload)

    def _parse_upon_action(self) -> Stmt:
        """The action of an `upon`: always a braced statement list.

        Braces are required even for a single statement.  A rule's action is
        a unit that fires as a whole, and the guard above it may span lines,
        so the braces are what say where the action begins and ends.  How it
        is printed is a separate matter -- the printer still renders a
        one-statement action on the header line."""
        if self.peek().kind == 'LBRACE':
            self.eat('LBRACE')
            body = self.parse_stmt_list()
            self.accept('SEMI')
            self.eat('RBRACE')
            return Stmt(kind='block', body=body)
        t = self.peek()
        raise SyntaxError(
            f'line {t.line}: the action of an `upon` must be wrapped in '
            f'braces, e.g. `upon P {{ x = 1; }}`')

    def _at_upon_index_list(self) -> bool:
        """True when the tokens after `upon` form an index list `( ID (, ID)* ) :`
        rather than a parenthesized predicate such as `upon (G[j] < 5) {...}`."""
        if self.peek().kind != 'LPAREN':
            return False
        k = 1
        while True:
            if self.peek(k).kind != 'ID':
                return False
            k += 1
            if self.peek(k).kind == 'COMMA':
                k += 1
                continue
            break
        return (self.peek(k).kind == 'RPAREN'
                and self.peek(k + 1).kind == 'COLON')

    def _parse_upon(self) -> Stmt:
        """upon [ (j) : ] PRED { STMT_LIST }   -- braces optional for one stmt.

        Sugar for `forbidden (j) : PRED => advance : { STMT_LIST }`: it builds
        the identical guard node, marked `is_upon` so the pretty printer can
        render the compact spelling.  The index list is optional; it defaults
        to the program's `setProcess(i)` index, and to `j` when the program
        declares none, which gives the process-local reading used by
        distributed programs."""
        indices = [self.process_index or 'j']
        if self._at_upon_index_list():
            self.eat('LPAREN')
            indices = [self.eat('ID').text]
            while self.accept('COMMA'):
                indices.append(self.eat('ID').text)
            self.eat('RPAREN')
            self.eat('COLON')
        # `upon receive(TAG, b1, ..., bk) from k { S }` -- a message rule.
        # The guard is implicit: "some sender has such a message queued for
        # me".  Binders are outputs: k is bound to the sender and the b's to
        # the payload fields declared for TAG.
        if self.peek().kind == 'KW_RECEIVE':
            self.eat('KW_RECEIVE')
            self.eat('LPAREN')
            rtag: Optional[str] = None
            binders: List[str] = []
            if self.peek().kind != 'RPAREN':
                rtag = self.eat('ID').text
                while self.accept('COMMA'):
                    binders.append(self.eat('ID').text)
            self.eat('RPAREN')
            if rtag is not None and binders:
                if rtag not in self.messages:
                    raise SyntaxError(
                        f'`upon receive({rtag}, ...)` binds payload fields, '
                        f'so `{rtag}` needs a `message {rtag}(...)` '
                        f'declaration to type them')
                want = len(self.messages[rtag])
                if want != len(binders):
                    raise SyntaxError(
                        f'message `{rtag}` has {want} payload field(s) but '
                        f'{len(binders)} binder(s) were given')
            # `from k` binds the sender; `from 0` names one particular
            # process, for a message that can only come from a distinguished
            # one -- the coordinator, the root, the environment.
            rfrom: Optional[str] = None
            rfromc: Optional[int] = None
            if self.peek().kind == 'ID' and self.peek().text == 'from':
                self.eat('ID')
                if self.peek().kind == 'INT':
                    rfromc = int(self.eat('INT').text)
                else:
                    rfrom = self.eat('ID').text
            # `when PRED` -- an extra condition on the rule, tested before the
            # message is taken.  It sees the guard index and program state,
            # not the binders, which do not exist until the message is
            # dequeued.  This is the figure's "wait until state = passive".
            rwhen: Optional[Expr] = None
            if self.peek().kind == 'ID' and self.peek().text == 'when':
                self.eat('ID')
                rwhen = self.parse_expr()
            adv = self._parse_upon_action()
            return Stmt(kind='guard', guard_index=indices,
                        guard_predicate=rwhen, guard_advance=adv,
                        is_upon=True, recv_tag=rtag, recv_binders=binders,
                        recv_from=rfrom, recv_from_const=rfromc)
        pred_line = self.peek().line
        try:
            pred = self.parse_expr()
        except SyntaxError as e:
            # The lexer discards newlines, so a brace-less action beginning
            # with `[` is swallowed as a second array index of the
            # predicate's trailing identifier and blows up inside the
            # expression parser.  Say so rather than reporting a stray
            # bracket.
            if 'LBRACK' in str(e):
                raise SyntaxError(
                    f'{e}\n  (line {pred_line}: an `upon` action beginning '
                    f'with `[` must be wrapped in braces, e.g. '
                    f'`upon P {{ [ A [] B ]; }}`)') from None
            raise
        adv = self._parse_upon_action()
        pick_var = None
        if (pred.kind == 'quant' and pred.quant == 'exists'
                and pred.pick_var is not None):
            pick_var = pred.pick_var
        return Stmt(kind='guard', guard_index=indices, guard_predicate=pred,
                    guard_advance=adv, guard_pick_var=pick_var, is_upon=True,
                    guard_line=pred_line)

    # Tokens that MUST terminate a stmt_list (cannot start a new stmt).
    _STMT_TERMINATORS = ('RBRACE', 'RBRACK', 'OROR', 'KW_ELSE', 'EOF')

    # Statement kinds whose textual representation ends in `}` -- a trailing
    # `;` is optional (Java treats it as an empty statement).  All other
    # stmts (assignments, calls, return, decls, guards, etc.) require a
    # terminating `;`, matching Java's expression-statement rule.
    _BLOCK_LIKE_KINDS = {'if', 'while', 'forall', 'par', 'block'}

    def parse_stmt_list(self) -> List[Stmt]:
        stmts: List[Stmt] = []
        while self.peek().kind not in self._STMT_TERMINATORS:
            s = self.parse_stmt()
            stmts.append(s)
            # A braced `upon` also ends in `}`, so its `;` is optional too;
            # a brace-less one is an ordinary statement and requires it.
            braced_upon = (s.is_upon and s.guard_advance is not None
                           and s.guard_advance.kind == 'block')
            if s.kind in self._BLOCK_LIKE_KINDS or braced_upon:
                self.accept('SEMI')      # optional terminator
            else:
                self.eat('SEMI')         # required terminator (Java-style)
        return stmts

    def parse_stmt(self) -> Stmt:
        if self.accept('KW_IF'):
            self.eat('LPAREN')
            cond = self.parse_expr()
            self.eat('RPAREN')
            self.eat('LBRACE')
            then_s = self.parse_stmt_list()
            self.accept('SEMI')
            self.eat('RBRACE')
            else_s: List[Stmt] = []
            if self.accept('KW_ELSE'):
                self.eat('LBRACE')
                else_s = self.parse_stmt_list()
                self.accept('SEMI')
                self.eat('RBRACE')
            return Stmt(kind='if', cond=cond, then_stmts=then_s, else_stmts=else_s)
        if self.accept('KW_FORALL'):
            var = self.eat('ID').text
            self.eat('KW_IN')
            lo, hi, setexpr = self.parse_range_or_set()
            self.eat('COLON')
            body = [self.parse_stmt()]
            return Stmt(kind='forall', loop_var=var, loop_lo=lo, loop_hi=hi,
                        loop_set=setexpr, body=body)
        # `while (cond) { stmts }` -- Java-style while loop.
        if self.accept('KW_WHILE'):
            self.eat('LPAREN')
            cond = self.parse_expr()
            self.eat('RPAREN')
            self.eat('LBRACE')
            body = self.parse_stmt_list()
            self.accept('SEMI')
            self.eat('RBRACE')
            return Stmt(kind='while', cond=cond, body=body)
        # `return X` -- early exit and result marker.  An operand is
        # mandatory; `return;` (without operand) is rejected at parse time.
        if self.accept('KW_RETURN'):
            if self.peek().kind in self._STMT_TERMINATORS or self.peek().kind == 'SEMI':
                t = self.peek()
                raise SyntaxError(
                    f'line {t.line}: `return` requires an operand '
                    f'(e.g. `return G;`)')
            val = self.parse_expr()
            return Stmt(kind='return', return_value=val)
        # `exit(EXPR)` -- the algorithm is finished, with that result.  A rule
        # has no caller to return to, so a rule that reaches a verdict exits
        # the program rather than returning from a method, as exit(2) does in
        # Unix.  Every rule stops firing afterwards.
        if self.accept('KW_EXIT'):
            self.eat('LPAREN')
            val = self.parse_expr()
            self.eat('RPAREN')
            return Stmt(kind='exit', return_value=val)
        # `always : TYPE var = expr` followed by `forbidden`/`ensure`.
        # Collect zero or more always clauses, then parse the guard.
        if self.accept('KW_ALWAYS'):
            always_clauses: list = []
            while True:
                self.eat('COLON')
                atype = self.parse_type()
                aname = self.eat('ID').text
                self.eat('ASSIGN')
                aexpr = self.parse_expr()
                always_clauses.append((atype, aname, aexpr))
                self.accept('SEMI')
                if not self.accept('KW_ALWAYS'):
                    break
            if self.accept('KW_FORBIDDEN'):
                g = self._parse_guard()
            elif self.accept('KW_ENSURE'):
                g = self._parse_ensure()
            elif self.accept('KW_UPON'):
                g = self._parse_upon()
            else:
                t = self.peek()
                raise SyntaxError(
                    f'line {t.line}: `always` must be followed by '
                    f'`forbidden`, `ensure`, or `upon`')
            g.guard_always = always_clauses
            return g
        # `forbidden (j) : PRED => advance : STMT` -- guarded command.
        if self.accept('KW_FORBIDDEN'):
            return self._parse_guard()
        # `ensure (j) : LHS op RHS` -- sugar that desugars to a guard.
        if self.accept('KW_ENSURE'):
            return self._parse_ensure()
        # `upon [ (j) : ] PRED { STMT }` -- process-local spelling of the
        # same guarded command; the index defaults to `j`.
        if self.accept('KW_UPON'):
            return self._parse_upon()
        # `send(DEST, PAYLOAD)` or `send(DEST, TAG, PAYLOAD)`.
        if self.accept('KW_SEND'):
            return self._parse_send()
        # `{ stmt_list }` block.
        if self.accept('LBRACE'):
            inner = self.parse_stmt_list()
            self.accept('SEMI')
            self.eat('RBRACE')
            return Stmt(kind='block', body=inner)
        # `[ B1 OP B2 (OP B3)* ]` -- bracket composition.  The operator
        # OP is one of:
        #   '[]'    parallel composition (Dijkstra-style; LBRACK RBRACK)
        #   '&&'    predicate conjunction (ANDAND)
        #   'or'    nondeterministic choice (contextual ID 'or')
        #   ';'     sequential composition (SEMI followed by another branch)
        #   '|'     pipelining (PIPE)
        #   '<>'    feedback composition (LT GT)
        #   'sync'  synchronous composition (contextual ID 'sync')
        # The first separator seen fixes OP for the whole bracket; mixing
        # operators inside one bracket is a syntax error.
        if self.accept('LBRACK'):
            branches: List[Stmt] = [self.parse_stmt()]
            par_op: Optional[str] = None
            while True:
                # '[]' -- two adjacent tokens
                if (self.peek().kind == 'LBRACK'
                        and self.peek(1).kind == 'RBRACK'):
                    sep = '[]'
                    self.eat('LBRACK'); self.eat('RBRACK')
                # '<>' -- two adjacent tokens (feedback)
                elif (self.peek().kind == 'LT'
                        and self.peek(1).kind == 'GT'):
                    sep = '<>'
                    self.eat('LT'); self.eat('GT')
                # '&&'
                elif self.peek().kind == 'ANDAND':
                    sep = '&&'
                    self.eat('ANDAND')
                # 'or' -- contextual ID, only inside this bracket form
                elif (self.peek().kind == 'ID'
                        and self.peek().text == 'or'):
                    sep = 'or'
                    self.eat('ID')
                # 'sync' -- contextual ID, only inside this bracket form
                elif (self.peek().kind == 'ID'
                        and self.peek().text == 'sync'):
                    sep = 'sync'
                    self.eat('ID')
                # ';' -- only counts as a composition separator when followed
                # by another branch (i.e., not the trailing ';' before ']')
                elif (self.peek().kind == 'SEMI'
                        and self.peek(1).kind != 'RBRACK'
                        and self.peek(1).kind != 'EOF'):
                    sep = ';'
                    self.eat('SEMI')
                # '|' -- pipelining; OROR (||) is the boolean OR, not pipe
                elif self.peek().kind == 'PIPE':
                    sep = '|'
                    self.eat('PIPE')
                else:
                    break
                if par_op is None:
                    par_op = sep
                elif par_op != sep:
                    raise SyntaxError(
                        f'line {self.peek().line}: mixed composition '
                        f'operators inside [ ... ]: {par_op!r} and {sep!r}')
                branches.append(self.parse_stmt())
            if par_op is None:
                par_op = '[]'  # single-branch default
            self.accept('SEMI')
            self.eat('RBRACK')
            return Stmt(kind='par', par_branches=branches, par_op=par_op)
        # Anonymous struct declaration: `struct [lo..hi] name { fields };`
        if self.peek().kind == 'KW_STRUCT':
            return self._parse_struct_decl()
        # Local variable declaration: <type> ID := expr.  A '[' here must
        # look like a type's `[lo..hi]` range, not an index expression --
        # `Comm[i][j].setValue(...)` as a statement also starts with a
        # capitalised ID followed by '[', but is a use of an
        # already-declared field/local, not a new declaration.
        if (self.peek().kind in ('KW_INT', 'KW_DOUBLE', 'KW_BOOLEAN')
                or (self.peek().kind == 'ID'
                    and self.peek().text in ('set', 'list')
                    and self.peek(1).kind == 'LT')
                or (self.peek().kind == 'ID'
                    and (self.peek().text in self.enums
                         or self.peek().text in self.records)
                    and (self.peek(1).kind == 'ID'
                         or (self.peek(1).kind == 'LBRACK'
                             and self._looks_like_type_bracket())))
                or (self.peek().kind == 'ID'
                    and self.peek().text[:1].isupper()
                    and self.peek().text not in self.enums
                    and self.peek().text not in self.records
                    and (self.peek(1).kind == 'ID'
                         or (self.peek(1).kind == 'LBRACK'
                             and self._looks_like_type_bracket())))):
            t = self.parse_type()
            name = self.eat('ID').text
            init = None
            if self.accept('ASSIGN'):
                init = self.parse_expr()
            return Stmt(kind='decl', decl_type=t, decl_name=name,
                        decl_init=init)
        # swap(lhs1, lhs2) as a statement
        if (self.peek().kind == 'ID' and self.peek().text == 'swap'
                and self.peek(1).kind == 'LPAREN'):
            self.eat('ID')
            self.eat('LPAREN')
            a = self.parse_expr()
            self.eat('COMMA')
            b = self.parse_expr()
            self.eat('RPAREN')
            return Stmt(kind='swap', swap_a=a, swap_b=b)
        # `q.add(x);` -- a method call in statement position.  Told apart
        # from an assignment by the `(` that closes the argument list being
        # followed by `;` rather than `=`.
        # A method call may sit on an indexed or field-selected target, not
        # only on a bare name: `S[i][j].add(m)`, `cut[i].incsend.clear()`.
        # Parse speculatively and keep it only if it really is a call.
        if self.peek().kind == 'ID' and self.peek(1).kind in ('DOT', 'LBRACK'):
            save = self.pos
            e = self.parse_expr()
            if e.kind == 'method_call' and self.peek().kind == 'SEMI':
                return Stmt(kind='exprstmt', rhs=e)
            self.pos = save
        # Method/program call at statement position: `name(args)`.  The
        # `(` lookahead distinguishes calls from assignments (which expect
        # `=` or `[ ... ] =` after the name).
        if (self.peek().kind == 'ID' and self.peek(1).kind == 'LPAREN'):
            name = self.eat('ID').text
            self.eat('LPAREN')
            args: List[Expr] = []
            if self.peek().kind != 'RPAREN':
                args.append(self.parse_expr())
                while self.accept('COMMA'):
                    args.append(self.parse_expr())
            self.eat('RPAREN')
            return Stmt(kind='call', call_name=name, call_args=args)
        # assignment: ID = expr | ID [expr] = expr | ID [expr] [expr] = expr
        name = self.eat('ID').text
        idx = None
        idx2 = None
        idx_extra: List[Expr] = []
        if self.accept('LBRACK'):
            idx = self.parse_expr()
            self.eat('RBRACK')
            if self.accept('LBRACK'):
                idx2 = self.parse_expr()
                self.eat('RBRACK')
                # 3rd+ chained index on the LHS, e.g. RT[a][b][c][d] = ...
                while self.accept('LBRACK'):
                    idx_extra.append(self.parse_expr())
                    self.eat('RBRACK')
        fld = None
        fidx = None
        if self.peek().kind == 'DOT':
            self.eat('DOT')
            fld = self.eat('ID').text
            if self.accept('LBRACK'):
                fidx = self.parse_expr()
                self.eat('RBRACK')
        self.eat('ASSIGN')
        rhs = self.parse_expr()
        return Stmt(kind='assign', lhs_name=name, lhs_index=idx,
                    lhs_index2=idx2, lhs_index_extra=idx_extra,
                    lhs_field=fld, lhs_field_index=fidx,
                    rhs=rhs)

    def parse_range(self) -> Tuple[Expr, Expr]:
        self.eat('LBRACK')
        lo = self.parse_expr()
        self.eat('DOTDOT')
        hi = self.parse_expr()
        self.eat('RBRACK')
        return lo, hi

    def parse_range_or_set(self) -> Tuple[Optional[Expr], Optional[Expr],
                                           Optional[Expr]]:
        """Accept either "[lo..hi]" or an indexed array expression like "pre[j]".
        Returns (lo, hi, None) for a range or (None, None, set_expr) otherwise.
        """
        if self.peek().kind == 'LBRACK':
            lo, hi = self.parse_range()
            return lo, hi, None
        # Otherwise: a set- or list-valued expression -- either a bare name
        # (`forall x in q`) or an indexed element (`forall k in pre[j]`).
        name = self.eat('ID').text
        e: Expr
        if self.peek().kind == 'LBRACK':
            self.eat('LBRACK')
            idx = self.parse_expr()
            self.eat('RBRACK')
            e = Expr(kind='index', name=name, idx=idx)
            if self.peek().kind == 'LBRACK':
                self.eat('LBRACK')
                idx2 = self.parse_expr()
                self.eat('RBRACK')
                e = Expr(kind='index2', name=name, idx=idx, rhs=idx2)
        else:
            e = Expr(kind='id', name=name)
        # `forall m in cut[i].incsend` -- the set may be a field of a record,
        # which is how the figure keeps a list inside each cut entry.
        while self.peek().kind == 'DOT':
            self.eat('DOT')
            fname = self.eat('ID').text
            e = Expr(kind='field_access', lhs=e, name=fname)
        return None, None, e

    # --- expressions ---
    def parse_expr(self) -> Expr:
        return self.parse_or()

    def parse_or(self) -> Expr:
        e = self.parse_and()
        while self.accept('OROR'):
            r = self.parse_and()
            e = Expr(kind='bin', op='or', lhs=e, rhs=r)
        return e

    def parse_and(self) -> Expr:
        e = self.parse_not()
        while self.accept('ANDAND'):
            r = self.parse_not()
            e = Expr(kind='bin', op='and', lhs=e, rhs=r)
        return e

    def parse_not(self) -> Expr:
        if self.accept('BANG'):
            e = self.parse_not()
            return Expr(kind='un', op='not', lhs=e)
        return self.parse_cmp()

    def parse_cmp(self) -> Expr:
        e = self.parse_add()
        if self.peek().kind in ('LT', 'LEQ', 'GT', 'GEQ', 'EQEQ', 'NEQ'):
            t = self.eat(self.peek().kind)
            # AST normalises `==` token text to op '=' for the existing
            # codegen / pretty-printer machinery.
            op = '=' if t.text == '==' else t.text
            r = self.parse_add()
            return Expr(kind='bin', op=op, lhs=e, rhs=r)
        # `X in Y`: set/array membership as a plain boolean expression, not
        # just the `Q in RANGE` clause of a quantifier (a separate, earlier
        # grammar position -- see the KW_IN handling in parse_quant/parse_
        # guard, which never reaches here since it doesn't go through
        # parse_expr for the variable being bound).
        if self.peek().kind == 'KW_IN':
            self.eat('KW_IN')
            r = self.parse_add()
            return Expr(kind='bin', op='in', lhs=e, rhs=r)
        return e

    def parse_add(self) -> Expr:
        e = self.parse_mul()
        while self.peek().kind in ('PLUS', 'MINUS'):
            t = self.eat(self.peek().kind)
            r = self.parse_mul()
            e = Expr(kind='bin', op=t.text, lhs=e, rhs=r)
        return e

    def parse_mul(self) -> Expr:
        e = self.parse_unary()
        while self.peek().kind in ('STAR', 'SLASH', 'PERCENT'):
            t = self.eat(self.peek().kind)
            r = self.parse_unary()
            e = Expr(kind='bin', op=t.text, lhs=e, rhs=r)
        return e

    def parse_unary(self) -> Expr:
        if self.accept('MINUS'):
            e = self.parse_unary()
            return Expr(kind='un', op='neg', lhs=e)
        return self.parse_primary()

    def parse_primary(self) -> Expr:
        t = self.peek()
        if t.kind == 'INT':
            self.eat('INT')
            return Expr(kind='int', value=int(t.text))
        if t.kind == 'REAL':
            self.eat('REAL')
            return Expr(kind='real', value=float(t.text))
        if t.kind == 'KW_TRUE':
            self.eat('KW_TRUE')
            return Expr(kind='bool', value=True)
        if t.kind == 'KW_FALSE':
            self.eat('KW_FALSE')
            return Expr(kind='bool', value=False)
        if t.kind == 'KW_INFINITY':
            self.eat('KW_INFINITY')
            return Expr(kind='infinity')
        if t.kind == 'KW_NEW':
            # `new <type>[<expr>]` or `new <type>[<expr>][<expr>]` --
            # Java-style 1D / 2D array allocation.
            self.eat('KW_NEW')
            # `new Name(f1, ..., fk)` -- a record value, so one can be put
            # into a list.  Fields are given positionally, as declared.
            if self.peek().kind == 'ID' and self.peek().text in self.records:
                rname = self.eat('ID').text
                self.eat('LPAREN')
                args: List[Expr] = []
                if self.peek().kind != 'RPAREN':
                    args.append(self.parse_expr())
                    while self.accept('COMMA'):
                        args.append(self.parse_expr())
                self.eat('RPAREN')
                want = len(self.records[rname])
                if len(args) != want:
                    raise SyntaxError(
                        f'record `{rname}` has {want} field(s) but '
                        f'{len(args)} were given')
                return Expr(kind='new_record', name=rname, args=args)
            base_tok = self.eat('KW_INT', 'KW_DOUBLE', 'KW_BOOLEAN')
            self.eat('LBRACK')
            size = self.parse_expr()
            self.eat('RBRACK')
            if self.accept('LBRACK'):
                # `new int[n][]` -- a jagged/ragged allocation, common when
                # each row is a different length assigned individually
                # afterward (e.g. LLPManOptimalStronglyStableMarriage's
                # `int[][] adj = new int[n][]; ... adj[j] = tmp;`). Only
                # the outer dimension is sized; leave idx2 unset, same as
                # the plain 1D `new int[n]` case -- an emitter that doesn't
                # already special-case a *sized* second dimension (idx2
                # present) has nothing extra to do for either.
                if self.peek().kind == 'RBRACK':
                    self.eat('RBRACK')
                    # `jagged=True` distinguishes this from a plain 1D
                    # `new int[n]` (no second bracket at all): the JS
                    # emitter treats both identically (untyped, so it
                    # doesn't matter), but Java needs to know, since `int[][]
                    # x = new int[n];` is a type error there -- it must emit
                    # the trailing `[]` (`new int[n][]`) to get an int[][].
                    return Expr(kind='new_array', name=base_tok.text, idx=size,
                                jagged=True)
                size2 = self.parse_expr()
                self.eat('RBRACK')
                # 3rd+ size dimension, e.g. `new int[n][m][n][m]`.
                extra_sizes: List[Expr] = []
                while self.accept('LBRACK'):
                    extra_sizes.append(self.parse_expr())
                    self.eat('RBRACK')
                return Expr(kind='new_array', name=base_tok.text,
                            idx=size, idx2=size2, idx_extra=extra_sizes)
            return Expr(kind='new_array', name=base_tok.text, idx=size)
        if t.kind == 'LPAREN':
            self.eat('LPAREN')
            e = self.parse_expr()
            self.eat('RPAREN')
            return e
        # quantifier: `min i in [a..b] : e`, `sum`, `max`, `count`, `exists`, `forall`,
        # `map`, `filter`.  `min`/`max`/`sum`/`count`/`map`/`filter` are
        # contextual identifiers (recognised by text); `exists`/`forall` remain
        # reserved.  For `map`, the body is a value expression (one per index
        # of the range); for `filter` it is a boolean predicate (the indices
        # satisfying it form the result).
        QUANT_NAMES = {'min', 'max', 'sum', 'count', 'exists', 'forall',
                       'map', 'filter'}
        if t.text in QUANT_NAMES and t.kind in ('ID', 'KW_EXISTS', 'KW_FORALL'):
            save = self.pos
            self.eat(t.kind)
            if self.peek().kind == 'ID' and self.peek(1).kind == 'KW_IN':
                var = self.eat('ID').text
                self.eat('KW_IN')
                lo, hi, setexpr = self.parse_range_or_set()
                # Optional `pick ID` clause -- witness capture for exists.
                pick_var = None
                if self.accept('KW_PICK'):
                    pick_var = self.eat('ID').text
                self.eat('COLON')
                body = self.parse_expr()
                qname = t.text
                return Expr(kind='quant', quant=qname, q_var=var,
                            q_lo=lo, q_hi=hi, q_set=setexpr, q_body=body,
                            pick_var=pick_var)
            # otherwise it's a function call: rewind and parse as ID
            self.pos = save
            name = self.eat(t.kind).text
            return self.finish_call_or_id(name)
        if t.kind == 'ID':
            self.eat('ID')
            return self.finish_call_or_id(t.text)
        raise SyntaxError(f'line {t.line}: unexpected token {t.text!r}')

    def finish_call_or_id(self, name: str) -> Expr:
        if self.accept('LPAREN'):
            args: List[Expr] = []
            if self.peek().kind != 'RPAREN':
                args.append(self.parse_expr())
                while self.accept('COMMA'):
                    args.append(self.parse_expr())
            self.eat('RPAREN')
            e: Expr = Expr(kind='call', name=name, args=args)
        elif self.accept('LBRACK'):
            i = self.parse_expr()
            self.eat('RBRACK')
            e = Expr(kind='index', name=name, idx=i)
            # Optional second index for 2D arrays: A[i][j]
            if self.accept('LBRACK'):
                j = self.parse_expr()
                self.eat('RBRACK')
                extra: List[Expr] = []
                # 3rd+ chained index (e.g. RT[a][b][c][d]): each additional
                # `[expr]` collects into idx_extra; the AST kind stays
                # 'index2' (a/b are still idx/idx2), so 1D/2D consumers
                # elsewhere are unaffected.
                while self.accept('LBRACK'):
                    extra.append(self.parse_expr())
                    self.eat('RBRACK')
                e = Expr(kind='index2', name=name, idx=i, rhs=j, idx_extra=extra)
        else:
            e = Expr(kind='id', name=name)
        # Postfix: chained `.method(args)` or `.field` (Java-style member access).
        while self.peek().kind == 'DOT':
            self.eat('DOT')
            mname = self.eat('ID').text
            if self.accept('LPAREN'):
                margs: List[Expr] = []
                if self.peek().kind != 'RPAREN':
                    margs.append(self.parse_expr())
                    while self.accept('COMMA'):
                        margs.append(self.parse_expr())
                self.eat('RPAREN')
                e = Expr(kind='method_call', lhs=e, name=mname, args=margs)
            else:
                # `.field`, optionally indexed: `cut[i].v[t]`.  The index
                # rides in `idx` rather than in a separate node so the
                # existing expression walkers need no new case.
                fidx = None
                if self.accept('LBRACK'):
                    fidx = self.parse_expr()
                    self.eat('RBRACK')
                e = Expr(kind='field_access', lhs=e, name=mname, idx=fidx)
        return e


# ============================================================
#  Java code generation
# ============================================================

JAVA_BINOP = {
    '+': '+', '-': '-', '*': '*', '/': '/', '%': '%',
    '<': '<', '<=': '<=', '>': '>', '>=': '>=',
    '=': '==', '!=': '!=',
    'and': '&&', 'or': '||',
}


# Placeholder line reserving the spot where componentwise array-comparison
# helpers go; which ones are needed is only known after the body is generated.
_VEC_SLOT = '\x00vec-helpers\x00'


# Boxed names, needed because a Java collection cannot hold a primitive.
_BOXED = {'int': 'Integer', 'double': 'Double', 'boolean': 'Boolean'}


def java_base_type(t: Type) -> str:
    if t.is_list:
        elem = 'LinkedList<' + _BOXED.get(t.base, t.base) + '>'
        if t.is_2d:
            return elem + '[][]'
        if t.is_array:
            return elem + '[]'
        return elem
    m = {'int': 'int', 'double': 'double', 'boolean': 'boolean'}
    # An enum type is emitted under its own name; anything not one of the
    # three primitives is one.
    base = m.get(t.base, t.base)
    if t.is_set and t.is_array:
        return base + '[][]'
    if t.is_set:
        return base + '[]'
    if t.is_2d:
        return base + '[]' * max(t.dims, 2)
    return base + ('[]' if t.is_array else '')


def _java_print_stmt(var: str, t: 'Type') -> str:
    """Java statement that prints a variable of the given type."""
    if t.is_2d:
        return f'System.out.println(Arrays.deepToString({var}));'
    if t.is_array:
        return f'System.out.println(Arrays.toString({var}));'
    return f'System.out.println({var});'


# Struct types of the program being generated.  `java_zero` needs them to
# tell a record base (which wants an instance) from an enum base (which does
# not), and it has no other access to the program.
_STRUCT_TYPES: set = set()
_ENUM_TYPES: set = set()
# Fields whose type names another program, composed in as a nested process
# (e.g. `DiningPhilosophers dining;` inside DrinkingPhilosophers).  Such a
# field is never allocated by its own program -- only the driver that knows
# the concrete topology can build and configure one -- so, like an enum with
# no initialiser, its Java default is `null` rather than `0`.
_CLASS_TYPES: set = set()


def _is_wholesale_init(init) -> bool:
    """True if an array declaration's initialiser produces the whole array
    (`T[] a = X.clone();`, `T[] a = f();`, `T[] a = new T[k];`) rather than a
    scalar to fill it with (`T[] a = 0;`, `T[] a = infinity;`).

    'index'/'index2' (e.g. `int[] Y = mpref[j][G[j]];`, indexing 2 levels
    into a 3D array to reach a 1D one) also count: since the target being
    declared is itself array-typed, a well-typed source program can only
    have gotten here if the indexed expression *also* evaluates to an
    array -- a source program where it evaluated to a scalar would already
    be a type error unrelated to this decision."""
    return init is not None and init.kind in ('method_call', 'call', 'new_array',
                                               'index', 'index2')


def java_zero(t: Type) -> str:
    if t.base == 'boolean':
        return 'false'
    if t.base == 'double':
        return '0.0'
    if not t.is_array and t.base in _STRUCT_TYPES:
        # `0` is not a value of a record; a scalar struct needs an instance
        # before any of its fields can be assigned.
        return f'new {t.base}()'
    if not t.is_array and t.base in _ENUM_TYPES:
        # Nor is `0` a value of an enum; with no initialiser it is simply
        # unset, which Java spells `null`.
        return 'null'
    if not t.is_array and t.base in _CLASS_TYPES:
        # A composed program field: left unset for the driver to build and
        # configure (it alone knows the topology-specific overrides, e.g.
        # neighbor()/adj()), exactly as an uninitialised enum is.
        return 'null'
    return '0'


def java_infinity(t: Type) -> str:
    if t.base == 'double':
        return 'Double.POSITIVE_INFINITY'
    return 'Integer.MAX_VALUE'


def _program_calls(prog: 'Program', target: str) -> bool:
    """Walk every statement of `prog` looking for a `call` stmt whose name
    equals `target`.  Used when deciding whether a subroutine constructor
    is needed for `target`."""
    def walk_stmts(stmts: List['Stmt']) -> bool:
        for s in stmts:
            if s.kind == 'call' and s.call_name == target:
                return True
            if s.kind == 'par' and walk_stmts(s.par_branches):
                return True
            if s.kind == 'if' and (walk_stmts(s.then_stmts)
                                    or walk_stmts(s.else_stmts)):
                return True
            if s.kind == 'forall' and walk_stmts(s.body):
                return True
            if s.kind == 'block' and walk_stmts(s.body):
                return True
            if s.kind == 'guard' and s.guard_advance is not None:
                if walk_stmts([s.guard_advance]):
                    return True
        return False
    return walk_stmts(prog.stmts)


def extract_top_comment(src: str) -> str:
    """Return the leading file-level // comment block of an .llp source,
    with each line de-indented and the leading "// " stripped.  Empty
    string when the source has no leading comment."""
    out = []
    for line in src.splitlines():
        s = line.strip()
        if not s:
            if out:
                break
            continue
        if s.startswith('//'):
            out.append(s[2:].lstrip())
            continue
        break
    return '\n'.join(out)


class CodeGen:
    def __init__(self, prog: Program, signatures: Optional[dict] = None,
                 noclass: bool = False, src_text: str = ''):
        self.p = prog
        self.src_text = src_text
        # Symbol table: name -> Decl (for type info)
        self.syms: dict = {d.name: d for d in prog.decls}
        # Cross-program signature table: name -> Program.  Populated by the
        # driver with sibling .llp files so that inter-program calls can
        # interleave forwarded fields with explicit scalar args.
        self.signatures: dict = signatures or {}
        # Counter for fresh variable names in quantifier expansion
        self._fresh = 0
        # The first 1-D array param defines `n`.
        self.primary_array = next(
            (p for p in prog.params
             if p.type.is_array and not p.type.is_2d),
            None)
        # Stack of variable names currently in scope for quantifier helpers.
        # Always starts with the rule's forbidden-index (filled in at emit()).
        self._q_scope: List[str] = []
        # Java type of a name in `_q_scope`, when it is not the usual int --
        # a receive binder is an array or an enum array, and a lifted
        # quantifier helper has to declare it correctly.
        self._scope_types: dict = {}
        # Current target type for `infinity` emission (flipped by _with_type).
        self._infty: str = 'Integer.MAX_VALUE'
        # `--noclass` mode: emit methods as `public static` so they can be
        # called as `ClassName.method(...)` rather than `new ClassName().method(...)`.
        self.noclass: bool = noclass
        # Componentwise array-comparison helpers actually reached, and the
        # element types they are needed for.
        # True while emitting a method declared void, where `return X` has
        # to come out as a bare `return;`.
        self._in_void_method: bool = False
        # Witness variables captured by a `pick` outside a guard.
        self._picks_used: set = set()
        self._vec_helpers: set = set()
        self._vec_bases: set = set()
        # LL types of the receive binders currently in scope.
        self._binder_types: dict = {}
        # Java type of each `pick` witness, by name.
        self._pick_types: dict = {}
        # Componentwise min/max on arrays actually reached: (name, base, kind).
        self._vec_minmax: set = set()
        # Element types whose lists are kept sorted by `insert`.
        self._sorted_lists: set = set()

    def _expr_type(self, e: 'Expr') -> Optional[Type]:
        """The declared type of an expression, for the few shapes where it can
        be read off directly: a name, an index into it, and a record field.
        Returns None when the type is not evident, which callers must treat as
        "not an array"."""
        k = e.kind
        if k == 'id':
            # A receive binder is typed by the message declaration.
            t = self._binder_types.get(e.name)
            if t is not None:
                return t
            for d in list(self.p.decls) + list(self.p.params):
                if d.name == e.name:
                    return d.type
            return None
        if k == 'index':
            base = self._expr_type(Expr(kind='id', name=e.name))
            if base is not None and base.is_2d:
                # One index off an N-D array leaves an (N-1)-D structure --
                # a row (N=2) or a smaller nested array (N>=3, e.g. one
                # index off a 3D int[][][] leaves a real int[][], not a
                # scalar). Falls back to "always a row" when dims is unset
                # (N assumed exactly 2), for exact backward compatibility.
                remaining = (base.dims - 1) if base.dims else 1
                if remaining >= 2:
                    return Type(base=base.base, is_array=True, is_2d=True,
                                dims=remaining, is_list=base.is_list)
                return Type(base=base.base, is_array=True, is_list=base.is_list,
                            range1=base.range2)
            if base is not None and base.is_array and base.is_list:
                # One index off an array of lists leaves a list.
                return Type(base=base.base, is_list=True)
            if base is not None and base.is_array:
                # One index off a plain 1D array (e.g. `double[] G`, `G[j]`)
                # leaves the scalar element type -- needed so a comparison
                # like `G[j] < infinity` can tell `infinity` apart from
                # Integer.MAX_VALUE (see gen_expr's 'bin' case).
                return Type(base=base.base)
            return None
        if k == 'index2':
            base = self._expr_type(Expr(kind='id', name=e.name))
            if base is not None and base.is_2d and base.is_list:
                # Both indices off a 2D array of lists leave a list.
                return Type(base=base.base, is_list=True)
            if base is not None and base.is_2d:
                # n_indices off an M-D array leave an (M - n_indices)-D
                # structure. Falls back to the classic "index2 off is_2d
                # is always a scalar" when dims is unset (M assumed exactly
                # 2) -- exact backward compatibility for every 2D-only use.
                # e.g. `mpref[j][G[j]]` off a 3D int[][][] mpref correctly
                # leaves an int[], not a scalar (LLPManOptimalStronglyStableMarriage).
                n_indices = 2 + len(e.idx_extra)
                total = base.dims if base.dims else 2
                remaining = total - n_indices
                if remaining <= 0:
                    return Type(base=base.base)
                if remaining == 1:
                    return Type(base=base.base, is_array=True)
                return Type(base=base.base, is_array=True, is_2d=True, dims=remaining)
            return None
        if k == 'field_access':
            owner = self._expr_type(
                Expr(kind='id', name=e.lhs.name)
                if e.lhs.kind in ('id', 'index') else e.lhs)
            if owner is None or owner.base not in self.p.records:
                return None
            for ftype, fname in self.p.records[owner.base]:
                if fname == e.name:
                    if e.idx is not None:
                        # An index into the field selects an element.
                        return Type(base=ftype.base)
                    return ftype
            return None
        return None

    def _is_vector(self, e: 'Expr') -> bool:
        """True if `e` denotes a whole numeric array, so a comparison against
        another such is componentwise."""
        t = self._expr_type(e)
        return (t is not None and t.is_array and not t.is_set
                and not t.is_2d and t.base in ('int', 'double'))

    # Comparison on whole arrays is componentwise.  `<=` is the pointwise
    # order and `<` is its strict part (`<=` and not equal), which is the
    # standard order on vector clocks and a *partial* order: `!(a < b)` does
    # not give `a >= b`.
    _VEC_CMP = {
        '<=': ('_vecLe', False), '>=': ('_vecLe', True),
        '<':  ('_vecLt', False), '>':  ('_vecLt', True),
        '==': ('_vecEq', False), '!=': ('_vecEq', False),
    }

    def _gen_vec_cmp(self, e: 'Expr') -> Optional[str]:
        if e.op not in self._VEC_CMP:
            return None
        if not (self._is_vector(e.lhs) and self._is_vector(e.rhs)):
            return None
        name, swap = self._VEC_CMP[e.op]
        t = self._expr_type(e.lhs)
        self._vec_bases.add(t.base)
        self._vec_helpers.add(name)
        if name == '_vecLt':
            self._vec_helpers.update(('_vecLe', '_vecEq'))
        a, b = self.gen_expr(e.lhs), self.gen_expr(e.rhs)
        if swap:
            a, b = b, a
        call = f'{name}({a}, {b})'
        return f'(!{call})' if e.op == '!=' else call

    def _elem_java_type(self, set_expr) -> str:
        """Java type of an element of a set- or list-valued expression."""
        t = self._expr_type(set_expr)
        if t is not None and t.is_list:
            return _BOXED.get(t.base, t.base)
        return 'int'

    def _gen_vec_minmax(self, name: str, args) -> Optional[str]:
        """`max(a, b)` on two whole numeric arrays is the componentwise max,
        which is what a matrix-clock merge means."""
        ts = [self._expr_type(a) for a in args]
        if any(t is None or t.is_set or not (t.is_array or t.is_2d)
               or t.base not in ('int', 'double') for t in ts):
            return None
        t = ts[0]
        kind = ('2d' if t.is_2d else '1d')
        self._vec_minmax.add((name, t.base, kind))
        fn = f'_{name}{"Mat" if t.is_2d else "Vec"}'
        return f'{fn}({self.gen_expr(args[0])}, {self.gen_expr(args[1])})'

    def _emit_sorted_insert_helpers(self, lines: List[str]) -> None:
        """`q.insert(x)` keeps the list in ascending order.  A record is
        ordered lexicographically by its fields in declaration order; only
        scalar fields take part, since an array field has no total order."""
        for base in sorted(self._sorted_lists):
            box = _BOXED.get(base, base)
            fn = '_insert' + base.capitalize()
            if base in self.p.records:
                cmp = '_cmp' + base.capitalize()
                fields = [(ft, fn2) for ft, fn2 in self.p.records[base]
                          if not (ft.is_array or ft.is_2d or ft.is_set
                                  or ft.is_list)]
                lines.append(f'  static int {cmp}({box} a, {box} b) {{')
                lines.append('    int c;')
                for ft, f2 in fields:
                    jb = {'int': 'Integer', 'double': 'Double',
                          'boolean': 'Boolean'}.get(ft.base, ft.base)
                    lines.append(f'    c = {jb}.compare(a.{f2}, b.{f2}); '
                                 f'if (c != 0) return c;')
                lines.append('    return 0;')
                lines.append('  }')
                test = f'{cmp}(l.get(k), x) <= 0'
            else:
                test = 'l.get(k).compareTo(x) <= 0'
            lines.append(f'  static void {fn}(LinkedList<{box}> l, {box} x) {{')
            lines.append('    int k = 0;')
            lines.append(f'    while (k < l.size() && {test}) k++;')
            lines.append('    l.add(k, x);')
            lines.append('  }')
        if self._sorted_lists:
            lines.append('')

    def _emit_vec_minmax_helpers(self, lines: List[str]) -> None:
        for name, base, kind in sorted(self._vec_minmax):
            fn = f'_{name}{"Mat" if kind == "2d" else "Vec"}'
            jt = base + ('[][]' if kind == '2d' else '[]')
            op = 'Math.' + name
            lines.append(f'  static {jt} {fn}({jt} a, {jt} b) {{')
            if kind == '2d':
                lines.append(f'    {jt} r = new {base}[a.length][];')
                lines.append('    for (int _u = 0; _u < a.length; _u++) {')
                lines.append(f'      r[_u] = new {base}[a[_u].length];')
                lines.append('      for (int _v = 0; _v < a[_u].length; _v++)')
                lines.append(f'        r[_u][_v] = {op}(a[_u][_v], b[_u][_v]);')
                lines.append('    }')
            else:
                lines.append(f'    {jt} r = new {base}[a.length];')
                lines.append('    for (int _u = 0; _u < a.length; _u++)')
                lines.append(f'      r[_u] = {op}(a[_u], b[_u]);')
            lines.append('    return r;')
            lines.append('  }')
        if self._vec_minmax:
            lines.append('')

    def _emit_vec_helpers(self, lines: List[str]) -> None:
        if not self._vec_helpers:
            return
        lo = self.p.base
        for base in sorted(self._vec_bases):
            jt = base + '[]'
            if '_vecLe' in self._vec_helpers:
                lines.append(f'  static boolean _vecLe({jt} a, {jt} b) {{')
                lines.append(f'    for (int _t = {lo}; _t < a.length; _t++) '
                             f'if (!(a[_t] <= b[_t])) return false;')
                lines.append('    return true;')
                lines.append('  }')
            if '_vecEq' in self._vec_helpers:
                lines.append(f'  static boolean _vecEq({jt} a, {jt} b) {{')
                lines.append(f'    for (int _t = {lo}; _t < a.length; _t++) '
                             f'if (a[_t] != b[_t]) return false;')
                lines.append('    return true;')
                lines.append('  }')
            if '_vecLt' in self._vec_helpers:
                lines.append(f'  static boolean _vecLt({jt} a, {jt} b) {{')
                lines.append('    return _vecLe(a, b) && !_vecEq(a, b);')
                lines.append('  }')
        lines.append('')

    def _with_type(self, t: Type, f):
        """Run `f` with self._infty temporarily set to match the target type."""
        prev = self._infty
        self._infty = ('Double.POSITIVE_INFINITY' if t.base == 'double'
                       else 'Integer.MAX_VALUE')
        try:
            return f()
        finally:
            self._infty = prev

    def fresh(self, prefix: str = 'q') -> str:
        self._fresh += 1
        return f'_{prefix}{self._fresh}'

    # --- expressions ---
    def gen_expr(self, e: Expr) -> str:
        k = e.kind
        if k == 'int':
            return str(e.value)
        if k == 'real':
            return f'{e.value}'
        if k == 'bool':
            return 'true' if e.value else 'false'
        if k == 'infinity':
            return self._infty
        if k == 'id':
            owner = self._enum_owner(e.name)
            if owner is not None:
                return f'{owner}.{e.name}'
            return e.name
        if k == 'index':
            return f'{e.name}[{self.gen_expr(e.idx)}]'
        if k == 'index2':
            dims = ''.join(f'[{self.gen_expr(x)}]'
                           for x in [e.idx, e.rhs, *e.idx_extra])
            return f'{e.name}{dims}'
        if k == 'bin':
            vec = self._gen_vec_cmp(e)
            if vec is not None:
                return vec
            if e.op == 'in':
                # `X in Y` (set/array membership) has no Java codegen yet --
                # unlike the JS emitter's Array.includes, Java has no
                # expression-level linear search over a plain int[]/int[][]
                # without a helper method. Fail loudly rather than emit
                # wrong code.
                raise NotImplementedError(
                    "'in' (set membership) is not supported by the Java "
                    "backend yet; it only has a JS emitter translation "
                    "(see llcjs.py's expr_to_js)")
            # `infinity` has no type of its own -- rendering it as
            # Integer.MAX_VALUE vs Double.POSITIVE_INFINITY has to come
            # from whichever side of the comparison is a real, typed
            # expression (e.g. `G[j] < infinity` where G is double[]).
            # Without this, a comparison against `infinity` always used
            # the emitter's ambient default (Integer.MAX_VALUE) regardless
            # of G's actual element type -- wrong for any double state.
            if e.lhs.kind == 'infinity' and e.rhs.kind != 'infinity':
                other_t = self._expr_type(e.rhs)
                l = (self._with_type(other_t, lambda: self.gen_expr(e.lhs))
                     if other_t is not None else self.gen_expr(e.lhs))
                r = self.gen_expr(e.rhs)
            elif e.rhs.kind == 'infinity' and e.lhs.kind != 'infinity':
                other_t = self._expr_type(e.lhs)
                l = self.gen_expr(e.lhs)
                r = (self._with_type(other_t, lambda: self.gen_expr(e.rhs))
                     if other_t is not None else self.gen_expr(e.rhs))
            else:
                l = self.gen_expr(e.lhs)
                r = self.gen_expr(e.rhs)
            op = JAVA_BINOP[e.op]
            return f'({l} {op} {r})'
        if k == 'un':
            s = self.gen_expr(e.lhs)
            if e.op == 'not':
                return f'(!{s})'
            if e.op == 'neg':
                return f'(-{s})'
        if k == 'call':
            name = e.name
            args = ', '.join(self.gen_expr(a) for a in e.args)
            # map built-ins
            if name in ('min', 'max') and len(e.args) == 2:
                # On whole arrays these are componentwise, as the comparison
                # operators are; on scalars they stay Math.min/Math.max.
                vec = self._gen_vec_minmax(name, e.args)
                if vec is not None:
                    return vec
                return f'Math.{name}({args})'
            if name == 'abs':
                return f'Math.abs({args})'
            # A call to a sibling program used as an EXPRESSION (its
            # return value captured, e.g. `int[] d = Dijkstra(w, s);`)
            # rather than as a bare statement -- the statement-level call
            # handler already does this instantiate-and-invoke rewrite
            # (`new X().entry(args)`) for a void, fire-and-forget call,
            # but this expression path had no equivalent: it fell through
            # to the generic `name(args)` case below, generating a bare
            # function call to a name that is actually a class, which
            # doesn't compile (confirmed: `new Dijkstra()` is required).
            # Excludes self-recursion (`name == self.p.name`): `signatures`
            # includes the program currently being compiled, and without
            # this check a recursive call like BinarySearch's own
            # `return BinarySearch(A, key, low, mid - 1);` turned into
            # `new BinarySearch().BinarySearch(...)`, allocating a fresh
            # (and here pointlessly, since nothing reads instance state
            # across the recursion) object on every recursive call instead
            # of the plain same-instance call the statement-level handler
            # already uses for this exact case.
            callee = self.signatures.get(name) if self.signatures else None
            if callee is not None and name != self.p.name:
                entry = callee.entry_name
                return f'new {name}().{entry}({args})'
            return f'{name}({args})'
        if k == 'new_record':
            return (f'new {e.name}('
                    + ', '.join(self.gen_expr(a) for a in e.args) + ')')
        if k == 'new_array':
            # name holds the base type, idx holds the (first) size
            # expression; idx2 (if set) is the second dimension, idx_extra
            # any 3rd+. jagged (idx2 unset) means `new T[n][]` -- only the
            # outer dimension sized, e.g. LLPManOptimalStronglyStableMarriage's
            # `int[][] adj = new int[n][];` -- Java needs the trailing `[]`
            # or the result is an int[], not an int[][], a type error at
            # every call site expecting int[][].
            if e.idx2 is not None:
                dims = ''.join(f'[{self.gen_expr(x)}]'
                               for x in [e.idx, e.idx2, *e.idx_extra])
                return f'new {e.name}{dims}'
            if e.jagged:
                return f'new {e.name}[{self.gen_expr(e.idx)}][]'
            return f'new {e.name}[{self.gen_expr(e.idx)}]'
        if k == 'method_call':
            obj  = self.gen_expr(e.lhs)
            args = ', '.join(self.gen_expr(a) for a in e.args)
            t = self._expr_type(e.lhs)
            if (e.name == 'insert' and len(e.args) == 1
                    and t is not None and t.is_list):
                # Sorted insertion.  Scalars order naturally; a record orders
                # lexicographically by its fields, in the order declared --
                # which is what "ordered by (timestamp, pid)" means.
                self._sorted_lists.add(t.base)
                return f'_insert{t.base.capitalize()}({obj}, {args})'
            if (e.name == 'first' and not e.args
                    and t is not None and t.is_list):
                return f'{obj}.getFirst()'
            if (e.name == 'remove' and len(e.args) == 1
                    and t is not None and t.is_list and t.base in _BOXED):
                # Java's List.remove(int) removes by index; the element is
                # meant here, so box it.
                box = _BOXED[t.base]
                return f'{obj}.remove({box}.valueOf({args}))'
            return f'{obj}.{e.name}({args})'
        if k == 'field_access':
            obj = self.gen_expr(e.lhs)
            if e.idx is not None:
                return f'{obj}.{e.name}[{self.gen_expr(e.idx)}]'
            return f'{obj}.{e.name}'
        if k == 'quant':
            return self.gen_quant_inline(e)
        raise ValueError(f'cannot generate expr kind {k}')

    def _try_emit_acc_assign(self, s: 'Stmt', indent: str) -> Optional[List[str]]:
        """If `s` has the form `lhs = max(scalar, max-quant)` or
        `lhs = max-quant` (or the min duals), emit the textbook
        accumulator pattern and return the lines.  Otherwise return None
        so the caller falls through to the lift-to-temp path.

            int m = <scalar-or-sentinel>;
            for (var : set)
                m = Math.max(m, body);
            lhs = m;
        """
        rhs = s.rhs
        quant = None
        init_expr = None
        op = None
        # Pattern A: rhs = call('max'|'min', [E, quant]) or [quant, E]
        if (rhs.kind == 'call' and rhs.name in ('max', 'min')
                and len(rhs.args) == 2):
            for i in (0, 1):
                cand = rhs.args[i]
                if (cand.kind == 'quant' and cand.quant == rhs.name
                        and not self._contains_quant(cand.q_body)):
                    other = rhs.args[1 - i]
                    if not self._contains_quant(other):
                        quant = cand
                        init_expr = other
                        op = rhs.name
                        break
        # Pattern B: rhs = max-quant or min-quant alone
        if quant is None and rhs.kind == 'quant' and rhs.quant in ('max', 'min'):
            if not self._contains_quant(rhs.q_body):
                quant = rhs
                init_expr = None
                op = rhs.quant
        if quant is None:
            return None
        # Don't trigger if the LHS index itself contains a quant.
        if (s.lhs_index is not None and self._contains_quant(s.lhs_index)):
            return None
        if (s.lhs_index2 is not None and self._contains_quant(s.lhs_index2)):
            return None
        if any(self._contains_quant(x) for x in s.lhs_index_extra):
            return None

        out: List[str] = []
        # Initialiser.
        if init_expr is not None:
            init_str = self.gen_expr(init_expr)
        else:
            init_str = ('Integer.MAX_VALUE' if op == 'min'
                        else 'Integer.MIN_VALUE')
        out.append(f'{indent}int m = {init_str};')
        # Loop header.
        if quant.q_set is not None:
            setexpr = self.gen_expr(quant.q_set)
            loop_hdr = f'{indent}for (int {quant.q_var} : {setexpr})'
        else:
            lo = self.gen_expr(quant.q_lo)
            test = self._for_loop_test(quant.q_var, quant.q_hi)
            loop_hdr = (f'{indent}for (int {quant.q_var} = {lo}; '
                        f'{test}; {quant.q_var}++)')
        # Body.
        self._q_scope.append(quant.q_var)
        try:
            body_expr = self.gen_expr(quant.q_body)
        finally:
            self._q_scope.pop()
        out.append(f'{loop_hdr} m = Math.{op}(m, {body_expr});')
        # Final assignment to the original LHS.
        lhs = s.lhs_name
        if s.lhs_index is not None:
            lhs = f'{s.lhs_name}[{self.gen_expr(s.lhs_index)}]'
            if s.lhs_index2 is not None:
                lhs = f'{lhs}[{self.gen_expr(s.lhs_index2)}]'
                for extra in s.lhs_index_extra:
                    lhs = f'{lhs}[{self.gen_expr(extra)}]'
        out.append(f'{indent}{lhs} = m;')
        return out

    def _emit_pred_procedural(self, expr: 'Expr', lines: List[str],
                                indent: str, stop: str = 'return false',
                                final_true: bool = True) -> bool:
        """Try to emit `expr` as a procedural sequence of early-return
        checks ending with `return true;`.  Used for the body of a
        boolean-returning method like `forbidden(int j)` to produce
        textbook-style code with no temp variables.

        Returns True on success.  Returns False (without writing to
        `lines`) if the predicate's shape is incompatible with the
        early-return style; the caller should then fall back to the
        lift-to-temp form."""
        conjuncts = self._flatten_and(expr)
        out: List[str] = []
        for c in conjuncts:
            check = self._emit_conjunct_check(c, indent, stop)
            if check is None:
                return False
            out.extend(check)
        if final_true:
            out.append(f'{indent}return true;')
        lines.extend(out)
        return True

    def _flatten_and(self, e: 'Expr') -> List['Expr']:
        if e.kind == 'bin' and e.op == 'and':
            return self._flatten_and(e.lhs) + self._flatten_and(e.rhs)
        return [e]

    def _emit_conjunct_check(self, c: 'Expr', indent: str,
                             stop: str = 'return false') -> Optional[List[str]]:
        """Emit `if (... violated ...) return false;` for one conjunct,
        or None if the conjunct is too complex (e.g. contains a non-
        forall quantifier) and the caller should fall back."""
        if c.kind == 'un' and c.op == 'not':
            # `not X` — violated when X is true.
            if self._contains_quant(c.lhs):
                return None
            v = self.gen_expr(c.lhs)
            return [f'{indent}if ({v}) {stop};']
        if c.kind == 'quant' and c.quant == 'forall':
            # `forall var in S : body` — violated when some body is false.
            if c.q_set is not None:
                setexpr = self.gen_expr(c.q_set)
                loop_hdr = f'{indent}for (int {c.q_var} : {setexpr})'
            else:
                lo = self.gen_expr(c.q_lo)
                test = self._for_loop_test(c.q_var, c.q_hi)
                loop_hdr = (f'{indent}for (int {c.q_var} = {lo}; '
                            f'{test}; {c.q_var}++)')
            self._q_scope.append(c.q_var)
            try:
                body_expr = self.gen_expr(c.q_body)
            finally:
                self._q_scope.pop()
            return [f'{loop_hdr} if (!{body_expr}) {stop};']
        if self._contains_quant(c):
            return None
        v = self.gen_expr(c)
        return [f'{indent}if (!({v})) {stop};']

    def _contains_quant(self, e: Optional['Expr']) -> bool:
        if e is None or not isinstance(e, Expr):
            return False
        if e.kind == 'quant':
            return True
        for attr in ('lhs', 'rhs', 'idx'):
            if self._contains_quant(getattr(e, attr, None)):
                return True
        for a in (getattr(e, 'args', None) or []):
            if self._contains_quant(a):
                return True
        return False

    def _inline_quant_in_expr(self, expr: 'Expr', pre_lines: List[str],
                                indent: str) -> 'Expr':
        """Lift any top-level quantifier inside `expr` into preceding
        statements appended to `pre_lines`, replacing each one with a
        temp identifier.  Lets the call site emit `gen_expr(rewritten)`
        without falling into the helper-method path of `emit_quant_helper`,
        so the resulting Java is direct (no `_qN(j)` artefacts)."""
        if expr is None:
            return expr
        k = expr.kind
        if k == 'quant':
            return self._lift_quant(expr, pre_lines, indent)
        if k == 'bin':
            new_lhs = self._inline_quant_in_expr(expr.lhs, pre_lines, indent)
            new_rhs = self._inline_quant_in_expr(expr.rhs, pre_lines, indent)
            return Expr(kind='bin', op=expr.op, lhs=new_lhs, rhs=new_rhs)
        if k == 'un':
            new_lhs = self._inline_quant_in_expr(expr.lhs, pre_lines, indent)
            return Expr(kind='un', op=expr.op, lhs=new_lhs)
        if k == 'call':
            new_args = [self._inline_quant_in_expr(a, pre_lines, indent)
                        for a in expr.args]
            return Expr(kind='call', name=expr.name, args=new_args)
        if k == 'index':
            new_idx = self._inline_quant_in_expr(expr.idx, pre_lines, indent)
            return Expr(kind='index', name=expr.name, idx=new_idx)
        if k == 'index2':
            new_idx = self._inline_quant_in_expr(expr.idx, pre_lines, indent)
            new_rhs = self._inline_quant_in_expr(expr.rhs, pre_lines, indent)
            new_extra = [self._inline_quant_in_expr(x, pre_lines, indent)
                         for x in expr.idx_extra]
            return Expr(kind='index2', name=expr.name,
                        idx=new_idx, rhs=new_rhs, idx_extra=new_extra)
        return expr

    def _lift_quant(self, e: 'Expr', pre_lines: List[str], indent: str) -> 'Expr':
        """Emit Java that runs the quantifier `e`'s loop into `pre_lines`,
        accumulating the result into a freshly-named temp variable.
        Returns an identifier Expr referring to that temp.  Quantifiers
        nested inside the body are recursively inlined too (their lifted
        statements are placed inside the outer loop), so the result is
        a flat tree of for-loops with no helper methods."""
        self._fresh += 1
        tmp = f't{self._fresh}'
        inner = indent + '  '
        # Emit the loop header.
        if e.q_set is not None:
            setexpr = self.gen_expr(e.q_set)
            loop_hdr = f'{indent}for (int {e.q_var} : {setexpr}) {{'
        else:
            lo = self.gen_expr(e.q_lo)
            test = self._for_loop_test(e.q_var, e.q_hi)
            loop_hdr = f'{indent}for (int {e.q_var} = {lo}; {test}; {e.q_var}++) {{'
        # Body expression: q_var must be in scope so nested expressions
        # see it.  Nested quantifiers in the body are lifted recursively
        # into `body_pre`, which is emitted INSIDE the outer loop body
        # right before the accumulator update.
        self._q_scope.append(e.q_var)
        try:
            body_pre: List[str] = []
            body_inlined = self._inline_quant_in_expr(e.q_body, body_pre, inner)
            body_expr = self.gen_expr(body_inlined)
        finally:
            self._q_scope.pop()
        if e.quant in ('min', 'max'):
            init = ('Integer.MAX_VALUE' if e.quant == 'min'
                    else 'Integer.MIN_VALUE')
            cmp = '<' if e.quant == 'min' else '>'
            pre_lines.append(f'{indent}int {tmp} = {init};')
            pre_lines.append(loop_hdr)
            pre_lines.extend(body_pre)
            pre_lines.append(f'{inner}int v = {body_expr};')
            pre_lines.append(f'{inner}if (v {cmp} {tmp}) {tmp} = v;')
            pre_lines.append(f'{indent}}}')
        elif e.quant == 'sum':
            pre_lines.append(f'{indent}int {tmp} = 0;')
            pre_lines.append(loop_hdr)
            pre_lines.extend(body_pre)
            pre_lines.append(f'{inner}{tmp} += {body_expr};')
            pre_lines.append(f'{indent}}}')
        elif e.quant == 'count':
            pre_lines.append(f'{indent}int {tmp} = 0;')
            pre_lines.append(loop_hdr)
            pre_lines.extend(body_pre)
            pre_lines.append(f'{inner}if ({body_expr}) {tmp}++;')
            pre_lines.append(f'{indent}}}')
        elif e.quant == 'forall':
            pre_lines.append(f'{indent}boolean {tmp} = true;')
            pre_lines.append(loop_hdr)
            pre_lines.extend(body_pre)
            pre_lines.append(f'{inner}if (!({body_expr})) {{ {tmp} = false; break; }}')
            pre_lines.append(f'{indent}}}')
        elif e.quant == 'exists':
            pre_lines.append(f'{indent}boolean {tmp} = false;')
            pre_lines.append(loop_hdr)
            pre_lines.extend(body_pre)
            pre_lines.append(f'{inner}if ({body_expr}) {{ {tmp} = true; break; }}')
            pre_lines.append(f'{indent}}}')
        elif e.quant == 'map':
            acc = f'{tmp}_acc'
            pre_lines.append(f'{indent}java.util.ArrayList<Integer> {acc} = new java.util.ArrayList<>();')
            pre_lines.append(loop_hdr)
            pre_lines.append(f'{indent}  {acc}.add({body_expr});')
            pre_lines.append(f'{indent}}}')
            pre_lines.append(f'{indent}int[] {tmp} = new int[{acc}.size()];')
            pre_lines.append(f'{indent}for (int _i = 0; _i < {tmp}.length; _i++) {tmp}[_i] = {acc}.get(_i);')
        elif e.quant == 'filter':
            acc = f'{tmp}_acc'
            pre_lines.append(f'{indent}java.util.ArrayList<Integer> {acc} = new java.util.ArrayList<>();')
            pre_lines.append(loop_hdr)
            pre_lines.append(f'{indent}  if ({body_expr}) {acc}.add({e.q_var});')
            pre_lines.append(f'{indent}}}')
            pre_lines.append(f'{indent}int[] {tmp} = new int[{acc}.size()];')
            pre_lines.append(f'{indent}for (int _i = 0; _i < {tmp}.length; _i++) {tmp}[_i] = {acc}.get(_i);')
        else:
            raise NotImplementedError(f'inline {e.quant!r}')
        return Expr(kind='id', name=tmp)

    def _for_loop_test(self, var: str, hi_expr: 'Expr') -> str:
        if (hi_expr.kind == 'bin' and hi_expr.op == '-'
                and hi_expr.rhs.kind == 'int' and hi_expr.rhs.value == 1):
            return f'{var} < {self.gen_expr(hi_expr.lhs)}'
        return f'{var} <= {self.gen_expr(hi_expr)}'

    def gen_quant_inline(self, e: Expr) -> str:
        """Quantifier inlined via an immediately-invoked lambda.

        Simpler alternative: lift quantifier into a helper method. For
        readability we emit a helper method instead when inside an
        expression context, via self.emit_quant_helper().
        """
        return self.emit_quant_helper(e)

    # Quantifier helpers are collected into a list and emitted as methods.
    def emit_quant_helper(self, e: Expr) -> str:
        # Every enclosing quantifier variable and the rule's forbidden-index
        # is passed in as a parameter so the helper can see them.
        mname = self.fresh('q')
        params_sig = ', '.join(
            f'{self._scope_types.get(v, "int")} {v}' for v in self._q_scope)
        call_args = ', '.join(self._q_scope)
        # lo / hi / set are in the outer scope -- evaluate them first.
        if e.q_set is not None:
            setexpr = self.gen_expr(e.q_set)
            et = self._elem_java_type(e.q_set)
            loop_hdr = f'  for ({et} {e.q_var} : {setexpr}) {{'
        else:
            lo = self.gen_expr(e.q_lo)
            test = self._for_loop_test(e.q_var, e.q_hi)
            loop_hdr = (f'  for (int {e.q_var} = {lo}; {test}; '
                        f'{e.q_var}++) {{')
        # Push the quantifier variable for any nested quantifier inside body.
        # Over a list it has the element type, not int, and a nested helper
        # that captures it has to declare it correctly.
        qt = self._elem_java_type(e.q_set) if e.q_set is not None else 'int'
        if qt != 'int':
            self._scope_types[e.q_var] = qt
        self._q_scope.append(e.q_var)
        try:
            body = self.gen_expr(e.q_body)
        finally:
            self._q_scope.pop()
            if qt != 'int':
                self._scope_types.pop(e.q_var, None)
        code = []
        if e.quant in ('min', 'max'):
            init = ('Integer.MAX_VALUE' if e.quant == 'min'
                    else 'Integer.MIN_VALUE')
            cmp = '<' if e.quant == 'min' else '>'
            code.append(f'private int {mname}({params_sig}) {{')
            code.append(f'  int acc = {init};')
            code.append(loop_hdr)
            code.append(f'    int v = {body};')
            code.append(f'    if (v {cmp} acc) acc = v;')
            code.append('  }')
            code.append('  return acc;')
            code.append('}')
        elif e.quant == 'sum':
            code.append(f'private int {mname}({params_sig}) {{')
            code.append('  int acc = 0;')
            code.append(loop_hdr)
            code.append(f'    acc += {body};')
            code.append('  }')
            code.append('  return acc;')
            code.append('}')
        elif e.quant == 'count':
            code.append(f'private int {mname}({params_sig}) {{')
            code.append('  int acc = 0;')
            code.append(loop_hdr)
            code.append(f'    if ({body}) acc++;')
            code.append('  }')
            code.append('  return acc;')
            code.append('}')
        elif e.quant == 'exists':
            code.append(f'private boolean {mname}({params_sig}) {{')
            code.append(loop_hdr)
            if e.pick_var:
                # Witness capture: stash the index that satisfied the body,
                # exactly as a `pick` guard does, so the caller can bind it.
                code.append(f'    if ({body}) {{ this.picked_{e.pick_var} = '
                            f'{e.q_var}; return true; }}')
                self._picks_used.add(e.pick_var)
            else:
                code.append(f'    if ({body}) return true;')
            code.append('  }')
            code.append('  return false;')
            code.append('}')
        elif e.quant == 'forall':
            code.append(f'private boolean {mname}({params_sig}) {{')
            code.append(loop_hdr)
            code.append(f'    if (!({body})) return false;')
            code.append('  }')
            code.append('  return true;')
            code.append('}')
        elif e.quant == 'map':
            # Body is a value expression; result is a new int[] of those
            # values, in iteration order.
            code.append(f'private int[] {mname}({params_sig}) {{')
            code.append('  java.util.ArrayList<Integer> _acc = new java.util.ArrayList<>();')
            code.append(loop_hdr)
            code.append(f'    _acc.add({body});')
            code.append('  }')
            code.append('  int[] arr = new int[_acc.size()];')
            code.append('  for (int _i = 0; _i < arr.length; _i++) arr[_i] = _acc.get(_i);')
            code.append('  return arr;')
            code.append('}')
        elif e.quant == 'filter':
            # Body is a boolean predicate; result is the int[] of indices
            # in the source range/set for which the body is true.
            code.append(f'private int[] {mname}({params_sig}) {{')
            code.append('  java.util.ArrayList<Integer> _acc = new java.util.ArrayList<>();')
            code.append(loop_hdr)
            code.append(f'    if ({body}) _acc.add({e.q_var});')
            code.append('  }')
            code.append('  int[] arr = new int[_acc.size()];')
            code.append('  for (int _i = 0; _i < arr.length; _i++) arr[_i] = _acc.get(_i);')
            code.append('  return arr;')
            code.append('}')
        else:
            raise ValueError(f'unknown quantifier {e.quant}')
        self.quant_methods.append('\n'.join('  ' + l for l in code))
        return f'{mname}({call_args})'

    def quant_return_type(self, e: Expr) -> str:
        if e.quant in ('exists', 'forall'):
            return 'boolean'
        if e.quant == 'count':
            return 'int'
        if e.quant in ('map', 'filter'):
            return 'int[]'
        return 'int'  # min, max, sum default; extend for real if needed

    # --- statements ---
    def gen_stmts(self, stmts: List[Stmt], indent: str) -> List[str]:
        out: List[str] = []
        saved = len(self._q_scope)
        for s in stmts:
            out.extend(self.gen_stmt(s, indent))
        # Restore scope: anything a local `decl` pushed goes out of scope
        # at the end of its enclosing block.
        del self._q_scope[saved:]
        return out

    def gen_stmt(self, s: Stmt, indent: str) -> List[str]:
        if s.kind == 'send':
            # Append to the channel from me to dest.  "Me" is the index of
            # the enclosing guard; outside a guard there is no process to
            # speak of, so the sender defaults to index 0.
            dest = self.gen_expr(s.send_dest)
            # The sender is the process, always -- never the guard's index,
            # which for a rule like `upon (f)' names a resource rather than a
            # process.  Reading it off the `me' field also frees a procedure
            # from depending on its caller passing the right index.
            me = ('me' if self._uses_msgs
                  else (getattr(self, '_cur_self', None) or '0'))
            payload = ', '.join(self.gen_expr(e) for e in s.send_payload)
            tag = f'"{s.send_tag}"' if s.send_tag is not None else '""'
            return [f'{indent}chan[{me}][{dest}].add('
                    f'new Msg({me}, {tag}, new Object[]{{{payload}}}));']
        if s.kind == 'swap':
            a = self.gen_expr(s.swap_a)
            b = self.gen_expr(s.swap_b)
            return [f'{indent}{{ int tmp = {a}; {a} = {b}; {b} = tmp; }}']
        if s.kind == 'assign':
            # Fast path: detect `lhs = max(scalar, max-quant)` (or its
            # min/quant-only variants) and emit the textbook accumulator
            # form `int m = init; for(...) m = Math.op(m, body); lhs = m;`.
            acc = self._try_emit_acc_assign(s, indent)
            if acc is not None:
                return acc
            pre: List[str] = []
            new_idx  = self._inline_quant_in_expr(s.lhs_index, pre, indent)
            new_idx2 = self._inline_quant_in_expr(s.lhs_index2, pre, indent)
            new_extra = [self._inline_quant_in_expr(x, pre, indent)
                        for x in s.lhs_index_extra]
            new_rhs  = self._inline_quant_in_expr(s.rhs, pre, indent)
            lhs = s.lhs_name
            if new_idx is not None:
                lhs = f'{s.lhs_name}[{self.gen_expr(new_idx)}]'
                if new_idx2 is not None:
                    lhs = f'{lhs}[{self.gen_expr(new_idx2)}]'
                    for extra in new_extra:
                        lhs = f'{lhs}[{self.gen_expr(extra)}]'
            if s.lhs_field is not None:
                lhs = f'{lhs}.{s.lhs_field}'
                if s.lhs_field_index is not None:
                    lhs = f'{lhs}[{self.gen_expr(s.lhs_field_index)}]'
            return pre + [f'{indent}{lhs} = {self.gen_expr(new_rhs)};']
        if s.kind == 'if' and getattr(self, '_uses_msgs', False):
            _inl = self._try_inline_pick_if(s, indent)
            if _inl is not None:
                return _inl
        if s.kind == 'if':
            out = [f'{indent}if ({self.gen_expr(s.cond)}) {{']
            # A `pick` in the condition binds its witness for the then-branch
            # only: if the condition was false there is no witness.
            for pv in self._cond_pick_vars(s.cond):
                pt = self._pick_types.get(pv, 'int')
                out.append(f'{indent}  {pt} {pv} = picked_{pv};')
                self._scope_types[pv] = pt
                self._q_scope.append(pv)
            try:
                out.extend(self.gen_stmts(s.then_stmts, indent + '  '))
            finally:
                for _ in self._cond_pick_vars(s.cond):
                    self._q_scope.pop()
            if s.else_stmts:
                out.append(f'{indent}}} else {{')
                out.extend(self.gen_stmts(s.else_stmts, indent + '  '))
            out.append(f'{indent}}}')
            return out
        if s.kind == 'while':
            cond = self.gen_expr(s.cond)
            out = [f'{indent}while ({cond}) {{']
            out.extend(self.gen_stmts(s.body, indent + '  '))
            out.append(f'{indent}}}')
            return out
        if s.kind == 'forall':
            if s.loop_set is not None:
                setexpr = self.gen_expr(s.loop_set)
                et = self._elem_java_type(s.loop_set)
                out = [f'{indent}for ({et} {s.loop_var} : {setexpr}) {{']
            else:
                lo = self.gen_expr(s.loop_lo)
                test = self._for_loop_test(s.loop_var, s.loop_hi)
                out = [f'{indent}for (int {s.loop_var} = {lo}; '
                       f'{test}; {s.loop_var}++) {{']
            # The loop variable is visible inside the body -- any nested
            # quantifier helper must receive it as a parameter, and with the
            # right type: iterating a list of records binds a record, not an
            # int, and a helper lifted out of the body must say so.
            self._q_scope.append(s.loop_var)
            had = s.loop_var in self._scope_types
            prev = self._scope_types.get(s.loop_var)
            if s.loop_set is not None:
                self._scope_types[s.loop_var] = self._elem_java_type(s.loop_set)
            try:
                out.extend(self.gen_stmts(s.body, indent + '  '))
            finally:
                self._q_scope.pop()
                if had:
                    self._scope_types[s.loop_var] = prev
                else:
                    self._scope_types.pop(s.loop_var, None)
            out.append(f'{indent}}}')
            return out
        if s.kind == 'decl':
            t = java_base_type(s.decl_type)
            # Generate the RHS BEFORE the name enters scope, so quantifier
            # helpers triggered during RHS evaluation don't try to capture
            # an undefined `decl_name`.  Lift quantifiers in the init into
            # preceding lines so they share the enclosing method scope --
            # this keeps `map`/`filter`/etc. able to read method params
            # without having to plumb them into a helper.
            pre: List[str] = []
            rhs = None
            if s.decl_init is not None:
                new_init = self._inline_quant_in_expr(s.decl_init, pre, indent)
                rhs = self.gen_expr(new_init)
            # Local int/real/bool declared in a body becomes visible to the
            # rest of the enclosing stmt_list, so push it onto the quantifier
            # scope now.
            if s.decl_type.base in ('int', 'double') and not s.decl_type.is_array:
                self._q_scope.append(s.decl_name)
            if s.decl_type.is_list and rhs is None:
                return pre + [f'{indent}{t} {s.decl_name} = new LinkedList<>();']
            if rhs is not None:
                arr = s.decl_type.is_array or s.decl_type.is_2d
                if arr and not _is_wholesale_init(s.decl_init):
                    # A scalar initialiser on an array type means "allocate
                    # and fill", not "assign the scalar to the reference".
                    d = Decl(kind='var', type=s.decl_type,
                             name=s.decl_name, init=s.decl_init)
                    if s.decl_type.is_2d:
                        return pre + self._emit_2d_alloc(
                            s.decl_name, d, indent, decl=t + ' ')
                    base = java_base_type(Type(s.decl_type.base))
                    if not self._has_guards:
                        # No `n` field exists outside a guard-based program
                        # (see self._has_guards) -- a scalar-filled array
                        # local here is always the "single register wrapped
                        # in an array" idiom (only ever indexed at [0], e.g.
                        # ClosestPair's `double[] G = infinity;`, matching
                        # the JS emitter's scalar_local convention). One
                        # slot, index 0 directly -- sidesteps this.p.base's
                        # 0-/1-indexed loop-bound complexity entirely, which
                        # doesn't apply here since there's exactly one slot.
                        out = pre + [f'{indent}{t} {s.decl_name} = new {base}[1];']
                        if rhs != java_zero(Type(s.decl_type.base)):
                            out.append(f'{indent}{s.decl_name}[0] = {rhs};')
                        return out
                    size = 'n + 1' if self.p.base == 1 else 'n'
                    out = pre + [f'{indent}{t} {s.decl_name} = '
                                 f'new {base}[{size}];']
                    if rhs != java_zero(Type(s.decl_type.base)):
                        lo, cmp = str(self.p.base), self._loop_cmp
                        out.append(f'{indent}for (int _u = {lo}; _u {cmp} n; '
                                   f'_u++) {s.decl_name}[_u] = {rhs};')
                    return out
                return pre + [f'{indent}{t} {s.decl_name} = {rhs};']
            if s.decl_type.is_array and not s.decl_type.is_2d:
                # An array declared with no initializer -- `SRSW[] W;` as a
                # plain statement, e.g. a per-call local buffer inside an
                # aux method -- still needs its backing storage allocated;
                # nothing else will do it, unlike a field, which
                # _init_body_lines allocates once up front.
                elem = java_base_type(Type(s.decl_type.base, False))
                size = self._array_size_expr(s.decl_type)
                return pre + [f'{indent}{t} {s.decl_name} = new {elem}[{size}];']
            return pre + [f'{indent}{t} {s.decl_name};']
        if s.kind == 'exprstmt':
            return [f'{indent}{self.gen_expr(s.rhs)};']
        if s.kind == 'par':
            op = getattr(s, 'par_op', '[]')
            # Composition flavor: emit a banner comment so the reader knows
            # which operator was used; sequential lowering is a placeholder
            # for all five operators (a future -parallel / -interleave flag
            # can refine each one).
            banner = {
                '[]': '/* [ ... [] ... ]: parallel branches (sequential for now) */',
                '&&': '/* [ ... && ... ]: predicate conjunction (sequential for now) */',
                'or': '/* [ ... or ... ]: nondeterministic choice (first branch for now) */',
                ';':  '/* [ ... ; ... ]: sequential composition */',
                '|':  '/* [ ... | ... ]: pipelining (sequential for now) */',
                '<>':   '/* [ ... <> ... ]:   feedback composition (sequential for now) */',
                'sync': '/* [ ... sync ... ]: synchronous composition (sequential for now) */',
            }.get(op, f'/* [ ... {op} ... ] */')
            out = [f'{indent}{banner}']
            for i, br in enumerate(s.par_branches):
                out.append(f'{indent}// branch {i}')
                out.extend(self.gen_stmt(br, indent))
            return out
        if s.kind == 'block':
            out = [f'{indent}{{']
            out.extend(self.gen_stmts(s.body, indent + '  '))
            out.append(f'{indent}}}')
            return out
        if s.kind == 'exit':
            val = self.gen_expr(s.return_value)
            return [f'{indent}this.result = {val};',
                    f'{indent}this.done = true;']
        if s.kind == 'return':
            if self._void_stop_signal:
                # A void program's advance still using `return null;` to
                # mean "stop the whole algorithm now" (see _void_stop_mode)
                # -- advance() here returns boolean, true = stop; a bare
                # `return;` (the ordinary void-method translation below)
                # would silently discard that intent instead of signaling
                # it to the fixpoint loop.
                return [f'{indent}return true;']
            # `return X` -- early exit returning the operand.  The operand
            # is mandatory in the new grammar, but defensive: if it's
            # missing (legacy AST), emit `return;`.
            if s.return_value is not None and not self._in_void_method:
                return [f'{indent}return {self.gen_expr(s.return_value)};']
            return [f'{indent}return;']
        if s.kind == 'guard':
            # Inline fixpoint loop using the helpers emitted by
            # _emit_all_guard_helpers.
            g_id = getattr(s, '_codegen_id', None)
            if g_id is None:
                raise ValueError('guard without codegen_id -- internal error')
            pick = s.guard_pick_var is not None
            if self._uses_msgs:
                # A reactive program has no fixpoint to run at start-up: its
                # rules fire as messages arrive, in whatever order a mode
                # interleaves them.  They are collected here and emitted as
                # step() instead, which is what every mode drives.
                self._step_guards.append(s)
                return []
            return self._emit_fixpoint_loop(f'_forbidden{g_id}',
                                            f'_advance{g_id}', pick, indent,
                                            indices=s.guard_index,
                                            returns=self._advance_returns(s))
        if s.kind == 'call':
            # Self-recursion to this program's entry method, OR a call to
            # an aux method declared in this class -- emit as a plain
            # method call (no `new`).
            aux_names = {m.name for m in self.p.aux_methods}
            if s.call_name in aux_names:
                args = ', '.join(self.gen_expr(a) for a in s.call_args)
                return [f'{indent}{s.call_name}({args});']
            if s.call_name == self.p.entry_name:
                args = self._build_self_call_args(s.call_args)
                return [f'{indent}{s.call_name}({args});']
            method_args = ', '.join(self.gen_expr(a) for a in s.call_args)
            callee = self.signatures.get(s.call_name) if self.signatures else None
            if callee is None and s.call_name[:1].islower():
                # Not a known program, and named like a procedure rather than
                # a class: a routine the harness supplies, so call it plainly.
                return [f'{indent}{s.call_name}({method_args});']
            # Cross-program call: empty constructor + entry method takes
            # the args, so the call shape is `new Foo().Foo(args)`.
            entry = callee.entry_name if callee else s.call_name
            return [f'{indent}new {s.call_name}().{entry}({method_args});']
        raise ValueError(f'unknown stmt kind {s.kind}')

    def _is_called_elsewhere(self) -> bool:
        """True if this program is referenced by any call statement (in any
        program in self.signatures, including itself -- recursion)."""
        me = self.p.name
        for other in self.signatures.values():
            if _program_calls(other, me):
                return True
        return False

    def _caller_has_field(self, name: str) -> bool:
        """True if the caller declares a field (param, input, output, var)
        of the given name."""
        if any(p.name == name for p in self.p.params):
            return True
        if any(d.name == name for d in self.p.decls):
            return True
        return False

    def _build_self_call_args(self, explicit: List['Expr']) -> str:
        """Build args for a self-recursive method call.  Explicit args bind
        to the method's params left-to-right; any trailing params not
        covered by explicit args are filled with `this.X` (typical for
        shared array state)."""
        parts: List[str] = []
        exp = list(explicit)
        for p in self.p.params:
            if exp:
                parts.append(self.gen_expr(exp.pop(0)))
            elif self._caller_has_field(p.name):
                parts.append(f'this.{p.name}')
            else:
                parts.append(f'this.{p.name}')
        return ', '.join(parts)

    def _build_call_args(self, callee_name: str, explicit: List['Expr']) -> str:
        """Build the full Java argument list for `new Callee(...)`.

        Strategy: look up the callee's signature in self.signatures (if
        available) and, for each callee param, either forward the caller's
        same-named array field (`this.X`) or consume the next explicit arg.
        Callee outputs are appended in order, each forwarded from the caller
        by name.  If the callee signature is unknown, fall back to the v0
        convention: explicit args first, then caller's inputs and outputs.
        """
        callee = self.signatures.get(callee_name) if self.signatures else None
        if callee is None:
            return ', '.join(self.gen_expr(a) for a in explicit)
        parts: List[str] = []
        exp = list(explicit)
        for p in callee.params:
            if exp:
                parts.append(self.gen_expr(exp.pop(0)))
            elif self._caller_has_field(p.name):
                parts.append(f'this.{p.name}')
            else:
                parts.append(f'this.{p.name}')
        # Subroutine constructor takes the callee's result var as the last
        # argument; forward the caller's same-named field.
        if callee.result_name is not None and self._caller_has_field(callee.result_name):
            parts.append(f'this.{callee.result_name}')
        return ', '.join(parts)

    def _walk_guards(self, stmts: List[Stmt]):
        """Yield every guard stmt nested anywhere in the given stmt list."""
        for s in stmts:
            if s.kind == 'guard':
                yield s
            elif s.kind == 'par':
                for b in s.par_branches:
                    for g in self._walk_guards([b]):
                        yield g
            elif s.kind == 'if':
                for g in self._walk_guards(s.then_stmts):
                    yield g
                for g in self._walk_guards(s.else_stmts):
                    yield g
            elif s.kind == 'forall':
                for g in self._walk_guards(s.body):
                    yield g
            elif s.kind == 'block':
                for g in self._walk_guards(s.body):
                    yield g

    # --- top-level emitter ---
    def emit(self) -> str:
        self.quant_methods: List[str] = []
        # Programs with `forbidden`/`advance` guards need shared class fields
        # so the auto-generated forbidden(j)/advance(j) helpers can read them.
        # Pure-method programs (D&C without guards) don't -- their state is
        # method-local, passed via parameters between aux methods.
        self._has_guards = any(True for _ in self._walk_guards(self.p.stmts))
        self._needs_no_exit_sentinel = any(
            self._contains_explicit_null_return(g.guard_advance)
            for g in self._walk_guards(self.p.stmts))
        # A *void* program (no return_type -- G is mutated via a param, not
        # returned) has no result to hand back, so _advance_returns is
        # always False for it and the whole sentinel/boxed-result machinery
        # above doesn't apply. But `return null;` inside such a program's
        # advance still means the same thing it does everywhere else --
        # "stop the whole algorithm now" -- and a void method has no way to
        # say that via its return value at all unless we give it one.
        # _void_stop_mode switches advance() to `boolean` (true = stop) for
        # just this purpose. Confirmed live: without this, MandatoryMST's
        # `return null;` was silently downgraded to a bare `return;` --
        # exiting advance() but never telling the fixpoint loop to stop,
        # which then looped forever on the same permanently-stuck index.
        self._void_stop_mode = self.p.return_type is None and self._needs_no_exit_sentinel
        self._void_stop_signal = False  # set/restored around each advance body
        self._uses_msgs = self._program_uses_messages()
        # Rules of a reactive program, in source order, for step().
        self._step_guards: List = []
        self._rule_method: dict = {}
        # name -> source, for types that travel in messages and so
        # are written as files of their own beside the program.
        self.shared_types: dict = {}
        # rule id -> True when the fused method scans its own
        # index, so step() must not wrap it in a loop.
        self._rule_scans: dict = {}
        lines: List[str] = []
        # File-level comment carried over from the .llp source.
        top_comment = extract_top_comment(self.src_text) if self.src_text else ''
        if top_comment:
            for cline in top_comment.splitlines():
                lines.append('// ' + cline if cline else '//')
            lines.append('')
        lines.append('import java.util.*;')
        lines.append('')
        global _STRUCT_TYPES, _ENUM_TYPES, _CLASS_TYPES
        _STRUCT_TYPES = {r for r in self.p.records
                         if r not in self.p.messages}
        _ENUM_TYPES = set(self.p.enums)
        # A field typed with an identifier that is neither a primitive, an
        # enum, nor a record names another program, composed in as a field
        # (see Parser._is_field_start).  Confirm it really is one -- against
        # every sibling .ll/.llp file, via scan_signatures -- so a typo'd
        # enum/record name fails with a clear message instead of silently
        # compiling to a bare, unresolvable Java type.  Skipped when no
        # signature table was supplied (e.g. compiling one file in isolation
        # outside its directory): there is nothing to check it against.
        _CLASS_TYPES = set()
        for d in list(self.p.decls) + list(self.p.params):
            base = d.type.base
            if base in ('int', 'double', 'boolean'):
                continue
            if base in self.p.enums or base in self.p.records:
                continue
            _CLASS_TYPES.add(base)
            if self.signatures and base not in self.signatures:
                raise SyntaxError(
                    f'`{base}` (type of `{d.name}`) is not an enum, a '
                    f'struct/message, or a program found beside this file; '
                    f'a composed field must name another .ll/.llp program '
                    f'in the same directory')
        if self._uses_msgs:
            lines.append(f'public class {self.p.name} extends LLProcess {{')
        else:
            lines.append(f'public class {self.p.name} {{')
        self._emit_enum_types(lines)
        self._emit_record_types(lines)
        vec_slot = len(lines)
        lines.append(_VEC_SLOT)
        if self._uses_msgs:
            self._emit_exit_fields(lines)
        if self._has_guards or self.p.stateful:
            if self._has_guards:
                lines.append('  int n;')
            sentinel_field = self._no_exit_sentinel_field()
            if sentinel_field is not None:
                lines.append(sentinel_field)
            # Names that get their own auto-generated field below (the
            # iteration-domain `n`, and each guard's own index variable(s))
            # must not *also* be declared here from a param or local decl
            # that happens to share the name -- e.g. LLPQHornSAT's own
            # `int j = 0;` (a plain loop counter, unrelated to the guard)
            # colliding with `forbidden (j) : ...`'s index variable, also
            # named `j`. Both would otherwise become the same duplicate
            # field, a javac error identical in shape to the `n` case this
            # guarded against before -- just under a different name.
            reserved_field_names: set = {'n'} if self._has_guards else set()
            for g in self._walk_guards(self.p.stmts):
                if g.guard_index:
                    reserved_field_names.update(g.guard_index)
            for d in self.p.params:
                if d.name in reserved_field_names:
                    continue
                lines.append(f'  {java_base_type(d.type)} {d.name};')
            for d in self.p.decls:
                if d.name in reserved_field_names:
                    continue
                lines.append(f'  {java_base_type(d.type)} {d.name};')
            # For each guard with `pick X`, declare a class field `picked_X`
            # that the auto-generated `forbidden` helper writes when it
            # finds a witness, and that the matching `advance` helper
            # reads in place of the (now removed) witness parameter.
            # Declare class fields for guard index variables so that
            # advance() can read them without parameters.
            idx_seen: set = set()
            for g in self._walk_guards(self.p.stmts):
                if g.guard_index:
                    for v in g.guard_index:
                        if v not in idx_seen:
                            idx_seen.add(v)
                            lines.append(f'  int {v};')
            picked_seen = set()
            # A `pick` anywhere in an expression -- not only in a guard --
            # needs the same witness field.
            self._pick_types = self._scan_pick_vars()
            for pv, pt in sorted(self._pick_types.items()):
                if pv not in picked_seen:
                    picked_seen.add(pv)
                    lines.append(f'  {pt} picked_{pv};')
            # `upon receive(...) from k` stashes its witness the same way a
            # `pick` guard does, so it needs the same field.
            for g in self._walk_guards(self.p.stmts):
                if g.recv_tag is not None or g.recv_binders or g.recv_from:
                    rv = g.recv_from or '_src'
                    if rv not in picked_seen:
                        picked_seen.add(rv)
                        lines.append(f'  int picked_{rv};')
            for g in self._walk_guards(self.p.stmts):
                if g.guard_pick_var and g.guard_pick_var not in picked_seen:
                    picked_seen.add(g.guard_pick_var)
                    lines.append(f'  int picked_{g.guard_pick_var};')
            # In default array mode the fixpoint loop scans 1..n directly,
            # so no forbiddenArr field is needed.  Linked-list and heap
            # modes still need their tracking structure.
            if self.p.mode == 'linkedlist':
                lines.append('  LinkedList<Integer> forbiddenList;')
            elif self.p.mode == 'heap':
                lines.append('  PriorityQueue<Integer> forbiddenHeap;')
            elif self.p.mode == 'deque':
                lines.append('  Deque<Integer> forbiddenDeque;')
            lines.append('')
        # Loop bounds: `lo`/`hi` params override the default `0..n-1`
        # (equivalently emitted as `j < n`).  When the user supplies
        # explicit lo/hi the fixpoint loop is half-open `[lo..hi)`.
        has_lo = any(p.name == 'lo' for p in self.p.params)
        has_hi = any(p.name == 'hi' for p in self.p.params)
        self._loop_lo = 'lo' if has_lo else str(self.p.base)
        self._loop_hi = 'hi' if has_hi else 'n'
        self._loop_cmp = '<=' if self.p.base == 1 else '<'
        # Detect the common "single guard stmt as the whole body" case; we
        # name its helpers `forbidden(j)` / `advance(j)` for byte-equivalence
        # with the v0 output.  Other programs use numbered helpers per guard.
        # A lone guard gets unnumbered `forbidden()`/`advance()` helpers, for
        # continuity with the v0 output.  A receive guard is excluded: it has
        # no predicate to emit, so it takes the general numbered path.
        self._single_guard = (
            len(self.p.stmts) == 1 and self.p.stmts[0].kind == 'guard'
            and not self._is_recv_guard(self.p.stmts[0]))
        self._guard_count = 0
        # Helper methods for guard statements must be defined in the class
        # so the constructor body can call them.  Emit them BEFORE the
        # constructor so the generated source is in declaration order.
        if self._has_guards:
            if self._single_guard:
                self._emit_single_guard_methods(lines, self.p.stmts[0])
            else:
                self._emit_all_guard_helpers(lines, self.p.stmts)
        # priority method for heap mode
        if self._has_guards and self.p.mode == 'heap':
            lines.append('  private double priority(int j) {')
            lines.append(f'    return {self.gen_expr(self.p.priority_expr)};')
            lines.append('  }')
            lines.append('')
        # No explicit constructor: Java's default no-arg constructor is
        # used.  All field initialisation lives at the top of the entry
        # method body, so callers write `new Foo().<entry>(args)`.
        # The recursive method (named after the class) holds the algorithm.
        self._emit_method(lines)
        # Auxiliary methods declared inside the same class.
        for m in self.p.aux_methods:
            self._emit_aux_method(lines, m)
        # quantifier helper methods collected during generation
        for m in self.quant_methods:
            lines.append(m); lines.append('')
        if self._uses_msgs:
            # A reactive program has no standalone run: it needs n processes,
            # channels between them, and a driver to interleave their steps.
            # That is what a mode is, and it lives in a driver class beside
            # this one -- which also supplies the runtime primitives above.
            lines.append('  // No main(): this algorithm runs as n processes')
            lines.append('  // under a driver.  See the companion Env class.')
            lines.append('')
        else:
            lines.extend(self.emit_main())
        lines.append('}')
        if self._uses_msgs:
            # A field only the old guard/action split needed is dead once the
            # rules are fused; leaving it would invite a reader to look for a
            # use that is not there.
            body = '\n'.join(lines)
            # Every name a rule used as its index.  Such a field exists only to
            # carry the index from the guard to the action; a fused rule holds
            # it in a local instead.
            idx_names = {v for g in self._walk_guards(self.p.stmts)
                         for v in (g.guard_index or [])}
            idx_names.add(self.p.process_index or '')
            keep = []
            for ln in lines:
                m = re.match(r'\s+\w[\w<>\[\], ]*\s+(picked_\w+);$', ln)
                if m and len(re.findall(r'\b' + m.group(1) + r'\b', body)) <= 1:
                    continue
                # an index field: dead once no rule assigns it
                m2 = re.match(r'\s+int (\w+);$', ln)
                if (m2 and m2.group(1) in idx_names
                        and f'this.{m2.group(1)} =' not in body):
                    continue
                keep.append(ln)
            lines[:] = keep
        vec_lines: List[str] = []
        self._emit_vec_helpers(vec_lines)
        self._emit_vec_minmax_helpers(vec_lines)
        self._emit_sorted_insert_helpers(vec_lines)
        lines[vec_slot:vec_slot + 1] = vec_lines
        return '\n'.join(lines)

    def _result_type(self) -> str:
        """Return type of the entry method, taken from the user's declared
        return type in the source header."""
        if self.p.return_type is None:
            return 'void'
        return java_base_type(self.p.return_type)

    def _emit_aux_method(self, lines: List[str], m: 'Method') -> None:
        """Emit an auxiliary method with the user-supplied signature."""
        ret = 'void' if m.return_type is None else java_base_type(m.return_type)
        sig = ', '.join(
            f'{java_base_type(d.type)} {d.name}' for d in m.params)
        modifier = 'static ' if self.noclass else ''
        lines.append(f'  public {modifier}{ret} {m.name}({sig}) {{')
        # Aux methods get the scalar params in `_q_scope` so quantifier
        # helpers see the right values.
        self._q_scope = [
            p.name for p in m.params
            if not p.type.is_array and not p.type.is_2d and not p.type.is_set]
        lines.extend(self.gen_stmts(m.body, '    '))
        lines.append('  }')
        lines.append('')

    def _emit_method(self, lines: List[str]) -> None:
        """Emit the entry method that holds the program body.  Its name may
        differ from the class name.  Calls in the body are routed:
        self-recursion via `entry_name` -> method call; aux calls -> method
        call; cross-program -> `new Other(...).<entry>(...)`.

        For guard-free programs (pure methods, no `forbidden`/`advance`),
        the program's field decls are emitted as locals at the top of the
        entry method, and `n` (auto-bound to the primary array length) is
        also a local.  This keeps the class clean: no fields, no constructor."""
        ret = self._result_type()
        sig = ', '.join(
            f'{java_base_type(d.type)} {d.name}' for d in self.p.params)
        modifier = 'static ' if self.noclass else ''
        lines.append(f'  public {modifier}{ret} {self.p.entry_name}({sig}) {{')
        if self._has_guards or self.p.stateful:
            lines.extend(self._init_body_lines('    '))
        if self._has_guards:
            lines.extend(self._mode_init_body_lines('    '))
        # Quantifier helpers emitted from the body must receive the method's
        # scalar params (which differ across recursive calls), not the
        # constructor-set fields.  Seed `_q_scope` accordingly.
        self._q_scope = [
            p.name for p in self.p.params
            if not p.type.is_array and not p.type.is_2d and not p.type.is_set]
        # For pure-method programs, emit the program's prog.decls as
        # locals at the top of the entry method.  Users must use explicit
        # `new T[expr]` allocations -- there is no auto-bound `n`.  A
        # `setStateful()` program's decls were already emitted as fields
        # and initialized above, so it is excluded here the same way a
        # guard-bearing program is.
        if not (self._has_guards or self.p.stateful):
            for d in self.p.decls:
                lines.extend(self._emit_local_decl(d, '    '))
        if self._single_guard:
            g = self.p.stmts[0]
            pick = g.guard_pick_var is not None
            lines.extend(
                self._emit_fixpoint_loop('forbidden', 'advance', pick,
                                          '    ', indices=g.guard_index,
                                          returns=self._advance_returns(g)))
        else:
            body_lines = self.gen_stmts(self.p.stmts, '    ')
            pidx = self.p.process_index
            if pidx and re.search(r'\b' + re.escape(pidx) + r'\b',
                                   '\n'.join(body_lines)):
                lines.append(f'    final int {pidx} = me;')
            lines.extend(body_lines)
        # Non-void methods need a trailing `return result_name;` because
        # `_scan_result` strips the user-written trailing return so the
        # body emission has the same shape as the v0 single-guard form.
        if ret != 'void':
            if self.p.result_name is not None:
                lines.append(f'    return {self.p.result_name};')
            elif not (self.p.stmts
                      and self._always_returns(self.p.stmts[-1])):
                # No result variable, and the body can fall off the end --
                # every `return` is a literal inside a terminal rule.  Java
                # needs a default there; reaching it means no rule fired.
                lines.append(f'    return {java_zero(self.p.return_type)};')
        lines.append('  }')
        lines.append('')
        self._emit_step(lines)
        if self._uses_msgs:
            self._emit_primitive_stubs(lines)

    # Names that are generated inline rather than called.
    _PRIM_BUILTINS = frozenset(('min', 'max', 'abs'))
    _PRIM_RANK = {'void': 0, 'int': 1, 'boolean': 2}

    def _scan_exits(self):
        """The exit statements of the program, if any, and the Java type of
        the value they hand back.  An `exit` gives the program a result
        without the entry method having a return type, so the type is read
        off the expression."""
        found = []

        def walk(node):
            if node is None or not hasattr(node, 'kind'):
                return
            if isinstance(node, Stmt) and node.kind == 'exit':
                found.append(node)
            for fl in dataclasses.fields(node):
                v = getattr(node, fl.name, None)
                if isinstance(v, (Stmt, Expr)):
                    walk(v)
                elif isinstance(v, list):
                    for x in v:
                        if isinstance(x, (Stmt, Expr)):
                            walk(x)

        for s in self.p.stmts:
            walk(s)
        for m in self.p.aux_methods:
            for s in m.body:
                walk(s)
        if not found:
            return [], None
        t = None
        try:
            t = self._expr_type(found[0].return_value)
        except Exception:
            t = None
        return found, (java_base_type(t) if t is not None else 'boolean')

    def _scan_primitives(self) -> dict:
        """Names the program calls but does not define: the runtime
        primitives the surrounding system supplies -- logicalClock,
        neighbor(f), next(i), and so on.  The notes files have always
        listed these; emitting them as overridable stubs is what lets the
        class compile on its own and a subclass supply the real thing.

        The return type is read off the context of the call: used as a
        statement it is void, used as an operand of a logical connective it
        is boolean, anything else is int.  A name used in more than one
        context takes the most specific reading."""
        known = {m.name for m in self.p.aux_methods}
        known.add(self.p.entry_name)
        known.add(self.p.name)
        known |= set(self.signatures or {})
        known |= self._PRIM_BUILTINS
        found: dict = {}

        # Names bound by the enclosing `receive`, with the types the message
        # declares for them.  A binder does not exist outside its rule, so
        # _expr_type cannot see it, but the message declaration says exactly
        # what it is -- which is how an enum-valued payload keeps its type.
        binders: dict = {}
        enum_consts = {c: nm for nm, cs in (self.p.enums or {}).items()
                       for c in cs}

        def argtypes(args) -> List[str]:
            """Java types for a primitive's parameters, read off the
            arguments passed to it.  Anything not inferable is an int,
            which is what a process or resource index is."""
            out: List[str] = []
            for a in (args or []):
                nm = getattr(a, 'name', None)
                if getattr(a, 'kind', None) == 'id' and nm in binders:
                    out.append(binders[nm])
                    continue
                if getattr(a, 'kind', None) == 'id' and nm in enum_consts:
                    out.append(enum_consts[nm])
                    continue
                t = None
                try:
                    t = self._expr_type(a)
                except Exception:
                    t = None
                out.append(java_base_type(t) if t is not None else 'int')
            return out

        def note(name: str, args, ret: str) -> None:
            if not name or name in known or not name[:1].islower():
                return
            prev = found.get(name)
            ptypes = argtypes(args)
            if prev is None or self._PRIM_RANK[ret] > self._PRIM_RANK[prev[1]]:
                found[name] = (ptypes, ret)

        def children(node):
            for fl in dataclasses.fields(node):
                v = getattr(node, fl.name, None)
                if isinstance(v, (Stmt, Expr)):
                    yield fl.name, v
                elif isinstance(v, list):
                    for x in v:
                        if isinstance(x, (Stmt, Expr)):
                            yield fl.name, x

        def visit(node, hint: str = 'int') -> None:
            if isinstance(node, Expr) and node.kind == 'call':
                note(node.name, node.args, hint)
                for a in (node.args or []):
                    visit(a)
                return
            if isinstance(node, Stmt) and node.kind == 'call':
                note(node.call_name, node.call_args, 'void')
                for a in (node.call_args or []):
                    visit(a)
                return
            # `x = prim(...)` tells us the primitive's return type, which
            # neither the statement position nor a logical context would.
            if (isinstance(node, Stmt) and node.kind == 'assign'
                    and isinstance(node.rhs, Expr) and node.rhs.kind == 'call'):
                d = next((d for d in self.p.decls
                          if d.name == node.lhs_name), None)
                if d is not None:
                    t = d.type
                    if node.lhs_index is not None:
                        t = Type(base=t.base, is_list=t.is_list)
                    jt = java_base_type(t)
                    nm = node.rhs.name
                    if (nm and nm not in known and nm[:1].islower()):
                        found[nm] = (argtypes(node.rhs.args), jt)
                        for a in (node.rhs.args or []):
                            visit(a)
                        return
            bools: set = set()
            if isinstance(node, Expr):
                if node.kind == 'un' and node.op == 'not':
                    bools = {'lhs'}
                elif node.kind == 'bin' and node.op in ('and', 'or'):
                    bools = {'lhs', 'rhs'}
            elif isinstance(node, Stmt):
                if node.kind == 'if':
                    bools = {'cond'}
                elif node.kind == 'guard':
                    bools = {'guard_predicate'}
            added: List[str] = []
            if isinstance(node, Stmt) and getattr(node, 'recv_binders', None):
                flds = self.p.messages.get(node.recv_tag, []) or []
                for pos, b in enumerate(node.recv_binders):
                    if pos < len(flds):
                        binders[b] = java_base_type(flds[pos][0])
                        added.append(b)
            try:
                for fname, child in children(node):
                    visit(child, 'boolean' if fname in bools else 'int')
            finally:
                for b in added:
                    binders.pop(b, None)

        for s in self.p.stmts:
            visit(s)
        for m in self.p.aux_methods:
            for s in m.body:
                visit(s)
        return found

    def _emit_primitive_stubs(self, lines: List[str]) -> None:
        """Declare the primitives so the algorithm compiles by itself, and
        leave them overridable so a driver can supply the real ones."""
        prims = self._scan_primitives()
        if not prims:
            return
        lines.append('  // ---- runtime primitives ------------------------'
                     '-----------------')
        lines.append('  // Supplied by the system the algorithm runs in, not'
                     ' by the algorithm.')
        lines.append('  // Override these in a subclass.')
        for name in sorted(prims):
            ptypes, ret = prims[name]
            params = ', '.join(f'{t} a{k}' for k, t in enumerate(ptypes))
            body = ('throw new UnsupportedOperationException('
                    f'"supply {name}() in a subclass");')
            lines.append(f'  public {ret} {name}({params}) {{ {body} }}')
        lines.append('')

    def _emit_step(self, lines: List[str]) -> None:
        """One attempt at one rule, for a reactive program.

        Every mode is a loop around this: local mode picks a process and
        calls it, parallel mode gives each process a thread that calls it,
        distributed mode calls it on the one process the node hosts.  The
        rules are tried in source order and the first enabled one runs, so a
        single call is one atomic step of P_me."""
        if not (self._uses_msgs and self._step_guards):
            return
        pidx = self.p.process_index
        lo, hi = self._loop_lo, self._loop_hi
        lines.append('  /** Run one enabled rule of this process, if any.')
        lines.append('   *  Returns false when no rule is enabled. */')
        lines.append('  public boolean step() {')
        if self._scan_exits()[1] is not None:
            lines.append('    if (done) return false;   // exit() was reached')
        # Several guard ids can name the same merged method (see
        # _emit_merged_recv_group); call it once, at the first of them.
        called: set = set()
        for g in self._step_guards:
            g_id = getattr(g, '_codegen_id', None)
            indices = list(g.guard_index or ['j'])
            # An index that names the process itself is bound to `me`; any
            # other index is a resource of this process and is scanned.
            loops = [v for v in indices if v != pidx]
            if self._rule_scans.get(g_id):
                loops = []          # the rule scans for itself
            name = self._rule_method.get(g_id)
            if name is not None and name in called:
                continue
            pad = '    '
            for v in loops:
                lines.append(f'{pad}for (int {v} = {lo}; '
                             f'{v} {self._loop_cmp} {hi}; {v}++) {{')
                pad += '  '
            if name is not None:
                called.add(name)
                lines.append(f'{pad}if ({name}()) return true;')
            else:
                args = ', '.join('me' if v == pidx else v for v in indices)
                sets = ' '.join(f'this.{v} = {"me" if v == pidx else v};'
                                for v in indices)
                lines.append(f'{pad}if (_forbidden{g_id}({args})) {{')
                lines.append(f'{pad}  {sets}')
                lines.append(f'{pad}  _advance{g_id}();')
                lines.append(f'{pad}  return true;')
                lines.append(f'{pad}}}')
            for _ in loops:
                pad = pad[:-2]
                lines.append(f'{pad}}}')
        lines.append('    return false;')
        lines.append('  }')
        lines.append('')

    def _emit_list_array_alloc(self, target: str, d: 'Decl',
                               indent: str, decl: str = '') -> List[str]:
        """An arrayed list needs the array allocated and every slot filled
        with an empty list, the way an array of records does."""
        # Java forbids creating an array of a parameterized type, so the
        # `new` uses the raw LinkedList; the declared type keeps the element
        # type.  This is what the channel matrix does too.
        elem = 'LinkedList'
        size = 'n + 1' if self.p.base == 1 else 'n'
        lo = str(self.p.base)
        cmp = self._loop_cmp
        out: List[str] = []
        if d.type.is_2d:
            out.append(f'{indent}{decl}{target} = new {elem}[{size}][{size}];')
            out.append(f'{indent}for (int _u = {lo}; _u {cmp} n; _u++)')
            out.append(f'{indent}  for (int _w = {lo}; _w {cmp} n; _w++)')
            out.append(f'{indent}    {target}[_u][_w] = new LinkedList<>();')
        else:
            out.append(f'{indent}{decl}{target} = new {elem}[{size}];')
            out.append(f'{indent}for (int _u = {lo}; _u {cmp} n; _u++) '
                       f'{target}[_u] = new LinkedList<>();')
        return out

    def _emit_2d_alloc(self, target: str, d: 'Decl', indent: str,
                       decl: str = '') -> List[str]:
        """A 2D array declared with a scalar initialiser (`int[1..n][1..n] M = 0`)
        needs allocating and, when the value is not the Java default, filling.
        Without this the scalar is assigned straight to the array reference."""
        base = java_base_type(Type(d.type.base))
        size = 'n + 1' if self.p.base == 1 else 'n'
        out = [f'{indent}{decl}{target} = new {base}[{size}][{size}];']
        rhs = self._with_type(d.type, lambda: self.gen_expr(d.init))
        if rhs != java_zero(Type(d.type.base)):
            lo = str(self.p.base)
            cmp = self._loop_cmp
            out.append(f'{indent}for (int _u = {lo}; _u {cmp} n; _u++) '
                       f'for (int _v = {lo}; _v {cmp} n; _v++) '
                       f'{target}[_u][_v] = {rhs};')
        return out

    def _emit_local_decl(self, d: 'Decl', indent: str) -> List[str]:
        """Emit a `prog.decl` as a local variable inside the entry method.
        Mirrors the field-init logic from `_emit_ctors` but produces locals
        (no `this.`)."""
        out: List[str] = []
        init = d.init
        t = java_base_type(d.type)
        if d.type.is_list:
            if d.type.is_array or d.type.is_2d:
                return self._emit_list_array_alloc(d.name, d, indent,
                                                   decl=t + ' ')
            out.append(f'{indent}{t} {d.name} = new LinkedList<>();')
            return out
        if d.type.is_set or d.type.is_2d:
            if d.type.is_2d and not _is_wholesale_init(init) and init is not None:
                out.extend(self._emit_2d_alloc(d.name, d, indent, decl=t + ' '))
            elif init is not None:
                rhs = self._with_type(d.type, lambda: self.gen_expr(init))
                out.append(f'{indent}{t} {d.name} = {rhs};')
            else:
                out.append(f'{indent}{t} {d.name};')
        elif d.type.is_array:
            if init is not None and _is_wholesale_init(init):
                # Was `init.kind in ('method_call', 'new_array')` -- a
                # narrower, independent check than _is_wholesale_init
                # (used everywhere else in this file for the identical
                # question), missing 'call' in particular. That gap meant
                # `int[] r = Dijkstra(w, s);` here -- a decl promoted to
                # this path because it's the leading statement in its
                # method (see _is_field_start) -- fell through to the
                # scalar-broadcast branch below and tried to fill a
                # single-slot int[1] with an entire array value, a type
                # error. Confirmed fixed: this is exactly the shape
                # needed to let Johnson call Dijkstra instead of
                # duplicating it (chat-companion review item 2).
                rhs = self._with_type(d.type, lambda: self.gen_expr(init))
                out.append(f'{indent}{t} {d.name} = {rhs};')
            else:
                # No auto-bound `n` exists for a pure-method program (see
                # this method's docstring) -- a scalar-filled array local
                # here is always the "single register wrapped in an array"
                # idiom (only ever indexed at [0]), e.g. ClosestPair's
                # `double[] G = infinity;`.  One slot, index 0 directly.
                elem = java_base_type(Type(d.type.base, False))
                out.append(f'{indent}{t} {d.name} = new {elem}[1];')
                if init is not None:
                    rhs = self._with_type(d.type, lambda: self.gen_expr(init))
                    if rhs != java_zero(d.type):
                        out.append(f'{indent}{d.name}[0] = {rhs};')
        else:
            if init is not None:
                rhs = self._with_type(d.type, lambda: self.gen_expr(init))
                out.append(f'{indent}{t} {d.name} = {rhs};')
            else:
                out.append(f'{indent}{t} {d.name} = {java_zero(d.type)};')
        if d.type.base in ('int', 'double') and not d.type.is_array:
            self._q_scope.append(d.name)
        return out

    def _array_size_expr(self, t: 'Type') -> str:
        """The Java size to allocate an array field with, for the codegen
        default `this.arr = new T[SIZE]`. Ordinarily this is the implicit
        process-count `n` (`n + 1` when 1-indexed, so index n itself is
        valid), which is right for a message-passing program's own
        `[1..n]`-ranged fields. A `setStateful()` program's array field may
        instead be declared with an explicit literal-0-based range, e.g.
        `boolean[0..maxVal-1] A` sized by one of its own parameters rather
        than by a process count; in that case size it as declared
        (`(hi) + 1`) instead of guessing at `n`. Any other explicit range
        (nonzero lower bound) still falls back to `n`/`n + 1`: shifting
        every access to such an array by its lower bound is unimplemented.
        """
        if t.range1 is not None:
            lo, hi = t.range1
            if lo.kind == 'int' and lo.value == 0:
                return f'({self.gen_expr(hi)}) + 1'
        return 'n + 1' if self.p.base == 1 else 'n'

    def _init_body_lines(self, indent: str = '    ') -> List[str]:
        """Generate the one-time setup statements for a guard-bearing
        program: store every input param into its same-named field,
        compute `n`, allocate the state vector G (and any other field
        decls), and prime the linked-list / heap if the program uses
        a non-default mode.  These run once at the top of the entry
        method body."""
        body: List[str] = []
        for d in self.p.params:
            body.append(f'{indent}this.{d.name} = {d.name};')
        if self.primary_array is not None:
            suffix = ' - 1' if self.p.base == 1 else ''
            body.append(f'{indent}this.n = {self.primary_array.name}.length{suffix};')
        elif any(p.name == 'hi' for p in self.p.params):
            body.append(f'{indent}this.n = hi;')
        else:
            # No 1-D array parameter at all to size `n` from (e.g.
            # LLPShortestPath(int s, int[][] pre, int[][] w): every array
            # parameter is 2D) -- fall back to a 2D array's own outer
            # length, the same thing a classical (non-guard) method does
            # by hand (see GaleShapley.match's own `int n = mpref.length
            # - 1;`). Without this, `n` stays Java's default 0, and the
            # very next lines below allocate this program's own state
            # arrays (G, fixed, ...) at length 0 -- confirmed to crash on
            # the first array access.
            #
            # Restricted to a 2D array actually named `pre`, rather than
            # just the first 2D array found: tried unconditionally first,
            # and it newly turned ConstrainedStableMatching from a
            # harmless no-op (n stuck at 0, so its forbidden/advance loop
            # never ran at all) into a crash -- its forbidden predicate
            # is already documented, in its own header comment, as an
            # incomplete encoding (the book's third disjunct "is not
            # expressible in the LL compiler; it is omitted here"), and
            # empirically overruns its own preference lists under nearly
            # every tested input, not just a synthesized one. Waking up
            # a program the compiler's own comments admit is incomplete
            # is a regression, not a fix, so this stays scoped to the one
            # param name (`pre`) actually confirmed to need it.
            pre_2d = next((p for p in self.p.params
                           if p.type.is_2d and p.name == 'pre'), None)
            if pre_2d is not None:
                body.append(f'{indent}this.n = {pre_2d.name}.length;')
        for d in self.p.decls:
            init = d.init
            if d.type.is_list:
                if d.type.is_array or d.type.is_2d:
                    body.extend(self._emit_list_array_alloc(
                        f'this.{d.name}', d, indent))
                else:
                    body.append(f'{indent}this.{d.name} = new LinkedList<>();')
                continue
            if d.type.is_set or d.type.is_2d:
                if d.type.is_2d and not _is_wholesale_init(init) and init is not None:
                    body.extend(self._emit_2d_alloc(f'this.{d.name}', d, indent))
                elif init is not None:
                    rhs = self._with_type(d.type, lambda: self.gen_expr(init))
                    body.append(f'{indent}this.{d.name} = {rhs};')
                elif d.type.is_2d:
                    # No initialiser still needs the array itself; the
                    # elements keep Java's default.
                    base = java_base_type(Type(d.type.base))
                    size = self._array_size_expr(d.type)
                    body.append(f'{indent}this.{d.name} = '
                                f'new {base}[{size}][{size}];')
                continue
            if d.type.is_array:
                if _is_wholesale_init(init):
                    rhs = self._with_type(d.type, lambda: self.gen_expr(init))
                    body.append(f'{indent}this.{d.name} = {rhs};')
                else:
                    elem = java_base_type(Type(d.type.base, False))
                    arr_size = self._array_size_expr(d.type)
                    body.append(f'{indent}this.{d.name} = new {elem}[{arr_size}];')
                    body.extend(self._record_alloc(
                        f'this.{d.name}', elem, arr_size, indent))
                    if init is not None:
                        lo = str(self.p.base)
                        cmp = self._loop_cmp
                        if init.kind == 'id' and init.name == '_i':
                            body.append(
                                f'{indent}for (int i = {lo}; i {cmp} n; i++) '
                                f'this.{d.name}[i] = i;')
                        else:
                            rhs = self._with_type(d.type, lambda: self.gen_expr(init))
                            if rhs != java_zero(d.type):
                                body.append(
                                    f'{indent}for (int i = {lo}; i {cmp} n; i++) '
                                    f'this.{d.name}[i] = {rhs};')
            else:
                if init is not None:
                    rhs = self._with_type(d.type, lambda: self.gen_expr(init))
                    body.append(f'{indent}this.{d.name} = {rhs};')
                else:
                    body.append(
                        f'{indent}this.{d.name} = {java_zero(d.type)};')
        if self._uses_msgs:
            self._emit_chan_init(body, indent)
        return body

    def _mode_init_body_lines(self, indent: str = '    ') -> List[str]:
        """Mode-specific bookkeeping populated at the top of the entry
        method body (so the entry can be re-invoked on the same
        instance).  The constructor handles only param/state init."""
        body: List[str] = []
        if self.p.mode == 'linkedlist':
            body.append(f'{indent}forbiddenList = new LinkedList<>();')
            body.append(
                f'{indent}for (int j = {self._loop_lo}; j {self._loop_cmp} {self._loop_hi}; j++) '
                f'forbiddenList.add(j);')
        elif self.p.mode == 'heap':
            body.append(f'{indent}forbiddenHeap = new PriorityQueue<>(')
            if self.p.priority_order == 'max':
                body.append(
                    f'{indent}  (a, b) -> Double.compare(priority(b), priority(a)));')
            else:
                body.append(
                    f'{indent}  (a, b) -> Double.compare(priority(a), priority(b)));')
            body.append(
                f'{indent}for (int j = {self._loop_lo}; j {self._loop_cmp} {self._loop_hi}; j++) '
                f'forbiddenHeap.add(j);')
        elif self.p.mode == 'deque':
            body.append(f'{indent}forbiddenDeque = new ArrayDeque<>();')
            body.append(
                f'{indent}for (int j = {self._loop_lo}; j {self._loop_cmp} {self._loop_hi}; j++) '
                f'forbiddenDeque.add(j);')
        return body

    def _emit_ctors(self, lines: List[str]) -> None:
        """Emit zero, one, or two constructors.  The constructor stores the
        program's params as fields (so aux methods and quantifier helpers
        can see them) and runs one-time field initialisation.  Unlike the
        old design, it does NOT call the entry method -- callers do that
        themselves via `instance.<Name>(args)`.  If the body would be empty
        the constructor is omitted entirely (Java's default no-arg ctor
        suffices)."""
        def _emit(include_result: bool):
            ctor_decls = list(self.p.params)
            if include_result and self.p.result_name is not None:
                ctor_decls += [d for d in self.p.decls
                               if d.name == self.p.result_name]
            body: List[str] = []
            for d in ctor_decls:
                body.append(f'    this.{d.name} = {d.name};')
            if self._uses_msgs:
                self._emit_chan_init(body)
            if self.primary_array is not None:
                body.append(f'    this.n = {self.primary_array.name}.length - 1;')
                body.append('    if (n < 0) n = 0;')
            elif any(p.name == 'hi' for p in self.p.params):
                body.append('    this.n = hi;')
            else:
                # See the sibling ctor path (_init_body_lines) above for
                # why this fallback exists, and why it is restricted to a
                # 2D array actually named `pre` -- same reasoning, kept
                # consistent across both emitter modes.
                pre_2d = next((p for p in self.p.params
                               if p.type.is_2d and p.name == 'pre'), None)
                if pre_2d is not None:
                    body.append(f'    this.n = {pre_2d.name}.length;')
            for d in self.p.decls:
                if include_result and d.name == self.p.result_name:
                    continue
                init = d.init
                if d.type.is_list:
                    body.append(f'    this.{d.name} = new LinkedList<>();')
                    continue
                if d.type.is_set or d.type.is_2d:
                    if (d.type.is_2d and not _is_wholesale_init(init)
                            and init is not None):
                        body.extend(self._emit_2d_alloc(
                            f'this.{d.name}', d, '    '))
                    elif init is not None:
                        rhs = self._with_type(d.type, lambda: self.gen_expr(init))
                        body.append(f'    this.{d.name} = {rhs};')
                    continue
                if d.type.is_array:
                    # Three shapes for an array initializer:
                    #   (a) `T[] G = X.clone();` -- wholesale assignment.
                    #   (b) `T[] G;` or `T[] G = 0/false;` -- Java's default
                    #       zero/false init suffices.
                    #   (c) `T[] G = SCALAR;` (e.g., infinity) -- allocate
                    #       and fill per-element.
                    if _is_wholesale_init(init):
                        rhs = self._with_type(d.type, lambda: self.gen_expr(init))
                        body.append(f'    this.{d.name} = {rhs};')
                    else:
                        elem = java_base_type(Type(d.type.base, False))
                        body.append(
                            f'    this.{d.name} = new {elem}[n + 1];')
                        body.extend(self._record_alloc(
                            f'this.{d.name}', elem, 'n + 1', '    '))
                        if init is not None:
                            rhs = self._with_type(d.type, lambda: self.gen_expr(init))
                            if rhs != java_zero(d.type):
                                body.append(
                                    f'    for (int _i = 0; _i <= n; _i++) '
                                    f'this.{d.name}[_i] = {rhs};')
                else:
                    if init is not None:
                        rhs = self._with_type(d.type, lambda: self.gen_expr(init))
                        body.append(f'    this.{d.name} = {rhs};')
                    else:
                        body.append(
                            f'    this.{d.name} = {java_zero(d.type)};')
            if self._has_guards:
                if self.p.mode == 'linkedlist':
                    body.append('    forbiddenList = new LinkedList<>();')
                    body.append(
                        f'    for (int j = {self._loop_lo}; j <= {self._loop_hi}; j++) '
                        'forbiddenList.add(j);')
                elif self.p.mode == 'array':
                    body.append('    forbiddenArr = new boolean[n + 1];')
                elif self.p.mode == 'heap':
                    body.append('    forbiddenHeap = new PriorityQueue<>(')
                    if self.p.priority_order == 'max':
                        body.append(
                            '      (a, b) -> Double.compare(priority(b), priority(a)));')
                    else:
                        body.append(
                            '      (a, b) -> Double.compare(priority(a), priority(b)));')
                    body.append(
                        f'    for (int j = {self._loop_lo}; j <= {self._loop_hi}; j++) '
                        'forbiddenHeap.add(j);')
            # If the constructor would be entirely empty, omit it -- Java's
            # default no-arg constructor is sufficient.
            if not body and not ctor_decls:
                return
            sig = ', '.join(
                f'{java_base_type(d.type)} {d.name}' for d in ctor_decls)
            lines.append(f'  public {self.p.name}({sig}) {{')
            lines.extend(body)
            lines.append('  }')
            lines.append('')
        _emit(include_result=False)
        # Emit a subroutine ctor only when the result is a `var` (not a
        # param) -- params are already passed in by the top-level ctor.
        result_is_var = self.p.result_name is not None and any(
            d.name == self.p.result_name for d in self.p.decls)
        if (self.p.params and result_is_var
                and self._is_called_elsewhere()):
            _emit(include_result=True)

    @staticmethod
    def _idx_params(indices: List[str]) -> str:
        """Build Java parameter list from guard indices, e.g. 'int i, int j'."""
        return ', '.join(f'int {v}' for v in indices)

    def _emit_always_vars(self, lines: List[str], g: Stmt,
                          indent: str = '    ') -> None:
        """Emit local variable assignments for any `always` clauses."""
        for atype, aname, aexpr in g.guard_always:
            jtype = java_base_type(atype)
            lines.append(f'{indent}{jtype} {aname} = {self.gen_expr(aexpr)};')

    @staticmethod
    def _is_recv_guard(g: Stmt) -> bool:
        return bool(g.recv_tag is not None or g.recv_binders or g.recv_from
                    or g.recv_from_const is not None)

    def _emit_single_guard_methods(self, lines: List[str], g: Stmt) -> None:
        """Emit `forbidden(j)` + `advance(j)` for a program whose body is
        exactly one guard statement.

        For guards with a `pick X` clause, `forbidden` returns boolean,
        writing the chosen witness into the class field `picked_X` on
        success.  `advance` takes only the forbidden index `j`; it reads
        the witness from `picked_X` (aliased as `X` at the top of the
        body so the user's advance code reads naturally)."""
        indices = g.guard_index
        idx_params = self._idx_params(indices)
        self._q_scope = list(indices)
        if g.guard_pick_var is not None:
            wvar = g.guard_pick_var
            quant = g.guard_predicate
            lines.append(f'  private boolean forbidden({idx_params}) {{')
            self._emit_always_vars(lines, g)
            wt = ('int' if quant.q_set is None
                  else self._elem_java_type(quant.q_set))
            self._scope_types[wvar] = wt
            self._q_scope.append(wvar)
            try:
                body_expr = self.gen_expr(quant.q_body)
            finally:
                self._q_scope.pop()
            if quant.q_set is not None:
                setexpr = self.gen_expr(quant.q_set)
                lines.append(f'    for ({wt} {wvar} : {setexpr}) {{')
            else:
                lo = self.gen_expr(quant.q_lo)
                test = self._for_loop_test(wvar, quant.q_hi)
                lines.append(
                    f'    for (int {wvar} = {lo}; {test}; {wvar}++) {{')
            lines.append(f'      if ({body_expr}) {{ this.picked_{wvar} = {wvar}; return true; }}')
            lines.append('    }')
            lines.append('    return false;')
            lines.append('  }')
            lines.append('')
            ret = self._advance_returns(g)
            sig = 'boolean' if self._void_stop_mode else (self._boxed_result() if ret else 'void')
            lines.append(f'  private {sig} advance() {{')
            self._emit_always_vars(lines, g)
            lines.append(f'    {wt} {wvar} = picked_{wvar};')
            self._q_scope.append(wvar)
            try:
                lines.extend(self._gen_guard_action(g.guard_advance, '    ', void=not ret))
            finally:
                self._q_scope.pop()
            if (ret or self._void_stop_mode) and self._advance_falls_through(g):
                lines.append(f'    return {"false" if self._void_stop_mode else self._null_signal()};')
            lines.append('  }')
            lines.append('')
        else:
            lines.append(f'  private boolean forbidden({idx_params}) {{')
            self._emit_always_vars(lines, g)
            body: List[str] = []
            if not self._emit_pred_procedural(g.guard_predicate, body, '    '):
                pre: List[str] = []
                new_pred = self._inline_quant_in_expr(
                    g.guard_predicate, pre, '    ')
                body.extend(pre)
                body.append(f'    return {self.gen_expr(new_pred)};')
            lines.extend(body)
            lines.append('  }')
            lines.append('')
            ret = self._advance_returns(g)
            sig = 'boolean' if self._void_stop_mode else (self._boxed_result() if ret else 'void')
            lines.append(f'  private {sig} advance() {{')
            self._emit_always_vars(lines, g)
            lines.extend(self._gen_guard_action(g.guard_advance, '    ', void=not ret))
            if (ret or self._void_stop_mode) and self._advance_falls_through(g):
                lines.append(f'    return {"false" if self._void_stop_mode else self._null_signal()};')
            lines.append('  }')
            lines.append('')

    @staticmethod
    def _stmt_returns(st: Stmt) -> bool:
        """True if a statement can execute a `return`, anywhere inside it."""
        if st is None:
            return False
        if st.kind == 'return':
            return True
        for sub in (st.body, st.then_stmts, st.else_stmts, st.par_branches):
            if sub:
                for x in sub:
                    if CodeGen._stmt_returns(x):
                        return True
        if st.guard_advance is not None and CodeGen._stmt_returns(st.guard_advance):
            return True
        return False

    @staticmethod
    def _always_returns(st: Stmt) -> bool:
        """True if every path through a statement ends in a `return`, so a
        following `return null;` would be unreachable."""
        if st is None:
            return False
        if st.kind == 'return':
            return True
        if st.kind == 'block':
            return bool(st.body) and CodeGen._always_returns(st.body[-1])
        if st.kind == 'if':
            return (bool(st.then_stmts) and bool(st.else_stmts)
                    and CodeGen._always_returns(st.then_stmts[-1])
                    and CodeGen._always_returns(st.else_stmts[-1]))
        return False

    def _advance_falls_through(self, g: Stmt) -> bool:
        return not self._always_returns(g.guard_advance)

    @staticmethod
    def _contains_explicit_null_return(st: Stmt) -> bool:
        """True if `return null;` (LL's "infeasible" result sentinel)
        appears anywhere inside st -- see _needs_no_exit_sentinel, which
        this drives, for why it matters."""
        if st is None:
            return False
        if st.kind == 'return':
            return (st.return_value is not None and st.return_value.kind == 'id'
                    and st.return_value.name == 'null')
        for sub in (st.body, st.then_stmts, st.else_stmts, st.par_branches):
            if sub:
                for x in sub:
                    if CodeGen._contains_explicit_null_return(x):
                        return True
        return False

    def _null_signal(self) -> str:
        """The Java value `advance()`'s callers compare against to detect
        "an early exit happened", and the value `advance()` itself falls
        back to when it completes without an explicit return (both must
        use the *same* value, by construction).

        Normally that value is literal `null`. But when the program's own
        advance body has an explicit `return null;` -- LL's spelling for
        "the algorithm's real result is null, stop now" (e.g. Mandatory:
        "return null" means the interval set is infeasible) -- literal
        `null` can no longer serve double duty: a genuine infeasible
        result and "nothing happened, keep looping" would both be `null`
        and indistinguishable, so the caller's `if (_r != null) return
        _r;` never fires for the *real* early exit either. Confirmed live:
        this exact collision made Mandatory/MandatoryEdges/
        MandatoryIntervals/MandatoryMST hang forever -- the fixpoint loop
        could never see that an early exit had actually happened.

        The fix is a dedicated sentinel object, compared by reference
        (`!=`), reserved for "no early exit" -- so literal `null` is free
        to mean only "the real result is null" again. See
        _no_exit_sentinel_field for where it's declared."""
        return '_NO_EARLY_EXIT' if self._needs_no_exit_sentinel else 'null'

    def _no_exit_sentinel_field(self) -> Optional[str]:
        """The static field declaration for _null_signal()'s sentinel, or
        None if this program doesn't need one. Only array/2D/set return
        types are supported (an empty array of the right element type,
        distinguished from any real result by reference identity, `!=`,
        never by content) -- the only shape `return null;` actually
        appears with anywhere in the current corpus; anything else falls
        back to the pre-existing plain-`null` convention untouched."""
        if not self._needs_no_exit_sentinel:
            return None
        t = self.p.return_type
        if t is None or not (t.is_array or t.is_2d or t.is_set):
            return None
        jt = java_base_type(t)
        dims = jt.count('[]')
        base = jt.replace('[]', '')
        empty_dims = '[0]' + '[]' * (dims - 1)
        return f'  private static final {jt} _NO_EARLY_EXIT = new {base}{empty_dims};'

    def _advance_returns(self, g: Stmt) -> bool:
        """True if this guard's advance returns a *result*, so the rule is
        terminal: reaching it ends the program rather than continuing the
        fixpoint.

        A void program has no result to hand back, so a `return` in its
        action is just an early exit from the action and the advance stays
        void."""
        return (self.p.return_type is not None
                and self._stmt_returns(g.guard_advance))

    def _boxed_result(self) -> str:
        """Java type an advance uses to report `no result yet` (null) or a
        value.  Boxed so null is available."""
        t = self.p.return_type
        if t is None:
            return 'Boolean'
        if t.is_array or t.is_2d or t.is_set:
            # An array reference is already nullable; boxing its element type
            # would give a type the value cannot be assigned to.
            return java_base_type(t)
        return {'int': 'Integer', 'double': 'Double',
                'boolean': 'Boolean'}.get(t.base, t.base)

    def _gen_guard_action(self, adv: Stmt, indent: str,
                          void: bool = True) -> List[str]:
        """Emit the body of a guard's advance action.  If `adv` is a block,
        its inner stmts are flattened (no extra braces) so the method body
        reads cleanly.  `void` says whether the enclosing advance method is
        void, which decides how a `return` inside it is emitted."""
        prev = self._in_void_method
        prev_stop = self._void_stop_signal
        self._in_void_method = void
        self._void_stop_signal = void and self._void_stop_mode
        try:
            if adv.kind == 'block':
                return self.gen_stmts(adv.body, indent)
            return self.gen_stmt(adv, indent)
        finally:
            self._in_void_method = prev
            self._void_stop_signal = prev_stop

    def _emit_all_guard_helpers(self, lines: List[str], stmts: List[Stmt]) -> None:
        """Assign each guard in the program a numeric id K and emit a
        corresponding `_forbiddenK(j)` / `_advanceK(j)` pair.  The emitted
        code for the guard stmt itself (in _emit_run) invokes these."""
        guards = list(self._walk_guards(stmts))
        for g in guards:
            g_id = self._guard_count
            self._guard_count += 1
            g._codegen_id = g_id

        # A channel holds at most one message at a time, so every receive
        # rule that would scan the same senders can be answered by a single
        # peek: switch on the tag once, instead of each rule re-scanning the
        # same senders looking for its own.  This is what turns three
        # separate onX()/onY()/onZ() scans into the one dispatch a
        # message-driven runtime would give the algorithm for free.
        grouped_ids = set()
        if self._uses_msgs:
            groups: dict = {}
            order: list = []
            for g in guards:
                if not self._is_recv_guard(g) or self._advance_returns(g):
                    continue          # terminal rules keep the old pair
                key = ('fixed', g.recv_from_const) if g.recv_from_const is not None \
                    else ('var', g.recv_from or '_src')
                if key not in groups:
                    groups[key] = []
                    order.append(key)
                groups[key].append(g)
            used_names: set = set()
            for key in order:
                group = groups[key]
                name = self._merged_recv_name(group, used_names)
                self._emit_merged_recv_group(lines, group, name)
                for g in group:
                    self._rule_method[g._codegen_id] = name
                    self._rule_scans[g._codegen_id] = True
                    grouped_ids.add(g._codegen_id)

        for g in guards:
            if g._codegen_id in grouped_ids:
                continue
            g_id = g._codegen_id
            indices = g.guard_index
            idx_params = self._idx_params(indices)
            self._q_scope = list(indices)
            # Inside a guard, "me" is its first index -- that is what a
            # `send` in the action uses as the sender.
            self._cur_self = indices[0]
            if self._is_recv_guard(g):
                self._emit_receive_guard(lines, g, g_id, indices, idx_params)
            elif (self._uses_msgs
                  and self._emit_fused_pred_rule(lines, g, g_id, indices)):
                pass
            elif g.guard_pick_var is not None:
                wvar = g.guard_pick_var
                quant = g.guard_predicate
                lines.append(f'  private boolean _forbidden{g_id}({idx_params}) {{')
                self._emit_always_vars(lines, g)
                wt = ('int' if quant.q_set is None
                      else self._elem_java_type(quant.q_set))
                self._scope_types[wvar] = wt
                self._q_scope.append(wvar)
                try:
                    body_expr = self.gen_expr(quant.q_body)
                finally:
                    self._q_scope.pop()
                if quant.q_set is not None:
                    setexpr = self.gen_expr(quant.q_set)
                    wt = self._elem_java_type(quant.q_set)
                    self._scope_types[wvar] = wt
                    lines.append(f'    for ({wt} {wvar} : {setexpr}) {{')
                else:
                    lo = self.gen_expr(quant.q_lo)
                    test = self._for_loop_test(wvar, quant.q_hi)
                    lines.append(
                        f'    for (int {wvar} = {lo}; {test}; {wvar}++) {{')
                lines.append(f'      if ({body_expr}) {{ this.picked_{wvar} = {wvar}; return true; }}')
                lines.append('    }')
                lines.append('    return false;')
                lines.append('  }')
                lines.append('')
                ret = self._advance_returns(g)
                sig = 'boolean' if self._void_stop_mode else (self._boxed_result() if ret else 'void')
                lines.append(f'  private {sig} _advance{g_id}() {{')
                self._emit_always_vars(lines, g)
                lines.append(f'    {wt} {wvar} = picked_{wvar};')
                self._q_scope.append(wvar)
                try:
                    lines.extend(self._gen_guard_action(g.guard_advance, '    ', void=not ret))
                finally:
                    self._q_scope.pop()
                if (ret or self._void_stop_mode) and self._advance_falls_through(g):
                    # Fell through without returning: no result yet.
                    lines.append(f'    return {"false" if self._void_stop_mode else self._null_signal()};')
                lines.append('  }')
                lines.append('')
            else:
                lines.append(f'  private boolean _forbidden{g_id}({idx_params}) {{')
                self._emit_always_vars(lines, g)
                body: List[str] = []
                if not self._emit_pred_procedural(g.guard_predicate, body, '    '):
                    pre: List[str] = []
                    new_pred = self._inline_quant_in_expr(
                        g.guard_predicate, pre, '    ')
                    body.extend(pre)
                    body.append(f'    return {self.gen_expr(new_pred)};')
                lines.extend(body)
                lines.append('  }')
                lines.append('')
                ret = self._advance_returns(g)
                sig = 'boolean' if self._void_stop_mode else (self._boxed_result() if ret else 'void')
                lines.append(f'  private {sig} _advance{g_id}() {{')
                self._emit_always_vars(lines, g)
                lines.extend(self._gen_guard_action(g.guard_advance, '    ', void=not ret))
                if (ret or self._void_stop_mode) and self._advance_falls_through(g):
                    # Fell through without returning: no result yet.
                    lines.append(f'    return {"false" if self._void_stop_mode else self._null_signal()};')
                lines.append('  }')
                lines.append('')



    def _try_inline_pick_if(self, s: 'Stmt', indent: str):
        """`if (exists v in S pick v : P) { ... }` as a plain search.

        The witness is a local the search fills in, so nothing has to travel
        through a `picked_v' field and no helper method is needed.  Works over
        a collection (absent witness is null) and over a literal range (absent
        witness is one below the low end)."""
        c = s.cond
        if (c is None or c.kind != 'quant' or c.quant != 'exists'
                or getattr(c, 'pick_var', None) is None):
            return None
        v = c.pick_var
        if c.q_set is not None:
            et = self._elem_java_type(c.q_set)
            src = f'for ({et} _x : {self.gen_expr(c.q_set)})'
            none, some = 'null', f'{v} != null'
        else:
            lo = self.gen_expr(c.q_lo)
            if not re.fullmatch(r'-?\d+', lo):
                return None      # no sentinel below a computed low end
            et = 'int'
            test = self._for_loop_test('_x', c.q_hi)
            src = f'for (int _x = {lo}; {test}; _x++)'
            none, some = str(int(lo) - 1), f'{v} >= {lo}'
        self._scope_types[v] = et
        self._q_scope.append(v)
        try:
            pred = self.gen_expr(c.q_body)
            body = self.gen_stmts(s.then_stmts, indent + '  ')
            alt = self.gen_stmts(s.else_stmts, indent + '  ')
        finally:
            self._q_scope.pop()
        # Every mention of the pick variable becomes the candidate, including a
        # bare one handed to a lifted quantifier helper -- `_q1(i, w)'.
        cand = re.sub(r'(?<![\w.])' + re.escape(v) + r'(?![\w])', '_x', pred)
        out = [f'{indent}{et} {v} = {none};',
               f'{indent}{src} if ({cand}) {{ {v} = _x; break; }}',
               f'{indent}if ({some}) {{']
        out += body
        if s.else_stmts:
            out.append(f'{indent}}} else {{')
            out += alt
        out.append(f'{indent}}}')
        return out

    def _upon_source(self, g) -> List[str]:
        """The `upon' this rule was written as, for its doc comment.  A
        predicate has no tag to name it after, so the source says it best.

        A predicate broken over several source lines is quoted over as many
        comment lines, so the rule is never shown half-stated."""
        ln = getattr(g, 'guard_line', None)
        if not ln or not self.src_text:
            return []
        src = self.src_text.split('\n')
        # The predicate's first token can sit a line below the `upon', as in
        # `upon exists w in reqlist pick w :' broken across lines.
        start = None
        for k in range(ln - 1, max(ln - 3, 0) - 1, -1):
            if 0 <= k < len(src) and 'upon' in src[k]:
                start = k
                break
        if start is None:
            return []
        out: List[str] = []
        for k in range(start, min(start + 5, len(src))):
            t = src[k].rstrip()
            done = t.endswith('{')
            t = t.rstrip('{').rstrip()
            if t:
                out.append(t.strip() if k == start else '    ' + t.strip())
            if done:
                return out
        return out[:1]          # no action brace in sight; quote the head only

    def _rule_name(self, g, g_id: int) -> str:
        """A predicate rule has nothing to name it after, so it is numbered
        among the predicate rules -- not after the order the compiler
        happened to walk all the guards in."""
        k = getattr(self, '_pred_rule_count', 0) + 1
        self._pred_rule_count = k
        return f'rule{k}'

    def _emit_fused_pred_rule(self, lines, g, g_id: int, indices) -> bool:
        """One method per predicate rule: find an index the guard holds for,
        then act on it.

        The guard/action split made the two halves communicate through an
        instance field -- `this.f = f' in step(), read back as a field in the
        action -- and put the scan in a third place again.  Fused, the index
        is an ordinary loop variable and the rule is one piece of code.

        Returns False for the shapes that still need the pair."""
        if self._advance_returns(g):
            return False                      # terminal rules keep the pair
        pidx = self.p.process_index
        idx = list(indices or [])
        loops = [v for v in idx if v != pidx]
        if len(loops) > 1:
            return False                      # a multi-index scan keeps the pair
        pick = g.guard_pick_var
        quant = g.guard_predicate
        if pick is not None and (quant is None or quant.q_set is None):
            return False        # a pick over a range has no null to test for
        lo, hi = self._loop_lo, self._loop_hi
        ind = '      ' if loops else '    '
        stop = 'continue' if loops else 'return false'

        head: List[str] = []
        body: List[str] = []
        ok = True
        if pick is not None:
            wt = self._elem_java_type(quant.q_set)
            setexpr = self.gen_expr(quant.q_set)
            self._scope_types[pick] = wt
            self._q_scope.append(pick)
            try:
                pred = self.gen_expr(quant.q_body)
            finally:
                self._q_scope.pop()
            cand = re.sub(r'(?<![\w.])' + re.escape(pick) + r'(?![\w])',
                          '_x', pred)
            # A primitive element type (set<int>/set<double>/set<boolean> --
            # _elem_java_type does not box these the way list<T> is boxed)
            # has no null to test for "not found"; use a boolean flag
            # instead of the reference-type null-sentinel below.
            _prim_default = {'int': '0', 'double': '0.0', 'boolean': 'false'}
            if wt in _prim_default:
                found = f'_{pick}_found'
                body.append(f'{ind}{wt} {pick} = {_prim_default[wt]};')
                body.append(f'{ind}boolean {found} = false;')
                body.append(f'{ind}for ({wt} _x : {setexpr}) '
                            f'if ({cand}) {{ {pick} = _x; {found} = true; break; }}')
                body.append(f'{ind}if (!{found}) {stop};')
            else:
                body.append(f'{ind}{wt} {pick} = null;')
                body.append(f'{ind}for ({wt} _x : {setexpr}) '
                            f'if ({cand}) {{ {pick} = _x; break; }}')
                body.append(f'{ind}if ({pick} == null) {stop};')
            self._q_scope.append(pick)
        else:
            if g.guard_predicate is not None:
                if not self._emit_pred_procedural(g.guard_predicate, body, ind,
                                                  stop=stop, final_true=False):
                    when_pre: List[str] = []
                    w = self._inline_quant_in_expr(g.guard_predicate,
                                                   when_pre, ind)
                    body.extend(when_pre)
                    body.append(f'{ind}if (!({self.gen_expr(w)})) {stop};')
        try:
            act = self._gen_guard_action(g.guard_advance, ind, void=True)
        finally:
            if pick is not None:
                self._q_scope.pop()
        if not ok:
            return False

        name = self._rule_name(g, g_id)
        self._rule_method[g_id] = name
        self._rule_scans[g_id] = bool(loops)   # the rule owns its own scan now
        out: List[str] = []
        upon = self._upon_source(g) or ["an `upon' rule"]
        if len(upon) == 1:
            out.append(f'  /** {upon[0]} */')
        else:
            out.append(f'  /** {upon[0]}')
            out.extend(f'   *  {t}' for t in upon[1:-1])
            out.append(f'   *  {upon[-1]} */')
        out.append(f'  private boolean {name}() {{')
        self._emit_always_vars(head, g, ind)
        if loops:
            v = loops[0]
            out.append(f'    for (int {v} = {lo}; '
                       f'{v} {self._loop_cmp} {hi}; {v}++) {{')
        out.extend(head)
        out.extend(body)
        out.extend(act)
        out.append(f'{ind}return true;')
        if loops:
            out.append('    }')
            out.append('    return false;')
        out.append('  }')
        out.append('')
        if pidx and re.search(r'\b' + re.escape(pidx) + r'\b',
                              '\n'.join(head + body + act)):
            at = out.index(f'  private boolean {name}() {{') + 1
            out.insert(at, f'    final int {pidx} = me;')
        lines.extend(out)
        return True

    def _merged_recv_name(self, group, used: set) -> str:
        """A name for the method serving one whole group of receive rules.

        A group of one is named after what it responds to, same as any
        single rule always has been.  A group of several answers for more
        than one tag, so no such name fits it; it is named after what it
        scans instead, which is the only thing its members still share."""
        if len(group) == 1:
            base = 'on' + group[0].recv_tag.capitalize()
        else:
            fixed = group[0].recv_from_const
            base = f'_recvFrom{fixed}' if fixed is not None else '_recv'
        name, k = base, 2
        while name in used:
            name = f'{base}{k}'
            k += 1
        used.add(name)
        return name

    def _emit_single_recv_rule(self, lines, g, name: str, fixed, src: str,
                              ind: str, stop: str) -> None:
        """The flat form for a rule that shares its senders with no one:
        check the one tag, act.  No switch, because there is nothing to
        switch between."""
        pidx = self.p.process_index
        flds = self.p.messages.get(g.recv_tag, []) if g.recv_tag else []
        binders = list(g.recv_binders)
        btypes = {b: (java_base_type(flds[k][0]) if k < len(flds) else 'int')
                  for k, b in enumerate(binders)}
        self._binder_types = {b: flds[k][0] for k, b in enumerate(binders)
                              if k < len(flds)}
        self._scope_types.update(btypes)
        added = list(binders) + ([] if fixed is not None else [src])
        self._q_scope.extend(added)
        prev_self = getattr(self, '_cur_self', None)
        self._cur_self = 'me'
        try:
            when, when_pre = None, []
            if g.guard_predicate is not None:
                w = self._inline_quant_in_expr(g.guard_predicate, when_pre, ind)
                when = self.gen_expr(w)
            act = self._gen_guard_action(g.guard_advance, ind, void=True)
        finally:
            self._cur_self = prev_self
            del self._q_scope[len(self._q_scope) - len(added):]
            for b in binders:
                self._scope_types.pop(b, None)
            self._binder_types = {}

        out: List[str] = []
        out.append(f'  /** upon receive({g.recv_tag}'
                   + (', ' + ', '.join(binders) if binders else '')
                   + f') from {"P" + str(fixed) if fixed is not None else src} */')
        out.append(f'  private boolean {name}() {{')
        if fixed is None:
            lo, hi = self._loop_lo, self._loop_hi
            out.append(f'    for (int {src} = {lo}; {src} {self._loop_cmp} {hi}; {src}++) {{')
        out.append(f'{ind}MsgQ _ch = chan[{src}][me];')
        out.append(f'{ind}if (_ch.isEmpty()) {stop};')
        out.append(f'{ind}Msg _m = _ch.peek();')
        out.append(f'{ind}if (!_m.tag.equals("{g.recv_tag}")) {stop};')
        for pos, b in enumerate(binders):
            out.append(f'{ind}{btypes[b]} {b} = ({btypes[b]}) _m.p[{pos}];')
        if when is not None:
            out.extend(when_pre)
            out.append(f'{ind}if (!({when})) {stop};')
        out.append(f'{ind}_ch.poll();')
        out.extend(act)
        out.append(f'{ind}return true;')
        if fixed is None:
            out.append('    }')
            out.append('    return false;')
        out.append('  }')
        out.append('')
        if re.search(r'\b' + re.escape(pidx) + r'\b',
                     '\n'.join(act + ([when] if when else []))):
            at = out.index(f'  private boolean {name}() {{') + 1
            out.insert(at, f'    final int {pidx} = me;')
        lines.extend(out)

    def _emit_merged_recv_group(self, lines, group, name: str) -> None:
        """One scan of the channels serves every rule in the group: a channel
        holds at most one message at a time, so peeking it once and switching
        on the tag finds whichever rule (if any) is ready.  This is the
        dispatch a message-driven runtime -- a `handleMsg(msg, src, tag)`
        called once per arrival -- gives an algorithm for free; LL cannot use
        that directly because `step()` must be askable rather than blocking
        (local mode interleaves many processes on one thread), but nothing
        stops every rule that would poll the same senders from sharing the
        one poll."""
        fixed = group[0].recv_from_const
        pidx = self.p.process_index
        src = str(fixed) if fixed is not None else (group[0].recv_from or '_src')
        ind = '    ' if fixed is not None else '      '
        stop = 'return false' if fixed is not None else 'continue'

        if len(group) == 1:
            # Nothing else polls these senders, so a switch would only add
            # its own scaffolding with nothing to share it with; the plain
            # single-rule form (check the one tag, act) says the same thing
            # in less code.
            self._emit_single_recv_rule(lines, group[0], name, fixed, src, ind, stop)
            return

        # Two rules can share a tag, told apart only by `when` -- SynchronousOrder
        # has an `upon receive(MSG, m) from j when j > i ...` and another `when
        # j < i`.  A Java switch cannot repeat a case label, so rules sharing a
        # tag become one case that tries them in source order, exactly the
        # if/else-if a hand-written dispatch would use.
        by_tag: dict = {}
        tag_order: list = []
        for g in group:
            if g.recv_tag not in by_tag:
                by_tag[g.recv_tag] = []
                tag_order.append(g.recv_tag)
            by_tag[g.recv_tag].append(g)

        cases: List[str] = []
        needs_me_idx = False
        for tag in tag_order:
            rules = by_tag[tag]
            # A rule alone on its tag reads flatly, as a single fused rule
            # always has.  One of several sharing a tag gets its own bare
            # block: its guard may read its binders (so they must be declared
            # before it, not after), and a sibling rule can bind the same
            # name (both of SynchronousOrder's MSG rules bind `m`), so each
            # needs a scope of its own.  `break` inside that block still
            # targets the switch, not the block, which is why a failed guard
            # here is a plain `if', not a `break' -- falling out of the `if'
            # already reaches the next rule's block.
            nested = len(rules) > 1
            bare_ind = ind + '  '                          # the bare block's braces
            decl_ind = ind + ('    ' if nested else '  ')  # binders, when, the `if'
            body_ind = ind + ('      ' if nested else '  ')  # poll, action, return
            branches: List[str] = []
            for g in rules:
                flds = self.p.messages.get(g.recv_tag, []) if g.recv_tag else []
                binders = list(g.recv_binders)
                btypes = {b: (java_base_type(flds[k][0]) if k < len(flds) else 'int')
                          for k, b in enumerate(binders)}
                self._binder_types = {b: flds[k][0] for k, b in enumerate(binders)
                                      if k < len(flds)}
                self._scope_types.update(btypes)
                added = list(binders) + ([] if fixed is not None else [src])
                self._q_scope.extend(added)
                prev_self = getattr(self, '_cur_self', None)
                self._cur_self = 'me'
                try:
                    when, when_pre = None, []
                    if g.guard_predicate is not None:
                        w = self._inline_quant_in_expr(g.guard_predicate, when_pre,
                                                        decl_ind)
                        when = self.gen_expr(w)
                    act = self._gen_guard_action(g.guard_advance, body_ind, void=True)
                finally:
                    self._cur_self = prev_self
                    del self._q_scope[len(self._q_scope) - len(added):]
                    for b in binders:
                        self._scope_types.pop(b, None)
                    self._binder_types = {}

                if re.search(r'\b' + re.escape(pidx) + r'\b',
                            '\n'.join(act + ([when] if when else []))):
                    needs_me_idx = True

                branch: List[str] = []
                branch.append(f'{bare_ind if nested else decl_ind}// upon receive({g.recv_tag}'
                             + (', ' + ', '.join(binders) if binders else '')
                             + f') from {"P" + str(fixed) if fixed is not None else src}')
                if nested:
                    branch.append(f'{bare_ind}{{')     # this rule's own scope
                for pos, b in enumerate(binders):
                    branch.append(f'{decl_ind}{btypes[b]} {b} = ({btypes[b]}) _m.p[{pos}];')
                # The guard can read the binders just declared -- CausalOrder's
                # `when' does -- so it is checked after them, never before.
                branch.extend(when_pre)
                if nested:
                    branch.append(f'{decl_ind}if ({when if when is not None else "true"}) {{')
                elif when is not None:
                    branch.append(f'{decl_ind}if (!({when})) break;')
                branch.append(f'{body_ind}_ch.poll();')
                branch.extend(act)
                branch.append(f'{body_ind}return true;')
                if nested:
                    branch.append(f'{decl_ind}}}')
                    branch.append(f'{bare_ind}}}')
                branches.append('\n'.join(branch))

            case: List[str] = []
            case.append(f'{ind}case "{tag}": {{')
            case.extend(branches)
            if nested:
                case.append(f'{ind}  break;'
                           + '   // no guard held yet -- try the next sender')
            case.append(f'{ind}}}')
            cases.append('\n'.join(case))

        tags = ', '.join(g.recv_tag for g in group)
        out: List[str] = []
        out.append('  /** the one message ready from '
                   + (f'P{fixed}' if fixed is not None else 'whichever sender has one')
                   + f': {tags} */')
        out.append(f'  private boolean {name}() {{')
        if needs_me_idx:
            out.append(f'    final int {pidx} = me;')
        if fixed is None:
            lo, hi = self._loop_lo, self._loop_hi
            out.append(f'    for (int {src} = {lo}; {src} {self._loop_cmp} {hi}; {src}++) {{')
        out.append(f'{ind}MsgQ _ch = chan[{src}][me];')
        out.append(f'{ind}if (_ch.isEmpty()) {stop};')
        out.append(f'{ind}Msg _m = _ch.peek();')
        out.append(f'{ind}switch (_m.tag) {{')
        out.extend(cases)
        out.append(f'{ind}  default: break;   // not one of ours -- try the next sender')
        out.append(f'{ind}}}')
        if fixed is None:
            out.append('    }')
            out.append('    return false;')
        else:
            out.append(f'{ind}return false;')
        out.append('  }')
        out.append('')
        lines.extend(out)

    def _emit_receive_guard(self, lines: List[str], g, g_id: int,
                            indices: List[str], idx_params: str) -> None:
        """Emit the forbidden/advance pair for `upon receive(TAG, b...) from k`.

        forbidden(j) scans the senders for a queued message with the right
        tag and stashes the witness, exactly as a `pick` guard does; advance
        polls that message and binds the payload fields."""
        me = 'me'          # the receiver is the process, not the guard index
        fixed = g.recv_from_const
        src = (str(fixed) if fixed is not None
               else (g.recv_from or '_src'))
        lo, hi = self._loop_lo, self._loop_hi
        test = self._for_loop_test(src, None) if False else \
            f'{src} {self._loop_cmp} {hi}'
        head = f'chan[{src}][{me}]'
        cond = f'!{head}.isEmpty()'
        _flds = self.p.messages.get(g.recv_tag, []) if g.recv_tag else []
        peek_names = list(g.recv_binders)
        peek_types = {
            b: (java_base_type(_flds[i][0]) if i < len(_flds) else 'int')
            for i, b in enumerate(peek_names)}
        # The binders' LL types, so array-valued ones are recognised as arrays.
        self._binder_types = {b: _flds[i][0] for i, b in enumerate(peek_names)
                              if i < len(_flds)}
        if g.recv_tag is not None:
            cond += f' && {head}.peek().tag.equals("{g.recv_tag}")'
        lines.append(f'  private boolean _forbidden{g_id}({idx_params}) {{')
        if fixed is not None:
            # One known sender: look at that channel, nothing to search for.
            lines.append(f'    if ({cond}) {{')
            if g.guard_predicate is not None:
                lines.append(f'      Msg _p = chan[{src}][{me}].peek();')
                self._scope_types.update(peek_types)
                self._q_scope.extend(peek_names)
                try:
                    for pos, bname in enumerate(g.recv_binders):
                        ft = peek_types[bname]
                        lines.append(f'      {ft} {bname} = ({ft}) _p.p[{pos}];')
                    pre0: List[str] = []
                    w0 = self._inline_quant_in_expr(g.guard_predicate, pre0,
                                                    '      ')
                    lines.extend(pre0)
                    lines.append(f'      return {self.gen_expr(w0)};')
                finally:
                    del self._q_scope[len(self._q_scope) - len(peek_names):]
                    for bname in peek_names:
                        self._scope_types.pop(bname, None)
            else:
                lines.append('      return true;')
            lines.append('    }')
            lines.append('    return false;')
            lines.append('  }')
            lines.append('')
            self._emit_recv_advance(lines, g, g_id, me, src, fixed,
                                    peek_names, peek_types)
            return
        lines.append(f'    for (int {src} = {lo}; {test}; {src}++) {{')
        if g.guard_predicate is None:
            lines.append(f'      if ({cond}) {{ this.picked_{src} = {src}; '
                         f'return true; }}')
        else:
            # `when PRED`: the message is only eligible if PRED holds of it,
            # so the head is examined without being taken -- the figure's
            # "buffered until it becomes eligible".  The binders are in scope
            # here, bound to that peeked message.
            lines.append(f'      if ({cond}) {{')
            lines.append(f'        Msg _p = chan[{src}][{me}].peek();')
            self._scope_types.update(peek_types)
            self._q_scope.append(src)
            self._q_scope.extend(peek_names)
            try:
                for pos, bname in enumerate(g.recv_binders):
                    ft = peek_types[bname]
                    lines.append(f'        {ft} {bname} = ({ft}) _p.p[{pos}];')
                pre: List[str] = []
                w = self._inline_quant_in_expr(g.guard_predicate, pre, '        ')
                lines.extend(pre)
                lines.append(f'        if ({self.gen_expr(w)}) '
                             f'{{ this.picked_{src} = {src}; return true; }}')
            finally:
                del self._q_scope[len(self._q_scope) - len(peek_names) - 1:]
                for bname in peek_names:
                    self._scope_types.pop(bname, None)
            lines.append('      }')
        lines.append('    }')
        lines.append('    return false;')
        lines.append('  }')
        lines.append('')
        self._emit_recv_advance(lines, g, g_id, me, src, None,
                                peek_names, peek_types)

    def _emit_recv_advance(self, lines: List[str], g, g_id: int, me: str,
                           src: str, fixed, peek_names, peek_types) -> None:
        """The advance of a receive rule: take the message and bind it."""
        ret = self._advance_returns(g)
        sig = 'boolean' if self._void_stop_mode else (self._boxed_result() if ret else 'void')
        lines.append(f'  private {sig} _advance{g_id}() {{')
        if fixed is None:
            lines.append(f'    int {src} = picked_{src};')
        lines.append(f'    Msg _m = chan[{src}][{me}].poll();')
        fields = self.p.messages.get(g.recv_tag, []) if g.recv_tag else []
        bound: List[str] = []
        for pos, bname in enumerate(g.recv_binders):
            ftype = java_base_type(fields[pos][0]) if pos < len(fields) else 'int'
            lines.append(f'    {ftype} {bname} = ({ftype}) _m.p[{pos}];')
            bound.append(bname)
            self._scope_types[bname] = ftype
        # The binders are locals of the advance, so a quantifier lifted out
        # of the action has to take them as parameters like any other name
        # in scope.
        nscope = len(bound) + (0 if fixed is not None else 1)
        if fixed is None:
            self._q_scope.append(src)
        self._q_scope.extend(bound)
        prev_self = getattr(self, '_cur_self', None)
        self._cur_self = me
        try:
            lines.extend(self._gen_guard_action(g.guard_advance, '    ', void=not ret))
        finally:
            self._cur_self = prev_self
            del self._q_scope[len(self._q_scope) - nscope:]
            for bname in bound:
                self._scope_types.pop(bname, None)
            self._binder_types = {}
        if (ret or self._void_stop_mode) and self._advance_falls_through(g):
            lines.append(f'    return {"false" if self._void_stop_mode else self._null_signal()};')
        lines.append('  }')
        lines.append('')

    def _enum_owner(self, name: str) -> Optional[str]:
        """The enum type declaring the constant `name`, or None if `name` is
        an ordinary identifier.  A qualified owner like `Foo.State` means the
        constant belongs to a *composed* program's enum (`dining.state ==
        thinking`, naming DiningPhilosophers' own `thinking`), which Java
        reaches as `Foo.State.thinking` from outside that class."""
        for ename, vals in self.p.enums.items():
            if name in vals:
                return ename
        # Only programs actually composed in as a field are searched here --
        # not every sibling .ll/.llp file in the directory (self.signatures)
        # -- so an unrelated program's same-named constant is never mistaken
        # for this one's.
        composed = {d.type.base for d in list(self.p.decls) + list(self.p.params)
                    if d.type.base in _CLASS_TYPES}
        for prog_name in composed:
            other = self.signatures.get(prog_name)
            if other is None:
                continue
            for ename, vals in other.enums.items():
                if name in vals:
                    return f'{other.name}.{ename}'
        return None

    def _record_alloc(self, target: str, elem: str, size: str,
                      indent: str) -> List[str]:
        """`new Cut[n+1]` leaves the array full of nulls, so a record array
        needs one allocation per element before any field can be assigned."""
        if elem not in self.p.records or elem in self.p.messages:
            return []
        inits = [(fn, self.p.record_inits[(elem, fn)])
                 for _ft, fn in self.p.records[elem]
                 if (elem, fn) in self.p.record_inits]
        body = f'{target}[_i] = new {elem}();'
        for fn, ex in inits:
            body += f' {target}[_i].{fn} = {self.gen_expr(ex)};'
        return [f'{indent}for (int _i = 0; _i < {size}; _i++) '
                f'{{ {body} }}' if inits else
                f'{indent}for (int _i = 0; _i < {size}; _i++) {target}[_i] = new {elem}();']

    def _payload_types(self) -> set:
        """The struct and enum names that travel inside a message.

        Such a type is part of the wire format, not of one program: if two
        programs that talk to each other each nested their own copy, the
        classes would be unrelated and the receiver's cast would fail.  They
        are emitted as top-level types in files of their own instead, so both
        sides name the same class."""
        out = set()
        for fields in self.p.messages.values():
            for ftype, _fn in fields:
                b = getattr(ftype, 'base', None)
                if b in self.p.records or b in self.p.enums:
                    out.add(b)
        # A struct reachable from a payload struct travels with it.
        changed = True
        while changed:
            changed = False
            for rname in list(out):
                for ftype, _fn in self.p.records.get(rname, []):
                    b = getattr(ftype, 'base', None)
                    if (b in self.p.records or b in self.p.enums) and b not in out:
                        out.add(b)
                        changed = True
        return out

    def _record_source(self, rname: str, fields, top: bool) -> List[str]:
        """One struct, either nested in the program or standing alone."""
        ind = '' if top else '  '
        ser = (' implements java.io.Serializable' if self._uses_msgs else '')
        out = [f'{ind}{"public " if top else ""}class {rname}{ser} {{']
        for ftype, fname in fields:
            out.append(f'{ind}  {"public " if top else ""}'
                       f'{java_base_type(ftype)} {fname};')
        out.append(f'{ind}  {"public " if top else ""}{rname}() {{ }}')
        params = ', '.join(java_base_type(ft) + ' ' + fn for ft, fn in fields)
        out.append(f'{ind}  {"public " if top else ""}{rname}({params}) {{')
        for _ft, fn in fields:
            out.append(f'{ind}    this.{fn} = {fn};')
        out.append(f'{ind}  }}')
        out.append(f'{ind}}}')
        return out

    def _emit_record_types(self, lines: List[str]) -> None:
        """A Java class per struct.  Message tags are records too, but they
        travel inside Msg as an Object[] payload, so only the structs need a
        named class here."""
        shared = self._payload_types()
        for rname in sorted(shared):
            if rname in self.p.records:
                self.shared_types[rname] = '\n'.join(
                    ['// A type that travels inside a message, so it belongs to',
                     '// neither program that names it.  Generated by lang/llc.py.',
                     '']
                    + self._record_source(rname, self.p.records[rname], True)) + '\n'
            elif rname in self.p.enums:
                self.shared_types[rname] = '\n'.join(
                    ['// A type that travels inside a message, so it belongs to',
                     '// neither program that names it.  Generated by lang/llc.py.',
                     '',
                     'public enum ' + rname + ' { '
                     + ', '.join(self.p.enums[rname]) + ' }']) + '\n'
        for rname, fields in self.p.records.items():
            if rname in self.p.messages or rname in shared:
                continue
            lines.extend(
                ['  static class' + l[len('  class'):] if l.startswith('  class ')
                 else l for l in self._record_source(rname, fields, False)])
        if any(r not in self.p.messages and r not in shared
               for r in self.p.records):
            lines.append('')

    def _emit_enum_types(self, lines: List[str]) -> None:
        shared = self._payload_types()
        rest = [e for e in self.p.enums if e not in shared]
        for ename in rest:
            lines.append('  enum ' + ename + ' { '
                         + ', '.join(self.p.enums[ename]) + ' }')
        if rest:
            lines.append('')

    def _program_uses_messages(self) -> bool:
        """True if the program sends or receives anything, in which case the
        emitted class needs the Msg type and the channel array."""
        def walk(stmts):
            for st in stmts:
                if st.kind == 'send':
                    return True
                if st.kind == 'guard' and st.recv_tag is not None:
                    return True
                if st.kind == 'guard' and st.guard_advance is not None:
                    if walk([st.guard_advance]):
                        return True
                for sub in (st.then_stmts, st.else_stmts, st.body,
                            st.par_branches):
                    if sub and walk(sub):
                        return True
            return False
        return walk(self.p.stmts) or any(
            walk(m.body) for m in self.p.aux_methods)

    def _emit_message_runtime(self, lines: List[str]) -> None:
        """The channel model: a message is a tag plus a payload, and the
        channel from a to b is a queue of them.  A `send` appends and a
        `receive` guard polls, so a message in transit is part of the
        program state -- which is what lets `upon receive(...)` be an
        ordinary state predicate."""
        lines.append('  static class Msg implements java.io.Serializable {')
        lines.append('    int src; String tag; Object[] p;')
        lines.append('    int ts;   // piggybacked logical clock, see Clock')
        lines.append('    Msg(int src, String tag, Object[] p) '
                     '{ this.src = src; this.tag = tag; this.p = p; }')
        lines.append('  }')
        lines.append('')
        lines.append('  /** The logical clock the timestamp algorithms assume')
        lines.append('   *  is "maintained by a separate protocol running')
        lines.append('   *  alongside": advanced before every send, and set')
        lines.append('   *  past any timestamp received.  It rides on the')
        lines.append('   *  channels rather than in the algorithm, which only')
        lines.append('   *  reads it through logicalClock(). */')
        lines.append('  public static class Clock {')
        lines.append('    private int t = 0;')
        lines.append('    public synchronized int tick()  { return ++t; }')
        lines.append('    public synchronized int value() { return t; }')
        lines.append('    public synchronized void observe(int s) '
                     '{ t = Math.max(t, s) + 1; }')
        lines.append('  }')
        lines.append('')
        lines.append('  /** One process\'s view of the channels, with the clock')
        lines.append('   *  protocol attached: stamp what leaves, and advance')
        lines.append('   *  past what arrives.  A process only ever touches')
        lines.append('   *  its own row and column, so wrapping per process is')
        lines.append('   *  enough and the underlying channels stay shared. */')
        lines.append('  public static MsgQ[][] withClock(MsgQ[][] base, int me,')
        lines.append('      int n, Clock c) {')
        lines.append('    MsgQ[][] w = new MsgQ[n + 2][n + 2];')
        lines.append('    for (int a = 0; a <= n + 1; a++) {')
        lines.append('      for (int b = 0; b <= n + 1; b++) {')
        lines.append('        final MsgQ q = base[a][b];')
        lines.append('        if (q == null) { w[a][b] = null; continue; }')
        lines.append('        if (a == me) {')
        lines.append('          w[a][b] = new MsgQ() {')
        lines.append('            public boolean isEmpty() { return q.isEmpty(); }')
        lines.append('            public Msg peek()        { return q.peek(); }')
        lines.append('            public Msg poll()        { return q.poll(); }')
        lines.append('            public void add(Msg m)   '
                     '{ m.ts = c.value(); q.add(m); }')
        lines.append('          };')
        lines.append('        } else if (b == me) {')
        lines.append('          w[a][b] = new MsgQ() {')
        lines.append('            public boolean isEmpty() { return q.isEmpty(); }')
        lines.append('            public Msg peek()        { return q.peek(); }')
        lines.append('            public void add(Msg m)   { q.add(m); }')
        lines.append('            public Msg poll() {')
        lines.append('              Msg m = q.poll();')
        lines.append('              if (m != null) c.observe(m.ts);')
        lines.append('              return m;')
        lines.append('            }')
        lines.append('          };')
        lines.append('        } else {')
        lines.append('          w[a][b] = q;')
        lines.append('        }')
        lines.append('      }')
        lines.append('    }')
        lines.append('    return w;')
        lines.append('  }')
        lines.append('')
        lines.append('  /** One directed channel.  The rules only ever ask a')
        lines.append('   *  channel these four things, so this is the whole')
        lines.append('   *  seam between the algorithm and the mode it runs')
        lines.append('   *  in: queues in memory, queues shared by threads,')
        lines.append('   *  or a socket to another machine. */')
        lines.append('  public interface MsgQ {')
        lines.append('    boolean isEmpty();')
        lines.append('    Msg peek();')
        lines.append('    Msg poll();')
        lines.append('    void add(Msg m);')
        lines.append('  }')
        lines.append('')
        lines.append('  /** A message is a value: what the sender keeps mutating afterwards')
        lines.append('   *  must not be visible to the receiver.  Distributed mode gets this')
        lines.append('   *  for free from serialization; in memory the payload is copied')
        lines.append('   *  instead, so all three modes agree. */')
        lines.append('  static Msg copyOf(Msg m) {')
        lines.append('    Object[] q = new Object[m.p.length];')
        lines.append('    for (int k = 0; k < q.length; k++) {')
        lines.append('      Object o = m.p[k];')
        lines.append('      if (o instanceof int[]) o = ((int[]) o).clone();')
        lines.append('      else if (o instanceof boolean[]) o = ((boolean[]) o).clone();')
        lines.append('      else if (o instanceof double[]) o = ((double[]) o).clone();')
        lines.append('      else if (o instanceof LinkedList)')
        lines.append('        o = new LinkedList<>((LinkedList<?>) o);')
        lines.append('      q[k] = o;')
        lines.append('    }')
        lines.append('    Msg c = new Msg(m.src, m.tag, q);')
        lines.append('    c.ts = m.ts;')
        lines.append('    return c;')
        lines.append('  }')
        lines.append('')
        lines.append('  /** Local mode: an ordinary FIFO queue in memory. */')
        lines.append('  public static class LocalQ implements MsgQ {')
        lines.append('    private final ArrayDeque<Msg> d = new ArrayDeque<>();')
        lines.append('    public boolean isEmpty() { return d.isEmpty(); }')
        lines.append('    public Msg peek()        { return d.peek(); }')
        lines.append('    public Msg poll()        { return d.poll(); }')
        lines.append('    public void add(Msg m)   { d.add(copyOf(m)); }')
        lines.append('  }')
        lines.append('')
        lines.append('  /** Parallel mode: the same FIFO, safe for one sender')
        lines.append('   *  thread and one receiver thread per channel. */')
        lines.append('  public static class SharedQ implements MsgQ {')
        lines.append('    private final java.util.concurrent.ConcurrentLinkedQueue<Msg> d')
        lines.append('      = new java.util.concurrent.ConcurrentLinkedQueue<>();')
        lines.append('    public boolean isEmpty() { return d.isEmpty(); }')
        lines.append('    public Msg peek()        { return d.peek(); }')
        lines.append('    public Msg poll()        { return d.poll(); }')
        lines.append('    public void add(Msg m)   { d.add(copyOf(m)); }')
        lines.append('  }')
        lines.append('')
        lines.append('  /** Distributed mode: an outgoing channel is a socket')
        lines.append('   *  to another machine.  Nothing reads an outgoing')
        lines.append('   *  channel -- a rule only ever inspects the channels')
        lines.append('   *  addressed to itself -- so only add() carries. */')
        lines.append('  public static class SocketQ implements MsgQ {')
        lines.append('    private final java.io.ObjectOutputStream out;')
        lines.append('    public SocketQ(java.io.ObjectOutputStream out) '
                     '{ this.out = out; }')
        lines.append('    public boolean isEmpty() { return true; }')
        lines.append('    public Msg peek()        { return null; }')
        lines.append('    public Msg poll()        { return null; }')
        lines.append('    public synchronized void add(Msg m) {')
        lines.append('      try {')
        lines.append('        out.writeObject(m);')
        lines.append('        out.reset();   // send values, not back-references')
        lines.append('        out.flush();')
        lines.append('      } catch (java.io.IOException e) {')
        lines.append('        throw new RuntimeException(e);')
        lines.append('      }')
        lines.append('    }')
        lines.append('  }')
        lines.append('')
        lines.append('  /** Wire up node `me` of n and return the channels it')
        lines.append('   *  sees: a socket out to each peer, and one inbound')
        lines.append('   *  queue per peer that a listener thread fills.')
        lines.append('   *  TCP keeps each pair FIFO, which is what the')
        lines.append('   *  timestamp algorithms assume. */')
        lines.append('  public static MsgQ[][] connect(int me, int n,')
        lines.append('      String[] host, int[] port) throws Exception {')
        lines.append('    MsgQ[][] c = new MsgQ[n + 2][n + 2];')
        lines.append('    final SharedQ[] in = new SharedQ[n + 2];')
        lines.append('    for (int j = 0; j <= n + 1; j++) {')
        lines.append('      in[j] = new SharedQ();')
        lines.append('      c[j][me] = in[j];')
        lines.append('    }')
        lines.append('    c[me][me] = in[me];')
        lines.append('    final java.net.ServerSocket ss '
                     '= new java.net.ServerSocket(port[me]);')
        lines.append('    Thread acc = new Thread(() -> {')
        lines.append('      try {')
        lines.append('        while (true) {')
        lines.append('          final java.net.Socket s = ss.accept();')
        lines.append('          Thread r = new Thread(() -> {')
        lines.append('            try {')
        lines.append('              java.io.ObjectInputStream ois =')
        lines.append('                new java.io.ObjectInputStream(s.getInputStream());')
        lines.append('              while (true) {')
        lines.append('                Msg m = (Msg) ois.readObject();')
        lines.append('                in[m.src].add(m);')
        lines.append('              }')
        lines.append('            } catch (Exception e) { /* peer went away */ }')
        lines.append('          });')
        lines.append('          r.setDaemon(true);')
        lines.append('          r.start();')
        lines.append('        }')
        lines.append('      } catch (Exception e) { /* listener closed */ }')
        lines.append('    });')
        lines.append('    acc.setDaemon(true);')
        lines.append('    acc.start();')
        lines.append('    // From 0, not 1: a distinguished process such as a')
        lines.append('    // coordinator sits at index 0, outside the range the')
        lines.append('    // rules scan.  Entries with no host are not peers.')
        lines.append('    for (int j = 0; j <= n; j++) {')
        lines.append('      if (j == me || host[j] == null) continue;')
        lines.append('      java.net.Socket s = null;')
        lines.append('      while (s == null) {')
        lines.append('        try { s = new java.net.Socket(host[j], port[j]); }')
        lines.append('        catch (java.io.IOException e) { Thread.sleep(50); }')
        lines.append('      }')
        lines.append('      // The reader blocks on the stream header, so the')
        lines.append('      // writer must be created and flushed first.')
        lines.append('      java.io.ObjectOutputStream oos =')
        lines.append('        new java.io.ObjectOutputStream(s.getOutputStream());')
        lines.append('      oos.flush();')
        lines.append('      c[me][j] = new SocketQ(oos);')
        lines.append('    }')
        lines.append('    return c;')
        lines.append('  }')
        lines.append('')
        lines.append('  public static MsgQ[][] channels(int n, boolean shared) {')
        lines.append('    MsgQ[][] c = new MsgQ[n + 2][n + 2];')
        lines.append('    for (int a = 0; a <= n + 1; a++)')
        lines.append('      for (int b = 0; b <= n + 1; b++)')
        lines.append('        c[a][b] = shared ? new SharedQ() : new LocalQ();')
        lines.append('    return c;')
        lines.append('  }')
        lines.append('')
        lines.append('  public MsgQ[][] chan;   // chan[src][dst]')
        lines.append('  public int me;          // this process\'s index')
        _, rt = self._scan_exits()
        if rt is not None:
            lines.append('')
            lines.append('  // Set by exit(): the program is finished, with')
            lines.append('  // this result.  No rule fires afterwards.')
            lines.append(f'  private {rt} result;')
            lines.append('  private boolean done;')
            lines.append(f'  public {rt} result() {{ return result; }}')
            lines.append('  public boolean done()  { return done; }')
        lines.append('')

    def _emit_exit_fields(self, lines: List[str]) -> None:
        """`exit(v)` gives the program a result, so a program that uses it
        carries the result and the finished flag.  Everything else about the
        message runtime is inherited from LLProcess."""
        _, rt = self._scan_exits()
        if rt is None:
            return
        lines.append('  // Set by exit(): the program is finished, with')
        lines.append('  // this result.  No rule fires afterwards.')
        lines.append(f'  private {rt} result;')
        lines.append('  private boolean done;')
        lines.append(f'  public {rt} result() {{ return result; }}')
        lines.append('  public boolean done()  { return done; }')
        lines.append('')

    def _emit_chan_init(self, body: List[str], indent: str = '    ') -> None:
        """Give this process channels if a driver has not already handed it
        a shared set.  Running one instance on its own then still works."""
        body.append(f'{indent}if (this.chan == null) '
                    f'this.chan = channels(n, false);')

    @staticmethod
    def _cond_pick_vars(cond) -> List[str]:
        """Witness variables a condition's `pick`s bind, in order."""
        found: List[str] = []

        def walk(e):
            if e is None or not hasattr(e, 'kind'):
                return
            if e.kind == 'quant' and getattr(e, 'pick_var', None):
                if e.pick_var not in found:
                    found.append(e.pick_var)
            for a in ('lhs', 'rhs', 'idx', 'q_body', 'q_lo', 'q_hi', 'q_set'):
                walk(getattr(e, a, None))
            for a in (getattr(e, 'args', None) or []):
                walk(a)

        walk(cond)
        return found

    def _scan_pick_vars(self) -> set:
        """Every witness variable named by a `pick`, anywhere in the program.
        Fields have to be declared before the body is generated, so this runs
        over the AST rather than collecting during emission."""
        found: dict = {}

        def expr(e):
            if e is None or not hasattr(e, 'kind'):
                return
            if e.kind == 'quant' and getattr(e, 'pick_var', None):
                found[e.pick_var] = (self._elem_java_type(e.q_set)
                                     if e.q_set is not None else 'int')
            for a in ('lhs', 'rhs', 'idx', 'q_body', 'q_lo', 'q_hi', 'q_set'):
                expr(getattr(e, a, None))
            for a in (getattr(e, 'args', None) or []):
                expr(a)

        def stmt(st):
            if st is None or not hasattr(st, 'kind'):
                return
            for a in ('rhs', 'cond', 'return_value', 'decl_init',
                      'guard_predicate', 'lhs_index', 'lhs_index2',
                      'lhs_field_index', 'send_dest'):
                expr(getattr(st, a, None))
            for a in ('call_args', 'send_payload'):
                for x in (getattr(st, a, None) or []):
                    expr(x)
            for a in ('body', 'then_stmts', 'else_stmts', 'par_branches'):
                for x in (getattr(st, a, None) or []):
                    stmt(x)
            stmt(getattr(st, 'guard_advance', None))

        for st in self.p.stmts:
            stmt(st)
        for m in self.p.aux_methods:
            for st in m.body:
                stmt(st)
        return found

    def _advance_call(self, aname: str, returns: bool) -> str:
        """The call to an advance inside a fixpoint loop.  A terminal rule
        hands back a result, which leaves the entry method immediately; the
        usual rule returns nothing and the loop carries on."""
        if self._void_stop_mode:
            # advance() returns boolean here (see _void_stop_mode): true
            # means the program's own `return null;` fired, so stop -- a
            # bare `return;` from this void entry method, since there's no
            # result to hand back.
            return f'if ({aname}()) return;'
        if not returns:
            return f'{aname}();'
        sig = self._null_signal()
        return (f'{self._boxed_result()} _r = {aname}(); '
                f'if (_r != {sig}) return _r;')

    def _emit_fixpoint_loop(self, fname: str, aname: str, pick: bool,
                             indent: str,
                             indices: Optional[List[str]] = None,
                             returns: bool = False) -> List[str]:
        """Emit the mode-appropriate fixpoint loop that repeatedly calls
        `fname(j)` / `aname(j)` until no index is forbidden.  `forbidden`
        is always boolean now (pick-witness, when relevant, is stashed
        in a class field by the helper itself), so the loop body is the
        same regardless of `pick`.

        For multi-index guards (e.g. forbidden(i, j)), nested for-loops
        are generated over each index variable."""
        lo, hi = self._loop_lo, self._loop_hi
        if indices is None:
            indices = ['j']
        call_args = ', '.join(indices)
        # Assignments to copy loop variables into class fields before
        # calling advance() (which reads them as fields, not params).
        set_fields = ' '.join(f'this.{v} = {v};' for v in indices)
        out: List[str] = []
        if self.p.mode == 'array':
            # Wrap in a block so multiple fixpoint loops in the same
            # method don't conflict on `changed`.
            out.append(f'{indent}{{')
            out.append(f'{indent}  boolean changed = true;')
            out.append(f'{indent}  while (changed) {{')
            out.append(f'{indent}    changed = false;')
            # Emit nested for-loops, one per index variable.
            depth = len(indices)
            for d, var in enumerate(indices):
                pad = '    ' + '  ' * d
                out.append(f'{indent}{pad}for (int {var} = {lo}; {var} {self._loop_cmp} {hi}; {var}++) {{')
            inner_pad = '    ' + '  ' * depth
            out.append(f'{indent}{inner_pad}if ({fname}({call_args})) {{')
            out.append(f'{indent}{inner_pad}  {set_fields} '
                       f'{self._advance_call(aname, returns)}')
            out.append(f'{indent}{inner_pad}  changed = true;')
            out.append(f'{indent}{inner_pad}}}')
            for d in range(depth - 1, -1, -1):
                pad = '    ' + '  ' * d
                out.append(f'{indent}{pad}}}')
            out.append(f'{indent}  }}')
            out.append(f'{indent}}}')
        elif self.p.mode == 'linkedlist':
            out.append(f'{indent}{{')
            out.append(f'{indent}  boolean changed = true;')
            out.append(f'{indent}  while (changed) {{')
            out.append(f'{indent}    changed = false;')
            out.append(f'{indent}    Iterator<Integer> it = forbiddenList.iterator();')
            out.append(f'{indent}    while (it.hasNext()) {{')
            out.append(f'{indent}      int j = it.next();')
            out.append(f'{indent}      if ({fname}({call_args})) {{ {set_fields} '
                       f'{self._advance_call(aname, returns)} changed = true; break; }}')
            out.append(f'{indent}    }}')
            out.append(f'{indent}  }}')
            out.append(f'{indent}}}')
        elif self.p.mode == 'heap':
            out.append(f'{indent}while (!forbiddenHeap.isEmpty()) {{')
            out.append(f'{indent}  int j = forbiddenHeap.poll();')
            out.append(f'{indent}  if ({fname}({call_args})) {{ {set_fields} '
                       f'{self._advance_call(aname, returns)} }}')
            out.append(f'{indent}}}')
        elif self.p.mode == 'deque':
            out.append(f'{indent}while (!forbiddenDeque.isEmpty()) {{')
            out.append(f'{indent}  int j = forbiddenDeque.pollFirst();')
            out.append(f'{indent}  if ({fname}({call_args})) {{ {set_fields} '
                       f'{self._advance_call(aname, returns)} }}')
            out.append(f'{indent}  else forbiddenDeque.addLast(j);')
            out.append(f'{indent}}}')
        return out

    def emit_main(self) -> List[str]:
        """Emit a runnable Java main() that constructs hard-coded inputs,
        calls the entry method, and prints the result.  Supports parameter
        types int / int[] / int[][] / boolean / boolean[].  Falls back to a
        stub when any parameter has an unsupported type (e.g. set<int>[])."""
        synth = self._synthesize_main_inputs()
        out = ['  public static void main(String[] args) {']
        if synth is None:
            out.append(f'    // No runnable example: {self.p.name}\'s parameters include a type')
            out.append('    // this synthesizer cannot safely construct on its own (e.g. set<int>[],')
            out.append('    // a 3D+ array, or another unsupported shape) -- not a compile error, just')
            out.append('    // nothing to call here. See the .llp source for the real signature, and')
            out.append('    // construct valid inputs by hand to exercise this method.')
            out.append('  }')
            return out
        decls, call_args = synth
        for d in decls:
            out.append('    ' + d)

        if self.noclass:
            call = f'{self.p.entry_name}({", ".join(call_args)})'
        else:
            out.append(f'    {self.p.name} prog = new {self.p.name}();')
            call = f'prog.{self.p.entry_name}({", ".join(call_args)})'

        rt = self.p.return_type
        is_void = (rt is None) or (
            getattr(rt, 'base', None) == 'void'
            and not getattr(rt, 'is_array', False)
            and not getattr(rt, 'is_2d', False))

        if is_void:
            out.append(f'    {call};')
            for arg, p in zip(call_args, self.p.params):
                if (p.type.is_array or p.type.is_2d) and arg == p.name:
                    out.append('    ' + _java_print_stmt(arg, p.type))
        else:
            out.append(f'    {java_base_type(rt)} result = {call};')
            out.append('    ' + _java_print_stmt('result', rt))
        out.append('  }')
        return out

    def _synthesize_main_inputs(self):
        """Return ([decl_lines], [arg_exprs]) or None.

        Decl lines are statements like `int[] A = new int[]{...};`.  Arg
        exprs are the corresponding expressions to pass to the entry method
        (often the variable name; for derived bounds like `hi`, an
        expression like `G.length - 1`)."""
        decls: List[str] = []
        args: List[str] = []
        # First int[] parameter, used to size lo/hi if present.
        first_int_arr = next(
            (p for p in self.p.params
             if p.type.is_array and not p.type.is_2d
             and p.type.base == 'int' and not p.type.is_set),
            None)
        # Defaults for successive int[] parameters.  When the program has
        # exactly one int[] param, give it an unsorted length-8 array
        # (works for general sorts and bitonic).  When two or more, many
        # of these programs are graph/interval algorithms with parallel
        # arrays (edge endpoints u[]/v[], a companion weight or
        # predecessor-index array) that are read as indices into each
        # other or into a same-length state array -- a mismatched length
        # or an out-of-range value crashes immediately (confirmed for
        # Kruskal/Boruvka/BellmanFord/WeightedInterval/ParQuadCRT/
        # LLPKruskal). So every slot here is deliberately the SAME length
        # (4). Every position j also satisfies value[j] < j (strictly --
        # not merely < 4): a first attempt used value[j] <= j, including
        # self-loops like value[3] == 3, which is exactly as fatal as a
        # forward reference for a "points to an earlier index" dependency
        # array (confirmed hang in LLPWeightedIntervalScheduling, whose
        # recurrence reads G[p[j]]; a self-loop there makes G[j] depend on
        # its own not-yet-final value and grow forever). Position 0 has no
        # valid value satisfying "< 0", so it is simply 0 throughout --
        # safe because index 0 is the base case in every observed
        # instance of this pattern (e.g. explicitly excluded from
        # `forbidden` by a `j >= 1` guard). Still fine as a plain small
        # index for edge-list/weight uses that don't care about direction,
        # though position 1 is forced to 0 in every slot (the only value
        # satisfying < 1), so two such arrays used together as parallel
        # edge endpoints will sometimes agree at index 1 (a harmless
        # self-loop edge, not a crash).
        int_array_count = sum(
            1 for q in self.p.params
            if q.type.is_array and not q.type.is_2d
            and q.type.base == 'int' and not q.type.is_set)
        if int_array_count >= 2:
            int_array_defaults = ['{0, 0, 1, 2}', '{0, 0, 1, 1}',
                                   '{0, 0, 0, 2}', '{0, 0, 2, 1}']
        else:
            # Was {5, 2, 4, 6, 1, 3, 8, 7}, which contains the out-of-range
            # value 8 -- every index into an 8-element array must be < 8,
            # and this slipped through because the single-array case was
            # never exercised by a program that uses its own values as
            # indices back into itself or a same-length sibling array,
            # e.g. a self-reference chain like Union-Find's
            # parent[parent[x]] or LLPBoruvka's G[G[j]] (confirmed both
            # crash on the old literal). A first attempt simply replaced
            # the 8 with a 0 (a permutation of [0..7]) -- that in turn
            # regressed the GCD/LCM family (EuclidGCD, GCD1, LLPGCD, ...),
            # which read this same single array as plain numbers and
            # compute `G[j] % G[i]`: a 0 anywhere is a division-by-zero
            # (or, for the subtract-based variant, an infinite loop
            # subtracting 0 forever). A plain permutation is also wrong
            # for the self-reference case for a second reason: every
            # element of a permutation lies on *some* cycle, but only
            # the index that happens to start a length-1 cycle (a fixed
            # point) terminates immediately -- chasing parent[parent[x]]
            # from any other index cycles forever (confirmed: still
            # crashed/looped with the 0-for-8 swap, from the synthesized
            # x = 0, which landed on a 4-cycle with no fixed point of its
            # own). This literal instead has zero cycles at all: every
            # chase of value[value[...[j]]] strictly decreases toward one
            # of two nonzero fixed points (value[1] = 1, value[6] = 6),
            # so it is simultaneously safe to chase from any starting
            # index, zero-free for the GCD family, and still visibly
            # unsorted for plain sort/search demos.
            int_array_defaults = ['{3, 1, 6, 1, 6, 3, 6, 4}']
        int_array_idx = 0
        int2d_idx = 0
        # Every 2D array parameter (int[][]/boolean[][]/double[][]) and
        # every 1D boolean[]/double[] parameter is sized to this same N --
        # matching whatever length the 1D int[] arrays above use (4 or 8)
        # -- rather than each type picking its own independent fixed size.
        # These parameters co-occur constantly in this corpus (a
        # preference/rank matrix alongside an index array that is its
        # intended length, an adjacency matrix alongside a vertex-count
        # array, LLPPrim's parent/fixed/C/W all describing the same n
        # vertices) and a per-type-independent size is exactly what
        # produced ~25 real ArrayIndexOutOfBoundsExceptions across the
        # corpus: a 3x3 default matrix paired with an 8-element index
        # array, or a 4-element boolean[] paired with an 8-element int[]
        # from the very same parameter list. Confirmed for GaleShapley
        # family (mpref/rank), LLPAntichain (chains/leq), TTC/
        # LLPHousingMarket (pref), LLPQHornSAT (body1/clauses2),
        # LeastJoinIrreducible (cap), JobScheduling (pre), LLPPrim
        # (fixed/W/C), LLPOptimalBinarySearchTree (G).
        shared_n = 4 if int_array_count >= 2 else 8

        def _square_int_literal(n, reverse=False):
            # Every row is a cyclic-shift permutation of [0..n-1], so
            # every entry is always a safe index into any length-n
            # sibling array -- whatever the specific semantic (a
            # preference list, a rank table, an adjacency list of vertex
            # indices) the entry is read as. `reverse` shifts the other
            # way: a program with two 2D int[] params (a preference
            # matrix `mpref` alongside a `rank` matrix, say) would
            # otherwise get the identical matrix twice over -- confirmed
            # to make LLPStableMarriage's proposal counter run past the
            # end of its own preference list (a worst-case cascade
            # specific to the resonance of two *identical* rotations),
            # fixed by making successive 2D int arrays in the same
            # parameter list rotate opposite ways.
            step = -1 if reverse else 1
            rows = ', '.join(
                '{' + ', '.join(str((i + step * k) % n) for k in range(n)) + '}'
                for i in range(n))
            return '{' + rows + '}'

        def _dag_int_literal(n):
            # Jagged, not square: row i holds exactly i entries, a
            # permutation of [0..i-1] -- so row 0 is genuinely *empty*
            # (vertex 0 has no predecessors at all) rather than square
            # and forced to contain some value, which for row 0
            # specifically has no option below its own index and must
            # fall back to a same-row self-reference. Used only for a 2D
            # int[][] parameter actually named `pre` (a predecessor list,
            # read as "the vertices this one depends on", consumed via
            # `while (k < pre[j].length)` -- genuinely jagged-aware code,
            # not an assumption of square shape) -- `_square_int_literal`'s
            # permutation is wrong there specifically: every one of its
            # entries lies on *some* cycle of the row-as-function graph,
            # and pre[j][k] being read as a dependency turns any such
            # cycle into a circular dependency with no valid finishing
            # time. A first attempt kept the square shape and forced row
            # 0 to all-zeros -- that is itself a self-loop (vertex 0
            # "depends on" vertex 0), still non-terminating (confirmed to
            # hang JobScheduling): row 0 has no safe *value* under a
            # square shape, only a safe *length* (zero) does the job.
            # Every row's own permutation of strictly-smaller indices can
            # never close a cycle back on itself, so this is
            # simultaneously safe to index with (every entry is < n) and
            # acyclic to chase -- at the cost of being a poor fit for a
            # genuine preference/rank *permutation*, which is why it is
            # not the default for every 2D int[][] (confirmed: using it
            # unconditionally in place of the permutation above regresses
            # GaleShapley, TTC, LLPHousingMarket, LLPOptimalHospitalResident,
            # LLPWeightedSetCover, none of which are named `pre`).
            rows = ', '.join(
                '{' + ', '.join(str(k) for k in range(i)) + '}'
                for i in range(n))
            return '{' + rows + '}'

        def _square_bool_literal(n):
            rows = ', '.join(
                '{' + ', '.join('true' if k == i else 'false'
                                 for k in range(n)) + '}'
                for i in range(n))
            return '{' + rows + '}'

        def _square_double_literal(n):
            rows = ', '.join(
                '{' + ', '.join(f'{float(i * n + k + 1)}' for k in range(n)) + '}'
                for i in range(n))
            return '{' + rows + '}'
        for p in self.p.params:
            ty = p.type
            name = p.name
            # Reject set<int>[] etc. -- too awkward to construct in main().
            if ty.is_set:
                return None
            if not ty.is_array and not ty.is_2d:
                # Scalar.
                if ty.base == 'int':
                    if name == 'lo' or name == 'low':
                        args.append('0'); continue
                    if name == 'hi' and first_int_arr:
                        args.append(f'{first_int_arr.name}.length - 1'); continue
                    if name == 'high' and first_int_arr:
                        # Two incompatible conventions share this name in
                        # this corpus: BinarySearch/ThreeWayPartition treat
                        # it as exclusive (one past the last valid index,
                        # `high=A.length`, confirmed to run correctly);
                        # MergeSort/CountingInversions treat it as
                        # inclusive (the last valid index, needing
                        # `A.length - 1` -- `A.length` crashed both).
                        # The two inclusive cases both also declare an aux
                        # method (Merge/mergeAndCount) that takes its own
                        # `high` parameter alongside a `mid`; neither
                        # exclusive case has any aux method at all. That
                        # co-occurrence is a reliable, corpus-wide way to
                        # tell which convention a given program uses.
                        has_aux_high = any(
                            any(ap.name == 'high' for ap in m.params)
                            for m in self.p.aux_methods)
                        if has_aux_high:
                            args.append(f'{first_int_arr.name}.length - 1'); continue
                        args.append(f'{first_int_arr.name}.length'); continue
                    if name == 'n' and first_int_arr:
                        args.append(f'{first_int_arr.name}.length'); continue
                    if name == 'src' or name == 'source':
                        args.append('0'); continue
                    if name == 's':
                        args.append('0'); continue
                    if name == 't' and len(self.p.params) >= 2:
                        # Network-flow sink: last vertex.  Best guess.
                        args.append('1'); continue
                    if name == 'pivot':
                        decls.append('int pivot = 5;')
                        args.append('pivot'); continue
                    if name == 'w':
                        decls.append('int w = 3;')
                        args.append('w'); continue
                    if name == 'v':
                        decls.append('int v = 5;')
                        args.append('v'); continue
                    if name == 'W':
                        decls.append('int W = 10;')
                        args.append('W'); continue
                    if name == 'dir':
                        args.append('1'); continue
                    if name == 'root':
                        args.append('0'); continue
                    if name == 'epsNum':
                        decls.append('int epsNum = 1;')
                        args.append('epsNum'); continue
                    if name == 'epsDen':
                        decls.append('int epsDen = 2;')
                        args.append('epsDen'); continue
                    if name == 'n' and not any(q.type.is_2d for q in self.p.params):
                        # No array to size `n` from (first_int_arr is None,
                        # else the branch above would have fired): a bare
                        # scalar `n` is usually a recursion depth/magnitude
                        # (e.g. Hanoi's disk count, Karatsuba's digit
                        # count), so 0 or 1 trivially satisfies -- and
                        # sometimes violates -- the base case without
                        # exercising any real recursion, and for several
                        # programs (Hanoi, Karatsuba) 0 crashes outright
                        # (the base case is n==1, so n=0 recurses forever).
                        # A small positive value exercises real recursion
                        # safely for every program observed so far.
                        #
                        # Excluded when a 2D array parameter is also
                        # present: that array's hardcoded default literal
                        # is a fixed 3x3 shape, never sized to match
                        # whatever `n` we pick, so an n other than the
                        # matrix's own dimension risks an out-of-bounds
                        # access the moment the program actually indexes
                        # up to n (confirmed for Constrained_Stable_Matching
                        # and ParallelCut, where n=3 awakens exactly that
                        # mismatch; n=0 previously made the loop a no-op
                        # and silently avoided it). Leaving `n` at the
                        # generic 0 default here isn't a real fix for
                        # those two -- they still don't demonstrate
                        # anything -- but it doesn't newly crash them
                        # either, and a proper fix needs the matrix
                        # default sized from the same chosen n, which is
                        # more than a mechanical, name-based change.
                        decls.append('int n = 3;')
                        args.append('n'); continue
                    if name in ('n1', 'n2'):
                        # LLPQHornSAT's only use of this convention: n1/n2
                        # size the G/H state arrays this same main allocates
                        # from them, and sibling 2D arrays (body1/clauses2)
                        # hold indices into those state arrays -- so n1/n2
                        # must match shared_n, the same size every sibling
                        # 2D array below is built to (was hardcoded 0,
                        # sizing G/H to length 0 and crashing immediately).
                        decls.append(f'int {name} = {shared_n};')
                        args.append(name); continue
                    decls.append(f'int {name} = 0;')
                    args.append(name)
                elif ty.base == 'boolean':
                    decls.append(f'boolean {name} = false;')
                    args.append(name)
                elif ty.base == 'double':
                    if name == 'W':
                        decls.append('double W = 10.0;')
                        args.append('W'); continue
                    decls.append(f'double {name} = 0.0;')
                    args.append(name)
                else:
                    return None
            elif ty.is_2d and ty.dims > 2:
                # 3D+ (e.g. LLPwithRejection's int[][][][] R): synthesizing
                # a correctly-shaped nested literal isn't worth it for a
                # throwaway demo harness -- fall back to the stub, same as
                # the ty.is_set case above.
                return None
            elif ty.is_2d:
                if ty.base == 'int':
                    if name == 'pre':
                        lit2d = _dag_int_literal(shared_n)
                    else:
                        lit2d = _square_int_literal(shared_n, reverse=(int2d_idx % 2 == 1))
                        int2d_idx += 1
                    decls.append(f'int[][] {name} = new int[][] {lit2d};')
                    args.append(name)
                elif ty.base == 'boolean':
                    decls.append(
                        f'boolean[][] {name} = new boolean[][] '
                        f'{_square_bool_literal(shared_n)};')
                    args.append(name)
                elif ty.base == 'double':
                    decls.append(
                        f'double[][] {name} = new double[][] '
                        f'{_square_double_literal(shared_n)};')
                    args.append(name)
                else:
                    return None
            else:  # 1D array
                if ty.base == 'int':
                    # `m` is exclusively used, corpus-wide, as a CRT-family
                    # modulus array (ParCRT/ParCRT2/ParQuadCRT) -- divided
                    # into, so it must never contain 0, unlike the
                    # index-safe-but-zero-containing default below. `p`
                    # paired with a sibling `e` is Ord's prime/exponent
                    # convention (also divided into); `p` on its own
                    # (WeightedInterval's predecessor-index array) is a
                    # different, unrelated use where 0 is a valid index
                    # and must stay available, so this check only fires
                    # for the specific p+e co-occurrence.
                    sibling_names = {q.name for q in self.p.params}
                    if name == 'm':
                        lit = '{3, 5, 7, 11}'
                    elif name == 'p' and 'e' in sibling_names:
                        lit = '{2, 3, 5, 7}'
                    elif name == 'I' and any(q.type.is_2d for q in self.p.params):
                        # `I`, alongside a 2D preference/rank matrix, is
                        # corpus-wide the initial *lattice-bottom* state
                        # for a monotone LLP computation (`forall k: G[k]
                        # = I[k]`) -- the stable-matching family's G[j] is
                        # a proposal counter meant to start at "nobody has
                        # proposed yet". Giving it the generic nonzero
                        # index-safe literal starts several counters already
                        # mid-flight with no consistent history behind
                        # them, which isn't a reachable state of the real
                        # algorithm at all: confirmed it drives G[j] past
                        # the end of j's own preference list within 2
                        # rounds (LLPStableMarriage), something no all-zero
                        # start ever did in randomized testing.
                        lit = '{' + ', '.join(['0'] * shared_n) + '}'
                    else:
                        lit = int_array_defaults[
                            min(int_array_idx, len(int_array_defaults) - 1)]
                        int_array_idx += 1
                    decls.append(f'int[] {name} = new int[] {lit};')
                    args.append(name)
                elif ty.base == 'boolean':
                    # Sized to shared_n (was a fixed length 4, independent
                    # of whatever length the 1D int[] siblings in the same
                    # parameter list use) -- LLPPrim's `fixed` boolean[]
                    # must be exactly as long as its sibling `parent`
                    # int[] (both describe the same n vertices), and a 4
                    # vs. 8 mismatch between them crashed it.
                    decls.append(
                        f'boolean[] {name} = new boolean[] '
                        '{' + ', '.join(['false'] * shared_n) + '};')
                    args.append(name)
                elif ty.base == 'double':
                    # Sized to shared_n for the same cross-array-length
                    # reason as boolean[] above (was a fixed length 4).
                    decls.append(
                        f'double[] {name} = new double[] '
                        '{' + ', '.join(f'{float(k + 1)}' for k in range(shared_n)) + '};')
                    args.append(name)
                else:
                    return None
        return decls, args


# ============================================================
#  Observer composition: `class NAME observes SUBJECT with OBSERVER;`
# ============================================================
#
# An observer program (Dijkstra-Scholten, say) is a template written
# against a generic message, meant to be attached to some specific
# application's sends and receives without editing the application. It
# exposes up to four hook methods, all optional:
#   onActivate(i)     -- this process is about to react to one event
#   onReceive(i, j)   -- it just received a message from j
#   beforeSend(i, j)  -- it is about to send a message to j
#   onDeactivate(i)   -- it is done reacting to the current event
# merge_observed_program weaves calls to whichever of these SUBJECT's
# observer defines around every `upon receive(...)`, every other guarded
# rule, `initiate` if present, and every `send` anywhere in SUBJECT
# (including inside its own auxiliary methods) -- SUBJECT's source is used
# unchanged. The observer's own fields are renamed to avoid colliding with
# SUBJECT's, and its own message types and rules (Dijkstra-Scholten's own
# `upon receive(SIG)`, for instance) are carried over as well.

_OBSERVER_HOOK_NAMES = ('onActivate', 'onReceive', 'beforeSend', 'onDeactivate')


def _rename_ids(node, rename_map: dict):
    """Recursively rename Expr(kind='id'|'index', name=X) and Stmt.lhs_name/
    decl_name references appearing anywhere in node to rename_map[X], for
    every X in rename_map.  Mutates node in place; node may be a Stmt, an
    Expr, a list of either, or None."""
    if node is None or not rename_map:
        return
    if isinstance(node, list):
        for x in node:
            _rename_ids(x, rename_map)
        return
    if isinstance(node, Expr):
        if node.kind in ('id', 'index') and node.name in rename_map:
            node.name = rename_map[node.name]
        for fl in dataclasses.fields(node):
            if fl.name == 'name':
                continue
            _rename_ids(getattr(node, fl.name), rename_map)
        return
    if isinstance(node, Stmt):
        if node.lhs_name in rename_map:
            node.lhs_name = rename_map[node.lhs_name]
        if node.decl_name in rename_map:
            node.decl_name = rename_map[node.decl_name]
        for fl in dataclasses.fields(node):
            if fl.name in ('lhs_name', 'decl_name'):
                continue
            _rename_ids(getattr(node, fl.name), rename_map)
        return
    # Decl, Type, str, int, dict, etc.: nothing under these to rename.


def _instrument_sends(stmts: List[Stmt], hook_name: Optional[str],
                       i_arg: 'Expr') -> List[Stmt]:
    """Return a new statement list like stmts, but with a call to
    hook_name(i_arg, dest) inserted immediately before every `send`
    statement, searched recursively through nested if/forall/block bodies
    and par branches.  A no-op, returning stmts unchanged, when hook_name
    is None (the observer does not define beforeSend)."""
    if hook_name is None:
        return stmts
    out: List[Stmt] = []
    for s in stmts:
        if s.kind == 'send':
            out.append(Stmt(kind='call', call_name=hook_name,
                             call_args=[i_arg, s.send_dest]))
            out.append(s)
            continue
        s2 = dataclasses.replace(s)
        if s.then_stmts:
            s2.then_stmts = _instrument_sends(s.then_stmts, hook_name, i_arg)
        if s.else_stmts:
            s2.else_stmts = _instrument_sends(s.else_stmts, hook_name, i_arg)
        if s.body:
            s2.body = _instrument_sends(s.body, hook_name, i_arg)
        if s.par_branches:
            s2.par_branches = _instrument_sends(s.par_branches, hook_name, i_arg)
        if s.guard_advance is not None:
            ga_body = (s.guard_advance.body if s.guard_advance.kind == 'block'
                       else [s.guard_advance])
            s2.guard_advance = Stmt(
                kind='block', body=_instrument_sends(ga_body, hook_name, i_arg))
        out.append(s2)
    return out


def _wrap_reactive_body(body: List[Stmt], i_arg: 'Expr',
                         j_arg: Optional['Expr'],
                         observer_methods: set) -> List[Stmt]:
    """One reactive step of SUBJECT -- an `upon` action, or `initiate`'s
    body -- wrapped with the observer's hooks: onActivate first, then
    onReceive if this step was triggered by a message from j_arg, then the
    original body with every send in it preceded by beforeSend, then
    onDeactivate last.  Any hook the observer does not define is omitted,
    not stubbed."""
    out: List[Stmt] = []
    if 'onActivate' in observer_methods:
        out.append(Stmt(kind='call', call_name='onActivate', call_args=[i_arg]))
    if j_arg is not None and 'onReceive' in observer_methods:
        out.append(Stmt(kind='call', call_name='onReceive',
                         call_args=[i_arg, j_arg]))
    before_send = 'beforeSend' if 'beforeSend' in observer_methods else None
    out.extend(_instrument_sends(body, before_send, i_arg))
    if 'onDeactivate' in observer_methods:
        out.append(Stmt(kind='call', call_name='onDeactivate', call_args=[i_arg]))
    return out


def merge_observed_program(subject: Program, observer: Program,
                            composed_name: str) -> Program:
    """Resolve `class COMPOSED observes SUBJECT with OBSERVER;` into a
    normal Program, by weaving OBSERVER's hooks around SUBJECT exactly as
    described in the module comment above."""
    observer_methods = {m.name for m in observer.aux_methods
                         if m.name in _OBSERVER_HOOK_NAMES}
    if not observer_methods:
        raise SyntaxError(
            f"'{observer.name}' defines none of the hook methods "
            f"{', '.join(_OBSERVER_HOOK_NAMES)}; an observer needs at "
            f"least one to do anything when composed")

    # Rename the observer's fields so they cannot collide with the
    # subject's, then make sure that actually worked.
    # camelCase, not underscore-joined: an underscore in an identifier
    # renders as a LaTeX subscript operator once llpp.py wraps it in
    # \mathit{...}, which either errors or mangles the figure.
    def _prefixed(name: str) -> str:
        return f'{observer.name}{name[0].upper()}{name[1:]}'
    rename_map = {d.name: _prefixed(d.name) for d in observer.decls}
    observer_decls: List[Decl] = []
    for d in observer.decls:
        d2 = dataclasses.replace(d)
        d2.name = rename_map[d.name]
        _rename_ids(d2.init, rename_map)
        observer_decls.append(d2)
    subject_names = {d.name for d in subject.decls}
    for d in observer_decls:
        if d.name in subject_names:
            raise SyntaxError(
                f"composing '{observer.name}' into '{subject.name}' still "
                f"collides on field '{d.name}' even after prefixing; "
                f"rename it in one of the two programs")

    # Message tags and enums: identical declarations may coexist (the same
    # name naturally means the same thing), but two different ones can't.
    messages = dict(subject.messages)
    for tag, flds in observer.messages.items():
        if tag in messages and messages[tag] != flds:
            raise SyntaxError(
                f"both '{subject.name}' and '{observer.name}' declare "
                f"message `{tag}` with different fields")
        messages[tag] = flds
    enums = dict(subject.enums)
    for ename, vals in observer.enums.items():
        if ename in enums and enums[ename] != vals:
            raise SyntaxError(
                f"both '{subject.name}' and '{observer.name}' declare "
                f"enum `{ename}` differently")
        enums[ename] = vals

    i_arg = Expr(kind='id', name=subject.process_index or 'i')

    # Weave the hooks into every one of the subject's top-level rules.
    # Anything that is not a guard (rare -- a leftover init-time statement
    # such as an array-filling forall) passes through untouched: it runs
    # before the process is ever active in the diffusing-computation sense.
    stmts: List[Stmt] = []
    for s in subject.stmts:
        if s.kind != 'guard' or s.guard_advance is None:
            stmts.append(s)
            continue
        j_arg = None
        if s.recv_tag is not None:
            j_name = s.recv_from or (s.guard_index[0] if s.guard_index else 'j')
            j_arg = Expr(kind='id', name=j_name)
        orig_body = (s.guard_advance.body if s.guard_advance.kind == 'block'
                     else [s.guard_advance])
        new_body = _wrap_reactive_body(orig_body, i_arg, j_arg, observer_methods)
        s2 = dataclasses.replace(s)
        s2.guard_advance = Stmt(kind='block', body=new_body)
        stmts.append(s2)
    # The observer's own rules (e.g. Dijkstra-Scholten's `upon receive(SIG)`)
    # are independent of the subject's and carry over with fields renamed.
    for s in observer.stmts:
        s2 = dataclasses.replace(s)
        _rename_ids(s2, rename_map)
        stmts.append(s2)

    # `initiate`, if the subject has one, is a reactive step exactly like an
    # `upon` action, just with no sender to report to onReceive. Every other
    # subject aux method only needs its own sends instrumented -- it is
    # always called from within an already-wrapped step.
    aux_methods: List[Method] = []
    before_send = 'beforeSend' if 'beforeSend' in observer_methods else None
    for m in subject.aux_methods:
        if m.name == 'initiate':
            m2 = dataclasses.replace(m)
            m2.body = _wrap_reactive_body(m.body, i_arg, None, observer_methods)
        else:
            m2 = dataclasses.replace(m)
            m2.body = _instrument_sends(m.body, before_send, i_arg)
        aux_methods.append(m2)
    # The observer's own hook methods (and any other helpers it defines)
    # are copied verbatim, field-renamed: they ARE the instrumentation, so
    # their own sends (Dijkstra-Scholten's SIG, say) must not be
    # instrumented again.
    for m in observer.aux_methods:
        m2 = dataclasses.replace(m, body=[dataclasses.replace(s) for s in m.body])
        _rename_ids(m2.body, rename_map)
        aux_methods.append(m2)

    return Program(
        name=composed_name,
        params=list(subject.params),
        decls=list(subject.decls) + observer_decls,
        mode=subject.mode,
        priority_expr=subject.priority_expr,
        priority_order=subject.priority_order,
        base=subject.base,
        process_name=subject.process_name,
        process_index=subject.process_index,
        channel=subject.channel,
        stmts=stmts,
        result_name=subject.result_name,
        trailing_return=subject.trailing_return,
        aux_methods=aux_methods,
        return_type=subject.return_type,
        entry_name=subject.entry_name,
        messages=messages,
        enums=enums,
        records={**subject.records, **observer.records},
        record_inits={**subject.record_inits, **observer.record_inits},
        anon_records=subject.anon_records | observer.anon_records,
    )


# ============================================================
#  Union composition: `class NAME union A, B, ...;`
# ============================================================
#
# Where `observes ... with ...` is a monitor bolted onto an application
# (asymmetric: the observer's fields are prefixed and its calls are woven
# around the subject's sends/receives), `union` is for two or more peer
# rule sets that all run at the same process index with no hook coupling
# between them -- e.g. a philosopher's own state transitions and a
# separate fork-granting protocol, each written and readable on its own,
# composed so that a rule from either fires whenever it is enabled. Every
# member's top-level guards are simply pooled: nothing is renamed, nothing
# is woven around anything else.
#
# A field, message, or enum may be declared by more than one member -- that
# is how they share state (e.g. every member that has rules touching
# `state` declares `State state;` itself, so it type-checks stand-alone) --
# but two members' declarations of the same name must actually agree:
# same type, and at most one of them may supply an initializer. This is
# checked, not assumed; a real conflict is a compile error naming both
# members, not a silent pick of one side.

def _merge_decls(existing: dict, member_name: str,
                  decls: List[Decl]) -> None:
    """Fold one member's decls into `existing` (name -> Decl), in place.
    A repeated name must match in type; at most one occurrence across all
    members may carry an initializer."""
    for d in decls:
        prev = existing.get(d.name)
        if prev is None:
            existing[d.name] = dataclasses.replace(d)
            continue
        if prev.type != d.type:
            raise SyntaxError(
                f"union: field '{d.name}' is declared with different types "
                f"by different members (one has {prev.type}, '{member_name}' "
                f"has {d.type}) -- rename one of them")
        if prev.init is not None and d.init is not None and prev.init != d.init:
            raise SyntaxError(
                f"union: field '{d.name}' is initialized differently by "
                f"different members -- only one member may initialize a "
                f"field shared this way; drop the other initializer")
        if prev.init is None and d.init is not None:
            existing[d.name] = dataclasses.replace(prev, init=d.init)
        # else: keep prev (either it already carries the initializer, or
        # neither member does).


def _merge_named_dict(existing: dict, member_name: str, kind: str,
                       table: dict) -> None:
    """Fold one member's `messages` or `enums` dict into `existing`, in
    place, raising if the same name is declared differently by two
    members (mirrors the equivalent check in merge_observed_program)."""
    for tag, val in table.items():
        if tag in existing and existing[tag] != val:
            raise SyntaxError(
                f"union: {kind} `{tag}` is declared differently by two "
                f"members (one of them is '{member_name}')")
        existing[tag] = val


def merge_union_programs(members: List[Program], composed_name: str) -> Program:
    """Resolve `class COMPOSED union A, B, ...;` into a normal Program, by
    pooling every member's fields, messages, enums, and top-level rules.
    See the module comment above for the sharing rules."""
    if len(members) < 2:
        raise SyntaxError(
            f"'{composed_name}': union needs at least two members")

    decls_by_name: dict = {}
    messages: dict = {}
    enums: dict = {}
    records: dict = {}
    record_inits: dict = {}
    anon_records: set = set()
    stmts: List[Stmt] = []
    aux_methods: List[Method] = []
    aux_method_names: dict = {}   # name -> owning member, for the error message

    # Fields that must agree across every member that specifies one
    # (anything but the default is a real, shared property of the process
    # this union runs as -- not something one member should silently
    # override another on).
    agreed = {
        'mode': ('mode', 'array'),
        'priority_expr': ('priority_expr', None),
        'priority_order': ('priority_order', 'min'),
        'base': ('base', 0),
        'process_name': ('process_name', 'P'),
        'process_index': ('process_index', None),
        'channel': ('channel', None),
        'result_name': ('result_name', None),
        'trailing_return': ('trailing_return', None),
        'return_type': ('return_type', None),
        'entry_name': ('entry_name', ''),
    }
    agreed_val = {k: default for k, (_, default) in agreed.items()}
    params: List[Decl] = []
    stateful = False

    for m in members:
        _merge_decls(decls_by_name, m.name, m.decls)
        _merge_named_dict(messages, m.name, 'message', m.messages)
        _merge_named_dict(enums, m.name, 'enum', m.enums)
        records.update(m.records)
        record_inits.update(m.record_inits)
        anon_records |= m.anon_records
        stmts.extend(dataclasses.replace(s) for s in m.stmts)
        for meth in m.aux_methods:
            if meth.name in aux_method_names:
                raise SyntaxError(
                    f"union: auxiliary method '{meth.name}' is defined by "
                    f"both '{aux_method_names[meth.name]}' and '{m.name}' -- "
                    f"rename one of them")
            aux_method_names[meth.name] = m.name
            aux_methods.append(dataclasses.replace(meth))
        if not params:
            params = list(m.params)
        stateful = stateful or m.stateful
        for field_name, (_, default) in agreed.items():
            v = getattr(m, field_name)
            if v == default:
                continue
            cur = agreed_val[field_name]
            if cur != default and cur != v:
                raise SyntaxError(
                    f"union: '{field_name}' is set inconsistently across "
                    f"members ('{m.name}' has {v!r}, another member has "
                    f"{cur!r}) -- every member sharing a process must agree")
            agreed_val[field_name] = v
    if not agreed_val['entry_name']:
        agreed_val['entry_name'] = 'init'

    return Program(
        name=composed_name,
        params=params,
        decls=list(decls_by_name.values()),
        stmts=stmts,
        aux_methods=aux_methods,
        messages=messages,
        enums=enums,
        records=records,
        record_inits=record_inits,
        anon_records=anon_records,
        stateful=stateful,
        **agreed_val,
    )


# ============================================================
#  Driver
# ============================================================

def scan_signatures(src_path: str, also_dirs: Optional[List[str]] = None) -> dict:
    """Parse every .llp/.ll file in the same directory as src_path -- and in
    each of also_dirs, e.g. for `observes ... with ...` composing a program
    from a directory other than its own -- and build a mapping
    {program_name: Program}.  Used by CodeGen to resolve inter-program
    calls.  Files that fail to parse are skipped (they may use features the
    caller does not care about).
    """
    out: dict = {}
    dirs = [os.path.dirname(os.path.abspath(src_path)) or '.']
    dirs.extend(also_dirs or [])
    for dir_ in dirs:
        for f in os.listdir(dir_):
            if not (f.endswith('.llp') or f.endswith('.ll')):
                continue
            try:
                with open(os.path.join(dir_, f)) as fh:
                    src = fh.read()
                stem = os.path.splitext(f)[0]
                prog = Parser(tokenize(src), default_name=stem).parse_program()
                out[prog.name] = prog
            except Exception:
                continue
    _resolve_signatures(out)
    return out


def _resolve_signatures(sigs: dict) -> None:
    """Resolve every `observes ... with ...`/`union` stub Program in
    `sigs`, in place, so a composed program used as another program's
    composed-field type -- e.g. DrinkingPhilosophers' `DiningPhilosophers
    dining;`, where DiningPhilosophers is itself `union Philosopher,
    ForkProtocol;` -- or as an outer composition's own member, is already
    the fully merged Program CodeGen/merge_union_programs/
    merge_observed_program expect (with real enums/messages/decls/stmts),
    not the two-line stub with none of those yet. Membership is resolved
    bottom-up, recursively, so a composed program may itself be built from
    other composed programs to any depth; a cycle is a compile error
    naming the program that closes it, not infinite recursion."""
    resolving: set = set()

    def resolve(name: str) -> Program:
        prog = sigs.get(name)
        if prog is None or (prog.union_members is None
                             and prog.observes_subject is None):
            return prog
        if name in resolving:
            raise SyntaxError(
                f"'{name}': composition cycle (a program cannot, even "
                f"indirectly, observe/union itself)")
        resolving.add(name)
        try:
            if prog.observes_subject is not None:
                subject = resolve(prog.observes_subject)
                observer = resolve(prog.observes_observer)
                if subject is None or observer is None:
                    return prog   # unresolved; let the original caller
                                   # report the specific missing name
                merged = merge_observed_program(subject, observer, prog.name)
            else:
                members = [resolve(m) for m in prog.union_members]
                if any(m is None for m in members):
                    return prog
                merged = merge_union_programs(members, prog.name)
        finally:
            resolving.discard(name)
        sigs[name] = merged
        return merged

    for name in list(sigs.keys()):
        resolve(name)


def main():
    if len(sys.argv) < 2:
        print('Usage: llc.py FILE.llp [--out DIR] [--noclass] '
              '[--also DIR ...]', file=sys.stderr)
        sys.exit(2)
    src_path = sys.argv[1]
    out_dir = '.'
    if '--out' in sys.argv:
        i = sys.argv.index('--out')
        out_dir = sys.argv[i + 1]
    noclass = '--noclass' in sys.argv
    # `--also DIR`, repeatable: extra directories to search for the SUBJECT
    # and OBSERVER of an `observes ... with ...` declaration, when they do
    # not live beside this file.
    also_dirs = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == '--also']
    with open(src_path) as f:
        src = f.read()
    toks = tokenize(src)
    stem = os.path.splitext(os.path.basename(src_path))[0]
    prog = Parser(toks, default_name=stem).parse_program()
    sigs = scan_signatures(src_path, also_dirs)
    sigs[prog.name] = prog   # always include self
    if prog.observes_subject is not None:
        subject = sigs.get(prog.observes_subject)
        observer = sigs.get(prog.observes_observer)
        if subject is None:
            print(f"llc: {src_path}: subject '{prog.observes_subject}' of "
                  f"`observes' not found (pass its directory with --also)",
                  file=sys.stderr)
            sys.exit(1)
        if observer is None:
            print(f"llc: {src_path}: observer '{prog.observes_observer}' of "
                  f"`observes' not found (pass its directory with --also)",
                  file=sys.stderr)
            sys.exit(1)
        prog = merge_observed_program(subject, observer, prog.name)
        sigs[prog.name] = prog
    if prog.union_members is not None:
        members = []
        for member_name in prog.union_members:
            member = sigs.get(member_name)
            if member is None:
                print(f"llc: {src_path}: union member '{member_name}' "
                      f"not found (pass its directory with --also)",
                      file=sys.stderr)
                sys.exit(1)
            members.append(member)
        prog = merge_union_programs(members, prog.name)
        sigs[prog.name] = prog
    cg = CodeGen(prog, signatures=sigs, noclass=noclass, src_text=src)
    java = cg.emit()
    os.makedirs(out_dir, exist_ok=True)
    # A type carried in a message is shared by every program that sends or
    # receives it, so it is written beside them rather than inside one.
    for tname, tsrc in cg.shared_types.items():
        tpath = os.path.join(out_dir, f'{tname}.java')
        if os.path.exists(tpath):
            have = open(tpath).read()
            if have != tsrc and have.startswith('// A type that travels'):
                # Two programs beside each other declare the same payload type
                # differently.  They cannot talk to each other -- one would
                # send what the other cannot read -- so say so here rather
                # than leave a cast to fail at run time.
                print(f'llc: {src_path}: `{tname}\' does not match the '
                      f'declaration already in {tpath}.\n'
                      f'  A type carried in a message is shared by every '
                      f'program that sends or receives it, so the '
                      f'declarations have to agree.', file=sys.stderr)
                sys.exit(1)
            if have == tsrc:
                continue
        with open(tpath, 'w') as f:
            f.write(tsrc)
        print(f'wrote {tpath}')

    out_path = os.path.join(out_dir, f'{prog.name}.java')
    with open(out_path, 'w') as f:
        f.write(java)
    print(f'wrote {out_path}')
    # A message program is only half a program without the runtime it extends,
    # so put a copy where javac will look for it: beside the output.
    if 'extends LLProcess' in java:
        rt_src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'LLProcess.java')
        rt_dst = os.path.join(out_dir, 'LLProcess.java')
        if os.path.exists(rt_src) and not os.path.samefile(
                os.path.dirname(rt_src) or '.', out_dir or '.'):
            with open(rt_src) as f:
                rt = f.read()
            if not (os.path.exists(rt_dst) and open(rt_dst).read() == rt):
                with open(rt_dst, 'w') as f:
                    f.write(rt)


if __name__ == '__main__':
    main()
