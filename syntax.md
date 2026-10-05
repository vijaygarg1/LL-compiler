# LL — Lattice Language

LL is a small domain-specific language for writing LLP programs in the same
style used throughout *A Systematic Approach to Parallel Algorithms*. An LL
source file compiles to a single Java class that drives the LLP main loop.

## Grammar (BNF, v1)

```
program      ::= "class" ID "{" entry_method aux_method* "}"
entry_method ::= return_type ID params "{" decls config* stmt_list "}"
                 // The entry method: a Java method whose name is
                 // independent of the class name (Java convention favours
                 // a noun class with a verb method, e.g., `class MergeSort
                 // { int[] sort(...) { ... } }`).  Its return type is
                 // declared explicitly.  The compiler emits a constructor
                 // that runs one-time field setup and the entry method
                 // itself.  External code uses `new Class(args).method(args)`.
aux_method   ::= return_type ID params "{" stmt_list "}"
                 // An auxiliary method declared inside the same class.
                 // Calls in the body to a name matching the entry or an
                 // aux method become method calls (no `new`); other names
                 // become `new Other(args).Other(args)` cross-class.
return_type  ::= "void" | type
params       ::= "(" param ("," param)* ")"
param        ::= type ID

decls        ::= decl*
decl         ::= type ID initializer? ";"
                 // Type-prefixed declarations at the top of the constructor
                 // body (before any non-decl statement) become class fields.
                 // The same shape appearing inside a nested block is a
                 // local variable.  A `return X` in the body designates
                 // which field is the program's result.
initializer  ::= "=" expr

type         ::= base_type ("[" "]")*
               | "set" "<" base_type ">" ("[" "]")?
base_type    ::= "int" | "double" | "boolean"

config       ::= "setMode"     "(" mode_kind ")" ";"?      (* omit for the default *)
               | "setPriority" "(" expr "," order ")" ";"?  (* PriorityQueue mode *)
               | "setBase"     "(" ("0" | "1") ")" ";"?     (* index base *)
               | "setProcess"  "(" ID ")" ";"?              (* index of `upon` rules *)
               | "setProcessName" "(" ID ")" ";"?           (* peer name in the figure *)
mode_kind    ::= "LinkedList" | "PriorityQueue"
order        ::= "min" | "max"

stmt_list    ::= stmt (";" stmt)* ";"?
stmt         ::= lhs "=" expr
               | "if" "(" expr ")" "{" stmt_list "}" ("else" "{" stmt_list "}")?
               | "while" "(" expr ")" "{" stmt_list "}"
               | "forall" ID "in" range ":" stmt             // single-stmt body
               | type ID (":=" expr)?                        // local decl
               | ID "(" (expr ("," expr)*)? ")"              // program call (ID uppercase)
               | "[" stmt (compose_op stmt)* ";"? "]"        // bracket composition
compose_op   ::= "[]"     // parallel (Dijkstra-style; existing)
               | "&&"     // predicate conjunction
               | "or"     // nondeterministic choice
               | ";"      // sequential
               | "|"      // pipelining
               | "<>"     // feedback
               | "sync"   // synchronous (contextual keyword)
               | "{" stmt_list ";"? "}"                      // explicit block
               | "swap" "(" expr "," expr ")"
               | "return" expr                               // early exit + result marker (operand required)
               | "forbidden" "(" ID ")" ":" expr "=>" "advance" ":" stmt
                                                             // guarded command
               | "ensure" "(" ID ")" ":" lhs cmp_op expr ";"?
                                                             // sugar for guard
               | "upon" index_list? expr upon_action
                                                             // sugar for guard
               | "send" "(" expr "," (ID ",")? expr ("," expr)* ")"
                                                             // message send
               | "upon" index_list? "receive" "(" (ID ("," ID)*)? ")"
                       ("from" ID)? upon_action              // message rule

program      ::= (enum_decl | message_decl)* ...             // before the class
enum_decl    ::= "enum" ID "{" ID ("," ID)* "}" ";"
struct_decl  ::= "struct" range? ID "{" (type ID ";")+ "}" ";"
                                                             // a declaration
message_decl ::= "message" ID "(" (type ID ("," type ID)*)? ")" ";"
index_list   ::= "(" ID ("," ID)* ")" ":"                    // optional; default (j):
upon_action  ::= "{" stmt_list "}"                           // trailing ";" optional
               | stmt                                        // single stmt; may not start with "["
lhs          ::= ID
               | ID "[" expr "]"
               | ID "[" expr "]" "[" expr "]"
               | lhs "." ID ("[" expr "]")?                 (* record field *)

range        ::= "[" expr ".." expr "]"

expr         ::= expr "||"   expr   // also accepts "or"
               | expr "&&"   expr   // also accepts "and"
               | "!" expr           // also accepts "not"
               | expr cmp_op expr
               | expr add_op expr
               | expr mul_op expr
               | unary_minus expr
               | primary
cmp_op       ::= "<" | "<=" | ">" | ">=" | "==" | "!="
add_op       ::= "+" | "-"
mul_op       ::= "*" | "/" | "%"

primary      ::= INT | REAL | "true" | "false" | "infinity"
               | ID
               | ID "[" expr "]"
               | ID "(" (expr ("," expr)*)? ")"             (* call *)
               | quantifier ID "in" range ":" expr
               | "(" expr ")"
               | "new" base_type "[" expr "]"               (* array allocation *)
               | primary "." ID ("(" (expr ("," expr)*)? ")")?
                                                            (* member access / method call *)
               | primary "." ID "[" expr "]"                (* indexed record field *)
quantifier   ::= "min" | "max" | "sum" | "count"           (* contextual *)
               | "map" | "filter"                          (* contextual *)
               | "exists" | "forall"                       (* reserved *)
```

Reserved words (27 total): `class void new forbidden advance ensure
always upon send receive message enum struct if else while forall exists
true false infinity in pick int double boolean return`.

`list` and `set` are *not* reserved: like `from` and `when` they are
recognised by text in the one position where they can occur, before a `<`
in a type.

`from` is *not* reserved: like `or` and `sync` it is recognised by text in
the one position where it can occur, after a `receive` pattern.

Boolean operators are Java spellings only: `&&`, `||`, `!`. The word forms
`and`/`or`/`not` are no longer accepted.

The names `min`, `max`, `sum`, `count`, `map`, `filter` are *not* reserved;
the parser recognises them by text + lookahead (`ID in`) in expression
position, the same trick used for `setMode`, `setPriority`, `LinkedList`,
and `PriorityQueue`.

`map i in [a..b] : E` evaluates `E` once per index `i` and returns the
resulting `int[]` in iteration order. `filter i in [a..b] : P` evaluates
the boolean `P` once per `i` and returns the `int[]` of indices for which
`P` is true. Both also work over `set<int>` ranges (e.g.,
`map i in pre[j] : G[i] + 1`).

Identifiers and integer/real literals follow standard C syntax. Line
comments start with `//` and run to end of line; block comments use
`/* ... */` and may span multiple lines.

Boolean operators use Java spelling: `&&`, `||`, `!`. The older forms
`and`, `or`, `not` are still accepted.

## Informal semantics

- Program **parameters** (the `(int[] A, int lo, int hi, ...)` header) become
  the constructor arguments of the emitted class. They are the single
  externally-supplied interface; anything previously declared with the
  `input` keyword lives here now. When `lo` and `hi` appear among the params
  the main-loop range is `[this.lo .. this.hi]` instead of `[1 .. n]`.
- `var` declares an internal state variable with optional initializer.
  All fields are `var`. The variable named in a `return X` statement is
  the program's *result* -- the only field treated specially by the
  subroutine constructor (it is forwarded by reference so recursive and
  cross-program calls can share it).
- The **program body** is a statement list that flows directly after the
  declarations (no `body :` keyword). It is lowered to `run()`.
- **`forbidden (j) : PRED => advance : STMT`** is a *statement* (not just a
  top-level rule): when reached, it enters its own fixpoint loop driven by
  the program's `mode`, iterating until no `j` satisfies `PRED`. `STMT` is a
  single statement; wrap multiple in `{ ... }`. `ensure` is syntactic sugar
  for the same form.
- **`[ A OP B (OP C)* ]`** is bracket composition. The operator `OP`
  selects the composition flavor; mixing operators inside one bracket
  is a syntax error.
  - `[]`  parallel composition (Dijkstra-style; LBRACK RBRACK).
  - `&&`  predicate conjunction: build a single LLP loop whose forbidden
          set is the union of the operands' forbidden indices.
  - `or`  nondeterministic choice between the operands.
  - `;`   sequential composition: run each branch in turn.
  - `|`   pipelining: data flows from one branch to the next.
  - `<>`    feedback composition (LT GT): the output of one branch
            loops back as the input of the other.
  - `sync`  synchronous composition (contextual keyword like `or`):
            an index advances only when forbidden in *both* branches,
            i.e., they advance in lockstep.  Distinguished from `&&`
            (predicate conjunction), which advances on the *union* of
            forbidden indices.
  Typical branches are program calls (which themselves run a fixpoint),
  or guard statements (which then compose as guarded commands under the
  selected operator). The Java back-end currently lowers all five
  operators to sequential branches with a banner comment naming the
  operator; a future `-parallel` flag can refine each one (e.g.,
  `[]` to `ForkJoinTask.invokeAll`, `&&` to a merged-loop runtime,
  etc.) without source edits.
- **`return X`** is an early exit; the operand is mandatory (no bare
  `return;`).  The named field `X` is also recorded as the program's
  result.  For most programs the result is `G`, so write `return G;`.
- **Constructor vs. method.** The LL `<ret> Name(params) { body }` block
  is compiled into two Java members: a constructor `Name(params)` that
  initialises the shared fields (e.g., `this.G = A.clone()`) and the
  recursive method `<ret> Name(params) { body }` that holds the
  algorithm.  The constructor does *not* call the method -- callers
  invoke it themselves via `instance.Name(args)`.  If the constructor
  body would be empty, it is omitted entirely (Java's default no-arg
  ctor is used).  Self-recursive `Name(args)` calls in the body become
  method calls (no `new`); cross-program `Other(args)` calls become
  `new Other(args).Other(args)` (instantiate, then run the method).
- `infinity` in a `var` initializer is emitted as
  `Double.POSITIVE_INFINITY` when the field's type is `double`, else as
  `Integer.MAX_VALUE`. Use this for decreasing-lattice fixpoints.
- The integer constant `n` is automatically bound at run-time to the length
  of the first `input` array declared (minus one, so indices 1..n are valid).
  If the program has no array input but has an `hi` parameter, `n` is set
  to `hi`.
- **`base` clause.** `base <expr>;` (optionally with `: <stmts>;`) is checked
  at the start of `run()`. If the expression is true, any base statements
  execute and `run()` returns immediately, short-circuiting the fixpoint loop.
- **`body` form.** A program may declare a single `body : <stmt_list>` in
  place of `forbidden`/`ensure` rules. The body is a composition recipe (no
  fixpoint loop); it is emitted as the contents of `run()` directly.
- **`par { S1 || S2 (|| S3)* }`** is a statement. Branches are independent
  (they operate on disjoint index sets of the shared state); the emitter
  lowers them to sequential Java with a `/* par */` marker. Each branch is
  a single statement; use `{ ... }` for multi-statement branches.
- **Program calls as statements.** Any statement of the form `Name(args)`
  where `Name` starts with an uppercase letter is treated as an invocation
  of another LL program. The compiler emits
  `new Name(args..., this.input1, ..., this.output1, ...).run();`, forwarding
  the caller's inputs and outputs by name (they must appear in the callee
  in the same order).
- **Subroutine constructor.** A program with parameters and at least one
  `output` field gets two constructors: a top-level one that initialises
  its outputs from `:=` expressions, and a subroutine one that accepts the
  outputs as arguments so recursive calls share a single reference.
- `setMode(K)` selects the data structure used to track candidate forbidden
  indices.  The default (omit `setMode`) is a plain `for j = 1..n`
  re-scan -- suitable when most indices may be forbidden at once
  (e.g., Bellman-Ford).  Otherwise pick a `java.util` class:
  - `LinkedList` — a `java.util.LinkedList<Integer>` holding all live
    indices; the main loop scans for the first forbidden node.
  - `PriorityQueue` — a `java.util.PriorityQueue<Integer>` keyed by
    `setPriority(EXPR, ORDER)` where `ORDER` is `min` or `max`.
- **`setProcess(i)`** declares the index at which this program's rules run,
  once, instead of repeating it on every rule. It is the default binder of
  an `upon` written without an index list, so a distributed program reads
  the way its figure does --- the header says `P_i ::` and each rule below
  says only `upon PRED`:

  ```
  setProcess(i);
  setBase(1);

  upon receive(REQUEST, ts) from j {
    q.insert(new Req(ts, j));
    send(j, ACK, logicalClock());
  }
  upon numAcks == n - 1 && smallest(i) { enterCriticalSection() }
  ```

  It is a *declaration*, not an effect: it names the binder of `upon` only.
  `forbidden` and `ensure` range over lattice coordinates and are untouched
  by it, and a rule that quantifies over something else --- a fork, a
  monitored process --- still writes its own index list, `upon (f) : PRED`,
  where the parentheses now mark exactly that departure. The declaration is
  read off the token stream before the body is parsed, so its position in
  the preamble does not matter.
- The `forbidden(j)` clause compiles to a Java method
  `boolean forbidden(int j)` that returns its predicate.
- The `advance : ...` clause compiles to `void advance(int j)`.
- Quantifiers (`min i in [a..b] : e`, `sum i in [a..b] : e`, ...) expand to
  a for-loop with an accumulator. `exists i in [a..b] : e` and
  `forall_pred i in [a..b] : e` expand to short-circuiting loops.
- `ensure (j) : LHS <op> RHS` is a shorthand for
  `forbidden (j) : not (LHS <op> RHS); advance : LHS := RHS`.
  The compiler desugars it before Java codegen; the pretty printer keeps
  the `ensure` form, matching the book's bold-italic notation.
- **`upon PRED { STMTS }`** is a third spelling of the same guarded
  command, meant for distributed programs, where the natural reading is
  "at this process, when PRED holds, do STMTS" rather than "index `j` is
  forbidden, so advance it". It is exactly equivalent to
  `forbidden (j) : PRED => advance : { STMTS }` and produces the identical
  Java; only the pretty printer distinguishes them, rendering `\Upon` /
  `\UponDo` instead of `\Forbidden` / `\Advance`.
  - The index list is **optional** and defaults to the program's
    `setProcess` index, or to `(j)` when the program declares none. Write
    `upon (i, j) : PRED { ... }` when another binding is wanted. The
    parser tells an index list from a parenthesized predicate by
    lookahead: `( ID (, ID)* ) :` is an index list, anything else starts
    the predicate, so `upon (G[j] < 5) { ... }` parses as intended.
  - The braces may be dropped for a **single** statement:
    `upon G[j] < t[j] G[j] = t[j];`. One restriction applies --- the
    brace-less action may not begin with `[`. The lexer discards newlines,
    so a leading `[` is absorbed as a second array index of the
    predicate's trailing identifier; wrap such an action in braces. The
    compiler detects this case and says so.
  - With braces the trailing `;` is optional, as for `if` and `while`;
    without braces it is required, as for any other statement.
  - `always : ... upon ...` chains just as it does with `forbidden` and
    `ensure`.
- **`send(DEST, PAYLOAD)`** and **`send(DEST, TAG, PAYLOAD)`** are statements
  that send a message. `DEST` is an expression naming the receiving process,
  so the destination is always explicit, matching the book's phrasing
  ("send the token to $M_j$", "send $(v, incsend, increcv)$ to the checker").
  - In the three-argument form the middle argument is a **message tag**: a
    plain identifier, conventionally upper case, emitted as a string
    constant. It *names* the message and is taken literally, so a variable
    cannot be used as a dynamic tag --- put such a value in the payload
    instead. A non-identifier in that position is a syntax error.
  - The pretty printer renders the two forms as
    `send PAYLOAD to DEST` and `send TAG(PAYLOAD) to DEST`.
  - The Java back end currently lowers `send` to a call on a runtime
    method of the same name. When channels become part of the program
    state --- which is what `receive` will require, so that a message
    arrival is a state predicate and `upon` keeps its present meaning ---
    this becomes an append to the channel from the sender to `DEST`, with
    no change to the syntax above.
  - Note that before `send` was reserved, `send(k, v)` parsed as an
    ordinary call and was miscompiled to `new send().send(k, v)`; it is
    now a distinct statement.
- **`list<T>` is an ordered, growable sequence.** `T` may be a scalar, an
  enum, or a named record. A list is declared without an initialiser and
  starts empty: `list<int> requestQ;`.
  - Operations are Java's, called with Java's syntax: `q.add(x)`,
    `q.addFirst(x)`, `q.removeFirst()`, `q.remove(x)`, `q.contains(x)`,
    `q.isEmpty()`, `q.size()`, `q.clear()`. A method call is also a
    statement, so `q.add(x);` stands on its own.
  - `forall x in q : S` iterates it, and `exists x in q : P` /
    `forall x in q : P` quantify over it --- the same `in` that already
    ranges over a set. The loop variable takes the element type, so a
    `list<Req>` yields `Req`s.
  - `pick` over a list binds an *element*, not an index.
  - It emits as `LinkedList<T>` with the element boxed. `q.remove(x)` on a
    `list<int>` removes the element `x`, not the index: Java's
    `List.remove(int)` would take it as an index, so the value is boxed.
- **`struct Name { type f; ... };` before the class declares a named record
  type**, usable wherever a type is --- including as a list's element type.
  Inside the class body the same keyword declares a *variable* with an
  anonymous record type (`struct[1..n] cut { ... };`); the two are told
  apart by where they appear, as in C.
  - `new Name(f1, ..., fk)` builds a record value, fields positional and in
    declaration order, so a record can be put into a list.
- **`pick` captures a witness**, and works wherever an `exists` may appear,
  not only in a guard. `if (exists j in [1..n] pick j : c[j] == red) { S }`
  binds `j` to an index satisfying the body, for use in `S`.
  - The binding covers the **then-branch only**: if the condition was false
    there is no witness. Which index is chosen is not specified.
  - It compiles the way a guard's `pick` already does --- the lifted
    quantifier helper stashes the index in `picked_j`, and the branch opens
    with `int j = picked_j;`.
  - The pretty printer does not show the word `pick`: the figures write
    `if (exists j : c[j] = red) then ... j ...` and let the binding be
    implied, which is ordinary mathematical usage.
- **Comparison of whole arrays is componentwise.** `a <= b` is the pointwise
  order (`forall t : a[t] <= b[t]`), `a == b` is pointwise equality, and
  `a < b` is the strict part of the order --- `a <= b` and `a != b` --- which
  is the standard order on vector clocks, so `cut[i].v < cut[j].v` says what
  the book's figures say. `>`/`>=` are the mirrors and `!=` is the negation
  of `==`.
  - It is a **partial** order: for incomparable vectors neither `a < b` nor
    `b < a` holds, so `!(a < b)` does *not* give `a >= b`. This is what makes
    `if (a < b) ... else if (b < a) ...` in paintState meaningful.
  - It applies only when *both* sides are whole `int`/`double` arrays, which
    the compiler reads off the declarations. Anything indexed down to a
    scalar compares as before.
  - Iteration starts at the program's index base, so under `setBase(1)` slot
    0 is not part of the vector.
- **`enum Name { v1, v2, ... };`** declares a value set. The values are
  used bare (`color[i] := red`), the name is a type (`Color[1..n] color`),
  and the pretty printer prints the type as its value set, which is how the
  book's figures write it: `color: array[1..n] of {red, green}`.
- **`struct [lo..hi] name { type f; ... };`** declares a *variable*, not a
  type. It is the LL spelling of the book's

    ```
    var cut: array[1..n] of struct
          v: array[1..n] of integer;
          color in {red, green};
        endstruct
    ```

  so the struct is anonymous and belongs to the variable --- there is no
  type name to invent or to repeat. `struct[1..n] cut { int[1..n] v; Color
  color; };` gives `cut[i].v` and `cut[i].color`, readable and assignable,
  and a field that is itself an array indexes further: `cut[a].v[t]`.
  - A Java class has to be called something, so one is derived from the
    variable name (`cut` -> `Cut`); it appears nowhere in LL or in the
    printed figure.
  - It goes with the other declarations at the top of the body, where a
    `struct` keyword in statement position is unambiguous.
  - `struct` and `message` are one mechanism: both declare a record, and a
    message is the subset of records that may be sent. So they share the
    field table, the arity check, and the emitter.
  - Java emits a nested class per struct. Because `new Cut[n+1]` leaves the
    array full of nulls, a record array is also filled element by element
    at construction, before any field assignment can run.
- **`message TAG(type name, ...);`** declares the payload shape of a tag.
  Declarations come before the program. They let `send` carry a tuple and
  they type the binders of `receive`; the arity is checked at both ends.
- **`upon receive(TAG, b1, ..., bk) from k { S }`** is a message rule. The
  binders are *outputs*: `k` is bound to the sender and the `b`'s to the
  payload fields declared for `TAG`. To react only to a particular sender,
  bind and then test (`if (k == leader) ...`); `from` never matches.
  - Both parts are optional: `upon receive(TAG, m)` ignores the sender,
    `upon receive() from k` fires on any message from `k`.
  - `from` also takes a *number*: `upon receive(TOKEN) from 0` is a message
    that can only come from one distinguished process --- a coordinator, a
    root, the environment. Nothing is bound, and the guard looks at that
    one channel instead of searching.
  - It desugars to an ordinary guard --- "some sender has such a message
    queued for me" --- so `upon` keeps its meaning. The witness is stashed
    exactly as a `pick` guard's is, and the dequeue is the first statement
    of the advance:
    ```
    upon receive(REQUEST, m) from k { S }
      ==  upon (j) : exists k pick k : pending(k, j, REQUEST)
                     { m := dequeue(k, j); S }
    ```
  - Because a message in transit is part of the state, a channel is
    emitted as `ArrayDeque<Msg>[][] chan` indexed `chan[src][dst]`;
    `send` appends and the guard polls. Inside a guard, "me" is the
    guard's first index, so `send(d, TAG, v)` from index `j` appends to
    `chan[j][d]`.
  - There is deliberately no `upon send`: a process controls its own
    sends, so the bookkeeping goes next to the `send` statement.
- **`class NAME observes SUBJECT with OBSERVER;`** composes two whole
  programs instead of declaring a body: `OBSERVER` is a template that
  reacts to `SUBJECT`'s activity through up to four optional hooks
  (`onActivate(i)`, `onReceive(i,j)`, `beforeSend(i,j)`, `onDeactivate(i)`),
  woven automatically around every one of `SUBJECT`'s guarded rules and
  every `send`, without editing `SUBJECT`. Its own fields are renamed to
  avoid colliding with `SUBJECT`'s, and its own rules (e.g. Dijkstra and
  Scholten's `upon receive(SIG)`) carry over unchanged. Asymmetric --
  `SUBJECT`'s code is used as-is; `OBSERVER` is instrumentation bolted onto
  it -- and meant for a monitor superimposed on an application (termination
  detection, Dijkstra-Scholten). See `merge_observed_program` in `llc.py`.
- **`class NAME union A, B, ...;`** composes two or more whole programs
  that are peers rather than monitor-and-subject: every member's top-level
  rules are simply pooled, so a rule from any member fires whenever it is
  enabled -- no hooks, no renaming, no asymmetry. Use it when a single
  process's behavior is naturally two (or more) independent rule sets, each
  readable and separately compiled on its own -- e.g. a philosopher's own
  `becomeHungry`/`finishEating` transitions versus a separate fork-granting
  protocol, both acting at the same index. A field, message, or enum
  declared by more than one member is how they share state, but two
  members' declarations of the same name must agree (same type, and at
  most one may carry an initializer) -- a real conflict is a compile
  error naming both members. `mode`, `base`, `process_index`, and the other
  process-wide settings must likewise agree wherever more than one member
  sets them. See `merge_union_programs` in `llc.py`.

## Mapping to Java (default skeleton)

The compiler emits one class per `llp` declaration:

```java
public class <Name> {
  int n;                           // from first input array
  <inputs>                         // constructor parameters
  <outputs/vars>                   // initialised from declarations

  boolean forbidden(int j) { return <forbidden-predicate>; }
  void advance(int j)      { <advance-stmts> }

  public void run() { <main-loop-for-selected-mode> }

  public static void main(String[] args) { /* demo harness */ }
}
```

The main loop is specialised per mode:

- **array** — `for j := 1..n : if forbidden(j) then advance(j); continue;`
- **linkedlist** — iterate the list, advance the first forbidden node,
  restart the scan; `delete(j)` removes the node.
- **heap** — `while (!pq.isEmpty()) { j = pq.poll(); if (forbidden(j))
  advance(j); }` with re-insertion of updated indices inside `advance`.

## Example 1 -- fixpoint program

A minimal MinMaxLate program in LL:

```
class MinMaxLate {
  int[] MinMaxLate(int[] t, int[] d) {
    int[] G = 0;
    int   lateness = 0;
    forbidden (j) : j > 1 && G[j] < G[j-1] + t[j-1] =>
      advance : {
        G[j] = G[j-1] + t[j-1];
        if (G[j] + t[j] > d[j]) {
          lateness = max(lateness, G[j] + t[j] - d[j])
        }
      };
    return G;
  }
}
```

The whole body is a single `forbidden-advance` statement; the constructor
runs the fixpoint loop. With no DSL constructs the file would parse as a
plain Java class.

## Example 2 -- divide-and-conquer with `[ A [] B ]`

MergeSort composes two parallel recursive calls with a Merge. The leaf LL
program is Merge (forbidden/advance); MergeSort is a composition.

```
class MergeSort {
  int[] sort(int lo, int hi, int[] A) {
    int[] G = 0;
    forall k in [0..n] : G[k] = A[k];
    MergeSort(lo, hi, A, G);
    return G;
  }

  void MergeSort(int lo, int hi, int[] A, int[] G) {
    if (lo < hi) {
      int mid = (lo + hi) / 2;
      [ MergeSort(lo, mid, A, G) [] MergeSort(mid+1, hi, A, G) ];
      Merge(lo, mid, hi, G);
    }
  }

  void Merge(int lo, int mid, int hi, int[] G) {
    int[] B = new int[n + 1];
    forall k in [lo..hi] : B[k] = G[k];
    int i = lo;  int j = mid + 1;  int k = lo;
    while (i <= mid && j <= hi) {
      if (B[i] <= B[j]) { G[k] = B[i]; i = i + 1; }
      else              { G[k] = B[j]; j = j + 1; };
      k = k + 1;
    };
    while (i <= mid) { G[k] = B[i]; i = i + 1; k = k + 1; };
    while (j <= hi)  { G[k] = B[j]; j = j + 1; k = k + 1; }
  }
}
```

- Each program is a Java class with one constructor; the constructor
  body holds the algorithm.
- `if (lo >= hi) { return };` is Java-style early exit.
- `[ A [] B ]` is the Dijkstra-style parallel composition (the `[]`
  separator is two adjacent `[`/`]` tokens, distinct from boolean `||`).
- The recursive `MergeSort(lo, mid)` call supplies only the varying scalar
  arguments; the compiler auto-forwards `this.A` (an array param matched by
  name) and `this.G` (a shared output) to the callee's subroutine
  constructor using the sibling-program signature map.

`par { X || Y }` keeps the operator notation of the composition chapter;
the emitter lowers to sequential Java today (with a `/* par */` comment) so
the code is correct on a single thread, and a future `-parallel` flag can
rewrite the block to `ForkJoinTask.invokeAll` without source edits.
