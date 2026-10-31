# LL — Lattice Language compiler

LL is a small domain-specific language for writing **lattice-linear
predicate (LLP)** algorithms — the forbidden/advance style of program used
throughout *A Systematic Approach to Algorithms* and *A Systematic
Approach to Parallel Algorithms* (Vijay K. Garg). An LL source file
(`.llp`) compiles to a self-contained Java class, a JavaScript module for
interactive browser demos, or a LaTeX `algorithm2e` fragment for the book
itself.

This repository holds the compiler and its companion emitters —
independent of any particular book's chapter content.

## This repository is frozen

`ll-1.0` is the language exactly as *A Systematic Approach to Algorithms*
ships it (published 2026-10-28). It is kept frozen so the programs printed
in that book can always be rebuilt with the compiler that produced them.

**Development continues in `github.com/vijaygarg1/LL2-Compiler`**, which
starts from this same code. The language stays upward compatible — every
`.llp` that compiles here compiles there — so the name does not change.
Nothing but a correctness fix to the 1.0 toolchain should land here.

## Tools

| Script | Emits | Usage |
|---|---|---|
| `llc.py` | Java | `python3 llc.py Prog.llp --out DIR/` |
| `llcpy.py` | Python | `python3 llcpy.py Prog.llp --out DIR/` |
| `llcjssrc.py` | JavaScript, the plain algorithm | `python3 llcjssrc.py Prog.llp --out DIR/` |
| `llcjs.py` | JavaScript, an `LLP.mount(container, spec)` interactive demo module | `python3 llcjs.py Prog.llp --out Prog.js` |
| `llpp.py` | LaTeX (`algorithm2e`, matching the book's `\Forbidden{...}\Advance{...}` macros) | `python3 llpp.py Prog.llp --out Prog.tex` |

All five are plain Python 3 with no external dependencies, and operate on
the same `.llp` source grammar (see `syntax.md`). The two JavaScript
emitters are different artifacts: one is the algorithm as plain source,
the other a steppable browser demo.

### Example

```
$ cat Smoke.llp
class Smoke {
  int[] run(int[] A) {
    int n = A.length;
    int[] G = new int[n];
    forall i in [0..n-1] : G[i] = A[i] * 2;
    return G;
  }
}

$ python3 llc.py Smoke.llp --out out/
wrote out/Smoke.java
$ javac -d out out/Smoke.java
```

## Language overview

An LL program is a class with one *entry method* (its return type and
parameters become the compiled class's public method) and any number of
auxiliary methods. Beyond straight-line/looping imperative code (`if`,
`while`, `forall`, arrays, `set<T>`, recursion, and calls to other LL
programs), the distinguishing construct is the **guarded command**:

```
forbidden (j) : <predicate on j and the program's state>
  => advance : <statement that repairs the violation at j>
```

The compiler wraps this in a fixed-point driver (`while (changed) { for
each j: if forbidden(j) then advance(j) }`) implementing the classical
lattice-linear-predicate-detection algorithm: pick any currently-forbidden
index, advance it, repeat until none remain. A program may also use the
`ensure` shorthand (`ensure : G[j] >= expr`) when the advance is simply
raising `G[j]` to a monotone target.

See `syntax.md` for the full grammar and `llc.py`'s own extensive internal
comments for the exact Java it emits for each construct.

## History

Extracted from the tooling built for *A Systematic Approach to
Algorithms* (github.com/vijaygarg1/systematic-algorithms) and *A
Systematic Approach to Parallel Algorithms*
(github.com/vijaygarg1/systematic-parallel-algorithms), where `.llp`
sources for each book chapter are compiled into the book's own
`algorithm2e` pseudocode and into the runnable Java/JavaScript published
on the books' companion website.
