---
name: boba-cali-macros
description: Place, move, or review `BOBA_CALI_MARK`, `BOBA_CALI_BEGIN`, `BOBA_CALI_SWITCH`, and `BOBA_CALI_END` in BoBa code. Use when refactoring backend wrappers, splitting algorithms across helper layers, or fixing profiling ownership so CALI scopes match the code that actually owns each phase.
---

# BOBA CALI Macros

Use this skill when editing BoBa profiling scopes.

## Ownership Rules

- Put `BOBA_CALI_MARK` on public entry points or meaningful top-level operations, not on every small helper.
- Keep each `BOBA_CALI_BEGIN(...)` and its matching `BOBA_CALI_END(...)` or `BOBA_CALI_SWITCH(...,...)` in the layer that owns the full profiled operation.
- Keep each `BOBA_CALI_SWITCH(from, to)` next to the code that actually transitions between those two phases.
- For each `BOBA_CALI_SWITCH(B, C)`, there must be a preceeding `BOBA_CALI_BEGIN(B)` OR `BOBA_CALI_SWITCH(A, B)` to match `B`
- For each `BOBA_CALI_SWITCH(B, C)`, there must be a following `BOBA_CALI_END(C)` OR `BOBA_CALI_SWITCH(C, D)` to match `C`
- Every function that has `BOBA_CALI...` profiling region must either (1) start with `BOBA_CALI_BEGIN(...)` OR (2) have  `BOBA_CALI_MARK` and no other `BOBA_CALI...` macros
- Every function that has `BOBA_CALI_BEGIN(...)` must have a  `BOBA_CALI_END(...)` that corresponds to the last  `BOBA_CALI_SWITCH(...,...)` or `BOBA_CALI_BEGIN(...)` if there are no switches
- Do not split ownership of one phase transition across caller and callee.
- Do not duplicate the same `SWITCH` in both the front-end wrapper and the backend helper.
- Ensure that if the function ends to hits a `return` that there is a `BOBA_CALI_END(...)` to close the profiling region
- You cannot have multiple ``BOBA_CALI_BEGIN(...)` in the same scope. Use `BOBA_CALI_SWITCH(...,...)` instead.

## Consistency Rules

- Prefer fewer, accurate phases over many weakly defined phases.
- Check whether a helper now performs multiple profiled phases but still has no local switches.
- Rebuild and run the affected target after moving CALI macros.
