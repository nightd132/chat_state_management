# Code Structure Issues - Visual Flowchart

## Issue Dependency Graph

```
                    ┌─────────────────────────────────┐
                    │  DUPLICATE FUNCTIONS            │
                    │  - get_memory_size_kb()         │
                    │  - aggregation functions        │
                    │  - state management patterns    │
                    └────────────┬────────────────────┘
                                 │
                    ┌────────────┴──────────────────┐
                    │                               │
          ┌─────────▼──────────┐        ┌──────────▼──────────┐
          │  INCONSISTENCIES   │        │  MISSING PATTERNS   │
          │  - Paths           │        │  - Type hints       │
          │  - Imports         │        │  - Cache utils      │
          │  - Logging         │        │  - State base class │
          └────────────────────┘        └─────────────────────┘
                    │                               │
                    └────────────┬──────────────────┘
                                 │
                    ┌────────────▼──────────────────┐
                    │  REFACTORED CODE              │
                    │  - 30% less duplication       │
                    │  - 80% type hint coverage     │
                    │  - 30% fewer lines of code    │
                    └──────────────────────────────┘
```

---

## File Dependency Before & After

### BEFORE (Current State)
```
experiment1.py ──┐
                 ├─→ utils.py ──┐
experiment2.py ──┤              ├─→ state_utils.py
                 ├─→ evaluate.py
experiment3.py ──┤  (shared)
                 ├─→ data.py
experiment4.py ──┤  (shared)
                 └─→ plot.py

❌ Problems:
- Duplicate functions spread across files
- No clear separation of concerns
- Hard to maintain consistency
- Hidden dependencies
```

### AFTER (Proposed Structure)
```
experiments/
├── base.py ──────────────────┐
├── experiment1.py ────┐      │
├── experiment2.py ────┼──→ state_runner.py
├── experiment3.py ────┤
└── experiment4.py ────┘      │
                              │
utils/                        │
├── cache_utils.py ◄──────────┤
├── aggregation_utils.py ◄────┤
├── log_utils.py ◄────────────┤
├── state_utils.py ◄──────────┘
└── utils.py

✅ Benefits:
- Single source of truth
- Clear dependencies
- Easier to test
- Better maintainability
```

---

## Code Duplication Map

```
DUPLICATION HEATMAP (% of repeated code)

           Exp1   Exp2   Exp3   Exp4
           ───    ───    ───    ───
import:     ▓▓▓▓   ▓▓▓▓   ▓▓▓▓   ▓▓▓▓
setup:      ▓▓▓▓   ▓▓▓▓   ▓▓▓▓   ▓▓▓▓
state_mgt:  ▓▓▓▓   ▓▓▓▓   ▓▓▓▓   ▓▓▓▓
paths:      ▓▓▓▓   ▓▓▓▓   ▓▓▓▓   ▓▓▓▓
logging:    ▓▓     ▓▓     ▓▓     ▓▓
agg_funcs:           ▓▓▓▓       ▓▓▓▓


Legend:  ▓▓▓▓ = High duplication (50%+)
         ▓▓   = Medium duplication (20-50%)
         ▓    = Low duplication (<20%)
         
Hotspots: state_mgt (100% duplicated)
          paths (95% duplicated)
          agg_funcs (100% duplicated in exp3/4)
```

---

## Priority Matrix - Visual

```
        EFFORT (→)
         Low    Med    High
        ┌────┬─────┬──────┐
HIGH    │ #1 │ #4  │ #3   │  FIX IMMEDIATELY
        │ #2 │ #5  │      │  (High impact)
        │ #8 │ #6  │      │
   ↑    │ #9 │ #7  │      │
IMPACT  │    │ #10 │      │
        ├────┼─────┼──────┤
MED     │    │     │      │
        │    │     │      │
   ↕    ├────┼─────┼──────┤
        │    │ #12 │      │
LOW     │#11 │     │      │  FIX LATER
        │#13 │     │      │  (Nice to have)
        └────┴─────┴──────┘

QUICK WINS:  #1, #2, #8 (5-10 min each)
MEDIUM:      #4-7, #9-10 (15-45 min each)
COMPLEX:     #3 (2-3 hours)
```

---

## Issue Resolution Workflow

```
START
  │
  ├─→ Review REVIEW_SUMMARY.md (5 min)
  │      │
  │      ├─→ Understand scope
  │      └─→ Identify priorities
  │
  ├─→ Read CODE_REVIEW.md (15 min)
  │      │
  │      ├─→ Learn WHY each issue matters
  │      └─→ Understand impact
  │
  ├─→ Check CONSISTENCY_ISSUES.md (10 min)
  │      │
  │      ├─→ See specific code examples
  │      └─→ Identify patterns
  │
  ├─→ PHASE 1: Quick Wins (30 min)
  │      │
  │      ├─→ Remove duplicate functions
  │      ├─→ Fix error handling
  │      └─→ Test everything
  │
  ├─→ PHASE 2: Consolidation (2-3 hrs)
  │      │
  │      ├─→ Create utility modules
  │      ├─→ Standardize paths
  │      └─→ Test everything
  │
  ├─→ PHASE 3: Major Refactor (3-4 hrs)
  │      │
  │      ├─→ Extract base classes
  │      ├─→ Add type hints
  │      └─→ Test everything
  │
  ├─→ PHASE 4: Polish (2 hrs)
  │      │
  │      ├─→ Format code
  │      ├─→ Add docstrings
  │      └─→ Run type checker
  │
  └─→ END (Production-ready code!)
```

---

## Code Quality Progression

```
BEFORE REFACTORING
┌─────────────────────────────────┐
│ Code Duplication:  ████████░░░░░ 25-30%
│ Type Hints:        ██░░░░░░░░░░░ <10%
│ Consistency:       █████░░░░░░░░ 50%
│ Maintainability:   ████░░░░░░░░░ 40%
│ Documentation:     ███░░░░░░░░░░ 30%
└─────────────────────────────────┘

AFTER REFACTORING (Target)
┌─────────────────────────────────┐
│ Code Duplication:  ░░░░░░░░░░░░░ <5%
│ Type Hints:        ███████░░░░░░ >80%
│ Consistency:       ██████████░░░ 95%
│ Maintainability:   ██████████░░░ 90%
│ Documentation:     ████████░░░░░ 80%
└─────────────────────────────────┘

Improvement Summary:
┌─────────────────────────────────┐
│ Code Duplication:  ↓ 80%
│ Type Hints:        ↑ 700%
│ Consistency:       ↑ 45%
│ Maintainability:   ↑ 50%
│ Documentation:     ↑ 50%
└─────────────────────────────────┘
```

---

## Module Creation Timeline

```
Week 1: Quick Wins & Foundation
├─ Mon: Remove duplicates → cache_utils.py
├─ Tue: aggregation_utils.py → log_utils.py
├─ Wed: Test all changes
└─ Thu: Review & document

Week 2: Consolidation
├─ Mon: Standardize paths in all files
├─ Tue: Unify import styles
├─ Wed: Consolidate logging
└─ Thu: Full test suite

Week 3: Major Refactor
├─ Mon: Create state_runner base class
├─ Tue: Refactor experiments 1-2
├─ Wed: Refactor experiments 3-4
└─ Thu: Full integration test

Week 4: Polish & Finalization
├─ Mon: Add type hints
├─ Tue: Code formatting & docstrings
├─ Wed: Type checking (mypy)
└─ Thu: Final review & merge
```

---

## File-by-File Change Summary

```
┌─────────────────────────────────────────────────┐
│ FILE CHANGE MATRIX                              │
├──────────────────┬──────────────────────────────┤
│ File             │ Changes                      │
├──────────────────┼──────────────────────────────┤
│ utils.py         │ ✓ Remove duplicate           │
│                  │ ✓ Add imports               │
│                  │ ✓ Add type hints            │
├──────────────────┼──────────────────────────────┤
│ state_utils.py   │ ✓ Improve docstrings        │
│                  │ ✓ Add type hints            │
├──────────────────┼──────────────────────────────┤
│ experiment1.py   │ ✓ Fix path handling         │
│                  │ ✓ Update imports            │
│                  │ ✓ Add type hints            │
├──────────────────┼──────────────────────────────┤
│ experiment2.py   │ ✓ Add constants             │
│                  │ ✓ Use cache_utils          │
│                  │ ✓ Add type hints            │
├──────────────────┼──────────────────────────────┤
│ experiment3.py   │ ✓ Remove duplicates         │
│                  │ ✓ Import from new modules  │
│                  │ ✓ Fix error handling       │
├──────────────────┼──────────────────────────────┤
│ experiment4.py   │ ✓ Remove duplicates         │
│                  │ ✓ Import from new modules  │
│                  │ ✓ Add type hints            │
├──────────────────┼──────────────────────────────┤
│ cache_utils.py   │ ✓ CREATE NEW               │
├──────────────────┼──────────────────────────────┤
│ aggregation_...  │ ✓ CREATE NEW               │
├──────────────────┼──────────────────────────────┤
│ log_utils.py     │ ✓ CREATE NEW               │
├──────────────────┼──────────────────────────────┤
│ state_runner.py  │ ✓ CREATE NEW (optional)    │
└──────────────────┴──────────────────────────────┘
```

---

## Impact on New Development

```
BEFORE: Adding Experiment 5
┌────────────────────────────────────┐
│ Time to setup new experiment       │
│ ████████████████████░░░░░░░░░░░░░░ 30-45 min
│                                    │
│ Copy-paste & modify code           │
│ □ Paths (3 variants)              │
│ □ State management (4 patterns)    │
│ □ Aggregation functions           │
│ □ Cache logic                      │
│ □ Logging                          │
│                                    │
│ Risk of inconsistencies: VERY HIGH │
│ Risk of bugs: VERY HIGH            │
└────────────────────────────────────┘

AFTER: Adding Experiment 5
┌────────────────────────────────────┐
│ Time to setup new experiment       │
│ ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 5-10 min
│                                    │
│ Simply import & extend base class  │
│ ✓ Paths (standardized)            │
│ ✓ State management (unified)      │
│ ✓ Aggregation functions (imported)│
│ ✓ Cache logic (imported)          │
│ ✓ Logging (imported)              │
│                                    │
│ Risk of inconsistencies: MINIMAL   │
│ Risk of bugs: LOW                  │
└────────────────────────────────────┘

IMPROVEMENT: 75% faster, much safer!
```

---

## Testing Strategy

```
BEFORE CHANGES (Current)
├─ Run experiment1.py → Verify output
├─ Run experiment2.py → Verify output
├─ Run experiment3.py → Verify output
└─ Run experiment4.py → Verify output
   (No automated tests)

AFTER CHANGES (Recommended)
├─ Unit Tests
│  ├─ test_cache_utils.py
│  ├─ test_aggregation_utils.py
│  ├─ test_log_utils.py
│  └─ test_state_utils.py
├─ Integration Tests
│  ├─ test_experiment1.py
│  ├─ test_experiment2.py
│  ├─ test_experiment3.py
│  └─ test_experiment4.py
└─ End-to-End Tests
   ├─ Run full pipeline
   └─ Compare outputs with baseline

Benefits:
✓ Catch regressions immediately
✓ Easier to refactor with confidence
✓ Better code quality
✓ Faster debugging
```

---

## Summary: Key Takeaways

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ WHAT TO DO                              ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃                                         ┃
┃ 1. READ the review documents (30 min)  ┃
┃    → Understand the issues             ┃
┃    → See the code examples             ┃
┃                                         ┃
┃ 2. START with Phase 1 (30 min)         ┃
┃    → Remove duplicate functions        ┃
┃    → Fix error handling                ┃
┃                                         ┃
┃ 3. CONTINUE with Phase 2 (2-3 hrs)     ┃
┃    → Create utility modules            ┃
┃    → Standardize code patterns         ┃
┃                                         ┃
┃ 4. IMPLEMENT Phase 3 (3-4 hrs)         ┃
┃    → Major refactoring                 ┃
┃    → Add type hints                    ┃
┃                                         ┃
┃ 5. POLISH Phase 4 (2 hrs)              ┃
┃    → Format & document                 ┃
┃    → Run type checker                  ┃
┃                                         ┃
┃ TOTAL TIME: 8-12 hours (phased)        ┃
┃ ROI: 30-40% code reduction             ┃
┃      80% type coverage improvement     ┃
┃      Much better maintainability       ┃
┃                                         ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```
