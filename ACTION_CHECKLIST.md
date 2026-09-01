# Action Checklist - Code Review Implementation

## 📋 Pre-Implementation Checklist

- [ ] Read README_REVIEW.md for overview (5 min)
- [ ] Read REVIEW_SUMMARY.md for visual overview (5-10 min)
- [ ] Read CODE_REVIEW.md for detailed analysis (15-20 min)
- [ ] Review CONSISTENCY_ISSUES.md for code patterns (10-15 min)
- [ ] Study REFACTORING_GUIDE.md for implementation details (20-30 min)
- [ ] Review VISUAL_GUIDE.md for flowcharts and diagrams
- [ ] Create a branch for refactoring: `refactor/code-cleanup`
- [ ] Ensure all tests pass before starting: `pytest` or `python -m src.experiment*`
- [ ] Document baseline metrics (duplication %, test pass rate, etc.)

---

## 🔴 PHASE 1: Quick Wins (30 minutes)

**Objective:** Remove critical code duplication and fix obvious bugs

### Task 1.1: Remove Duplicate `get_memory_size_kb()`
- [ ] Open `src/utils.py` (line 5-10)
- [ ] Delete the `get_memory_size_kb()` function
- [ ] Add import: `from src.state_utils import get_memory_size_kb`
- [ ] Verify all usages in utils.py still work
- [ ] Test: `python -c "from src.utils import get_memory_size_kb; print(get_memory_size_kb('/tmp'))"`
- [ ] **Expected:** Function works, no ImportError

### Task 1.2: Fix Silent Exception Handling
- [ ] Open `src/experiment3.py` (line 50-54)
- [ ] Replace try-except block:
  ```python
  # OLD:
  try:
      del ssm_states
      del conv_states
  except Exception:
      pass
  
  # NEW: (only catch expected errors)
  try:
      del ssm_states
      del conv_states
  except (NameError, AttributeError, UnboundLocalError):
      pass
  ```
- [ ] Test: `python -m src.experiment3` (verify no silent failures)
- [ ] **Expected:** Code runs cleanly, no hidden errors

### Completion Checklist for Phase 1
- [ ] All Python files pass basic import check
- [ ] experiment3.py runs without warnings
- [ ] git diff shows 2 files changed, 2 functions removed
- [ ] Run tests: All pass ✓

---

## 🟡 PHASE 2: Consolidation (2-3 hours)

**Objective:** Create utility modules and standardize patterns

### Task 2.1: Create `src/cache_utils.py`
- [ ] Create new file: `src/cache_utils.py`
- [ ] Copy template from REFACTORING_GUIDE.md (cache_utils.py section)
- [ ] Implement all functions with proper docstrings
- [ ] Add type hints to all functions
- [ ] Test each function:
  ```python
  pytest -xvs tests/test_cache_utils.py
  ```
- [ ] **Expected:** All tests pass, 100% code coverage

### Task 2.2: Create `src/aggregation_utils.py`
- [ ] Create new file: `src/aggregation_utils.py`
- [ ] Copy `aggregate_by_boundary_offset()` from experiment3.py
- [ ] Copy `aggregate_boundary_only_by_session()` from experiment3.py
- [ ] Copy `summarize_boundary_health()` from experiment3.py
- [ ] Add comprehensive docstrings
- [ ] Add type hints
- [ ] Test functions independently
- [ ] **Expected:** Functions work identically to original

### Task 2.3: Create `src/log_utils.py`
- [ ] Create new file: `src/log_utils.py`
- [ ] Copy template from REFACTORING_GUIDE.md (log_utils.py section)
- [ ] Implement all logging functions
- [ ] Add docstrings and type hints
- [ ] Test print output format
- [ ] **Expected:** Consistent, professional output format

### Task 2.4: Update `src/experiment3.py`
- [ ] Remove duplicate aggregation functions (lines 70-96)
- [ ] Add imports:
  ```python
  from src.aggregation_utils import (
      aggregate_by_boundary_offset,
      aggregate_boundary_only_by_session,
      summarize_boundary_health
  )
  ```
- [ ] Update all print statements to use `log_utils`
- [ ] Update path handling to use `pathlib.Path`
- [ ] Test: `python -m src.experiment3` (verify same output)
- [ ] **Expected:** 50+ lines removed, same functionality

### Task 2.5: Update `src/experiment4.py`
- [ ] Remove duplicate aggregation functions (lines 90-116)
- [ ] Add imports from `aggregation_utils`
- [ ] Update print statements to use `log_utils`
- [ ] Update path handling to use `pathlib.Path`
- [ ] Test: `python -m src.experiment4` (verify same output)
- [ ] **Expected:** 50+ lines removed, same functionality

### Task 2.6: Update All Experiments - Path Handling
For each file (experiment1.py, experiment2.py, experiment3.py, experiment4.py):
- [ ] Replace all string concatenation with pathlib.Path:
  ```python
  # OLD: str(root) + "/" + paths["output_dir"]
  # NEW: Path(root) / paths["output_dir"]
  ```
- [ ] Use Path.mkdir(parents=True, exist_ok=True) consistently
- [ ] Test that paths work correctly on your system
- [ ] **Expected:** All paths work, platform-independent

### Task 2.7: Update Imports to be Consistent
For all experiment files:
- [ ] Use consistent import style:
  ```python
  # Choose ONE style and use everywhere:
  # Option A (recommended):
  from src.evaluate import evaluate_baseline, evaluate_injected
  
  # Option B (if preferred):
  from src import evaluate
  # Usage: evaluate.evaluate_baseline(...)
  ```
- [ ] Update all usages to match chosen style
- [ ] Test: All experiments run correctly
- [ ] **Expected:** Consistent import style across all files

### Completion Checklist for Phase 2
- [ ] 3 new utility modules created (cache, aggregation, log)
- [ ] experiment3.py and experiment4.py updated
- [ ] All path handling uses pathlib
- [ ] All imports follow one style
- [ ] All experiments run: `python -m src.experiment{1,2,3,4}`
- [ ] Output is identical to before refactoring
- [ ] git diff shows 5-6 files changed, 30%+ duplication removed
- [ ] Run tests: All pass ✓

---

## 🔵 PHASE 3: Major Refactor (3-4 hours)

**Objective:** Extract common patterns and improve code structure

### Task 3.1: Create Base State Runner (Optional but Recommended)
- [ ] Create new file: `src/state_runner.py` (or `src/utils/state_runner.py`)
- [ ] Design base class for state management pattern
- [ ] Copy template from REFACTORING_GUIDE.md
- [ ] Implement common logic from experiments
- [ ] Add comprehensive docstrings
- [ ] Add type hints
- [ ] Test independently
- [ ] **Expected:** Reduces code duplication by 40%+

### Task 3.2: Add Type Hints to All Files
For each Python file in src/:
- [ ] [ ] autoencoder.py
- [ ] [ ] autoencoder_train.py
- [ ] [ ] data.py
- [ ] [ ] evaluate.py
- [ ] [ ] experiment1.py
- [ ] [ ] experiment2.py
- [ ] [ ] experiment3.py
- [ ] [ ] experiment4.py
- [ ] [ ] model_loader.py
- [ ] [ ] mamba2_stateful.py
- [ ] [ ] plot.py
- [ ] [ ] utils.py
- [ ] [ ] state_utils.py
- [ ] [ ] cache_utils.py (NEW)
- [ ] [ ] aggregation_utils.py (NEW)
- [ ] [ ] log_utils.py (NEW)

For each file, add type hints:
```python
# Add to function signatures:
def run_experiment(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    sessions: List[Dict],
    device: str
) -> Dict[int, Dict[str, float]]:
    pass

# Add to variable declarations where needed:
output_data: Dict[int, Dict[str, float]] = {}
history_text: str = ""
```

- [ ] Test with mypy (if installed):
  ```bash
  pip install mypy
  mypy src/ --ignore-missing-imports
  ```
- [ ] Fix any type errors reported
- [ ] **Expected:** >80% type hint coverage

### Task 3.3: Add Constants for Magic Numbers
In `src/experiment2.py`:
- [ ] Add at module level (after imports):
  ```python
  # Model constants
  NUM_HEADS = 24
  HEAD_DIM = 64
  D_STATE = 128
  ```
- [ ] Replace all instances of `24, 64, 128` with constants
- [ ] Search for patterns: `view(1, 24,`, `view(1, 24, 64,`
- [ ] Test: `python -m src.experiment2` (verify same output)
- [ ] **Expected:** No magic numbers in code, clearer intent

### Task 3.4: Refactor Experiment Files (if using state_runner.py)
For each experiment file:
- [ ] [ ] experiment1.py
- [ ] [ ] experiment2.py
- [ ] [ ] experiment3.py
- [ ] [ ] experiment4.py

For each file:
- [ ] Import base class: `from src.state_runner import StateRunner`
- [ ] Refactor to use base class methods where possible
- [ ] Simplify setup and teardown code
- [ ] Test: Run experiment and verify output matches baseline
- [ ] **Expected:** 20-30% code reduction, cleaner structure

### Task 3.5: Add Comprehensive Docstrings
For all public functions and classes:
- [ ] Add docstring in Google/NumPy style:
  ```python
  def aggregate_by_boundary_offset(
      output_data: Dict,
      exclude_first_session: bool = True
  ) -> Dict[int, Dict[str, float]]:
      """Aggregate metrics by turns since session boundary.
      
      This function groups output data by the number of turns
      elapsed since a session boundary (turn_id == 0).
      
      Args:
          output_data: Dictionary with keys (session_id, turn_id)
          exclude_first_session: Whether to exclude first session
          
      Returns:
          Dictionary mapping offset → aggregated metrics
          
      Example:
          >>> output = {(0, 1): {...}, (0, 2): {...}}
          >>> result = aggregate_by_boundary_offset(output)
      """
  ```
- [ ] Ensure all public APIs documented
- [ ] Test docstrings render correctly
- [ ] **Expected:** Professional documentation

### Completion Checklist for Phase 3
- [ ] Base state runner created (optional but recommended)
- [ ] >80% type hint coverage achieved
- [ ] All magic numbers replaced with constants
- [ ] All public functions have docstrings
- [ ] Experiments still produce identical output
- [ ] `mypy` passes with minimal errors
- [ ] git diff shows significant refactoring
- [ ] Code duplication reduced to <5%
- [ ] Run tests: All pass ✓

---

## 🟢 PHASE 4: Polish & Finalization (2 hours)

**Objective:** Code formatting, type checking, and final validation

### Task 4.1: Format Code with Black
- [ ] Install black: `pip install black`
- [ ] Run formatter:
  ```bash
  black src/ --line-length 88
  ```
- [ ] Review changes: `git diff` before committing
- [ ] Commit formatted code
- [ ] **Expected:** Consistent code style across all files

### Task 4.2: Run Type Checker (mypy)
- [ ] Install mypy: `pip install mypy`
- [ ] Run type checker:
  ```bash
  mypy src/ --ignore-missing-imports --no-error-summary
  ```
- [ ] Note: You may need to skip some third-party libraries
- [ ] Fix type errors if any appear
- [ ] Aim for <20 errors on large codebase
- [ ] **Expected:** Type safety validated

### Task 4.3: Run Linter (pylint)
- [ ] Install pylint: `pip install pylint`
- [ ] Run linter:
  ```bash
  pylint src/ --disable=R0903,C0114,C0115,C0116
  ```
- [ ] Review high-severity issues
- [ ] Fix obvious issues (unused imports, undefined variables)
- [ ] Ignore style issues if not critical
- [ ] **Expected:** No critical errors

### Task 4.4: Run All Tests End-to-End
- [ ] Verify experiment1 runs: `python -m src.experiment1`
- [ ] Verify experiment2 runs (if autoencoder trained)
- [ ] Verify experiment3 runs: `python -m src.experiment3`
- [ ] Verify experiment4 runs: `python -m src.experiment4`
- [ ] Compare outputs with baseline (should be identical)
- [ ] **Expected:** All experiments pass

### Task 4.5: Update Documentation
- [ ] [ ] Update README.md with new module structure
- [ ] [ ] Add CONTRIBUTING.md with coding guidelines
- [ ] [ ] Create DEVELOPMENT.md with setup instructions
- [ ] [ ] Document new utility modules usage
- [ ] [ ] Add examples to docstrings
- [ ] **Expected:** Clear, helpful documentation

### Task 4.6: Create Commit Messages
- [ ] Ensure all changes are well-committed with clear messages:
  ```
  Commit 1: "refactor: remove duplicate get_memory_size_kb function"
  Commit 2: "feat: add cache_utils, aggregation_utils, log_utils modules"
  Commit 3: "refactor: standardize path handling with pathlib"
  Commit 4: "refactor: consolidate import styles"
  Commit 5: "refactor: extract state management patterns"
  Commit 6: "feat: add comprehensive type hints"
  Commit 7: "docs: improve docstrings and comments"
  Commit 8: "style: format code with black"
  ```
- [ ] Each commit is small and focused
- [ ] Each commit passes tests independently
- [ ] **Expected:** Clean git history

### Completion Checklist for Phase 4
- [ ] Code formatted with black
- [ ] Type hints added (>80% coverage)
- [ ] mypy passes with <20 errors
- [ ] pylint passes with no critical errors
- [ ] All experiments run successfully
- [ ] Outputs identical to baseline
- [ ] Documentation updated
- [ ] Clean commit history
- [ ] Ready for code review
- [ ] Ready to merge to main
- [ ] Final metrics: <5% duplication, >80% type coverage
- [ ] Run final tests: All pass ✓

---

## ✅ Post-Implementation Verification

### Metrics Validation
- [ ] Measure code duplication (before: 25-30%, target: <5%)
  ```bash
  # Install radon: pip install radon
  radon cc src/ -s -a
  ```
- [ ] Verify type hint coverage (before: <10%, target: >80%)
  ```bash
  # Rough estimate: lines with type hints / total lines
  ```
- [ ] Count lines of code (before: ~2000, target: ~1400)
  ```bash
  wc -l src/*.py
  ```
- [ ] Check path consistency (100% using pathlib.Path)
  ```bash
  grep -r "+" src/ | grep -c "Path"
  ```

### Functionality Validation
- [ ] experiment1 produces same output as baseline
- [ ] experiment2 produces same output as baseline
- [ ] experiment3 produces same output as baseline
- [ ] experiment4 produces same output as baseline
- [ ] Performance is not degraded
- [ ] Memory usage is not increased

### Code Quality Validation
- [ ] No duplicate functions
- [ ] No duplicate code blocks
- [ ] Consistent naming conventions
- [ ] Consistent import styles
- [ ] Consistent logging/printing
- [ ] Comprehensive type hints
- [ ] Comprehensive docstrings
- [ ] Error handling is appropriate

### Documentation Validation
- [ ] README.md updated with new structure
- [ ] All modules have module-level docstrings
- [ ] All public functions have docstrings
- [ ] All public classes have docstrings
- [ ] Type hints are accurate and complete

### Final Sign-Off
- [ ] All checklist items completed ✓
- [ ] Code review approved ✓
- [ ] Tests pass ✓
- [ ] Metrics validated ✓
- [ ] Ready for production ✓

---

## 📊 Metrics Tracking

### Before Refactoring
```
Code Duplication:    25-30%
Type Hint Coverage:  <10%
Lines of Code:       ~2000
Path Consistency:    30%
Avg Line Length:     95 chars
Cyclomatic Complexity: HIGH
Test Coverage:       N/A
```

### After Phase 1
```
Code Duplication:    20%
Type Hint Coverage:  <10%
Lines of Code:       ~1950
Path Consistency:    30%
Avg Line Length:     95 chars
Cyclomatic Complexity: HIGH
```

### After Phase 2
```
Code Duplication:    8-10%
Type Hint Coverage:  <15%
Lines of Code:       ~1700
Path Consistency:    100%
Avg Line Length:     88 chars
Cyclomatic Complexity: MEDIUM
```

### After Phase 3
```
Code Duplication:    <5%
Type Hint Coverage:  >80%
Lines of Code:       ~1400
Path Consistency:    100%
Avg Line Length:     85 chars
Cyclomatic Complexity: MEDIUM
```

### After Phase 4
```
Code Duplication:    <5%
Type Hint Coverage:  >80%
Lines of Code:       ~1400
Path Consistency:    100%
Avg Line Length:     <85 chars
Cyclomatic Complexity: MEDIUM
Test Coverage:       >70%
```

---

## 🎯 Success Criteria

The refactoring is **COMPLETE** when:

✅ Code Quality
- [ ] Code duplication < 5% (down from 25-30%)
- [ ] Type hint coverage > 80% (up from <10%)
- [ ] Lines of code reduced by 30% (2000 → 1400)
- [ ] Path handling 100% consistent (using pathlib)
- [ ] No magic numbers in code

✅ Functionality
- [ ] All experiments produce identical output to baseline
- [ ] All tests pass
- [ ] No performance regression
- [ ] No memory usage increase

✅ Maintainability
- [ ] Clear module structure
- [ ] No code duplication
- [ ] Consistent naming conventions
- [ ] Consistent error handling
- [ ] Professional logging

✅ Documentation
- [ ] All modules documented
- [ ] All functions documented
- [ ] Clear usage examples
- [ ] Contributing guidelines

✅ Process
- [ ] Changes in logical commits
- [ ] Code reviewed by peer
- [ ] CI/CD pipeline passes
- [ ] Ready for production

---

## 🆘 Troubleshooting

### Import Errors After Changes
```python
# If you get: ModuleNotFoundError: No module named 'src.cache_utils'
# Solution: Make sure new files are in src/ directory and have __init__.py

# Add to src/__init__.py if needed:
from src import cache_utils, aggregation_utils, log_utils
```

### Test Failures After Refactoring
```python
# If experiments produce different output:
# 1. Check path handling (string vs Path differences)
# 2. Verify cache operations work the same
# 3. Check logging isn't affecting data
# 4. Compare outputs character-by-character

# If type checker fails:
# 1. Install missing type stubs: pip install types-<package>
# 2. Add # type: ignore comments for unavoidable issues
# 3. Use Union for multiple types if needed
```

### Performance Issues
```python
# If code runs slower:
# 1. Profile with: python -m cProfile -s cumtime script.py
# 2. Check for unnecessary Path conversions
# 3. Verify cache operations not slowing things
# 4. Check type hints aren't causing overhead (they shouldn't)
```

---

## 📚 Additional Resources

- Python Type Hints: https://docs.python.org/3/library/typing.html
- Black Code Formatter: https://github.com/psf/black
- MyPy Type Checker: http://mypy-lang.org/
- PEP 8 Style Guide: https://www.python.org/dev/peps/pep-0008/
- Google Style Docstrings: https://google.github.io/styleguide/pyguide.html

---

**Status:** Ready to begin implementation ✓  
**Estimated Total Time:** 8-12 hours (phased over 4 weeks)  
**Expected Outcome:** Production-ready codebase with 80%+ improved quality metrics
