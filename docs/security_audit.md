# Security and Secret Exposure Audit

Audit Date: 2026-02-20
Scope: Full repository scan
Auditor: Automated static analysis

---

## Summary

| Severity | Count |
|:---------|:------|
| CRITICAL | 1 |
| HIGH | 1 |
| MEDIUM | 2 |
| LOW | 0 |

---

## Findings

### Finding 1 -- CRITICAL: Live Google API Key in .env

| Field | Value |
|:------|:------|
| Severity | CRITICAL |
| File | `.env` |
| Line | 5 |
| Content | `LLM_API_KEY=AIzaSyBcW1Ytcad1hOhsuGxmYwsgyo19zoCSFBg` |
| Type | Google Generative AI API Key |

**Description**: A live Google API key is present in the `.env` file. This key follows the `AIzaSy` prefix pattern used by Google Cloud API keys and is fully functional. If committed to a public repository, this key grants unauthorized access to the associated Google Cloud project.

**Mitigation Status**: `.env` is listed in `.gitignore`, which prevents it from being tracked by Git going forward. However, if `.env` was ever committed before `.gitignore` was created, the key remains in Git history.

**Recommendation**:

1. Rotate this API key immediately in the Google Cloud Console.
2. Verify that `.env` has never been committed to Git history. Run:
   ```bash
   git log --all --full-history -- .env
   ```
3. If `.env` was previously committed, use `git filter-branch` or BFG Repo Cleaner to purge it from history before pushing to any remote.
4. Enable API key restrictions in Google Cloud Console (restrict by HTTP referrer or IP).

---

### Finding 2 -- HIGH: Hardcoded Placeholder Secrets in Settings

| Field | Value |
|:------|:------|
| Severity | HIGH |
| File | `config/settings.py` |
| Lines | 46, 47 |

**Content**:
```python
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
API_KEY_ENTERPRISE = os.getenv("API_KEY_ENTERPRISE", "change-me-in-production")
```

**Description**: Both `JWT_SECRET` and `API_KEY_ENTERPRISE` use `"change-me-in-production"` as the default fallback value. If the `.env` file does not define these variables (which it currently does not), the application runs with a known, guessable secret. Any attacker who reads the source code can forge JWT tokens or access the enterprise API.

**Recommendation**:

1. Remove the default fallback values. Use empty strings or `None` and fail fast if the secret is required but missing.
2. Add `JWT_SECRET` and `API_KEY_ENTERPRISE` to `.env` with strong, randomly generated values.
3. Document in `.env.example` that these must be set before production deployment.

---

### Finding 3 -- MEDIUM: API Key Interpolated into URL in Debug Script

| Field | Value |
|:------|:------|
| Severity | MEDIUM |
| File | `debug_llm.py` |
| Line | 9 |

**Content**:
```python
url = f"https://generativelanguage.googleapis.com/v1beta/models/{LLM_MODEL}:generateContent?key={LLM_API_KEY}"
```

**Description**: The debug script constructs a URL with the API key in a query parameter. If this URL is logged, printed, or appears in error tracebacks, the key is exposed in plaintext. Query parameters are also typically recorded in server access logs and browser history.

**Note**: This is a debug-only script not used in production. The risk is limited to developer workstations.

**Recommendation**:

1. Move the API key from query parameter to an `Authorization` header (already done elsewhere in the same file at line 31).
2. Consider removing `debug_llm.py` from the repository before public release, or add it to `.gitignore`.

---

### Finding 4 -- MEDIUM: .env.example Contains Placeholder Secrets

| Field | Value |
|:------|:------|
| Severity | MEDIUM |
| File | `.env.example` |
| Lines | 24, 25 |

**Content**:
```
JWT_SECRET=change-me-in-production
API_KEY_ENTERPRISE=change-me-in-production
```

**Description**: While `.env.example` is intended to be committed as a template, the placeholder values are identical to the hardcoded defaults in `config/settings.py`. A developer who copies `.env.example` to `.env` without modifying these values will run the application with known secrets.

**Recommendation**:

1. Change the placeholder values to clearly unusable strings:
   ```
   JWT_SECRET=REPLACE_WITH_SECURE_RANDOM_STRING
   API_KEY_ENTERPRISE=REPLACE_WITH_SECURE_RANDOM_STRING
   ```
2. Add a comment indicating these must be changed before deployment.

---

## Environment File Status

| Check | Result |
|:------|:-------|
| `.env` exists | Yes |
| `.env` contains live credentials | Yes (Google API key) |
| `.env` is in `.gitignore` | Yes |
| `.env` was previously committed to Git | UNKNOWN -- requires `git log` verification |
| `.env.example` exists | Yes |
| `.env.example` contains live credentials | No (empty or placeholder values only) |

---

## Pattern Scan Results

| Pattern | Files Scanned | Matches |
|:--------|:--------------|:--------|
| `sk-` (OpenAI key prefix) | All source files | 0 |
| `gpt-` (model reference) | All source files | 0 (only in `.env.example` as model name) |
| `SECRET_KEY` | All source files | 0 |
| Hardcoded Bearer tokens | All `.py` files | 0 |
| `print()` leaking secrets | All `.py` files | 0 |
| `logging.*` leaking secrets | All `.py` files | 0 |
| Hardcoded `AIza*` in `.py` files | All `.py` files | 0 |
| `token = "..."` assignments | All source files | 0 |
| Base64 credential strings | All source files | 0 |
| `.DS_Store` | Entire repository | 0 |
| `*.bak` files | Entire repository | 0 |
| Jupyter notebooks | Entire repository | 0 |
| `node_modules/` | Entire repository | 0 |

---

## Git Hygiene Status

| Item | Status |
|:-----|:-------|
| `.gitignore` exists | Yes |
| `.env` excluded | Yes |
| `__pycache__/` excluded | Yes |
| `.pytest_cache/` excluded | Yes |
| `outputs/` excluded | Yes |
| `*.log` excluded | Yes |
| `*.pdf` excluded | Yes |
| `.venv/` excluded | Yes |

---

## Verdict

**DO NOT PUSH until Finding 1 is resolved.**

The live Google API key in `.env` must be rotated before the repository is pushed to any remote, even if `.env` is gitignored. If `.env` was ever committed in a previous Git operation, the key is permanently in Git history and must be purged.

After resolving Finding 1 and reviewing Findings 2-4, the repository is safe to push.

### Action Items (Priority Order)

1. Rotate the Google API key in Google Cloud Console.
2. Verify `.env` is not in Git history.
3. Replace placeholder defaults in `config/settings.py` with fail-fast logic.
4. Update `.env.example` placeholder values to clearly unusable strings.
5. Consider removing `debug_llm.py` from the repository.
