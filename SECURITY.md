# Security model

What this application defends, how, and what it does not yet cover. Written so a
reviewer can check the claims against the code rather than take them on trust.

## Authentication

| Control | Implementation |
|---|---|
| Password hashing | Argon2id via `argon2-cffi` (`aish/security.py`), t=3, m=64 MiB, p=2, 16-byte salt |
| Password normalisation | NFKC before hashing, per NIST SP 800-63B |
| Password policy | Minimum 12 characters, breach-list check, no email-in-password, repetition check. Length over composition rules — no forced symbol classes |
| Rehash on login | Work factors raised in config are applied on the user's next sign-in |
| Account enumeration | Login returns one message for every failure; an unknown address still incurs a dummy Argon2 verification so timing does not distinguish it. A duplicate registration returns the same page as a pending one |
| Brute force | Per-account lockout after 5 failures (15 minutes) plus a per-IP budget. Counters are persisted, so a restart does not reset an attacker's budget |
| Registration | Email is the username; institution is chosen from a server-side list (`packs/euibas.yaml`) and validated against it — the submitted value is never trusted |

## Sessions

- Opaque 256-bit token in a cookie; the database stores only its SHA-256 digest, so
  read access to the database cannot forge a login.
- `HttpOnly`, `Secure`, `SameSite=Strict`, `Path=/`.
- Absolute lifetime 8 hours, idle timeout 1 hour, both enforced server-side.
- Server-side revocation: sign-out, "sign out other sessions", and a password change
  all revoke rows immediately.
- A password change invalidates every other session.

## CSRF

Double-submit token bound to the session row, checked on every non-idempotent
request (`verify_csrf` in `aish/web/deps.py`), combined with a `SameSite=Strict`
session cookie. A cross-site POST has neither the cookie nor the token.

## Stored provider credentials

- Encrypted with AES-256-GCM. The key is derived from `AISH_ENCRYPTION_KEY` via
  HKDF-SHA256, separate from the cookie-signing key so that rotating one does not
  destroy the other.
- The owner's user id is the AEAD associated data, so a ciphertext copied to another
  user's row fails to decrypt rather than silently working.
- Plaintext exists only in memory for the duration of one outbound call. It is never
  written to the database, a log, a template or an error page: `redact()` strips it
  from any message that reaches a stored result.
- The UI shows only the last four characters.

## Outbound requests (SSRF)

User-supplied endpoints are validated in `validate_endpoint()`:

- Scheme restricted to http/https; credentials in the URL refused.
- Host resolved, and loopback, link-local, multicast, reserved and the cloud metadata
  address (169.254.169.254) refused unconditionally.
- Private ranges are permitted, because EUIBA on-prem models legitimately live there.
  Set `AISH_ALLOW_PRIVATE_ENDPOINTS=false` where that is not wanted.
- Redirects are not followed, so a validated endpoint cannot hand the request to an
  unvalidated one.
- The check runs again immediately before every outbound call, not only when the
  target was saved. A save-time-only check would be a time-of-check/time-of-use gap.

## Web hardening

- `Content-Security-Policy: default-src 'self'` with **no** `unsafe-inline` anywhere.
  There are no inline styles and no inline scripts in the application; charts are
  inline SVG whose geometry comes from attributes, and the sequential heatmap is
  quantised onto stylesheet classes precisely so that no inline fill is needed.
- `frame-ancestors 'none'`, `base-uri 'none'`, `object-src 'none'`,
  `require-trusted-types-for 'script'`.
- `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
  `Referrer-Policy: same-origin`, `Cross-Origin-Opener-Policy: same-origin`,
  a restrictive `Permissions-Policy`, and `Cache-Control: no-store` on every response.
- HSTS (2 years, includeSubDomains, preload) in production.
- Host header validated against `AISH_PUBLIC_HOST`; anything else gets a 400.
- Request bodies over 1 MB rejected before routing.
- The interactive API explorer and the OpenAPI schema are disabled.
- Autoescaping is on for every template variable.

## Authorisation

Every row lookup is scoped to the owning user in the query itself, not filtered after
the fact. Supplying another account's target or run id returns 404, not 403 — the
existence of the row is not disclosed. This is covered by tests
(`test_one_user_cannot_see_another_users_target`,
`test_one_user_cannot_read_another_users_run`).

## Data handling

- SQLAlchemy ORM throughout; no string-built SQL.
- Audit events record a salted hash of the client IP, never the address itself.
- The `/healthz` probe discloses nothing about internals.
- Unhandled exceptions are logged server-side; the user sees a generic page with an
  HTTP status and no stack trace.
- A suite marked `restricted: true` in `packs/` cannot target a cloud model; the pack
  validator rejects the configuration, so hazardous probe content cannot be sent to a
  third-party API by misconfiguration.

## Production start-up invariants

The application refuses to start with `AISH_ENV=production` unless:

- `AISH_SECRET_KEY` and `AISH_ENCRYPTION_KEY` are both set, at least 32 characters,
  and different from each other;
- `AISH_COOKIE_SECURE` is true;
- `AISH_DEBUG` is false.

Failing closed is deliberate: a production instance running on development defaults
is worse than one that does not start.

## Dependencies

- `requirements.txt` pins exact versions.
- CI runs `pip-audit` against both requirement files and fails the build on any known
  vulnerability. Both are clean as of the last run.
- CI runs `bandit` over the application, harness and scripts. The only suppressions
  are four annotated `# nosec` markers on seeded `random.Random` calls (reproducible
  sampling and bootstrap resampling, not secrets) and the development seed script's
  fixture password.
- No npm dependency tree: the front end is server-rendered with one small
  first-party script. There is no bundler, no framework and no CDN.
- The container image builds in one stage and runs from another, so no compiler or
  header package ships to production. It runs as an unprivileged user, read-only
  except for its data volume, with all capabilities dropped and
  `no-new-privileges`.

## Known gaps

Stated rather than glossed over.

1. **No password reset.** There is no mail transport configured, so a user who
   forgets their password needs an administrator to intervene at the database. A
   reset flow needs SMTP and a signed, single-use, time-limited token.
2. **No second factor.** Worth adding TOTP before this holds anything sensitive.
3. **No administrator UI.** `AISH_REQUIRE_ADMIN_APPROVAL` gates sign-in, but approving
   an account is currently a database operation.
4. **Rate limiting is per-instance.** The counters live in the application database,
   which is correct for a single instance; a multi-instance deployment needs a shared
   store.
5. **SQLite by default.** Fine for a working-group instance. A concurrent,
   multi-instance deployment should point `AISH_DATABASE_URL` at PostgreSQL.
6. **The judge path is not implemented.** Suites that declare an LLM judge
   (AILuminate) cannot run yet; the deterministic scorers are what execute today, and
   the UI says which was used.

## Reporting

Report a suspected vulnerability to the ICDT ET Working Group rather than opening a
public issue.
