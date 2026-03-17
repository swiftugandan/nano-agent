# Code Review: src/tools.rs

## Summary
The `tools.rs` module provides the tool definitions and dispatch implementation for the nano-agent's built-in tools. The review found the code to be generally well-structured and safe, with a few recommendations for improvement.

## Review Date
2026-03-17

## Reviewer
lead (developer agent)

---

## 1. Architecture & Design ✅

### Strengths:
- **Clean separation of concerns**: tool definitions are separate from implementation
- **Sandboxing**: Uses `PathSandbox` to prevent path traversal attacks
- **Dispatch pattern**: Clear function pointer table pattern
- **Modular helpers**: `resolve_path`, `format_external_output`, etc.

### Observations:
- The dispatch table is built once at startup with workspace-specific closures
- Each tool has its own sandbox instance, providing isolation

---

## 2. Security ✅

### Positive Security Features:
- **Command blocklist**: Dangerous commands like `rm -rf /`, `mkfs`, `fork bomb` are blocked
- **Path sandbox**: All file operations are confined to workspace
- **Output truncation**: Prevents memory exhaustion (50KB limit)
- **Timeout mechanism**: Prevents runaway processes

### Recommendations:
1. **Consider command allowlist**: A blocklist can be bypassed. Consider only allowing specific safe commands or using a restricted shell.
2. **Add resource limits**: Consider adding CPU/memory limits for subprocesses (e.g., using `setrlimit`). This is Rust-specific and requires libc.
3. **Inspect environment**: Sanitize environment variables passed to subprocesses to avoid injection attacks.

---

## 3. Code Quality ✅

### Good Practices:
- Consistent formatting (after f23d36b refactor)
- Clear function names and comments
- Proper error handling with user-friendly messages
- Helper functions (`format_command_output`, `truncate_output`) are reusable

### Minor Issues:
1. **Line 187**: Use of `crate::util::truncate_at_boundary`
   - The helper exists and works correctly
   - Ensure it's properly exported in `util.rs` (it is)

2. **Readability**: Some nested closures could be extracted to named functions for testability, but current approach is acceptable for this scale.

---

## 4. Correctness & Bugs ⚠️

### Potential Issues:

#### 4.1 Output Truncation Implementation (Line 187)
The current implementation calls `crate::util::truncate_at_boundary(&s, MAX_OUTPUT_BYTES)`.

Let me verify the helper:
```rust
pub fn truncate_at_boundary(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        let mut end = max;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end]
    }
}
```

✅ **This is correct** - it properly finds a UTF-8 char boundary.

#### 4.2 Thread Safety in Bash Tool
The bash implementation spawns two threads for stdout/stderr capture. Need to ensure:
- Thread handles are joined even on timeout/kill (they are)
- No deadlocks: The polling loop is reasonable (50ms sleep)
- ✅ Code properly joins threads in all exit paths

#### 4.3 Race Condition in Timeout Handling
When a timeout occurs:
```rust
let _ = child.kill();
let _ = child.wait();
let stdout_data = stdout_handle.join().unwrap_or_default();
let stderr_data = stderr_handle.join().unwrap_or_default();
```

Potential issue: `child.kill()` sends signal, then `child.wait()` reaps the child. The threads reading from pipes should exit because the pipe is closed. However, if the child spawns grandchildren that outlive the shell, they could become orphaned. This is minor and typical for shell execution.

#### 4.4 `ls` Tool Empty Directory Message
When `names.is_empty()`, returns `"(empty directory)"`. This is a special case that might be unexpected by callers expecting a list. Consider returning an empty string or a consistent empty representation.

#### 4.5 `find` Tool Default Limit
Default limit is 1000 results. This is reasonable but should be documented in the tool schema description.

---

## 5. Performance ⚠️

### Observations:

1. **Thread overhead**: Bash command with timeout spawns 2 extra threads. For a CLI agent, this is acceptable.
2. **External dependencies**: Uses system `rg` and `fd` commands. Ensure they are available or handle errors gracefully (they do).
3. **Output truncation**: 50KB limit prevents memory issues but could truncate useful info. Consider configurable limit (future enhancement).

---

## 6. Error Handling ✅

### Good:
- All tools return `String` error messages
- Errors are descriptive and user-friendly
- `?` operator is used where appropriate

### Minor Suggestions:
- Consider using `Result<Output, Error>` type for more structured errors, but current string-based approach is simpler for JSON dispatch.
- Some errors like `(exit code -1)` could be more informative.

---

## 7. Tool Definitions (JSON Schema) ✅

All tools have proper JSON schemas with:
- Required fields marked
- Descriptions for parameters
- Types correctly specified

### Missing Enhancements (Optional):
- Add examples for tool parameters in description
- Add deprecation notices if any tools will be replaced

---

## 8. Consistency with Other Modules ✅

- Uses `PathSandbox` consistently across file tools
- Uses `Dispatch` type from `types.rs`
- Follows Rust idioms (closures, `Box::new`, etc.)

---

## 9. Testing ⚠️

### Current State:
- There are integration tests in `tests/test_scenarios.rs`
- Unit tests for individual tools appear to be minimal

### Recommendations:
1. **Add unit tests** for `truncate_output`, `format_command_output`, `resolve_path`
2. **Test edge cases**:
   - Very large outputs
   - Non-UTF8 output (should be handled by `String::from_utf8_lossy`)
   - Path traversal attempts with `../`
   - Timeout edge cases (exact timeout boundary)
3. **Test blocklist bypass attempts**

---

## 10. Documentation ⚠️

### Inline Documentation:
- Function comments are present but could be expanded
- Complex logic (timeout handling) is mostly self-documenting

### Recommendations:
- Add module-level documentation explaining the dispatch mechanism
- Document safety guarantees and threat model
- Add doc comments to public functions in future if making them public

---

## Final Verdict

**Status**: **APPROVE** with minor recommendations

The tools.rs implementation is solid, secure, and follows good practices. The changes from recent commits (adding timeout, new tools, refactoring) are well-executed.

### Priority Recommendations (high):
1. None - code is production-ready

### Medium Priority (nice-to-have):
1. Add unit tests for helper functions
2. Document external tool dependencies (`rg`, `fd`) in README
3. Consider making `MAX_OUTPUT_BYTES` configurable

### Low Priority:
1. Extract closure logic to named functions for testability
2. Add module-level documentation
3. Consider using `Default` trait for some structs if applicable

---

## Specific Commands to Consider

To run tests:
```bash
cargo test --all
```

To lint:
```bash
cargo clippy --all-targets --all-features
```

---

## Conclusion

The tools.rs module is well-designed and safe for use in the nano-agent system. The security measures (sandboxing, blocklist, output limits) are appropriate. I recommend merging these changes with the understanding that testing should be comprehensive before production deployment.
