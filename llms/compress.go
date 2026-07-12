package llms

import (
	"fmt"
	"regexp"
	"strings"
)

// goFileLineRe matches a Go log file:line token, e.g. "compress.go:58".
// B110: only such a token preceding ": " is a real log-location prefix to strip.
var goFileLineRe = regexp.MustCompile(`^\S+\.go:\d+$`)

// CompressToolOutput reduces tool output to fit within a character budget.
// Applies format-aware compression strategies: dedup, timestamp stripping,
// line collapsing, and smart truncation. Budget is in characters.
func CompressToolOutput(output string, budget int) string {
	if len(output) <= budget {
		return output
	}

	// Stage 1: Strip common noise
	output = stripTimestamps(output)

	if len(output) <= budget {
		return output
	}

	// Stage 2: Deduplicate repeated lines
	output = deduplicateLines(output)

	if len(output) <= budget {
		return output
	}

	// Stage 3: Collapse JSON arrays and null fields
	if looksLikeJSON(output) {
		output = compressJSON(output)
		if len(output) <= budget {
			return output
		}
	}

	// Stage 4: Smart truncate — keep first and last lines
	return smartTruncate(output, budget)
}

// stripTimestamps removes common log timestamp prefixes.
// Patterns: "2026/04/10 14:32:35", "Apr 10 14:32:35 hostname", syslog-style.
func stripTimestamps(s string) string {
	lines := strings.Split(s, "\n")
	for i, line := range lines {
		// Syslog: "Apr 10 14:32:35 hostname service[pid]: "
		if len(line) > 30 {
			// Check for "Mon DD HH:MM:SS host" pattern
			if len(line) > 15 && line[3] == ' ' && line[6] == ' ' && line[9] == ':' && line[12] == ':' {
				if idx := strings.Index(line[15:], ": "); idx >= 0 {
					lines[i] = line[15+idx+2:]
					continue
				}
			}
		}
		// Go log: "2026/04/10 14:32:35 file.go:123: " or "2026/04/10 14:32:35 msg"
		if len(line) > 20 && line[4] == '/' && line[7] == '/' && line[10] == ' ' {
			rest := strings.TrimLeft(line[19:], " ")
			// B110: only strip through ": " when the token before it is a real
			// file:line location prefix; otherwise the message itself contains
			// ": " and we'd delete its severity/component/function prefix.
			if idx := strings.Index(rest, ": "); idx >= 0 {
				fields := strings.Fields(rest[:idx])
				if len(fields) > 0 && goFileLineRe.MatchString(fields[len(fields)-1]) {
					lines[i] = rest[idx+2:]
					continue
				}
			}
			lines[i] = rest
		}
	}
	return strings.Join(lines, "\n")
}

// deduplicateLines collapses consecutive identical or near-identical lines.
// B111: only collapse a run of >= minRun lines whose shared non-digit skeleton
// is long enough to be structural; shorter/number-heavy runs (distinct data
// rows) are emitted verbatim so we don't fold distinct numeric rows into one.
func deduplicateLines(s string) string {
	lines := strings.Split(s, "\n")
	if len(lines) <= 3 {
		return s
	}

	const minSkeletonLen = 8 // require enough non-digit structure to treat as a repeat
	const minRun = 3         // only collapse runs of at least this many similar lines

	var result []string
	var run []string // buffered consecutive similar (after digit-strip) lines

	flush := func() {
		if len(run) == 0 {
			return
		}
		if len(run) >= minRun && len(stripDigits(run[0])) >= minSkeletonLen {
			result = append(result, fmt.Sprintf("[x%d] %s", len(run), run[0]))
		} else {
			result = append(result, run...)
		}
		run = run[:0]
	}

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}

		if len(run) == 0 || similarLines(trimmed, run[len(run)-1]) {
			run = append(run, trimmed)
			continue
		}

		flush()
		run = append(run, trimmed)
	}
	flush()

	return strings.Join(result, "\n")
}

// similarLines returns true if two lines differ only in numbers/timestamps.
func similarLines(a, b string) bool {
	if a == b {
		return true
	}
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	// Strip all digits and compare structure
	return stripDigits(a) == stripDigits(b)
}

func stripDigits(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		if r >= '0' && r <= '9' {
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}

func looksLikeJSON(s string) bool {
	s = strings.TrimSpace(s)
	return (strings.HasPrefix(s, "{") && strings.HasSuffix(s, "}")) ||
		(strings.HasPrefix(s, "[") && strings.HasSuffix(s, "]"))
}

// compressJSON strips null/empty values and truncates arrays.
func compressJSON(s string) string {
	lines := strings.Split(s, "\n")
	var result []string
	arrayDepth := 0
	arrayItems := 0
	skipping := false

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)

		// Skip null/empty fields
		if strings.Contains(trimmed, ": null") || strings.Contains(trimmed, ": \"\"") ||
			strings.Contains(trimmed, ": []") || strings.Contains(trimmed, ": {}") {
			continue
		}

		// Track array depth for truncation
		if strings.HasSuffix(trimmed, "[") {
			arrayDepth++
			arrayItems = 0
			skipping = false
			result = append(result, line)
			continue
		}
		if strings.HasPrefix(trimmed, "]") {
			if skipping {
				result = append(result, fmt.Sprintf("      ... (%d more items)", arrayItems-3))
				skipping = false
			}
			arrayDepth--
			result = append(result, line)
			continue
		}

		if arrayDepth > 0 {
			arrayItems++
			if arrayItems > 3 {
				skipping = true
				continue
			}
		}

		result = append(result, line)
	}

	return strings.Join(result, "\n")
}

// smartTruncate keeps the first and last portions of text with an omission notice.
func smartTruncate(s string, budget int) string {
	lines := strings.Split(s, "\n")
	if len(lines) <= 6 {
		// Few lines — just hard truncate
		if len(s) > budget {
			return s[:budget-20] + "\n... (truncated)"
		}
		return s
	}

	// Keep first 40% and last 20% of budget
	headBudget := budget * 2 / 5
	tailBudget := budget / 5
	omitNotice := ""

	var head []string
	headLen := 0
	for _, line := range lines {
		if headLen+len(line)+1 > headBudget {
			break
		}
		head = append(head, line)
		headLen += len(line) + 1
	}

	var tail []string
	tailLen := 0
	for i := len(lines) - 1; i >= 0; i-- {
		if tailLen+len(lines[i])+1 > tailBudget {
			break
		}
		tail = append([]string{lines[i]}, tail...)
		tailLen += len(lines[i]) + 1
	}

	omitted := len(lines) - len(head) - len(tail)
	if omitted > 0 {
		omitNotice = fmt.Sprintf("\n... (%d lines omitted)\n", omitted)
	}

	return strings.Join(head, "\n") + omitNotice + strings.Join(tail, "\n")
}
