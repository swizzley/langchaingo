package claudecode

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/swizzley/langchaingo/llms"
)

var (
	ErrNotFound      = errors.New("claude binary not found in PATH")
	ErrEmptyResponse = errors.New("no response from claude CLI")
)

// LLM implements llms.Model by shelling out to the Claude Code CLI in print mode.
type LLM struct {
	claudePath string
	model      string
}

var _ llms.Model = (*LLM)(nil)

// New creates a new Claude Code CLI provider.
func New(opts ...Option) (*LLM, error) {
	o := options{
		claudePath: "claude",
		model:      "opus",
	}
	for _, opt := range opts {
		opt(&o)
	}

	// Verify the binary exists
	resolved, err := exec.LookPath(o.claudePath)
	if err != nil {
		return nil, fmt.Errorf("%w: %s", ErrNotFound, o.claudePath)
	}

	return &LLM{
		claudePath: resolved,
		model:      o.model,
	}, nil
}

// Call implements the simplified text-only interface.
func (l *LLM) Call(ctx context.Context, prompt string, options ...llms.CallOption) (string, error) {
	return llms.GenerateFromSinglePrompt(ctx, l, prompt, options...)
}

// cliResponse is the JSON shape returned by `claude -p --output-format json`.
type cliResponse struct {
	Result           string          `json:"result"`
	StructuredOutput json.RawMessage `json:"structured_output,omitempty"`
	Usage            struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
	Model      string `json:"model"`
	StopReason string `json:"stop_reason"`
}

// GenerateContent implements llms.Model by invoking the claude CLI.
func (l *LLM) GenerateContent(ctx context.Context, messages []llms.MessageContent, options ...llms.CallOption) (*llms.ContentResponse, error) {
	opts := &llms.CallOptions{}
	for _, opt := range options {
		opt(opts)
	}

	// Extract system and user prompts from messages
	var systemPrompt, userPrompt string
	for _, msg := range messages {
		for _, part := range msg.Parts {
			if tc, ok := part.(llms.TextContent); ok {
				switch msg.Role {
				case llms.ChatMessageTypeSystem:
					if systemPrompt != "" {
						systemPrompt += "\n"
					}
					systemPrompt += tc.Text
				case llms.ChatMessageTypeHuman:
					if userPrompt != "" {
						userPrompt += "\n"
					}
					userPrompt += tc.Text
				}
			}
		}
	}

	// Build CLI arguments
	args := []string{
		"--print",
		"--output-format", "json",
		"--model", l.model,
		"--allowedTools", "",
		"--no-session-persistence",
	}

	if systemPrompt != "" {
		args = append(args, "--system-prompt", systemPrompt)
	}

	// If tools are provided, use --json-schema with the first tool's parameters
	var toolName string
	if len(opts.Tools) > 0 && opts.Tools[0].Function != nil {
		toolName = opts.Tools[0].Function.Name
		schemaBytes, err := json.Marshal(opts.Tools[0].Function.Parameters)
		if err == nil {
			args = append(args, "--json-schema", string(schemaBytes))
		}
	}

	cmd := exec.CommandContext(ctx, l.claudePath, args...)

	// Pipe user prompt via stdin to avoid ARG_MAX limits
	cmd.Stdin = strings.NewReader(userPrompt)

	// Filter CLAUDECODE env var — CLI refuses to launch inside another session
	env := os.Environ()
	filtered := make([]string, 0, len(env))
	for _, e := range env {
		if !strings.HasPrefix(e, "CLAUDECODE=") {
			filtered = append(filtered, e)
		}
	}
	cmd.Env = filtered

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, fmt.Errorf("claude CLI: %w (stderr: %s)", err, strings.TrimSpace(stderr.String()))
	}

	// Parse JSON response
	var resp cliResponse
	if err := json.Unmarshal(stdout.Bytes(), &resp); err != nil {
		return nil, fmt.Errorf("claude CLI: parse response: %w (raw: %s)", err, stdout.String()[:min(200, stdout.Len())])
	}

	if resp.Result == "" && len(resp.StructuredOutput) == 0 {
		return nil, ErrEmptyResponse
	}

	// Build ContentResponse
	choice := &llms.ContentChoice{
		StopReason: resp.StopReason,
		GenerationInfo: map[string]any{
			"InputTokens":  resp.Usage.InputTokens,
			"OutputTokens": resp.Usage.OutputTokens,
		},
	}

	if len(resp.StructuredOutput) > 0 && toolName != "" {
		// Wrap structured output as a tool call for compatibility with
		// callLLMWithTools / callLLMWithScoringTool consumers
		choice.ToolCalls = []llms.ToolCall{
			{
				ID: "claudecode_1",
				FunctionCall: &llms.FunctionCall{
					Name:      toolName,
					Arguments: string(resp.StructuredOutput),
				},
			},
		}
		// Also set Content in case callers check text fallback
		choice.Content = resp.Result
	} else {
		choice.Content = resp.Result
	}

	return &llms.ContentResponse{
		Choices: []*llms.ContentChoice{choice},
	}, nil
}
