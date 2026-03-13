package claudecode

// Option is a function that configures the Claude Code CLI provider.
type Option func(*options)

type options struct {
	claudePath string // path to claude binary
	model      string // model alias (opus, sonnet, haiku) or full ID
}

// WithClaudePath sets the path to the claude CLI binary.
// Defaults to "claude" (found via PATH).
func WithClaudePath(path string) Option {
	return func(o *options) {
		o.claudePath = path
	}
}

// WithModel sets the model to use (e.g. "opus", "sonnet", "claude-opus-4-6").
// Defaults to "opus".
func WithModel(model string) Option {
	return func(o *options) {
		o.model = model
	}
}
