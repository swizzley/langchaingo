package ollama

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/swizzley/langchaingo/callbacks"
	"github.com/swizzley/langchaingo/llms"
	"github.com/swizzley/langchaingo/llms/ollama/internal/ollamaclient"
)

var (
	ErrEmptyResponse       = errors.New("no response")
	ErrIncompleteEmbedding = errors.New("not all input got embedded")
	ErrPullError           = errors.New("ollama model pull error")
	ErrPullTimeout         = errors.New("ollama model pull deadline exceeded")
)

// LLM is a ollama LLM implementation.
type LLM struct {
	CallbacksHandler callbacks.Handler
	client           *ollamaclient.Client
	options          options
}

var (
	_ llms.Model          = (*LLM)(nil)
	_ llms.ReasoningModel = (*LLM)(nil)
)

// New creates a new ollama LLM implementation.
func New(opts ...Option) (*LLM, error) {
	o := options{}
	for _, opt := range opts {
		opt(&o)
	}

	client, err := ollamaclient.NewClient(o.ollamaServerURL, o.httpClient)
	if err != nil {
		return nil, err
	}

	return &LLM{client: client, options: o}, nil
}

// SupportsReasoning implements the ReasoningModel interface.
// Returns true if the current model supports reasoning/thinking.
func (o *LLM) SupportsReasoning() bool {
	// Check if the model supports reasoning based on model name patterns
	model := strings.ToLower(o.options.model)

	// Ollama models that support reasoning/thinking:
	// - deepseek-r1 models (DeepSeek reasoning models)
	// - qwq models (Alibaba's QwQ reasoning models)
	// - Models with "reasoning" or "thinking" in the name
	if strings.Contains(model, "deepseek-r1") ||
		strings.Contains(model, "qwq") ||
		strings.Contains(model, "reasoning") ||
		strings.Contains(model, "thinking") {
		return true
	}

	// Future: could check model capabilities via Ollama API when available
	return false
}

// Call Implement the call interface for LLM.
func (o *LLM) Call(ctx context.Context, prompt string, options ...llms.CallOption) (string, error) {
	return llms.GenerateFromSinglePrompt(ctx, o, prompt, options...)
}

// GenerateContent implements the Model interface.
func (o *LLM) GenerateContent(ctx context.Context, messages []llms.MessageContent, options ...llms.CallOption) (*llms.ContentResponse, error) { // nolint: lll, cyclop, funlen
	if o.CallbacksHandler != nil {
		o.CallbacksHandler.HandleLLMGenerateContentStart(ctx, messages)
	}

	opts := llms.CallOptions{}
	for _, opt := range options {
		opt(&opts)
	}

	// Check if context caching is enabled
	var contextCache *ContextCache
	if opts.Metadata != nil {
		if cache, ok := opts.Metadata["context_cache"].(*ContextCache); ok {
			contextCache = cache
		}
	}

	// Override LLM model if set as llms.CallOption
	model := o.options.model
	if opts.Model != "" {
		model = opts.Model
	}

	// Pull model if enabled
	if o.options.pullModel {
		if err := o.pullModelIfNeeded(ctx, model); err != nil {
			return nil, fmt.Errorf("%w: %w", ErrPullError, err)
		}
	}

	// Our input is a sequence of MessageContent, each of which potentially has
	// a sequence of Part that could be text, images etc.
	// We have to convert it to a format Ollama understands: ChatRequest, which
	// has a sequence of Message, each of which has a role and content - single
	// text + potential images + optional tool calls.
	chatMsgs := make([]*ollamaclient.Message, 0, len(messages))
	for _, mc := range messages {
		msg := &ollamaclient.Message{Role: typeToRole(mc.Role)}

		// Look at all the parts in mc; expect to find a single Text part and
		// any number of binary parts or tool call/response parts.
		var text string
		foundText := false
		var images []ollamaclient.ImageData

		for _, p := range mc.Parts {
			switch pt := p.(type) {
			case llms.TextContent:
				if foundText {
					return nil, errors.New("expecting a single Text content")
				}
				foundText = true
				text = pt.Text
			case llms.BinaryContent:
				images = append(images, ollamaclient.ImageData(pt.Data))
			case llms.ToolCallResponse:
				// Tool result message — set content to the tool output
				text = pt.Content
			case llms.ToolCall:
				// AI message with tool call — convert to Ollama format
				if pt.FunctionCall != nil {
					var args map[string]any
					if pt.FunctionCall.Arguments != "" {
						_ = json.Unmarshal([]byte(pt.FunctionCall.Arguments), &args)
					}
					msg.ToolCalls = append(msg.ToolCalls, ollamaclient.ToolCall{
						Function: ollamaclient.ToolCallFunction{
							Name:      pt.FunctionCall.Name,
							Arguments: args,
						},
					})
				}
			}
		}

		msg.Content = text
		msg.Images = images
		chatMsgs = append(chatMsgs, msg)
	}

	format := o.options.format
	if opts.JSONMode {
		format = "json"
	}

	// Get our ollamaOptions from llms.CallOptions
	ollamaOptions := makeOllamaOptionsFromOptions(o.options.ollamaOptions, opts)

	// Handle thinking mode if specified via metadata
	if opts.Metadata != nil {
		if config, ok := opts.Metadata["thinking_config"].(*llms.ThinkingConfig); ok {
			if o.SupportsReasoning() {
				if config.Mode == llms.ThinkingModeNone {
					// Explicitly disable thinking (needed for models like qwen3 that think by default)
					f := false
					ollamaOptions.Think = &f
				} else {
					// Enable thinking
					t := true
					ollamaOptions.Think = &t
				}
			}
		}
	}
	// Convert llms.Tool to ollamaclient.Tool
	var ollamaTools []ollamaclient.Tool
	for _, t := range opts.Tools {
		if t.Function != nil {
			ollamaTools = append(ollamaTools, ollamaclient.Tool{
				Type: "function",
				Function: ollamaclient.ToolFunction{
					Name:        t.Function.Name,
					Description: t.Function.Description,
					Parameters:  t.Function.Parameters,
				},
			})
		}
	}

	req := &ollamaclient.ChatRequest{
		Model:    model,
		Format:   format,
		Messages: chatMsgs,
		Tools:    ollamaTools,
		Options:  ollamaOptions,
		Stream:   opts.StreamingFunc != nil,
	}

	keepAlive := o.options.keepAlive
	if keepAlive != "" {
		req.KeepAlive = keepAlive
	}

	var fn ollamaclient.ChatResponseFunc
	streamedResponse := ""
	var resp ollamaclient.ChatResponse

	fn = func(response ollamaclient.ChatResponse) error {
		if opts.StreamingFunc != nil && response.Message != nil {
			if err := opts.StreamingFunc(ctx, []byte(response.Message.Content)); err != nil {
				return err
			}
		}
		if response.Message != nil {
			streamedResponse += response.Message.Content
		}
		if !req.Stream || response.Done {
			resp = response
			resp.Message = &ollamaclient.Message{
				Role:    "assistant",
				Content: streamedResponse,
			}
		}
		return nil
	}

	err := o.client.GenerateChat(ctx, req, fn)
	if err != nil {
		if o.CallbacksHandler != nil {
			o.CallbacksHandler.HandleLLMError(ctx, err)
		}
		return nil, err
	}

	// Handle case where Message might be nil (e.g., context cancelled during streaming)
	content := ""
	if resp.Message != nil {
		content = resp.Message.Content
	}

	// Build generation info with standardized fields
	genInfo := map[string]any{
		"CompletionTokens": resp.EvalCount,
		"PromptTokens":     resp.PromptEvalCount,
		"TotalTokens":      resp.EvalCount + resp.PromptEvalCount,
		// Add empty thinking fields for cross-provider compatibility
		"ThinkingContent": "", // Ollama doesn't separate thinking content
		"ThinkingTokens":  0,  // Ollama doesn't track thinking tokens separately
	}

	// If context caching is enabled, track cache usage
	if contextCache != nil {
		if cacheEntry, hit := contextCache.Get(messages); hit {
			// Cache hit - we reused cached context
			genInfo["CachedTokens"] = cacheEntry.ContextTokens
			genInfo["CacheHit"] = true
		} else {
			// Cache miss - store for future use
			contextCache.Put(messages, resp.PromptEvalCount)
			genInfo["CachedTokens"] = 0
			genInfo["CacheHit"] = false
		}
	}

	// Note: Ollama may include thinking in the main content when Think mode is enabled
	// Future versions may provide separate thinking content
	if ollamaOptions.Think != nil && *ollamaOptions.Think && o.SupportsReasoning() {
		genInfo["ThinkingEnabled"] = true
	}

	// Convert tool calls from Ollama response to llms.ToolCall
	var toolCalls []llms.ToolCall
	if resp.Message != nil && len(resp.Message.ToolCalls) > 0 {
		for i, tc := range resp.Message.ToolCalls {
			argsJSON, _ := json.Marshal(tc.Function.Arguments)
			toolCalls = append(toolCalls, llms.ToolCall{
				ID:   fmt.Sprintf("call_%d", i),
				Type: "function",
				FunctionCall: &llms.FunctionCall{
					Name:      tc.Function.Name,
					Arguments: string(argsJSON),
				},
			})
		}
	}

	choices := []*llms.ContentChoice{
		{
			Content:        content,
			GenerationInfo: genInfo,
			ToolCalls:      toolCalls,
		},
	}

	response := &llms.ContentResponse{Choices: choices}

	if o.CallbacksHandler != nil {
		o.CallbacksHandler.HandleLLMGenerateContentEnd(ctx, response)
	}

	return response, nil
}

func (o *LLM) CreateEmbedding(ctx context.Context, inputTexts []string) ([][]float32, error) {
	// Pull model if enabled
	if o.options.pullModel {
		if err := o.pullModelIfNeeded(ctx, o.options.model); err != nil {
			return nil, err
		}
	}

	embeddings := [][]float32{}

	for _, input := range inputTexts {
		req := &ollamaclient.EmbeddingRequest{
			Input: input,
			Model: o.options.model,
		}
		if o.options.keepAlive != "" {
			req.KeepAlive = o.options.keepAlive
		}

		embedding, err := o.client.CreateEmbedding(ctx, req)
		if err != nil {
			return nil, err
		}

		if len(embedding.Embeddings) == 0 {
			return nil, ErrEmptyResponse
		}

		embeddings = append(embeddings, embedding.Embeddings...)
	}

	if len(inputTexts) != len(embeddings) {
		return embeddings, ErrIncompleteEmbedding
	}

	return embeddings, nil
}

func typeToRole(typ llms.ChatMessageType) string {
	switch typ {
	case llms.ChatMessageTypeSystem:
		return "system"
	case llms.ChatMessageTypeAI:
		return "assistant"
	case llms.ChatMessageTypeHuman:
		fallthrough
	case llms.ChatMessageTypeGeneric:
		return "user"
	case llms.ChatMessageTypeFunction:
		return "function"
	case llms.ChatMessageTypeTool:
		return "tool"
	}
	return ""
}

func makeOllamaOptionsFromOptions(ollamaOptions ollamaclient.Options, opts llms.CallOptions) ollamaclient.Options {
	// Load back CallOptions as ollamaOptions
	ollamaOptions.NumPredict = opts.MaxTokens
	ollamaOptions.Temperature = float32(opts.Temperature)
	ollamaOptions.Stop = opts.StopWords
	ollamaOptions.TopK = opts.TopK
	ollamaOptions.TopP = float32(opts.TopP)
	ollamaOptions.Seed = opts.Seed
	ollamaOptions.RepeatPenalty = float32(opts.RepetitionPenalty)
	ollamaOptions.FrequencyPenalty = float32(opts.FrequencyPenalty)
	ollamaOptions.PresencePenalty = float32(opts.PresencePenalty)

	// Extract thinking configuration for models that support it
	if opts.Metadata != nil {
		if config, ok := opts.Metadata["thinking_config"].(*llms.ThinkingConfig); ok {
			if config.Mode == llms.ThinkingModeNone {
				f := false
				ollamaOptions.Think = &f
			} else {
				t := true
				ollamaOptions.Think = &t
			}
		}
	}

	return ollamaOptions
}

// pullModelIfNeeded pulls the model if it's not already available.
func (o *LLM) pullModelIfNeeded(ctx context.Context, model string) error {
	// Try to use the model first. If it fails with a model not found error,
	// then pull the model.
	// This is a simple implementation. In production, you might want to
	// implement a more sophisticated check (e.g., using a list endpoint).

	// Apply timeout if configured
	pullCtx := ctx
	if o.options.pullTimeout > 0 {
		var cancel context.CancelFunc
		pullCtx, cancel = context.WithTimeoutCause(ctx, o.options.pullTimeout, ErrPullTimeout)
		defer func() {
			if cancel != nil {
				cancel()
			}
		}()
	}

	// For now, we'll just pull the model without checking.
	// This ensures the model is available but may result in unnecessary pulls.
	req := &ollamaclient.PullRequest{
		Model:  model,
		Stream: false,
	}

	err := o.client.Pull(pullCtx, req)
	if err != nil {
		// Check if the error is due to context timeout
		if errors.Is(err, context.DeadlineExceeded) {
			return err
		}
		// Check if the context has a cause
		if cause := context.Cause(pullCtx); cause != nil {
			return fmt.Errorf("%w: %w", cause, err)
		}
	}
	return err
}
