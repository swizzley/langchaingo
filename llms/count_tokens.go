package llms

import (
	"crypto/sha1"
	"encoding/base64"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/pkoukk/tiktoken-go"
)

const (
	_tokenApproximation = 4
)

// B112: tiktoken-go's default BPE loader downloads encoding files with a bare
// http.Get (no timeout), so CountTokens can block forever on first use. Install
// a loader backed by an http.Client with a bounded timeout, with the same local
// caching behavior so downloads happen at most once.
func init() {
	tiktoken.SetBpeLoader(&timeoutBpeLoader{
		client: &http.Client{Timeout: 15 * time.Second},
	})
}

type timeoutBpeLoader struct {
	client *http.Client
}

func (l *timeoutBpeLoader) LoadTiktokenBpe(tiktokenBpeFile string) (map[string]int, error) {
	contents, err := l.readFileCached(tiktokenBpeFile)
	if err != nil {
		return nil, err
	}

	bpeRanks := make(map[string]int)
	for _, line := range strings.Split(string(contents), "\n") {
		if line == "" {
			continue
		}
		parts := strings.Split(line, " ")
		if len(parts) < 2 {
			continue
		}
		token, err := base64.StdEncoding.DecodeString(parts[0])
		if err != nil {
			return nil, err
		}
		rank, err := strconv.Atoi(parts[1])
		if err != nil {
			return nil, err
		}
		bpeRanks[string(token)] = rank
	}
	return bpeRanks, nil
}

func (l *timeoutBpeLoader) readFile(blobpath string) ([]byte, error) {
	if !strings.HasPrefix(blobpath, "http://") && !strings.HasPrefix(blobpath, "https://") {
		return os.ReadFile(blobpath)
	}
	resp, err := l.client.Get(blobpath)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

func (l *timeoutBpeLoader) readFileCached(blobpath string) ([]byte, error) {
	cacheDir := os.Getenv("TIKTOKEN_CACHE_DIR")
	if cacheDir == "" {
		cacheDir = os.Getenv("DATA_GYM_CACHE_DIR")
	}
	if cacheDir == "" {
		cacheDir = filepath.Join(os.TempDir(), "data-gym-cache")
	}

	cacheKey := fmt.Sprintf("%x", sha1.Sum([]byte(blobpath)))
	cachePath := filepath.Join(cacheDir, cacheKey)
	if _, err := os.Stat(cachePath); err == nil {
		return os.ReadFile(cachePath)
	}

	contents, err := l.readFile(blobpath)
	if err != nil {
		return nil, err
	}

	if err := os.MkdirAll(cacheDir, 0o755); err == nil {
		if tmp, err := os.CreateTemp(cacheDir, cacheKey+".*.tmp"); err == nil {
			if _, werr := tmp.Write(contents); werr == nil {
				tmp.Close()
				_ = os.Rename(tmp.Name(), cachePath)
			} else {
				tmp.Close()
				_ = os.Remove(tmp.Name())
			}
		}
	}
	return contents, nil
}

const (
	_gpt35TurboContextSize    = 16385  // gpt-3.5-turbo default context
	_gpt35Turbo16KContextSize = 16385  // gpt-3.5-turbo-16k
	_gpt4ContextSize          = 8192   // gpt-4
	_gpt432KContextSize       = 32768  // gpt-4-32k
	_gpt4TurboContextSize     = 128000 // gpt-4-turbo models
	_gpt4oContextSize         = 128000 // gpt-4o models
	_gpt4oMiniContextSize     = 128000 // gpt-4o-mini
	_textDavinci3ContextSize  = 4097
	_textBabbage1ContextSize  = 2048
	_textAda1ContextSize      = 2048
	_textCurie1ContextSize    = 2048
	_codeDavinci2ContextSize  = 8000
	_codeCushman1ContextSize  = 2048
	_defaultContextSize       = 4096
)

// nolint:gochecknoglobals
var modelToContextSize = map[string]int{
	// GPT-3.5 models
	"gpt-3.5-turbo":      _gpt35TurboContextSize,
	"gpt-3.5-turbo-16k":  _gpt35Turbo16KContextSize,
	"gpt-3.5-turbo-0125": _gpt35TurboContextSize,
	"gpt-3.5-turbo-1106": _gpt35TurboContextSize,
	// GPT-4 models
	"gpt-4":          _gpt4ContextSize,
	"gpt-4-32k":      _gpt432KContextSize,
	"gpt-4-0613":     _gpt4ContextSize,
	"gpt-4-32k-0613": _gpt432KContextSize,
	// GPT-4 Turbo models
	"gpt-4-turbo":            _gpt4TurboContextSize,
	"gpt-4-turbo-preview":    _gpt4TurboContextSize,
	"gpt-4-turbo-2024-04-09": _gpt4TurboContextSize,
	"gpt-4-1106-preview":     _gpt4TurboContextSize,
	"gpt-4-0125-preview":     _gpt4TurboContextSize,
	// GPT-4o models
	"gpt-4o":                 _gpt4oContextSize,
	"gpt-4o-2024-05-13":      _gpt4oContextSize,
	"gpt-4o-2024-08-06":      _gpt4oContextSize,
	"gpt-4o-mini":            _gpt4oMiniContextSize,
	"gpt-4o-mini-2024-07-18": _gpt4oMiniContextSize,
	// Ollama / local models
	"qwen3-coder:30b": 32768,
	"qwen3:14b":       32768,
	"qwen3.5:latest":  32768,
	"qwen3.5:7b":      32768,
	// Legacy models
	"text-davinci-003": _textDavinci3ContextSize,
	"text-curie-001":   _textCurie1ContextSize,
	"text-babbage-001": _textBabbage1ContextSize,
	"text-ada-001":     _textAda1ContextSize,
	"code-davinci-002": _codeDavinci2ContextSize,
	"code-cushman-001": _codeCushman1ContextSize,
}

// GetModelContextSize gets the max number of tokens for a language model. If the model
// name isn't recognized the default value 2048 is returned.
func GetModelContextSize(model string) int {
	contextSize, ok := modelToContextSize[model]
	if !ok {
		return _defaultContextSize
	}
	return contextSize
}

// CountTokens gets the number of tokens the text contains.
func CountTokens(model, text string) int {
	e, err := tiktoken.EncodingForModel(model)
	if err != nil {
		e, err = tiktoken.GetEncoding("gpt2")
		if err != nil {
			log.Printf("[WARN] Failed to calculate number of tokens for model, falling back to approximate count")
			return len([]rune(text)) / _tokenApproximation
		}
	}
	return len(e.Encode(text, nil, nil))
}

// CalculateMaxTokens calculates the max number of tokens that could be added to a text.
func CalculateMaxTokens(model, text string) int {
	return GetModelContextSize(model) - CountTokens(model, text)
}

// EstimateTokens provides a fast char-based token estimate without tiktoken overhead.
// Approximation: ~4 chars per token for English text. Good enough for budget tracking.
func EstimateTokens(text string) int {
	return len([]rune(text)) / _tokenApproximation
}
