package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"vertex2api/openai"
)

const Hydrogen = "gemini-4-hydrogen"

var Models = []string{Hydrogen, "gemini-4-carbon", "gemini-4-argon"}

type Vertex struct {
	Project, Location, Key string
	Client                 *http.Client
}

func NewVertex(project, location, key string) Vertex {
	return Vertex{Project: project, Location: location, Key: key, Client: &http.Client{
		Timeout:       5 * time.Minute,
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}}
}

type Part struct {
	Text    string `json:"text"`
	Thought bool   `json:"thought,omitempty"`
}

type Content struct {
	Role  string `json:"role,omitempty"`
	Parts []Part `json:"parts"`
}

type GenerationConfig struct {
	MaxOutputTokens *int            `json:"maxOutputTokens,omitempty"`
	Temperature     *float64        `json:"temperature,omitempty"`
	TopP            *float64        `json:"topP,omitempty"`
	StopSequences   []string        `json:"stopSequences,omitempty"`
	Thinking        *ThinkingConfig `json:"thinkingConfig,omitempty"`
}

type ThinkingConfig struct {
	Level string `json:"thinkingLevel"`
}

type Request struct {
	Contents []Content        `json:"contents"`
	System   *Content         `json:"systemInstruction,omitempty"`
	Config   GenerationConfig `json:"generationConfig"`
}

type Candidate struct {
	Content Content `json:"content"`
	Finish  string  `json:"finishReason"`
}

func (c Candidate) Text() string {
	var result strings.Builder
	for _, part := range c.Content.Parts {
		if !part.Thought {
			result.WriteString(part.Text)
		}
	}
	return result.String()
}

type Response struct {
	Candidates     []Candidate `json:"candidates"`
	PromptFeedback struct {
		BlockReason string `json:"blockReason"`
	} `json:"promptFeedback"`
	Usage *Usage          `json:"usageMetadata"`
	Error json.RawMessage `json:"error"`
}

type Usage struct {
	Prompt     int `json:"promptTokenCount"`
	Candidates int `json:"candidatesTokenCount"`
	Thoughts   int `json:"thoughtsTokenCount"`
	Cached     int `json:"cachedContentTokenCount"`
	Total      int `json:"totalTokenCount"`
}

func (u *Usage) OpenAI() *openai.Usage {
	if u == nil {
		return nil
	}
	result := &openai.Usage{Prompt: u.Prompt, Completion: u.Candidates + u.Thoughts, Total: u.Total}
	result.PromptDetails.Cached = u.Cached
	result.CompletionDetails.Reasoning = u.Thoughts
	if result.Total == 0 {
		result.Total = result.Prompt + result.Completion
	}
	return result
}

func messageParts(raw json.RawMessage) ([]Part, error) {
	var text *string
	if json.Unmarshal(raw, &text) == nil && text != nil {
		return []Part{{Text: *text}}, nil
	}
	var items []struct {
		Type string  `json:"type"`
		Text *string `json:"text"`
	}
	if json.Unmarshal(raw, &items) != nil || len(items) == 0 {
		return nil, errors.New("messages must contain text")
	}
	parts := make([]Part, 0, len(items))
	for _, item := range items {
		if item.Type != "text" || item.Text == nil {
			return nil, errors.New("only text message parts are supported")
		}
		parts = append(parts, Part{Text: *item.Text})
	}
	return parts, nil
}

func Translate(input openai.Request, model string) (Request, error) {
	if (input.N != nil && *input.N != 1) || len(input.Tools) > 0 || len(input.Functions) > 0 ||
		(input.ResponseFormat != nil && input.ResponseFormat.Type != "text") {
		return Request{}, errors.New("only one text choice without tools or structured output is supported")
	}
	var output Request
	for _, message := range input.Messages {
		parts, err := messageParts(message.Content)
		if err != nil {
			return Request{}, err
		}
		if len(message.ToolCalls) > 0 && string(message.ToolCalls) != "null" && string(message.ToolCalls) != "[]" ||
			len(message.Function) > 0 && string(message.Function) != "null" {
			return Request{}, errors.New("tool calls are unsupported")
		}
		switch message.Role {
		case "system", "developer":
			if len(output.Contents) > 0 {
				return Request{}, errors.New("place instructions before the conversation")
			}
			if output.System == nil {
				output.System = &Content{}
			}
			output.System.Parts = append(output.System.Parts, parts...)
		case "user", "assistant":
			role := "user"
			if message.Role == "assistant" {
				role = "model"
			}
			last := len(output.Contents) - 1
			if last >= 0 && output.Contents[last].Role == role {
				output.Contents[last].Parts = append(output.Contents[last].Parts, parts...)
			} else {
				output.Contents = append(output.Contents, Content{Role: role, Parts: parts})
			}
		default:
			return Request{}, errors.New("unsupported message role")
		}
	}
	if len(output.Contents) == 0 || output.Contents[0].Role != "user" {
		return Request{}, errors.New("conversation must start with a user message")
	}
	max := input.MaxCompletionTokens
	if max == nil {
		max = input.MaxTokens
	}
	if max != nil && *max < 1 {
		return Request{}, errors.New("token limit must be positive")
	}
	output.Config = GenerationConfig{MaxOutputTokens: max, Temperature: input.Temperature, TopP: input.TopP}
	if len(input.Stop) > 0 && string(input.Stop) != "null" {
		var stop string
		if json.Unmarshal(input.Stop, &stop) == nil {
			output.Config.StopSequences = []string{stop}
		} else if json.Unmarshal(input.Stop, &output.Config.StopSequences) != nil {
			return Request{}, errors.New("stop must be a string or an array of strings")
		}
	}
	switch input.ReasoningEffort {
	case "":
	case "low", "medium", "high":
		output.Config.Thinking = &ThinkingConfig{Level: strings.ToUpper(input.ReasoningEffort)}
	case "xhigh":
		if model != Hydrogen {
			return Request{}, errors.New("xhigh requires Gemini 4 Hydrogen")
		}
		output.Config.Thinking = &ThinkingConfig{Level: "XHIGH"}
	default:
		return Request{}, errors.New("unsupported reasoning_effort")
	}
	return output, nil
}

func (v Vertex) Generate(ctx context.Context, model string, body Request, stream bool) (*http.Response, error) {
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	host := "aiplatform.googleapis.com"
	switch v.Location {
	case "global":
	case "us", "eu":
		host = "aiplatform." + v.Location + ".rep.googleapis.com"
	default:
		host = v.Location + "-aiplatform.googleapis.com"
	}
	action := "generateContent"
	if stream {
		action = "streamGenerateContent?alt=sse"
	}
	endpoint := fmt.Sprintf("https://%s/v1/projects/%s/locations/%s/publishers/google/models/%s:%s",
		host, v.Project, v.Location, model, action)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-api-key", v.Key)
	return v.Client.Do(req)
}
