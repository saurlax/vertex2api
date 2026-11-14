package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httputil"
	"net/url"
	"regexp"
	"strings"
	"time"
)

var advertisedModels = []string{"gemini-4-hydrogen", "gemini-4-carbon", "gemini-4-argon"}

type vertexClient struct {
	Project, Location, Key string
	Client                 *http.Client
}

func newVertex(project, location, key string) vertexClient {
	return vertexClient{Project: project, Location: location, Key: key, Client: &http.Client{
		Timeout:       5 * time.Minute,
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}}
}

type vertexPart struct {
	Text             *string                 `json:"text,omitempty"`
	Thought          bool                    `json:"thought,omitempty"`
	ThoughtSignature string                  `json:"thoughtSignature,omitempty"`
	FunctionCall     *vertexFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *vertexFunctionResponse `json:"functionResponse,omitempty"`
}

type vertexContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []vertexPart `json:"parts"`
}

type generationConfig struct {
	MaxOutputTokens *int            `json:"maxOutputTokens,omitempty"`
	Temperature     *float64        `json:"temperature,omitempty"`
	TopP            *float64        `json:"topP,omitempty"`
	StopSequences   []string        `json:"stopSequences,omitempty"`
	ResponseMIME    string          `json:"responseMimeType,omitempty"`
	ResponseSchema  json.RawMessage `json:"responseJsonSchema,omitempty"`
	Thinking        *thinkingConfig `json:"thinkingConfig,omitempty"`
}

type thinkingConfig struct {
	Level string `json:"thinkingLevel"`
}

type vertexRequest struct {
	Contents    []vertexContent   `json:"contents"`
	System      *vertexContent    `json:"systemInstruction,omitempty"`
	Config      generationConfig  `json:"generationConfig"`
	Tools       []vertexTool      `json:"tools,omitempty"`
	ToolConfig  *vertexToolConfig `json:"toolConfig,omitempty"`
	toolHistory bool
}

type vertexCandidate struct {
	Content vertexContent `json:"content"`
	Finish  string        `json:"finishReason"`
}

func (c vertexCandidate) Text() string {
	var result strings.Builder
	for _, part := range c.Content.Parts {
		if !part.Thought && part.Text != nil {
			result.WriteString(*part.Text)
		}
	}
	return result.String()
}

type vertexResponse struct {
	Candidates     []vertexCandidate `json:"candidates"`
	PromptFeedback struct {
		BlockReason string `json:"blockReason"`
	} `json:"promptFeedback"`
	Usage *vertexUsage    `json:"usageMetadata"`
	Error json.RawMessage `json:"error"`
}

type vertexUsage struct {
	Prompt     int `json:"promptTokenCount"`
	Candidates int `json:"candidatesTokenCount"`
	Thoughts   int `json:"thoughtsTokenCount"`
	Cached     int `json:"cachedContentTokenCount"`
	Total      int `json:"totalTokenCount"`
}

func (u *vertexUsage) OpenAI() *chatUsage {
	if u == nil {
		return nil
	}
	result := &chatUsage{Prompt: u.Prompt, Completion: u.Candidates + u.Thoughts, Total: u.Total}
	result.PromptDetails.Cached = u.Cached
	result.CompletionDetails.Reasoning = u.Thoughts
	if result.Total == 0 {
		result.Total = result.Prompt + result.Completion
	}
	return result
}

func messageParts(raw json.RawMessage) ([]vertexPart, error) {
	var text *string
	if json.Unmarshal(raw, &text) == nil && text != nil {
		return []vertexPart{{Text: text}}, nil
	}
	var items []struct {
		Type string  `json:"type"`
		Text *string `json:"text"`
	}
	if json.Unmarshal(raw, &items) != nil || len(items) == 0 {
		return nil, errors.New("messages must contain text")
	}
	parts := make([]vertexPart, 0, len(items))
	for _, item := range items {
		if item.Type != "text" || item.Text == nil {
			return nil, errors.New("only text message parts are supported")
		}
		parts = append(parts, vertexPart{Text: item.Text})
	}
	return parts, nil
}

func translate(input chatRequest) (vertexRequest, error) {
	if input.N != nil && *input.N != 1 {
		return vertexRequest{}, errors.New("chat completions supports one choice per request; set n to 1")
	}
	output, err := translateTools(input)
	if err != nil {
		return vertexRequest{}, err
	}
	for i := 0; i < len(input.Messages); i++ {
		message := input.Messages[i]
		parts, err := message.parts()
		if err != nil {
			return vertexRequest{}, err
		}
		switch message.Role {
		case "system", "developer":
			if len(output.Contents) > 0 {
				return vertexRequest{}, errors.New("place instructions before the conversation")
			}
			if output.System == nil {
				output.System = &vertexContent{}
			}
			output.System.Parts = append(output.System.Parts, parts...)
		case "user", "assistant":
			role := "user"
			if message.Role == "assistant" {
				role = "model"
			}
			last := len(output.Contents) - 1
			if last >= 0 && output.Contents[last].Role == role &&
				output.Contents[last].Parts[0].FunctionResponse == nil {
				output.Contents[last].Parts = append(output.Contents[last].Parts, parts...)
			} else {
				output.Contents = append(output.Contents, vertexContent{Role: role, Parts: parts})
			}
			if len(message.ToolCalls) > 0 || message.Function != nil {
				output.toolHistory = true
				results, err := message.toolResults(input.Messages[i+1:])
				if err != nil {
					return vertexRequest{}, err
				}
				output.Contents = append(output.Contents, vertexContent{Role: "user", Parts: results})
				i += len(results)
			}
		case "tool", "function":
			return vertexRequest{}, errors.New("tool result has no preceding assistant call")
		default:
			return vertexRequest{}, errors.New("unsupported message role")
		}
	}
	if len(output.Contents) == 0 || output.Contents[0].Role != "user" {
		return vertexRequest{}, errors.New("conversation must start with a user message")
	}
	max := input.MaxCompletionTokens
	if max == nil {
		max = input.MaxTokens
	}
	if max != nil && *max < 1 {
		return vertexRequest{}, errors.New("token limit must be positive")
	}
	output.Config = generationConfig{MaxOutputTokens: max, Temperature: input.Temperature, TopP: input.TopP}
	if len(input.Stop) > 0 && string(input.Stop) != "null" {
		var stop string
		if json.Unmarshal(input.Stop, &stop) == nil {
			output.Config.StopSequences = []string{stop}
		} else if json.Unmarshal(input.Stop, &output.Config.StopSequences) != nil {
			return vertexRequest{}, errors.New("stop must be a string or an array of strings")
		}
	}
	if format := input.ResponseFormat; format != nil {
		switch format.Type {
		case "text":
		case "json_object":
			output.Config.ResponseMIME = "application/json"
		case "json_schema":
			if len(format.JSONSchema.Schema) == 0 {
				return vertexRequest{}, errors.New("missing response_format JSON schema")
			}
			output.Config.ResponseMIME = "application/json"
			output.Config.ResponseSchema = format.JSONSchema.Schema
		default:
			return vertexRequest{}, errors.New("unsupported response_format")
		}
	}
	if input.ReasoningEffort != "" {
		level := strings.ToUpper(input.ReasoningEffort)
		model := strings.TrimPrefix(input.Model, "google/")
		if model == "gemini-4-argon" || model == "gemini-4-carbon" ||
			strings.HasPrefix(model, "gemini-4-argon-") || strings.HasPrefix(model, "gemini-4-carbon-") {
			switch level {
			case "NONE", "MINIMAL":
				level = "LOW"
			case "XHIGH":
				level = "HIGH"
			}
		}
		output.Config.Thinking = &thinkingConfig{Level: level}
	}
	return output, nil
}

func (v vertexClient) endpoint(version, action string) url.URL {
	host := "aiplatform.googleapis.com"
	switch v.Location {
	case "global":
	case "us", "eu":
		host = "aiplatform." + v.Location + ".rep.googleapis.com"
	default:
		host = v.Location + "-aiplatform.googleapis.com"
	}
	return url.URL{Scheme: "https", Host: host, Path: fmt.Sprintf("/%s/projects/%s/locations/%s/publishers/google/models/%s",
		version, v.Project, v.Location, action)}
}

func (v vertexClient) generate(ctx context.Context, model string, body vertexRequest, stream bool) (*http.Response, error) {
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	action := model + ":generateContent"
	if stream {
		action = model + ":streamGenerateContent"
	}
	version := "v1"
	if len(body.Tools) > 0 || body.toolHistory {
		version = "v1beta1"
	}
	endpoint := v.endpoint(version, action)
	if stream {
		endpoint.RawQuery = "alt=sse"
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint.String(), bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-api-key", v.Key)
	return v.Client.Do(req)
}

var modelAction = regexp.MustCompile("^[a-z0-9][a-z0-9-]*:[a-zA-Z][a-zA-Z0-9]*$")

func (v vertexClient) native(w http.ResponseWriter, r *http.Request) {
	version, action := r.PathValue("version"), r.PathValue("action")
	if (version != "v1" && version != "v1beta1") || !modelAction.MatchString(action) {
		writeError(w, http.StatusBadRequest, "invalid Vertex version or model method")
		return
	}
	query, err := url.ParseQuery(r.URL.RawQuery)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid query")
		return
	}
	for key := range query {
		switch strings.ToLower(key) {
		case "key", "$key", "access_token", "oauth_token", "$httpheaders":
			query.Del(key)
		}
	}
	endpoint := v.endpoint(version, action)
	endpoint.RawQuery = query.Encode()
	ctx, cancel := context.WithTimeout(r.Context(), v.Client.Timeout)
	defer cancel()
	proxy := httputil.ReverseProxy{
		Transport: v.Client.Transport,
		Rewrite: func(req *httputil.ProxyRequest) {
			req.Out.URL = &endpoint
			req.Out.Host = endpoint.Host
			req.Out.Header = http.Header{}
			for _, key := range []string{"Accept", "Content-Type", "Content-Encoding"} {
				if value := req.In.Header.Get(key); value != "" {
					req.Out.Header.Set(key, value)
				}
			}
			req.Out.Header.Set("x-goog-api-key", v.Key)
		},
		ModifyResponse: func(resp *http.Response) error {
			resp.Header.Del("Location")
			return nil
		},
		ErrorHandler: func(w http.ResponseWriter, _ *http.Request, err error) {
			if errors.Is(err, context.DeadlineExceeded) {
				writeError(w, http.StatusGatewayTimeout, "Vertex request timed out")
				return
			}
			writeError(w, http.StatusBadGateway, "Vertex request failed")
		},
	}
	proxy.ServeHTTP(w, r.WithContext(ctx))
}
