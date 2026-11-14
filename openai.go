package main

import (
	"bufio"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"
)

type handler struct{ vertex vertexClient }

type completion struct {
	ID      string     `json:"id"`
	Object  string     `json:"object"`
	Created int64      `json:"created"`
	Model   string     `json:"model"`
	Choices []choice   `json:"choices"`
	Usage   *chatUsage `json:"usage,omitempty"`
}

type choice struct {
	Index   int      `json:"index"`
	Message *message `json:"message,omitempty"`
	Delta   *message `json:"delta,omitempty"`
	Finish  *string  `json:"finish_reason"`
}

type message struct {
	Role      string         `json:"role,omitempty"`
	Content   *string        `json:"content,omitempty"`
	ToolCalls []chatToolCall `json:"tool_calls,omitempty"`
	Function  *chatFunction  `json:"function_call,omitempty"`
}

func (h handler) chat(w http.ResponseWriter, r *http.Request) {
	var input chatRequest
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 2<<20))
	if err := decoder.Decode(&input); err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			writeError(w, 413, "request exceeds 2 MiB")
			return
		}
		writeError(w, 400, "invalid JSON request")
		return
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		writeError(w, 400, "expected one JSON object")
		return
	}
	model := strings.TrimPrefix(input.Model, "google/")
	if !resourceID.MatchString(model) {
		writeError(w, 400, "invalid model ID")
		return
	}
	body, err := translate(input)
	if err != nil {
		writeError(w, 400, err.Error())
		return
	}

	upstream, err := h.vertex.generate(r.Context(), model, body, input.Stream)
	if err != nil {
		var timeout net.Error
		if errors.As(err, &timeout) && timeout.Timeout() {
			writeError(w, 504, "Vertex request timed out")
			return
		}
		writeError(w, 502, "Vertex request failed")
		return
	}
	defer upstream.Body.Close()
	if upstream.StatusCode != 200 {
		status := upstream.StatusCode
		if status < 400 {
			status = 502
		}
		writeError(w, status, fmt.Sprintf("Vertex rejected %s (HTTP %d)", model, upstream.StatusCode))
		return
	}
	reply := completion{ID: "chatcmpl-" + rand.Text(), Object: "chat.completion", Created: time.Now().Unix(), Model: model}
	legacy := len(input.Functions) > 0
	if input.Stream {
		stream(w, upstream, reply, input.StreamOptions.IncludeUsage, legacy)
		return
	}
	var result vertexResponse
	if json.NewDecoder(io.LimitReader(upstream.Body, 16<<20)).Decode(&result) != nil || len(result.Error) > 0 {
		writeError(w, 502, "invalid Vertex response")
		return
	}
	candidate, ok := firstCandidate(result)
	if !ok {
		writeError(w, 502, "Vertex returned no completion")
		return
	}
	reason, err := finishReason(candidate.Finish)
	if err != nil || reason == "" {
		writeError(w, 502, "invalid Vertex finish reason")
		return
	}
	text := candidate.Text()
	response := message{Role: "assistant", Content: &text}
	calls := candidate.toolCalls(0, false)
	if len(calls) > 0 {
		if text == "" {
			response.Content = nil
		}
		if err := response.setToolCalls(calls, legacy); err != nil {
			writeError(w, 502, err.Error())
			return
		}
		reason = toolFinishReason(reason, legacy)
	}
	reply.Choices = []choice{{Message: &response, Finish: &reason}}
	reply.Usage = result.Usage.OpenAI()
	writeJSON(w, 200, reply)
}

func firstCandidate(result vertexResponse) (vertexCandidate, bool) {
	if len(result.Candidates) > 0 {
		return result.Candidates[0], true
	}
	if reason := result.PromptFeedback.BlockReason; reason != "" && reason != "BLOCK_REASON_UNSPECIFIED" {
		return vertexCandidate{Finish: "SAFETY"}, true
	}
	return vertexCandidate{}, false
}

func finishReason(reason string) (string, error) {
	switch reason {
	case "", "FINISH_REASON_UNSPECIFIED":
		return "", nil
	case "STOP":
		return "stop", nil
	case "MAX_TOKENS":
		return "length", nil
	case "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "MODEL_ARMOR":
		return "content_filter", nil
	default:
		return "", errors.New("unsupported Vertex finish reason")
	}
}

func stream(w http.ResponseWriter, upstream *http.Response, reply completion, includeUsage, legacy bool) {
	if !strings.HasPrefix(upstream.Header.Get("Content-Type"), "text/event-stream") {
		writeError(w, 502, "Vertex did not return an event stream")
		return
	}
	controller := http.NewResponseController(w)
	send := func(value any) error {
		data, err := json.Marshal(value)
		if err != nil {
			return err
		}
		if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
			return err
		}
		return controller.Flush()
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	reply.Object = "chat.completion.chunk"
	reply.Choices = []choice{{Delta: &message{Role: "assistant"}}}
	if err := send(reply); err != nil {
		return
	}

	var usage *chatUsage
	var finish string
	callCount := 0
	err := readEvents(upstream.Body, func(result vertexResponse) error {
		if len(result.Error) > 0 {
			return errors.New("Vertex stream returned an error")
		}
		if result.Usage != nil {
			usage = result.Usage.OpenAI()
		}
		candidate, ok := firstCandidate(result)
		if !ok {
			return nil
		}
		reason, err := finishReason(candidate.Finish)
		if err != nil {
			return err
		}
		if reason != "" {
			finish = reason
		}
		var delta message
		if text := candidate.Text(); text != "" {
			delta.Content = &text
		}
		calls := candidate.toolCalls(callCount, true)
		if legacy && callCount+len(calls) > 1 {
			return errors.New("Vertex returned multiple calls for legacy functions")
		}
		if err := delta.setToolCalls(calls, legacy); err != nil {
			return err
		}
		callCount += len(calls)
		if delta.Content != nil || len(calls) > 0 {
			reply.Choices = []choice{{Delta: &delta}}
			return send(reply)
		}
		return nil
	})
	if err != nil || finish == "" {
		_ = send(errorBody(502, "Vertex stream ended before completion"))
		return
	}
	if callCount > 0 {
		finish = toolFinishReason(finish, legacy)
	}
	reply.Choices = []choice{{Delta: &message{}, Finish: &finish}}
	if err := send(reply); err != nil {
		return
	}
	if includeUsage && usage != nil {
		reply.Choices = []choice{}
		reply.Usage = usage
		if err := send(reply); err != nil {
			return
		}
	}
	_, _ = io.WriteString(w, "data: [DONE]\n\n")
	_ = controller.Flush()
}

func readEvents(body io.Reader, handle func(vertexResponse) error) error {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 4096), 8<<20)
	var data []string
	dispatch := func() error {
		if len(data) == 0 {
			return nil
		}
		var event vertexResponse
		err := json.Unmarshal([]byte(strings.Join(data, "\n")), &event)
		data = data[:0]
		if err != nil {
			return err
		}
		return handle(event)
	}
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if err := dispatch(); err != nil {
				return err
			}
		} else if value, ok := strings.CutPrefix(line, "data:"); ok {
			data = append(data, strings.TrimPrefix(value, " "))
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	return dispatch()
}

type chatMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`
	Name       string          `json:"name"`
	ToolCallID string          `json:"tool_call_id"`
	ToolCalls  []chatToolCall  `json:"tool_calls"`
	Function   *chatFunction   `json:"function_call"`
}

type chatRequest struct {
	Model         string        `json:"model"`
	Messages      []chatMessage `json:"messages"`
	Stream        bool          `json:"stream"`
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
	MaxTokens           *int                  `json:"max_tokens"`
	MaxCompletionTokens *int                  `json:"max_completion_tokens"`
	Temperature         *float64              `json:"temperature"`
	TopP                *float64              `json:"top_p"`
	Stop                json.RawMessage       `json:"stop"`
	ReasoningEffort     string                `json:"reasoning_effort"`
	N                   *int                  `json:"n"`
	Tools               []chatTool            `json:"tools"`
	Functions           []functionDeclaration `json:"functions"`
	ToolChoice          json.RawMessage       `json:"tool_choice"`
	FunctionChoice      json.RawMessage       `json:"function_call"`
	ResponseFormat      *struct {
		Type       string `json:"type"`
		JSONSchema struct {
			Schema json.RawMessage `json:"schema"`
		} `json:"json_schema"`
	} `json:"response_format"`
}

type chatUsage struct {
	Prompt        int `json:"prompt_tokens"`
	Completion    int `json:"completion_tokens"`
	Total         int `json:"total_tokens"`
	PromptDetails struct {
		Cached int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
	CompletionDetails struct {
		Reasoning int `json:"reasoning_tokens"`
	} `json:"completion_tokens_details"`
}
