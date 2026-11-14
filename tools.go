package main

import (
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"errors"
	"strings"
)

type functionDeclaration struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type chatTool struct {
	Type     string              `json:"type"`
	Function functionDeclaration `json:"function"`
}

type chatToolCall struct {
	ID       string         `json:"id"`
	Type     string         `json:"type"`
	Index    *int           `json:"index,omitempty"`
	Function chatFunction   `json:"function"`
	Extra    *functionExtra `json:"extra_content,omitempty"`
}

const signedToolCallPrefix = "call_vertex2api_v1_"

type signedToolCall struct {
	ID        string `json:"id,omitempty"`
	Signature string `json:"signature"`
	Nonce     string `json:"nonce,omitempty"`
}

// Some OpenAI clients only round-trip the standard tool call fields.
func signedToolCallID(id, signature string) string {
	identity := signedToolCall{ID: id, Signature: signature}
	if id == "" {
		identity.Nonce = rand.Text()
	}
	data, _ := json.Marshal(identity)
	return signedToolCallPrefix + base64.RawURLEncoding.EncodeToString(data)
}

func (c chatToolCall) signedIdentity() signedToolCall {
	encoded, ok := strings.CutPrefix(c.ID, signedToolCallPrefix)
	if !ok {
		return signedToolCall{}
	}
	data, err := base64.RawURLEncoding.DecodeString(encoded)
	if err != nil {
		return signedToolCall{}
	}
	var identity signedToolCall
	if json.Unmarshal(data, &identity) != nil || identity.Signature == "" {
		return signedToolCall{}
	}
	return identity
}

type chatFunction struct {
	Name      string         `json:"name"`
	Arguments string         `json:"arguments"`
	Extra     *functionExtra `json:"extra_content,omitempty"`
}

type functionExtra struct {
	Google struct {
		Signature string `json:"thought_signature"`
		VertexID  string `json:"vertex_function_call_id,omitempty"`
	} `json:"google"`
}

type vertexTool struct {
	Declarations []vertexFunctionDeclaration `json:"functionDeclarations"`
}

type vertexFunctionDeclaration struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parametersJsonSchema,omitempty"`
}

type vertexToolConfig struct {
	Calling vertexCallingConfig `json:"functionCallingConfig"`
}

type vertexCallingConfig struct {
	Mode    string   `json:"mode"`
	Allowed []string `json:"allowedFunctionNames,omitempty"`
}

type vertexFunctionCall struct {
	ID   string          `json:"id,omitempty"`
	Name string          `json:"name"`
	Args json.RawMessage `json:"args,omitempty"`
}

type vertexFunctionResponse struct {
	ID       string          `json:"id,omitempty"`
	Name     string          `json:"name"`
	Response json.RawMessage `json:"response"`
}

func translateTools(input chatRequest) (vertexRequest, error) {
	var output vertexRequest
	if len(input.Tools) > 0 && len(input.Functions) > 0 {
		return output, errors.New("specify tools or functions, not both")
	}
	definitions := input.Functions
	for _, tool := range input.Tools {
		if tool.Type != "function" {
			return output, errors.New("only function tools are supported")
		}
		definitions = append(definitions, tool.Function)
	}
	names := make(map[string]bool, len(definitions))
	var declarations []vertexFunctionDeclaration
	for _, definition := range definitions {
		if definition.Name == "" || names[definition.Name] {
			return output, errors.New("function names must be nonempty and unique")
		}
		names[definition.Name] = true
		declarations = append(declarations, vertexFunctionDeclaration(definition))
	}
	if len(declarations) > 0 {
		output.Tools = []vertexTool{{Declarations: declarations}}
	}

	raw := input.ToolChoice
	if len(raw) == 0 || string(raw) == "null" {
		raw = input.FunctionChoice
	}
	if len(raw) == 0 || string(raw) == "null" {
		return output, nil
	}
	calling, err := translateToolChoice(raw, names)
	if err != nil {
		return output, err
	}
	if len(declarations) > 0 {
		output.ToolConfig = &vertexToolConfig{Calling: calling}
	}
	return output, nil
}

func translateToolChoice(raw json.RawMessage, names map[string]bool) (vertexCallingConfig, error) {
	var choice string
	if json.Unmarshal(raw, &choice) == nil {
		switch choice {
		case "auto", "none":
			return vertexCallingConfig{Mode: strings.ToUpper(choice)}, nil
		case "required":
			if len(names) == 0 {
				return vertexCallingConfig{}, errors.New("required tool_choice needs at least one function")
			}
			return vertexCallingConfig{Mode: "ANY"}, nil
		}
		return vertexCallingConfig{}, errors.New("invalid tool choice")
	}
	var selected struct {
		Type     string       `json:"type"`
		Function chatFunction `json:"function"`
		Name     string       `json:"name"`
	}
	if json.Unmarshal(raw, &selected) != nil {
		return vertexCallingConfig{}, errors.New("invalid tool choice")
	}
	name := selected.Name
	if selected.Type == "function" {
		name = selected.Function.Name
	} else if selected.Type != "" {
		return vertexCallingConfig{}, errors.New("only function tool choices are supported")
	}
	if !names[name] {
		return vertexCallingConfig{}, errors.New("tool choice must name a declared function")
	}
	return vertexCallingConfig{Mode: "ANY", Allowed: []string{name}}, nil
}

func (m chatMessage) parts() ([]vertexPart, error) {
	if len(m.ToolCalls) == 0 && m.Function == nil {
		return messageParts(m.Content)
	}
	if m.Role != "assistant" || (m.Function != nil && len(m.ToolCalls) > 0) {
		return nil, errors.New("invalid assistant function calls")
	}
	var parts []vertexPart
	if len(m.Content) > 0 && string(m.Content) != "null" && string(m.Content) != `""` {
		var err error
		parts, err = messageParts(m.Content)
		if err != nil {
			return nil, err
		}
	}
	if m.Function != nil {
		part, err := m.Function.vertexPart()
		return append(parts, part), err
	}
	ids := make(map[string]bool, len(m.ToolCalls))
	for _, call := range m.ToolCalls {
		if call.Type != "function" || call.ID == "" || ids[call.ID] {
			return nil, errors.New("tool calls must be functions with unique nonempty IDs")
		}
		ids[call.ID] = true
		function := call.Function
		function.Extra = call.Extra
		part, err := function.vertexPart()
		if err != nil {
			return nil, err
		}
		part.FunctionCall.ID = call.vertexID()
		if part.ThoughtSignature == "" {
			part.ThoughtSignature = call.signedIdentity().Signature
		}
		parts = append(parts, part)
	}
	return parts, nil
}

func (c chatToolCall) vertexID() string {
	if identity := c.signedIdentity(); identity.Signature != "" {
		return identity.ID
	}
	// IDs generated for the OpenAI client are absent in the Vertex replay.
	if strings.HasPrefix(c.ID, "call_vertex2api_") {
		return ""
	}
	return c.ID
}

func (f chatFunction) vertexPart() (vertexPart, error) {
	args := json.RawMessage(f.Arguments)
	if f.Name == "" || !json.Valid(args) || !isJSONObject(args) {
		return vertexPart{}, errors.New("function call requires a name and JSON object arguments")
	}
	part := vertexPart{FunctionCall: &vertexFunctionCall{Name: f.Name, Args: args}}
	if f.Extra != nil {
		part.ThoughtSignature = f.Extra.Google.Signature
		part.FunctionCall.ID = f.Extra.Google.VertexID
	}
	return part, nil
}

func (m chatMessage) toolResults(history []chatMessage) ([]vertexPart, error) {
	if m.Function != nil {
		if len(history) == 0 || history[0].Role != "function" || history[0].Name != m.Function.Name {
			return nil, errors.New("missing result for assistant function call")
		}
		id := ""
		if m.Function.Extra != nil {
			id = m.Function.Extra.Google.VertexID
		}
		part, err := history[0].resultPart(m.Function.Name, id)
		return []vertexPart{part}, err
	}
	count := len(m.ToolCalls)
	if len(history) < count {
		return nil, errors.New("missing results for assistant tool calls")
	}
	results := make(map[string]chatMessage, count)
	for _, result := range history[:count] {
		if result.Role != "tool" || result.ToolCallID == "" || results[result.ToolCallID].Role != "" {
			return nil, errors.New("expected a result for each assistant tool call")
		}
		results[result.ToolCallID] = result
	}
	parts := make([]vertexPart, 0, count)
	for _, call := range m.ToolCalls {
		result, ok := results[call.ID]
		if !ok {
			return nil, errors.New("tool_call_id must match a preceding assistant tool call")
		}
		part, err := result.resultPart(call.Function.Name, call.vertexID())
		if err != nil {
			return nil, err
		}
		parts = append(parts, part)
	}
	return parts, nil
}

func (m chatMessage) resultPart(name, id string) (vertexPart, error) {
	parts, err := messageParts(m.Content)
	if err != nil {
		return vertexPart{}, err
	}
	result := vertexCandidate{Content: vertexContent{Parts: parts}}.Text()
	payload := json.RawMessage(result)
	if !json.Valid(payload) {
		payload, err = json.Marshal(result)
		if err != nil {
			return vertexPart{}, err
		}
	}
	if !isJSONObject(payload) {
		payload = append(append(json.RawMessage(`{"output":`), payload...), '}')
	}
	return vertexPart{FunctionResponse: &vertexFunctionResponse{ID: id, Name: name, Response: payload}}, nil
}

func isJSONObject(raw json.RawMessage) bool {
	return strings.HasPrefix(strings.TrimSpace(string(raw)), "{")
}

func (c vertexCandidate) toolCalls(start int, stream bool) []chatToolCall {
	var calls []chatToolCall
	for _, part := range c.Content.Parts {
		call := part.FunctionCall
		if call == nil {
			continue
		}
		id := call.ID
		if part.ThoughtSignature != "" {
			id = signedToolCallID(id, part.ThoughtSignature)
		} else if id == "" {
			id = "call_vertex2api_" + rand.Text()
		}
		args := string(call.Args)
		if args == "" || args == "null" {
			args = "{}"
		}
		result := chatToolCall{ID: id, Type: "function", Function: chatFunction{Name: call.Name, Arguments: args}}
		if part.ThoughtSignature != "" {
			extra := &functionExtra{}
			extra.Google.Signature = part.ThoughtSignature
			result.Extra = extra
		}
		if stream {
			index := start + len(calls)
			result.Index = &index
		}
		calls = append(calls, result)
	}
	return calls
}

func (m *message) setToolCalls(calls []chatToolCall, legacy bool) error {
	if !legacy {
		m.ToolCalls = calls
	} else if len(calls) == 1 {
		function := calls[0].Function
		function.Extra = calls[0].Extra
		id := calls[0].vertexID()
		if id != "" {
			if function.Extra == nil {
				function.Extra = &functionExtra{}
			}
			function.Extra.Google.VertexID = id
		}
		m.Function = &function
	} else if len(calls) > 1 {
		return errors.New("Vertex returned multiple calls for legacy functions")
	}
	return nil
}

func toolFinishReason(reason string, legacy bool) string {
	if reason != "stop" {
		return reason
	}
	if legacy {
		return "function_call"
	}
	return "tool_calls"
}
