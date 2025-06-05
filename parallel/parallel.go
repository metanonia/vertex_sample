package parallel

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"

	"cloud.google.com/go/vertexai/genai"
)

// parallelFunctionCalling shows how to execute multiple function calls in parallel
// and return their results to the model for generating a complete response.
func parallelFunctionCalling(w io.Writer, projectID string, location map[string]string, modelName string) error {
	// location = "us-central1"
	// modelName = "gemini-1.5-flash-002"
	ctx := context.Background()
	client, err := genai.NewClient(ctx, projectID, "us-central1")
	if err != nil {
		return fmt.Errorf("failed to create GenAI client: %w", err)
	}
	defer client.Close()

	model := client.GenerativeModel(modelName)
	// Set temperature to 0.0 for maximum determinism in function calling.
	model.SetTemperature(0.0)

	funcName := "getCurrentWeather"
	funcDecl := &genai.FunctionDeclaration{
		Name:        funcName,
		Description: "Get the current weather in a given location",
		Parameters: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"location": {
					Type: genai.TypeString,
					Description: "The location for which to get the weather. " +
						"It can be a city name, a city name and state, or a zip code. " +
						"Examples: 'San Francisco', 'San Francisco, CA', '95616', etc.",
				},
			},
			Required: []string{"location"},
		},
	}
	// Add the weather function to our model toolbox.
	model.Tools = []*genai.Tool{
		{
			FunctionDeclarations: []*genai.FunctionDeclaration{funcDecl},
		},
	}

	prompt := genai.Text("Get weather details in New Delhi and San Francisco?")
	resp, err := model.GenerateContent(ctx, prompt)

	if err != nil {
		return fmt.Errorf("failed to generate content: %w", err)
	}
	if len(resp.Candidates) == 0 {
		return errors.New("got empty response from model")
	} else if len(resp.Candidates[0].FunctionCalls()) == 0 {
		return errors.New("got no function call suggestions from model")
	}

	// In a production environment, consider adding validations for function names and arguments.
	for _, fnCall := range resp.Candidates[0].FunctionCalls() {
		fmt.Fprintf(w, "The model suggests to call the function %q with args: %v\n", fnCall.Name, fnCall.Args)
		// Example response:
		// The model suggests to call the function "getCurrentWeather" with args: map[location:New Delhi]
		// The model suggests to call the function "getCurrentWeather" with args: map[location:San Francisco]
	}

	// Use synthetic data to simulate responses from the external API.
	// In a real application, this would come from an actual weather API.
	mockAPIResp1, err := json.Marshal(map[string]string{
		"location":         "New Delhi",
		"temperature":      "42",
		"temperature_unit": "C",
		"description":      "Hot and humid",
		"humidity":         "65",
	})
	if err != nil {
		return fmt.Errorf("failed to marshal function response to JSON: %w", err)
	}

	mockAPIResp2, err := json.Marshal(map[string]string{
		"location":         "San Francisco",
		"temperature":      "36",
		"temperature_unit": "F",
		"description":      "Cold and cloudy",
		"humidity":         "N/A",
	})
	if err != nil {
		return fmt.Errorf("failed to marshal function response to JSON: %w", err)
	}

	// Note, that the function calls don't have to be chained. We can obtain both responses in parallel
	// and return them to Gemini at once.
	funcResp1 := &genai.FunctionResponse{
		Name: funcName,
		Response: map[string]any{
			"content": mockAPIResp1,
		},
	}
	funcResp2 := &genai.FunctionResponse{
		Name: funcName,
		Response: map[string]any{
			"content": mockAPIResp2,
		},
	}

	// Return both API responses to the model allowing it to complete its response.
	resp, err = model.GenerateContent(ctx, prompt, funcResp1, funcResp2)
	if err != nil {
		return fmt.Errorf("failed to generate content: %w", err)
	}
	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Content.Parts) == 0 {
		return errors.New("got empty response from model")
	}

	fmt.Fprintln(w, resp.Candidates[0].Content.Parts[0])
	// Example response:
	// The weather in New Delhi is hot and humid with a humidity of 65 and a temperature of 42°C. The weather in San Francisco ...

	return nil
}
