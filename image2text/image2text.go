package image2text

import (
	"context"
	"fmt"
	"io"
	"mime"
	"path/filepath"

	"cloud.google.com/go/vertexai/genai"
)

// generateMultimodalContent generates a response into w, based upon the  provided image.
func generateMultimodalContent(w io.Writer, projectID, location, modelName string) error {
	// location := "us-central1"
	// model := "gemini-1.5-flash-001"
	ctx := context.Background()

	client, err := genai.NewClient(ctx, projectID, location)
	if err != nil {
		return fmt.Errorf("unable to create client: %w", err)
	}
	defer client.Close()

	model := client.GenerativeModel(modelName)
	model.SetTemperature(0.4)
	// configure the safety settings thresholds
	model.SafetySettings = []*genai.SafetySetting{
		{
			Category:  genai.HarmCategoryHarassment,
			Threshold: genai.HarmBlockLowAndAbove,
		},
		{
			Category:  genai.HarmCategoryDangerousContent,
			Threshold: genai.HarmBlockLowAndAbove,
		},
	}

	// Given an image file URL, prepare image file as genai.Part
	img := genai.FileData{
		//MIMEType: mime.TypeByExtension(filepath.Ext("320px-Felis_catus-cat_on_snow.jpg")),
		//FileURI:  "gs://cloud-samples-data/generative-ai/image/320px-Felis_catus-cat_on_snow.jpg",
		MIMEType: mime.TypeByExtension(filepath.Ext("/background.jpeg")),
		FileURI:  "https://metanonia.com/images/background.jpeg",
	}

	res, err := model.GenerateContent(ctx, img, genai.Text("describe this image using Korean."))
	if err != nil {
		return fmt.Errorf("unable to generate contents: %w", err)
	}

	fmt.Fprintf(w, "generated response: %s\n", res.Candidates[0].Content.Parts[0])
	return nil
}
