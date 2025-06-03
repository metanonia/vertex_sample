package main

import (
	"context"
	"fmt"
	"google.golang.org/genai"
	"log"
	"os"
)

func main() {
	ctx := context.Background()
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		Project:  "metanonia-53f36", // GCP 프로젝트 ID
		Location: "us-central1",     // 예: "asia-northeast3"
		Backend:  genai.BackendVertexAI,
	})
	if err != nil {
		log.Fatal(err)
	}

	// 예시: 텍스트 생성
	result, err := client.Models.GenerateContent(ctx, "gemini-2.0-flash", genai.Text("서울에 대해 알려줘"), nil)
	if err != nil {
		log.Fatal(err)
	}
	log.Println(result.Candidates[0].Content.Parts[0].Text)

	// 이미지 생성 요청
	resp, err := client.Models.GenerateImages(ctx,
		"imagen-3.0-generate-002", // Imagen 모델 이름
		"A landscape-style painting depicting a giant whale soaring freely through a celestial blue cosmos. The whale is rendered with delicate brushstrokes, evoking a cloud-like ethereal quality. The deep azure cosmic expanse is sprinkled with subtle starlight, creating an enigmatic atmosphere. The whale’s body glows with a faint cerulean hue, contrasting vividly against the rich navy and violet tones of the spatial background. Its tail and fins, adorned with soft curvilinear forms, convey dynamic motion, while distant shimmering planets enhance the boundless mystery of the universe. The harmonious blend of the whale and cosmic elements creates a dreamlike, visually stunning composition that embodies otherworldly beauty",
		&genai.GenerateImagesConfig{},
	)
	if err != nil {
		log.Fatal(err)
	}

	// 결과 처리 (Base64 또는 GCS URI)
	fmt.Println("=============")
	fmt.Printf("%+v\n", resp)
	for idx, img := range resp.GeneratedImages {
		fmt.Printf("%+v\n", img)
		ofile := fmt.Sprintf("images/image_%d.png", idx)
		err := os.WriteFile(ofile, img.Image.ImageBytes, 0644) // 파일명과 권한 지정
		if err != nil {
			log.Fatal(err)
		}
	}

	// 이미지 생성 요청
	resp, err = client.Models.GenerateImages(ctx,
		"imagen-3.0-generate-002", // Imagen 모델 이름
		"따뜻한 봄날, 경복궁을 거닐고 있는 한복을 입은 아이돌",
		&genai.GenerateImagesConfig{
			Language:      genai.ImagePromptLanguageKo,
			EnhancePrompt: true,
		},
	)
	if err != nil {
		log.Fatal(err)
	}

	// 결과 처리 (Base64 또는 GCS URI)
	fmt.Println("=============")
	fmt.Printf("%+v\n", resp)
	for idx, img := range resp.GeneratedImages {
		fmt.Printf("%+v\n", img)
		ofile := fmt.Sprintf("images/image2_%d.png", idx)
		err := os.WriteFile(ofile, img.Image.ImageBytes, 0644) // 파일명과 권한 지정
		if err != nil {
			log.Fatal(err)
		}
	}
}
