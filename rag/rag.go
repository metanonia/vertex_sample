package main

import (
	"context"
	"fmt"
	"log"
	"math"

	aiplatform "cloud.google.com/go/aiplatform/apiv1"
	"cloud.google.com/go/aiplatform/apiv1/aiplatformpb"
	"cloud.google.com/go/vertexai/genai"
	"google.golang.org/api/option"
	"google.golang.org/protobuf/types/known/structpb"
)

// 전역 클라이언트
var (
	genaiClient      *genai.Client
	predictionClient *aiplatform.PredictionClient
	projectID        = "metanonia-53f36"
	location         = "us-central1"
	embeddingModel   = "text-multilingual-embedding-002"
	geminiModel      = "gemini-2.0-flash"
	endpointPrefix   = "projects/" + projectID + "/locations/" + location + "/publishers/google/models/"
)

func initClients(ctx context.Context) error {
	var err error
	genaiClient, err = genai.NewClient(ctx, projectID, location)
	if err != nil {
		return fmt.Errorf("genai.NewClient: %v", err)
	}
	predictionClient, err = aiplatform.NewPredictionClient(ctx,
		option.WithEndpoint(location+"-aiplatform.googleapis.com:443"))
	if err != nil {
		return fmt.Errorf("aiplatform.NewPredictionClient: %v", err)
	}
	return nil
}

func main() {
	ctx := context.Background()
	if err := initClients(ctx); err != nil {
		log.Fatal(err)
	}
	defer genaiClient.Close()
	defer predictionClient.Close()

	// 1. 문서 임베딩 생성
	documents := map[string]string{
		"doc1": "Vertex AI는 Google Cloud의 ML 플랫폼입니다",
		"doc2": "RAG는 검색과 생성을 결합한 AI 접근법",
	}
	documentEmbeddings := make(map[string][]float32)
	for id, content := range documents {
		emb, err := getDocumentEmbedding(ctx, content)
		if err != nil {
			log.Fatalf("문서 임베딩 실패: %v", err)
		}
		documentEmbeddings[id] = emb
	}

	// 2. 쿼리 임베딩 생성
	query := "Vertex AI로 RAG를 어떻게 구현하나요?"
	queryEmbedding, err := getQueryEmbedding(ctx, query)
	if err != nil {
		log.Fatalf("쿼리 임베딩 실패: %v", err)
	}

	// 3. 유사도 기반 문서 검색
	mostSimilarDocID := findMostSimilar(documentEmbeddings, queryEmbedding)
	mostSimilarDocContent := documents[mostSimilarDocID]

	// 4. Gemini 모델을 이용한 응답 생성
	model := genaiClient.GenerativeModel(geminiModel)
	prompt := fmt.Sprintf(`다음 문서를 기반으로 질문에 답하세요:
문서: %s
질문: %s`, mostSimilarDocContent, query)

	resp, err := model.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		log.Fatalf("Gemini 응답 생성 실패: %v", err)
	}

	fmt.Println(resp.Candidates[0].Content.Parts[0])
}

// 문서 임베딩 생성 함수
func getDocumentEmbedding(ctx context.Context, text string) ([]float32, error) {
	req := &aiplatformpb.PredictRequest{
		Endpoint: endpointPrefix + embeddingModel,
		Instances: []*structpb.Value{
			structpb.NewStructValue(&structpb.Struct{
				Fields: map[string]*structpb.Value{
					"content": structpb.NewStringValue(text),
				},
			}),
		},
		Parameters: structpb.NewStructValue(&structpb.Struct{
			Fields: map[string]*structpb.Value{
				"outputDimensionality": structpb.NewNumberValue(256),
			},
		}),
	}
	resp, err := predictionClient.Predict(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("임베딩 API 호출 실패: %v", err)
	}
	if len(resp.Predictions) == 0 {
		return nil, fmt.Errorf("빈 임베딩 응답")
	}
	values := resp.Predictions[0].GetListValue().GetValues()
	embedding := make([]float32, len(values))
	for i, v := range values {
		embedding[i] = float32(v.GetNumberValue())
	}
	return embedding, nil
}

// 쿼리 임베딩 생성 함수
func getQueryEmbedding(ctx context.Context, query string) ([]float32, error) {
	req := &aiplatformpb.PredictRequest{
		Endpoint: endpointPrefix + embeddingModel,
		Instances: []*structpb.Value{
			structpb.NewStructValue(&structpb.Struct{
				Fields: map[string]*structpb.Value{
					"content":   structpb.NewStringValue(query),
					"task_type": structpb.NewStringValue("RETRIEVAL_QUERY"),
				},
			}),
		},
		Parameters: structpb.NewStructValue(&structpb.Struct{
			Fields: map[string]*structpb.Value{
				"outputDimensionality": structpb.NewNumberValue(256),
			},
		}),
	}
	resp, err := predictionClient.Predict(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("임베딩 API 호출 실패: %v", err)
	}
	if len(resp.Predictions) == 0 {
		return nil, fmt.Errorf("빈 임베딩 응답")
	}
	values := resp.Predictions[0].GetListValue().GetValues()
	embedding := make([]float32, len(values))
	for i, v := range values {
		embedding[i] = float32(v.GetNumberValue())
	}
	return embedding, nil
}

// 코사인 유사도 계산 함수
func cosineSimilarity(a, b []float32) float32 {
	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

// 가장 유사한 문서 찾기
func findMostSimilar(docs map[string][]float32, queryEmb []float32) string {
	var maxScore float32 = -1
	var bestDocID string
	for docID, docEmb := range docs {
		score := cosineSimilarity(queryEmb, docEmb)
		if score > maxScore {
			maxScore = score
			bestDocID = docID
		}
	}
	return bestDocID
}
