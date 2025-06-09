package main

import (
	"context"
	"fmt"
	"google.golang.org/protobuf/types/known/structpb"
	"log"

	aiplatform "cloud.google.com/go/aiplatform/apiv1"
	"cloud.google.com/go/aiplatform/apiv1/aiplatformpb"
	"cloud.google.com/go/vertexai/genai"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"
	pgxvec "github.com/pgvector/pgvector-go/pgx"
	"google.golang.org/api/option"
)

// PostgreSQL 연결 정보
const (
	dbHost     = "34.132.111.179"
	dbPort     = 5432
	dbUser     = "admin"
	dbPassword = "m2tan0n1a!"
	dbName     = "vector_test"
)

var (
	genaiClient      *genai.Client
	predictionClient *aiplatform.PredictionClient
	dbPool           *pgxpool.Pool
	projectID        = "metanonia-53f36"
	location         = "us-central1"
	embeddingModel   = "text-multilingual-embedding-002"
	geminiModel      = "gemini-2.0-flash"
	endpointPrefix   = "projects/" + projectID + "/locations/" + location + "/publishers/google/models/"
)

func initClients(ctx context.Context) error {
	var err error

	// GenAI 클라이언트 초기화
	genaiClient, err = genai.NewClient(ctx, projectID, location)
	if err != nil {
		return fmt.Errorf("genai.NewClient: %v", err)
	}

	// Vertex AI 예측 클라이언트 초기화
	predictionClient, err = aiplatform.NewPredictionClient(ctx,
		option.WithEndpoint(location+"-aiplatform.googleapis.com:443"))
	if err != nil {
		return fmt.Errorf("aiplatform.NewPredictionClient: %v", err)
	}

	// PostgreSQL 연결 풀 초기화
	connStr := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
		dbHost, dbPort, dbUser, dbPassword, dbName)

	config, err := pgxpool.ParseConfig(connStr)
	if err != nil {
		return fmt.Errorf("pgxpool.ParseConfig: %v", err)
	}

	config.AfterConnect = func(ctx context.Context, conn *pgx.Conn) error {
		err := pgxvec.RegisterTypes(ctx, conn)
		fmt.Errorf("AfterConnect: %v\n", err)
		return nil
	}

	dbPool, err = pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		return fmt.Errorf("pgxpool.Connect: %v", err)
	}

	return initDB(ctx)
}

func initDB(ctx context.Context) error {
	// pgvector 확장 활성화
	_, err := dbPool.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS vector")
	if err != nil {
		return fmt.Errorf("CREATE EXTENSION vector: %v", err)
	}

	// 문서 저장 테이블 생성
	_, err = dbPool.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS documents (
			id TEXT PRIMARY KEY,
			content TEXT,
			embedding VECTOR(256)
		)
	`)
	return err
}

func main() {
	ctx := context.Background()
	if err := initClients(ctx); err != nil {
		log.Fatal(err)
	}
	defer genaiClient.Close()
	defer predictionClient.Close()
	defer dbPool.Close()

	// 1. 문서 임베딩 생성 및 저장
	documents := map[string]string{
		"doc1": "Vertex AI는 Google Cloud의 ML 플랫폼입니다",
		"doc2": "RAG는 검색과 생성을 결합한 AI 접근법",
	}

	for id, content := range documents {
		emb, err := getEmbedding(ctx, content, "RETRIEVAL_DOCUMENT")
		if err != nil {
			log.Fatalf("문서 임베딩 실패: %v", err)
		}

		// PostgreSQL에 문서 저장
		_, err = dbPool.Exec(ctx,
			"INSERT INTO documents (id, content, embedding) VALUES ($1, $2, $3)ON CONFLICT (id) DO NOTHING",
			id, content, pgvector.NewVector(emb),
		)
		if err != nil {
			log.Fatalf("문서 저장 실패: %v", err)
		}
	}

	// 2. 쿼리 임베딩 생성
	query := "Vertex AI로 RAG를 어떻게 구현하나요?"
	queryEmb, err := getEmbedding(ctx, query, "RETRIEVAL_QUERY")
	if err != nil {
		log.Fatalf("쿼리 임베딩 실패: %v", err)
	}

	// 3. pgvector를 이용한 유사도 검색
	var similarContent string
	err = dbPool.QueryRow(ctx, `
		SELECT content 
		FROM documents 
		ORDER BY embedding <=> $1 
		LIMIT 1
	`, pgvector.NewVector(queryEmb)).Scan(&similarContent)

	if err != nil {
		log.Fatalf("유사도 검색 실패: %v", err)
	}

	// 4. Gemini 모델 응답 생성
	model := genaiClient.GenerativeModel(geminiModel)
	prompt := fmt.Sprintf(`다음 문서를 기반으로 질문에 답하세요:
문서: %s
질문: %s`, similarContent, query)

	resp, err := model.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		log.Fatalf("Gemini 응답 생성 실패: %v", err)
	}

	fmt.Println(resp.Candidates[0].Content.Parts[0])
}

// 임베딩 생성 공통 함수
// 수정된 getEmbedding 함수
func getEmbedding(ctx context.Context, text string, taskType string) ([]float32, error) {
	req := &aiplatformpb.PredictRequest{
		Endpoint: endpointPrefix + embeddingModel,
		Instances: []*structpb.Value{
			structpb.NewStructValue(&structpb.Struct{
				Fields: map[string]*structpb.Value{
					"content":   structpb.NewStringValue(text),
					"task_type": structpb.NewStringValue(taskType),
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

	// 임베딩 데이터 구조 파싱 (중첩 필드 처리)
	predictionStruct := resp.Predictions[0].GetStructValue()
	if predictionStruct == nil {
		return nil, fmt.Errorf("예측 결과 형식 오류: 구조체 아님")
	}

	embeddingsField := predictionStruct.Fields["embeddings"]
	if embeddingsField == nil {
		return nil, fmt.Errorf("embeddings 필드 없음")
	}

	embeddingsStruct := embeddingsField.GetStructValue()
	if embeddingsStruct == nil {
		return nil, fmt.Errorf("embeddings 구조체 아님")
	}

	valuesField := embeddingsStruct.Fields["values"]
	if valuesField == nil {
		return nil, fmt.Errorf("values 필드 없음")
	}

	listValue := valuesField.GetListValue()
	if listValue == nil {
		return nil, fmt.Errorf("리스트 형식 아님")
	}

	values := listValue.GetValues()
	if len(values) == 0 {
		return nil, fmt.Errorf("임베딩 데이터가 비어있음")
	}

	embedding := make([]float32, len(values))
	for i, v := range values {
		embedding[i] = float32(v.GetNumberValue())
	}
	return embedding, nil
}
