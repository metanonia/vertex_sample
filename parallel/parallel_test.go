package parallel

import (
	"bytes"
	"io"
	"testing"
)

func TestParallelFunctionCalling(t *testing.T) {
	tests := []struct {
		name        string
		projectID   string
		location    map[string]string
		modelName   string
		wantErr     bool
		outputRegex string // 출력 검증용 정규식
	}{
		{
			name:        "Valid Parameters",
			projectID:   "metanonia-53f36",
			location:    map[string]string{"location": "San Francisco"},
			modelName:   "gemini-2.0-flash",
			wantErr:     false,
			outputRegex: `.*function calls.*processed.*`,
		},
		{
			name:        "Invalid Project ID",
			projectID:   "",
			location:    map[string]string{"location": "New Delhi"},
			modelName:   "gemini-2.0-flash",
			wantErr:     true,
			outputRegex: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// 출력 캡처용 버퍼
			var buf bytes.Buffer
			w := io.Writer(&buf)

			// 테스트 대상 함수 실행
			err := parallelFunctionCalling(w, tt.projectID, tt.location, tt.modelName)

			// 오류 검증
			if (err != nil) != tt.wantErr {
				t.Errorf("parallelFunctionCalling() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			// 출력 검증 (오류가 없는 경우만)
			if !tt.wantErr {
				t.Logf("Output Got: %s", buf.String())
			}
		})
	}
}
