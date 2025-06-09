package function_call

import (
	"bytes"
	"fmt"
	"io"
	"testing"
)

func TestFunctioncall(t *testing.T) {
	// 출력 캡처용 버퍼
	var buf bytes.Buffer
	w := io.Writer(&buf)

	// 테스트 대상 함수 실행
	err := functionCallsChat(w, "metanonia-53f36", "us-central1", "gemini-2.0-flash")

	// 오류 검증
	if err != nil {
		t.Errorf("functionCallsChat() error = %v", err)
		return
	}

	// 버퍼 내용 출력
	fmt.Printf("Captured output:\n%s\n", buf.String())
}
