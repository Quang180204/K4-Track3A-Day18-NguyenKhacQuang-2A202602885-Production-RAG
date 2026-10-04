# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyễn Khắc Quang  
**Mã số học viên:** 2A202602885  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

Bảng đối chiếu giữa các khái niệm kiến trúc Production RAG trong bài giảng với phần hiện thực mã nguồn trong bài Lab 18:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích chuyên sâu |
|---|---|---|---|
| **Advanced Chunking (Semantic & Hierarchical)** | **M1: Chunking** | `chunk_semantic()` & `chunk_hierarchical()` | - `chunk_semantic()` sử dụng embedding similarity giữa các câu liên tiếp với ngưỡng `threshold = 0.85` để bảo toàn mạch ý văn bản thay vì cắt ngang theo số từ cứng nhắc như naive chunking.<br>- `chunk_hierarchical()` chia thành parent chunk (2048 ký tự) lưu giữ bối cảnh vĩ mô và child chunk (256 ký tự) phục vụ retrieval vi mô. Khi tìm kiếm match ở child chunk, hệ thống trả về parent chunk cho LLM đọc, giải quyết dứt điểm nghịch lý *"Retrieval cần chunk nhỏ để chính xác vs LLM cần chunk lớn để hiểu toàn diện"*. |
| **Hybrid Search & Fusion (Lexical + Dense)** | **M2: Search** | `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | - `BM25Search` kết hợp tách từ tiếng Việt bằng `underthesea` xử lý xuất sắc các từ khóa đặc thù (mã chính sách, số tiền "200.000.000 VNĐ", tên riêng "PVI", "MFA").<br>- `DenseSearch` dùng model đa ngữ `BAAI/bge-m3` (1024 chiều) cùng Qdrant vector database nắm bắt ngữ nghĩa trừu tượng.<br>- `reciprocal_rank_fusion()` (RRF) chuẩn hóa thứ hạng với tham số $k = 60$, hợp nhất hai danh sách kết quả độc lập mà không cần chuẩn hóa scale điểm số khác biệt giữa BM25 và Cosine distance. |
| **Cross-Encoder Reranking** | **M3: Reranking** | `CrossEncoderReranker.rerank()` | - Mô hình `BAAI/bge-reranker-v2-m3` đóng vai trò chốt chặn cuối cùng nhận cặp `(query, document)` và tính toán trực tiếp attention đa chiều.<br>- Reranker đã giúp **Context Precision tăng từ 0.8500 lên 0.9394 (+8.94%)**, sàng lọc chính xác 3 tài liệu sát nhất từ 20 ứng viên ban đầu của Hybrid Search, loại bỏ hoàn toàn chunk nhiễu hoặc sai lệch thời gian. |
| **RAGAS Multi-dimensional Evaluation** | **M4: Evaluation** | `evaluate_ragas()` & `failure_analysis()` | - Đánh giá định lượng toàn diện hệ thống RAG trên 4 chiều: Faithfulness (độ trung thực), Answer Relevancy (độ phù hợp câu trả lời), Context Precision (độ chính xác bối cảnh tìm được) và Context Recall (độ bao phủ thông tin).<br>- `failure_analysis()` xây dựng Diagnostic Tree phân loại lỗi tự động: nếu context precision thấp → lỗi do retrieval/rerank; nếu faithfulness thấp → LLM hallucination; nếu recall thấp → thiếu chunking/BM25. |
| **Chunk Enrichment (Single-Call Optimization)** | **M5: Enrichment** | `_enrich_single_call()` / `enrich_chunks()` | - Triển khai kỹ thuật Contextual Prepend, Question Generation (HyQA) và Structured Metadata Extraction.<br>- Tối ưu hóa chi phí API bằng giải pháp gom toàn bộ 4 tác vụ enrichment vào duy nhất 1 lần gọi prompt JSON (`_enrich_single_call`), giảm 75% số lượng LLM request và độ trễ so với việc gọi tuần tự 4 hàm riêng lẻ. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

Trong quá trình triển khai hệ thống Production RAG, tôi đã gặp và giải quyết 3 bài toán kỹ thuật phức tạp:

### 1. Lỗi Google Gemini OpenAI Endpoint: "Multiple candidates is not enabled for this model" (HTTP 400)
- **Exact Error:**
  ```text
  BadRequestError: Error code: 400 - [{'error': {'code': 400, 'message': 'Multiple candidates is not enabled for this model', 'status': 'INVALID_ARGUMENT'}}]
  ```
- **Nguyên nhân gốc rễ:**
  Mặc định thư viện RAGAS metric `answer_relevancy` khởi tạo thuộc tính `self.strictness = 3`. Thuộc tính này ra lệnh cho LangChain gọi `llm.generate([prompt], n=self.strictness)` để sinh đồng thời 3 câu hỏi đảo ngược phục vụ tính cosine similarity trung bình. Tuy nhiên, endpoint tương thích OpenAI của Google Gemini (`https://generativelanguage.googleapis.com/v1beta/openai/`) chưa hỗ trợ tham số $n > 1$ và lập tức trả về lỗi `HTTP 400 INVALID_ARGUMENT`.
- **Cách debug & xử lý:**
  Can thiệp trực tiếp vào metric trước khi đánh giá: gán `answer_relevancy.strictness = 1`. Khi đó Ragas chỉ sinh 1 câu hỏi tương đương với $n = 1$, hoàn toàn tương thích với Gemini API, loại bỏ triệt để lỗi 400.

### 2. Quota Rate Limiting & Concurrency Throttling (HTTP 429 RESOURCE_EXHAUSTED)
- **Exact Error:**
  ```text
  RateLimitError: Error code: 429 - Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 15 (RPM)
  ```
- **Nguyên nhân gốc rễ:**
  Mặc định RAGAS chạy hàm `evaluate()` với cấu hình `RunConfig(max_workers=16)`. Đối với gói Gemini Free Tier (giới hạn 15 Requests Per Minute), 16 worker song song lập tức làm cạn kiệt RPM quota chỉ sau vài giây đầu tiên, gây ra TimeoutError và đứt kết nối.
- **Cách debug & xử lý:**
  - Cấu hình lại `RunConfig(max_workers=2, timeout=120, max_retries=10, max_wait=60)`. Việc giới hạn tối đa 2 workers kết hợp độ trễ tự nhiên của LLM (~1.5s/request) giúp nhịp độ gọi API luôn duy trì dưới ngưỡng 15 RPM.
  - Tối ưu hóa Module 5 Enrichment: tích hợp bộ đệm nhớ (`dict cache`) và áp dụng hybrid enrichment (chỉ gọi LLM cho 10 chunk mẫu, các chunk còn lại sử dụng thuật toán trích xuất ngữ cảnh cục bộ dựa trên tiêu đề section và câu đầu), bảo toàn 95% quota API phục vụ khâu evaluation RAGAS.
  - Linh hoạt chuyển đổi model sang `gemini-3.1-flash-lite` khi một mã model đạt trần 500 requests/ngày.

### 3. Qdrant HTTP Read Timeout khi tải CPU cao
- **Exact Error:**
  ```text
  httpx.ReadTimeout: timed out -> qdrant_client.http.exceptions.ResponseHandlingException: timed out
  ```
- **Nguyên nhân gốc rễ:**
  Lớp `DenseSearch.__init__` khởi tạo kết nối QdrantClient với giá trị `timeout=2` giây. Khi máy tính thực hiện đồng thời tính toán vector `bge-m3` và rerank cross-encoder, tài nguyên CPU đạt đỉnh, khiến một số truy vấn HTTP nội bộ đến Docker Qdrant mất khoảng 2.1 - 2.5 giây, dẫn đến văng exception timeout.
- **Cách debug & xử lý:**
  Tăng tham số kết nối lên `timeout=60` trong `QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)` để đảm bảo khả năng chịu tải ổn định của database trong môi trường production.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

Dựa trên những kinh nghiệm thực chiến từ Lab 18, tôi xây dựng kế hoạch nâng cấp hệ thống RAG cho dự án cá nhân:

### Project: Trợ lý AI Hỏi - Đáp Tra Cứu Tài Liệu Kỹ Thuật & Quy Trình Doanh Nghiệp (Internal Enterprise Copilot)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Sử dụng LangChain RecursiveCharacterTextSplitter (chunk_size=500, overlap=50), lưu trữ ChromaDB và truy vấn Dense Retrieval thuần với OpenAI Embeddings text-embedding-3-small.
- **Vấn đề / Bottlenecks đang gặp:**
  - Precision thấp: Đối với các văn bản có cấu trúc phân tầng (Chương - Điều - Khoản), chunking thuần làm đứt đoạn ngữ cảnh khiến LLM trả lời thiếu hoặc sai chủ thể áp dụng.
  - Dễ nhầm lẫn phiên bản: Khi tài liệu cập nhật phiên bản mới (v1.0 lên v2.0), dense search thuần thường lấy cả 2 phiên bản và LLM bị ảo giác (hallucination) trộn lẫn quy định cũ và mới.
  - Chưa có cơ chế đánh giá tự động: Đang đánh giá thủ công bằng mắt (human eye-balling).

#### 2. Kế hoạch cải tiến áp dụng kiến trúc Lab 18
1. **Chunking Strategy:**
   - Chuyển sang **Hierarchical Chunking (Parent 2048 / Child 256)** kết hợp **Structure-Aware** cho các tài liệu markdown/PDF có tiêu đề điều khoản. Retrieve theo child nhưng inject parent vào prompt.
2. **Search Retrieval:**
   - Triển khai **Hybrid Search**: BM25 (với phân tích tiếng Việt `underthesea`) + Dense Search (`BAAI/bge-m3`).
   - Sử dụng **Reciprocal Rank Fusion (RRF)** ($k=60$) để dung hòa điểm số, đảm bảo tra cứu chính xác cả mã văn bản, số hiệu quyết định lẫn câu hỏi ngữ nghĩa tự nhiên.
3. **Reranking:**
   - Bắt buộc tích hợp **Cross-Encoder Reranker (`bge-reranker-v2-m3`)** lấy Top 3 từ Top 25 ứng viên hybrid. Giúp tăng mạnh Context Precision và loại bỏ hoàn toàn các văn bản trích dẫn sai phiên bản.
4. **Evaluation:**
   - Xây dựng benchmark test set 50 câu hỏi đa dạng (Factoid, Multi-hop, Negation, Temporal).
   - Thiết lập CI/CD pipeline tự động chấm điểm với **RAGAS 4 metrics**, đặt ngưỡng chặn (Quality Gate): Faithfulness $\ge 0.85$, Context Precision $\ge 0.90$.
5. **Enrichment & Temporal Filtering:**
   - Trích xuất metadata có cấu trúc: `doc_id`, `version`, `effective_date`, `status` (`active` / `superseded`).
   - Tự động lọc metadata trước khi search: loại bỏ ngay các văn bản mang trạng thái `superseded`.

#### 3. Timeline triển khai (2 tuần)
- **Tuần 1:**
  - Ngày 1-2: Refactor module Ingestion, cài đặt Qdrant Docker production và tích hợp Hierarchical Chunking + Structure-Aware parsing.
  - Ngày 3-5: Xây dựng BM25 tiếng Việt và tích hợp RRF Hybrid Search, kiểm thử độ nhạy từ khóa.
- **Tuần 2:**
  - Ngày 6-8: Tích hợp Cross-Encoder Reranker và xây dựng bộ metadata filter theo phiên bản tài liệu.
  - Ngày 9-10: Thiết lập luồng kiểm thử tự động RAGAS Evaluation, tối ưu hóa prompt generation và viết tài liệu hướng dẫn vận hành.
