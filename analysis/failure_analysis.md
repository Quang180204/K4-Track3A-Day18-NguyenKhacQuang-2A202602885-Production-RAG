# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Khắc Quang  
**Mã số học viên:** 2A202602885  
**Khóa:** K4 - Track 3A  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ | Đánh giá |
|--------|---------------|------------|---|----------|
| **Faithfulness** | 0.9815 | 0.8750 | -0.1065 | Đạt ngưỡng chuẩn (>0.75). Câu trả lời bám sát context retrieved. |
| **Answer Relevancy** | 0.7552 | 0.6337 | -0.1216 | Câu trả lời cô đọng, tuy nhiên do prompt strictness và context ngắn khiến similarity của generated question thấp hơn baseline. |
| **Context Precision** | 0.8500 | **0.9394** | **+0.0894** | **Tăng vượt bậc (+8.94%)** nhờ kết hợp Hybrid Search (BM25 + Dense) và Cross-Encoder Reranking (`bge-reranker-v2-m3`). Các chunk nhiễu bị loại bỏ ở top rank. |
| **Context Recall** | 0.8250 | 0.7750 | -0.0500 | Ổn định quanh mức 0.78-0.82. Việc giới hạn RERANK_TOP_K = 3 đôi khi cắt bỏ 1 chunk phụ chứa thông tin chi tiết. |

---

## Bottom-5 Failures

Dưới đây là 5 câu hỏi có điểm số RAGAS cần cải thiện được phân tích theo Diagnostic Tree:

### #1
- **Question:** Bảo hiểm sức khỏe PVI có hạn mức bao nhiêu cho nhân viên?
- **Expected:** Hạn mức bảo hiểm sức khỏe PVI cho nhân viên là 200.000.000 VNĐ/năm, bao gồm nội trú, ngoại trú và nha khoa.
- **Got:** Hạn mức bảo hiểm sức khỏe PVI là 200.000.000 VNĐ/năm cho nhân viên chính thức (gồm nội trú, ngoại trú, nha khoa).
- **Worst metric:** Faithfulness (0.0 do network timeout trong quá trình eval của evaluator LLM)
- **Error Tree:** Output đúng nội dung thực tế → Context trích xuất đầy đủ → Query retrieve chính xác → Đứt đoạn tại bước Evaluator LLM API call gặp timeout/rate limit.
- **Root cause:** Trong quá trình RAGAS gọi async parallel nhiều metrics cùng lúc, một số job evaluator của LLM gặp connection timeout/rate limit dẫn đến metric fallback trả về 0.0.
- **Suggested fix:** Cấu hình `RunConfig(max_workers=2, timeout=120, max_retries=10)` với exponential backoff và lưu checkpoint cache cho từng câu hỏi eval.

---

### #2
- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** Theo chính sách hiện hành (v2024), nhân viên được nghỉ 15 ngày phép năm có lương. Chính sách cũ (v2023) là 12 ngày nhưng đã bị thay thế.
- **Got:** Nhân viên được nghỉ 15 ngày phép năm theo chính sách năm 2024 (chính sách cũ 2023 là 12 ngày).
- **Worst metric:** Answer Relevancy (0.0)
- **Error Tree:** Output trả lời đúng sự thật → Context chứa cả v2023 và v2024 → Reranker đẩy v2024 lên đầu → Evaluator question-generation đảo ngữ tạo câu hỏi khác cấu trúc query.
- **Root cause:** Cả hai phiên bản quy chế `nghi_phep_nam_v2023.md` (12 ngày) và `nghi_phep_nam_v2024.md` (15 ngày) đều tồn tại trong corpus. Dù reranker đã đẩy v2024 lên vị trí số 1, việc xuất hiện cả hai con số 12 và 15 khiến câu trả lời cần giải thích ngữ cảnh phiên bản thay thế, làm loãng độ tương đồng cosine khi RAGAS sinh câu hỏi kiểm tra relevancy.
- **Suggested fix:** Áp dụng Temporal Metadata Filtering trong Module 5: tự động trích xuất `effective_year` / `is_active` và ưu tiên metadata `status: active` / `version: latest` trước khi rerank.

---

### #3
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Nhân viên Senior có 9 năm thâm niên được nghỉ 18 ngày phép năm. Mức lương thuộc bậc Senior từ 20 đến 35 triệu VNĐ/tháng.
- **Worst metric:** Context Recall (0.0 do multi-hop retrieval phân mảnh)
- **Error Tree:** Câu hỏi dạng Multi-hop (Phép năm thâm niên + Thang bảng lương) → Retrieval Dense/BM25 chỉ tập trung vào tài liệu có nhiều từ khóa trùng lặp nhất (chính sách phép năm) → Thiếu chunk quy chế lương P3-P4 trong Top 3.
- **Root cause:** Câu hỏi đa ý (multi-hop) đòi hỏi 2 tài liệu hoàn toàn độc lập (`nghi_phep_nam_v2024.md` và `bang_luong_senior.md`). Khi `RERANK_TOP_K = 3`, các chunk về phép năm và thâm niên chiếm trọn top 3 điểm rerank cao nhất, đẩy chunk về thang bảng lương xuống hạng 4-5.
- **Suggested fix:** Sử dụng Query Decomposition (chia sub-queries) trong RAG pipeline: tách thành (1) "Senior 9 năm thâm niên được bao nhiêu ngày phép?" và (2) "Thang lương bậc Senior là bao nhiêu?", sau đó truy vấn song song và merge contexts.

---

### #4
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Laptop 30 triệu nằm trong khoảng 5-50 triệu nên cần Giám đốc phòng ban (Director) phê duyệt. Ngoài ra, mua sắm thiết bị CNTT cần có xác nhận cấu hình kỹ thuật từ phòng CNTT trước khi đề xuất. Cần đính kèm ít nhất 3 báo giá vì trên 10 triệu.
- **Got:** Cần Giám đốc phòng ban phê duyệt và xác nhận cấu hình kỹ thuật từ phòng CNTT.
- **Worst metric:** Context Recall (0.5)
- **Error Tree:** Retrieval lấy được quy trình mua sắm CNTT → Bỏ sót điều kiện về số lượng báo giá (3 báo giá) ở phần phụ lục tài liệu mua sắm.
- **Root cause:** Chunking size nhỏ (256 tokens child chunk) tách rời bảng phân quyền phê duyệt khỏi điều khoản phụ về số lượng báo giá tối thiểu (3 báo giá) nằm ở phần cuối văn bản.
- **Suggested fix:** Kỹ thuật Hierarchical Chunking: Khi retrieve được child chunk về laptop CNTT, trả về parent chunk (2048 tokens) chứa toàn bộ điều kiện đính kèm báo giá và thẩm quyền phê duyệt.

---

### #5
- **Question:** Thông tin lương thuộc cấp độ phân loại dữ liệu nào?
- **Expected:** Theo quy chế chi trả lương, thông tin lương được phân loại là dữ liệu Bí mật, cấm chia sẻ với đồng nghiệp. Theo chính sách phân loại dữ liệu, dữ liệu Bí mật (cấp 3) phải mã hóa khi truyền và hạn chế truy cập theo need-to-know.
- **Got:** Thông tin lương được phân loại là Dữ liệu Bí mật (Cấp độ 3), tuyệt đối không được tiết lộ cho đồng nghiệp và phải mã hóa khi lưu trữ/truyền tải.
- **Worst metric:** Faithfulness (0.0) / Context Recall (0.5)
- **Error Tree:** Trả lời chính xác → Context trích xuất được file quy chế bảo mật → Đánh giá faithfulness bị ảnh hưởng do evaluator LLM so khớp chuỗi định dạng (Cấp độ 3 vs cấp 3).
- **Root cause:** Định dạng văn phong của câu trả lời có thêm từ ngữ chuẩn hóa ("Dữ liệu Bí mật (Cấp độ 3)"), trong khi context chỉ ghi "cấp 3 - Bí mật".
- **Suggested fix:** Điều chỉnh prompt generation thêm chỉ thị: "Chỉ trích xuất nguyên văn thuật ngữ phân loại có trong tài liệu, không tự ý suy diễn hoặc diễn đạt lại".

---

## Case Study (cho presentation)

**Question chọn phân tích:**  
> *"Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"*

### Error Tree walkthrough:
1. **Output đúng?**  
   - Output trả về: 18 ngày phép (15 ngày cơ bản + 3 ngày thâm niên) và dải lương 20 - 35 triệu VNĐ/tháng. Về mặt thông tin thực tế là chính xác hoàn toàn.
2. **Context đúng & đầy đủ?**  
   - Chưa tối ưu. Do `RERANK_TOP_K = 3`, top 3 kết quả bị chiếm bởi các chunk thuộc `nghi_phep_nam_v2024.md` (giải thích chi tiết cách tính 1 ngày cho mỗi 3 năm thâm niên). Chunk chứa thang bảng lương Senior nằm ở rank 4 của Cross-Encoder.
3. **Query rewrite / Decomposition OK?**  
   - Pipeline hiện tại chỉ đưa câu hỏi phức hợp gốc vào hybrid search. BM25 và Dense Search bị "thiên lệch" (bias) vào các từ khóa "nhân viên Senior", "thâm niên", "ngày phép năm", làm giảm điểm số của chunk thuần về thang bảng lương.
4. **Fix ở bước:**  
   - **Bước Query Pre-processing (M2/Orchestrator):** Triển khai Sub-question Query Decomposition.
   - **Bước Chunk Retrieval (M1/M5):** Tăng `RERANK_TOP_K` từ 3 lên 5 đối với các câu hỏi phát hiện có liên từ "và" / "đồng thời".

---

## Nếu có thêm 1 giờ, sẽ optimize:

1. **Sub-question Query Decomposition:** Xử lý tự động các câu hỏi phức multi-hop thành các truy vấn đơn lẻ trước khi đưa vào Hybrid Search.
2. **Metadata Filtering theo phiên bản hiệu lực:** Loại trừ hoàn toàn các tài liệu cũ (`superseded`, ví dụ `v2023` khi đã có `v2024`) ngay từ giai đoạn lọc metadata của Qdrant.
3. **Adaptive K cho Reranking:** Tự động mở rộng `top_k` (từ 3 lên 5) nếu độ tin cậy điểm số của candidate thứ 3 và thứ 4 chênh lệch dưới ngưỡng $\epsilon = 0.05$.
4. **Local Evaluation Cache:** Lưu trữ kết quả đánh giá RAGAS theo hàm băm `(question, answer, context, model)` để không phải gọi lại API khi re-run pipeline.

---

## Latency Breakdown Report (Bảng thời gian từng bước)

Báo cáo phân rã độ trễ (latency breakdown) chi tiết của toàn bộ Production RAG Pipeline (lưu tại `reports/latency_report.json`):

| Giai đoạn (Stage) | Thời gian (s) | Tỷ lệ (%) | Chi tiết thực thi |
|---|:---:|:---:|---|
| **1. Chunking (M1)** | `0.2s` | 0.01% | Phân tích 26 file markdown/PDF thành 105 child chunks (256 chars) và parent chunks (2048 chars). |
| **2. Enrichment (M5)** | `91.5s` | 3.85% | Áp dụng `_enrich_single_call` kết hợp hybrid memory cache cho 105 chunks. |
| **3. Hybrid Indexing (M2)** | `15.3s` | 0.64% | Sinh vector dense `BAAI/bge-m3` (1024-dim) và tokenize BM25 bằng `underthesea` đẩy vào Qdrant. |
| **4. Reranker Loading (M3)** | `2.1s` | 0.09% | Load model `BAAI/bge-reranker-v2-m3` vào RAM với class-level singleton cache. |
| **5. Query Inference (20 câu)** | `18.4s` | 0.77% | **Trung bình 920ms/câu**: Hybrid search (~85ms) + Cross-Encoder rerank (~185ms) + Gemini answer gen (~650ms). |
| **6. RAGAS Evaluation (M4)** | `2251.2s` | 94.64% | Chấm điểm 80 lượt đánh giá (4 metrics × 20 câu) với `RunConfig(max_workers=2)` đảm bảo không vượt 15 RPM. |
| **Tổng cộng (End-to-End)** | **`2378.7s`** | **100%** | Toàn bộ pipeline sẵn sàng cho môi trường production thực tế. |

