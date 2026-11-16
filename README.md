# Day 7 — Agentic RAG

**Bài cá nhân · Python 3.11+ · tiếp nối Day 6**

Dùng embedding và vector store từ Day 6 để xây một agent biết tìm bằng chứng, trả lời có nguồn và nhận biết khi không đủ thông tin. Bạn tự chọn kiến trúc, số tool và cách tổ chức retrieval.

> **Tùy chọn:** [Gợi ý dùng AI để lập trình](AGENT_SETUP.md) giúp bạn làm lab; phần này không bắt buộc và không tính điểm.

## Bài cần làm

![Agent tìm tài liệu trong kho Day 6 và trả lời có nguồn](assets/retrieval-pipeline.svg)

1. Kết nối dữ liệu, embedding và vector store Day 6.
2. Xây agent dùng model thật và ít nhất một khả năng truy xuất dữ liệu. Agent có thể dùng nhiều tool và nhiều lượt gọi.
3. Thử **ít nhất 3 phương pháp retrieval khác nhau**, trong đó có dense retrieval dùng vector store Day 6. Tự chọn các phương pháp còn lại; xem [gợi ý](GUIDE.md#2-thử-các-phương-pháp-retrieval).
4. Thử nhiều cấu hình, benchmark, phân tích lỗi và chọn cấu hình tốt trên **dev**; sau đó đánh giá trên **test**.

Benchmark tối thiểu có **24 câu**: dev gồm 15 câu có đáp án và 2 câu thiếu bằng chứng; test gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. Xem [benchmark](BENCHMARK.md) và [rubric](RUBRIC.md).

## Thử thách nâng cao · +15 điểm

Index một tài liệu thực tế từ **100 trang** có bảng, ảnh và biểu đồ; xây agent truy xuất và trả lời các câu hỏi về nội dung đó. Hoàn thành [thử thách nâng cao](ADVANCED.md) và đạt **ít nhất 90% benchmark test** để nhận 15 điểm bonus. Phần cơ bản tối đa 85 điểm; toàn bài tối đa 100 điểm.

## Bắt đầu

1. Đọc [hướng dẫn](GUIDE.md), [rubric](RUBRIC.md) và [quy tắc](RULES.md).
2. Tạo repo cá nhân theo [cách nộp bài](SUBMISSION.md), chuẩn bị đầu ra Day 6 và model hỗ trợ tool calling.
3. Dùng hoặc thay thế [starter](starter/README.md); bạn cần kết nối model và vector store thật.
4. Chạy `python -m unittest discover -s tests -v`. Test mock chỉ kiểm tra code, không phải benchmark agent thật.

![Các bước thử nghiệm trên dev và đánh giá trên test](assets/experiment-loop.svg)

Nộp code, cách chạy, cấu hình, benchmark, trace và [báo cáo](report/individual/MSSV_HoTen.md). Thời gian ở [checkpoints](CHECKPOINTS.md) là gợi ý, có thể điều chỉnh theo lớp và phạm vi thử nghiệm.

---

## Bài làm — Hoàng Xuân Lĩnh (2A202604017)

Báo cáo: [report/individual/2A202604017_HoangXuanLinh.md](report/individual/2A202604017_HoangXuanLinh.md)

### Thành phần

| Đường dẫn | Nội dung |
|---|---|
| `src/export_day6.py` | Export index A (455 chunk) từ pgvector Day 6 sang `data/day7/chunks.json` |
| `src/build_benchmark.py` | Gộp câu hỏi Day 6 (+ sửa nhãn có ghi lý do trong `FIXES`) và câu thiếu bằng chứng → `data/day7/questions.json` |
| `src/embeddings.py` | Embedder Day 6 (`google/embeddinggemma-2`, 768 chiều), giữ nguyên để query tương thích index |
| `src/retrievers.py` | 3 phương pháp: Dense (pgvector), BM25, Hybrid RRF |
| `src/context.py` | Neighbor expansion (thêm chunk liền trước/sau) |
| `src/eval_retrieval.py` | Hit@1, Evidence Hit@3, Context Evidence Rate theo split |
| `src/llm.py`, `src/agent.py` | Client OpenAI + agent tool-calling (`search_book`, `answer`, `abstain`) |
| `src/run_agent_eval.py` | Chạy agent trên một split, lưu trace + mẫu nhãn |
| `data/day6/questions.json` | Câu hỏi Day 6 bản gốc (không sửa tay) |
| `data/day7/` | Chunk, câu thiếu bằng chứng, benchmark Day 7 (26 câu) |
| `results/retrieval/`, `results/agent/` | Summary, trace, nhãn chấm tay, metric từng cấu hình |
| `results/selection.json` | Tiêu chí và cấu hình đã chốt trước khi chạy test |
| `results/context_review_test.json` | Chấm context 0/1/2 trên test |

### Cách chạy lại

Yêu cầu: Python 3.11+, Docker, OpenAI API key. Trên Windows dùng `python -X utf8` để đọc/ghi UTF-8.

1. **Vector store Day 6.** Trong repo Day 6 (`K5-L3B-DAY06-HoangXuanLinh-2A202604017-DataIngestion/starter`):
   ```bash
   docker compose -p day6_student_starter up -d
   # nếu volume trống: python -m src.run ingest   (tạo lại index A, B; khoảng 5 phút trên CPU)
   ```
   Kiểm tra: `docker compose -p day6_student_starter exec postgres psql -U day6 -d day6_lab -c "SELECT strategy, count(*) FROM chunks GROUP BY strategy;"` phải ra A = 455, B = 354.
2. **Môi trường Day 7** (thư mục gốc repo này):
   ```bash
   python -m venv venv && source venv/Scripts/activate      # Linux/macOS: venv/bin/activate
   pip install -r requirements.txt
   cp starter/.env.example starter/.env                     # điền OPENAI_API_KEY
   python -X utf8 -m unittest discover -s tests -v           # 25 test, không cần DB/API
   ```
3. **Dữ liệu và benchmark** (đã có sẵn trong repo, chạy lại để kiểm chứng):
   ```bash
   python -X utf8 src/export_day6.py
   python -X utf8 src/build_benchmark.py
   ```
4. **Retrieval trên dev** (không tốn phí):
   ```bash
   python -X utf8 src/eval_retrieval.py --split dev --tag baseline
   python -X utf8 src/eval_retrieval.py --split dev --neighbors 1 --budget 2000 --tag n1-budget2000
   python -X utf8 src/eval_retrieval.py --split dev --neighbors 1 --budget 5500 --tag n2-budget5500
   ```
5. **Agent trên dev** (khoảng 0.03–0.05 USD mỗi cấu hình):
   ```bash
   python -X utf8 src/run_agent_eval.py --split dev --method dense      --neighbors 0 --max-searches 3 --tag dense-n0-s3
   python -X utf8 src/run_agent_eval.py --split dev --method hybrid_rrf --neighbors 1 --max-searches 3 --tag hybrid-n1-s3
   python -X utf8 src/run_agent_eval.py --split dev --method hybrid_rrf --neighbors 1 --max-searches 1 --tag hybrid-n1-s1
   ```
6. **Test** (chỉ chạy được khi đã có `results/selection.json`):
   ```bash
   python -X utf8 src/eval_retrieval.py --split test --methods dense --tag baseline
   python -X utf8 src/eval_retrieval.py --split test --methods hybrid_rrf --neighbors 1 --budget 2000 --tag selected
   python -X utf8 src/run_agent_eval.py --split test --method dense      --neighbors 0 --max-searches 3 --tag dense-n0-s3
   python -X utf8 src/run_agent_eval.py --split test --method hybrid_rrf --neighbors 1 --max-searches 1 --tag hybrid-n1-s1
   ```
7. **Tổng hợp nhãn chấm tay** (`reviews.json` trong từng thư mục kết quả):
   ```bash
   d=results/agent/test-hybrid-n1-s1
   python -X utf8 starter/agent_metrics.py --cases $d/cases.json --traces $d/traces.json --reviews $d/reviews.json --split test --output $d/summary_metrics.json
   ```

Chạy lại agent sẽ ghi đè trace và cho câu trả lời khác đôi chút (LLM không tất định); nhãn trong `reviews.json` ứng với trace đã lưu trong repo.

Dữ liệu: *Principles of Data Science* (OpenStax, Rice University, 2025), giấy phép CC BY-NC-SA 4.0; mỗi chunk giữ dòng attribution trong `metadata`.

### Phần nâng cao (`advanced/`)

Index toàn bộ sách (569 trang) bằng text, ảnh trang và vector interleaved (`google/embeddinggemma-2`). Agent gửi ảnh gốc của trang cho gpt-5.4-mini. Kết quả test là **24/30 (80%)**, chưa đạt ngưỡng 90%. Chi tiết nằm trong báo cáo, mục "Thử thách nâng cao".

1. Tải PDF về `advanced/data/raw/Principles-of-Data-Science-WEB.pdf` (bản web của OpenStax). Cài thư viện: `pip install -r requirements.txt` (có thêm `pymupdf`).
2. Lọc trang có hình, rồi xây index. Bước này chạy trên CPU, không tốn API, mất khoảng 80 phút:
   ```bash
   python -X utf8 advanced/profile_pdf.py
   python -X utf8 advanced/build_index.py --kinds text image interleaved
   ```
3. Kiểm tra benchmark:
   ```bash
   python -X utf8 advanced/benchmark/check_benchmark.py
   ```
4. Đánh giá retrieval theo trang (không tốn phí). Lệnh chạy test chỉ chạy được khi đã có `advanced/results/selection.json`:
   ```bash
   python -X utf8 advanced/eval_retrieval_adv.py --split dev
   ```
5. Chạy agent. `--max-usd` là trần của tổng sổ chi phí `advanced/data/api_costs.jsonl`; script dừng khi chạm trần:
   ```bash
   python -X utf8 advanced/agent_adv.py --split dev  --method hybrid_inter --top-k 5 --tag hinter-k5-v4 --max-usd 1.7
   python -X utf8 advanced/agent_adv.py --split test --method hybrid_inter --top-k 5 --tag hinter-k5-v4 --max-usd 2.4
   ```
6. Tổng hợp nhãn chấm tay (`reviews.json`):
   ```bash
   python -X utf8 advanced/grade_summary.py dev-hinter-k5-v4 test-hinter-k5-v4
   ```

Lưu ý:
- File `.npy`, ảnh render của các trang và PDF không được commit, vì tạo lại được bằng các bước trên.
- Chạy lại agent sẽ ghi đè trace đã chấm.
