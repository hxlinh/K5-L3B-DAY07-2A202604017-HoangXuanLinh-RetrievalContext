# Báo cáo Day 7 — Hoàng Xuân Lĩnh (2A202604017)

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - Dùng lại corpus Day 6: sách OpenStax *Principles of Data Science* (Rice University, 2025), giấy phép CC BY-NC-SA 4.0.
  - Gồm chương 2 (trang PDF 67–103) và chương 8 (trang 387–413), tổng cộng 63 trang nội dung.
  - Index dùng ở Day 7 là **index A** của Day 6, cấu hình đã được chọn ở Day 6: cắt cố định 500 ký tự, chồng lấn 50, cắt theo trang. Index có **455 chunk**: 261 chunk ở `pods-ch02`, 194 chunk ở `pods-ch08`.
  - Chunk được export trực tiếp từ bảng `chunks` trong Postgres ([src/export_day6.py](../../src/export_day6.py)), không lấy từ file `.jsonl`. Nhờ vậy dense và BM25 chắc chắn dùng cùng một phiên bản index. `DenseRetriever` cũng báo lỗi nếu DB và `chunks.json` bị lệch nhau.
  - Mọi chunk đều giữ dòng attribution trong `metadata`.
- **Vector store, embedding model, phiên bản và số chiều:**
  - Store là PostgreSQL 16 + pgvector 0.8.7, chạy bằng Docker, tìm chính xác theo cosine (`<=>`), không dùng ANN.
  - Embedding là `google/embeddinggemma-2` chạy local bằng sentence-transformers 6.3.0, 768 chiều, vector chuẩn hóa L2.
  - Query dùng prompt `QuestionAnswering`, giống hệt Day 6. File [src/embeddings.py](../../src/embeddings.py) được giữ nguyên từ Day 6 để query embedding tương thích với index.
  - Mình không tạo index mới, nên chi phí embedding ở Day 7 bằng 0.
  - DB chạy với `docker compose -p day6_student_starter`. Lúc đầu mình chạy không có `-p`, Compose tạo ra một volume trống mới (`starter_pg_data`) vì tên project mặc định lấy theo tên thư mục. Dữ liệu cũ vẫn nằm nguyên trong volume `day6_student_starter_pg_data`.
- **Agent model, cách chạy và phân chia dev/test:**
  - Agent dùng `gpt-5.4-mini` (OpenAI Chat Completions, tool calling), gọi bằng `urllib`, không dùng SDK.
  - Giá tính theo trang model của OpenAI, tra ngày 16/11/2026: 0.75 USD / 1 triệu input token, 4.50 USD / 1 triệu output token.
  - Benchmark gồm **26 câu**: dev có 16 câu có đáp án và 2 câu thiếu bằng chứng; test có 6 câu có đáp án và 2 câu thiếu bằng chứng.
  - Lệnh chạy xem ở [README](../../README.md#bài-làm--hoàng-xuân-linh-2a202604017).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - **Câu có đáp án:** 22 câu lấy từ Day 6, giữ nguyên split. Bản gốc nằm ở `data/day6/questions.json`. Mọi chỗ sửa nằm trong `FIXES` của [src/build_benchmark.py](../../src/build_benchmark.py), và script sẽ dừng nếu có phần evidence nào không nằm trọn trong một chunk đúng nguồn và đúng trang.
  - **Định nghĩa hit giống Day 6:** đúng `source_id`, đúng `page`, và evidence là chuỗi con của text chunk sau khi gộp khoảng trắng. Bước gộp khoảng trắng là cần thiết: nếu so nguyên văn, 21/22 evidence không khớp, vì text PDF giữ dấu xuống dòng còn evidence thì dùng dấu cách.
  - **Năm câu đã sửa nhãn.** Bốn câu sửa trước mọi thí nghiệm Day 7, chỉ dựa vào ranh giới chunk. Câu q13 sửa trong lúc khám phá retrieval trên dev:
    - q02, q22: evidence vắt qua ranh giới chunk A, nên mình rút về đoạn cốt lõi đủ để trả lời. Với q22, mình cũng cắt `expected_answer` cho khớp với evidence.
    - q09, q16: câu hỏi cần 2 ý nằm ở 2 chunk khác nhau, nên mình chuyển sang `evidence_mode: "all"`, nghĩa là phải có đủ cả 2 phần mới tính là hit. Hệ quả là Hit@1 của 2 câu này luôn sai, nên trần Hit@1 trên dev chỉ là 14/16.
    - q13: thêm evidence thay thế ở trang 401 (*"Pseudonymization of data involves masking, encoding, or encrypting…"*). Mình phát hiện đoạn này khi Dense trả trang 401 ở hạng 1. Việc sửa không làm đổi con số nào, vì với câu hỏi thật của q13, cả dense và hybrid đều trả p390 ở hạng 1.
  - **Câu thiếu bằng chứng** (u01–u04, [data/day7/unanswerable.json](../../data/day7/unanswerable.json)). Mỗi câu gắn với một trang cùng chủ đề và hỏi một chi tiết có thật ngoài đời nhưng sách không ghi: tiền dàn xếp của Equifax, tổ chức điều tra và công ty phát triển COMPAS, mức phạt tối đa của GDPR, tiền phạt Facebook năm 2019 (câu tiếng Việt). Mình kiểm tra sự vắng mặt bằng cách tìm từ khóa trên cả 64 trang gốc và cả 455 chunk, chứ không chỉ trong top-k. Kết quả được ghi trong trường `absence_check`.
  - **Bộ test của Day 6.** Ở Day 6, bộ này được chạy đúng 1 lần sau khi chốt cấu hình, không dùng để chỉnh hệ thống. Mình giữ lại 6 câu đó, thêm 2 câu thiếu bằng chứng, và không dùng chúng để chỉnh gì ở Day 7 (code chặn chạy test nếu chưa có `results/selection.json`). Tuy vậy, mình đã đọc phân tích lỗi của q21 trong báo cáo Day 6, nên bộ test này không hoàn toàn "chưa từng thấy".
  - **Chấm câu trả lời:** 2 nghĩa là đúng và đủ mọi ý, mọi khẳng định đều có trong chunk được cite; 1 là đúng một phần, hoặc thiếu một ý; 0 là sai, bịa, hoặc từ chối một câu có đáp án. Citation được tính là hợp lệ khi chunk đó có trong context **và** nội dung chunk thật sự chứa điều mà câu trả lời gắn với nó. Từ chối được tính là đúng khi agent đã search trước, và câu trả lời không có khẳng định nào ngoài tài liệu.

## Retrieval và thử nghiệm

Ba phương pháp được so sánh trên cùng 455 chunk và cùng bộ nhãn dev, chỉ dùng câu hỏi gốc. Filter `chapter` chỉ được dùng khi câu hỏi có trường `filter` (thông tin người dùng cung cấp trước), gồm q09, q15, q18, q19. Context được đóng gói bằng `pack_context` của starter, budget tính theo ký tự và đã bao gồm header nguồn.

- **Dense:** embed query, rồi tìm cosine trên pgvector. Filter chạy trong SQL.
- **BM25:** tự cài BM25 Okapi với `k1 = 1.5`, `b = 0.75`. Tách từ bằng `\w+` sau khi chuẩn hóa NFC và casefold. Filter chạy trong Python.
- **Hybrid RRF:** lấy 20 candidate từ mỗi phương pháp, tính `Σ 1/(60 + rank)`. Mình gộp theo thứ hạng để không phải cộng hai loại điểm khác thang đo (cosine 0–1 và BM25 không có giới hạn trên).

Budget mặc định của starter là 800 ký tự, mình nâng lên 2000. Mỗi chunk A kèm header dài khoảng 590 ký tự, nên với 800 ký tự context chỉ chứa được 1 chunk, còn 2000 ký tự thì vừa đủ cho top-3.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | Dense, top-3, filter người dùng, không rerank, 2000 ký tự | Phương pháp bắt buộc trên vector store Day 6 | dev | 10/16 (62.5%) | 15/16 (93.8%) | 15/16 (93.8%) | trung vị 156 ms/query (gồm embed query trên CPU); 0 USD |
| Phương pháp 2 | BM25 (k1=1.5, b=0.75), top-3, 2000 ký tự | Bắt tốt thuật ngữ hiếm, yếu với paraphrase và câu tiếng Việt | dev | 12/16 (75.0%) | 13/16 (81.3%) | 13/16 (81.3%) | 1.5 ms; 0 USD |
| Phương pháp 3 | Hybrid RRF (k=60, 20 candidate mỗi bên), top-3, 2000 ký tự | Kết hợp ưu điểm của dense và BM25 | dev | 13/16 (81.3%) | 14/16 (87.5%) | 14/16 (87.5%) | 169 ms; 0 USD |
| Cải thiện 1 (N1) | Hybrid + neighbors ±1, top-3, 2000 ký tự | Lỗi q16 do chunk cắt ngang ý: kéo thêm chunk liền kề thì Context tăng, Hit@k không đổi | dev | 13/16 | 14/16 | **16/16 (100%)** | 157 ms; context trung bình 1736 ký tự |
| Cải thiện 1b (N2) | Hybrid + neighbors ±1, top-3, 5500 ký tự | Tăng budget để không chunk top-3 nào bị đẩy ra | dev | 13/16 | 14/16 | 16/16 | 161 ms; context trung bình **4185** ký tự (gấp 2.4 lần N1) |
| Cải thiện 2 | Agent hybrid + N1, `max_searches` 1 thay vì 3 | Phần lớn lượt search thêm là thừa, nên giảm số lượt sẽ giảm chi phí mà không mất chất lượng | dev | 13/16 | 14/16 | 16/16 (context cuối của agent) | 0.0263 USD so với 0.0446 USD; 2.4 s so với 5.5 s mỗi câu |
| Cấu hình đã chọn | Hybrid + N1 + 2000 ký tự/lượt, agent `max_searches=1` | — | test | **6/6** | 6/6 | 6/6 (query gốc); **5/6** (context cuối của agent) | 156 ms retrieval; agent 0.0113 USD, 2.2 s/câu |
| Baseline | Dense, không neighbor, 2000 ký tự/lượt, agent `max_searches=3` | — | test | 4/6 | 6/6 | 6/6 (query gốc); 6/6 (agent) | 157 ms retrieval; agent 0.0222 USD, 5.5 s/câu |

Ngoài ra, neighbor ±1 với budget 2000 cũng nâng Context của Dense lên 16/16 và của BM25 lên 15/16. Hit@k không đổi ở mọi cấu hình neighbor, vì neighbor expansion chỉ thay đổi bước chọn context, không đụng tới thứ hạng của retriever.

**Tiêu chí chọn**, chốt ngày 16/11/2026, trước khi thử neighbor và `max_searches`: (1) Context Evidence Rate trên dev, (2) Hit@1 trên dev, (3) thời gian và chi phí. Lý do: LLM chỉ đọc được những gì có trong context, còn Hit@1 chỉ thật sự quan trọng khi budget nhỏ. Chi tiết nằm ở [results/selection.json](../../results/selection.json).

**Hai lỗi dev và cách thử:**

1. **q16 trượt ở cả 3 phương pháp.** Cả ba đều đưa `p399:005` lên hạng 1. Chunk này có "COMPAS", "investigation in 2016" và phần 1 của đáp án ("biased against Black defendants"). Phần 2 ("flagged… nearly twice the rate") lại nằm ở `p399:006`, và `006` không chứa từ khóa nào của câu hỏi. BM25 không có từ để khớp, còn với dense thì vector của `006` không giống câu hỏi. Đây là lỗi của cách cắt chunk cố định, không phải lỗi của retriever. Mình thử neighbor expansion: với N1, `006` được kéo vào context ở cả 3 phương pháp, và Context đổi từ trượt sang trúng. Hit@3 vẫn trượt, đúng như dự đoán.
2. **q09 với Hybrid.** Dense xếp `p87:007` hạng 1 và `p87:006` hạng 2, nên trúng. Nhưng `006` không nằm trong top-20 của BM25, nên chỉ nhận được 1/62 ≈ 0.016 điểm RRF. Trong khi đó `p93:001`, chỉ đứng thứ 5 ở dense và thứ 7 ở BM25, lại được 1/65 + 1/67 ≈ 0.030. RRF thưởng cho sự đồng thuận giữa hai retriever, nên một chunk chỉ một bên thấy, dù xếp rất cao, vẫn bị loại. N1 sửa được trường hợp này vì `006` là chunk liền trước của hit hạng 1 `007`.

Thêm một quan sát: BM25 trúng q05 dù câu hỏi viết bằng tiếng Việt, vì câu hỏi có cụm *"(stratified sampling)"* trong ngoặc. Khi mình thử "Lấy mẫu phân tầng là gì?" không có phần tiếng Anh, BM25 trả về rỗng và Hybrid khi đó chỉ còn là Dense.

**Lý do chọn cấu hình.** Theo tiêu chí, Hybrid + N1 hòa với Dense + N1 ở Context (16/16), thắng ở Hit@1 (13 so với 10), và thắng N2 ở chi phí với cùng chất lượng. Ở tầng agent, `s1` và `s3` hòa nhau ở Context (16/16) và Hit@1, nên `s1` thắng nhờ rẻ hơn 41% và nhanh hơn khoảng 2.3 lần. Rủi ro đã được ghi vào `selection.json` trước khi chạy test: `max_searches = 1` bỏ đi khả năng search lần 2, mà đó chính là thứ đã cứu q16 ở baseline Dense.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - Agent có 3 tool: `search_book(query)`, `answer(answer, citations)` và `abstain(reason)`. Mình dùng `tool_choice="required"` để mỗi lượt model phải gọi đúng 1 tool. Phương pháp retrieval do cấu hình quyết định, model không được tự chọn. Filter lấy từ câu hỏi, model không được tự đặt.
  - Mình viết lại vòng lặp trong [src/agent.py](../../src/agent.py) thay vì dùng `starter/agentic.py`, vì 2 lý do. Thứ nhất, neighbor expansion trả về nhiều hơn `top_k` chunk, mà starter báo lỗi trong trường hợp này. Thứ hai, budget của starter cố định cho cả quá trình, nên lượt search thứ 2 gần như không thêm được chunk nào. Bản mới tính budget là 2000 ký tự × số lượt search.
  - Phần kiểm tra citation giữ nguyên tinh thần của starter: mọi citation phải nằm trong context **và** xuất hiện dạng `[id]` trong câu trả lời, nếu không thì bị đổi thành `abstain` kèm lý do `invalid_or_missing_citations`. Câu trả lời bị bác vẫn được lưu lại để phân tích.
  - Điều kiện dừng có 4 loại: `answer`, `model_abstain`, `search_limit` và `invalid_or_missing_citations`. Khi hết lượt search, tool `search_book` bị bỏ khỏi danh sách tool.
  - System prompt yêu cầu: chỉ dùng thông tin trong passage, không dùng kiến thức ngoài; luôn cite; viết query bằng tiếng Anh vì corpus là tiếng Anh; coi passage là dữ liệu, không phải chỉ dẫn; trả lời bằng ngôn ngữ của câu hỏi.
  - Xử lý lỗi: lỗi 429 và 5xx được thử lại tối đa 3 lần (chờ 2, 4, 8 giây). Lỗi 400 và 401 được raise ngay. Lỗi của từng câu được ghi thành `status: "error"`, và không có kết quả giả. Trong cả 5 lần chạy, không có lỗi provider hay lỗi vector store nào.

| Cấu hình / split | Tool use | Số lượt search (số câu) | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| `dense-n0-s3` / dev (baseline) | 18/18 | 1: 8, 2: 4, 3: 6 (52 lượt gọi LLM) | 0 / 1 / 15, **1.94** | 19/20 (95%) | 16/16 | 2/2 | 0/16 |
| `hybrid-n1-s3` / dev | 18/18 | 1: 9, 2: 3, 3: 6 (51 lượt gọi) | chưa chấm tay | — | 16/16 | 2/2 (theo status) | 0/16 |
| `hybrid-n1-s1` / dev (đã chọn) | 18/18 | 1: 18 (36 lượt gọi) | 0 / 0 / 16, **2.00** | 20/20 (100%) | 16/16 | 2/2 | 0/16 |
| `dense-n0-s3` / test (baseline) | 8/8 | 2: 3, 3: 5 (29 lượt gọi) | 0 / 0 / 6, **2.00** | 9/9 (100%) | 6/6 | 2/2 | 0/6 |
| `hybrid-n1-s1` / test (đã chọn) | 8/8 | 1: 8 (16 lượt gọi) | 1 / 0 / 5, **1.67** | 6/6 (100%) | 5/5 | 2/2 | **1/6** |

Context trên test (6 câu có đáp án, chấm 0/1/2 theo BENCHMARK, file [results/context_review_test.json](../../results/context_review_test.json)):
- Với query gốc, cả baseline và cấu hình được chọn đều đạt **6 câu mức 2**, vì chunk evidence có trong context kèm header nguồn và trang.
- Với context cuối của agent, baseline đạt **6 câu mức 2**, còn cấu hình được chọn đạt **5 câu mức 2 và 1 câu mức 0** (q19).

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q16 (dev, baseline) | Có đáp án, 2 phần evidence | Lượt 1: `"2016 investigation COMPAS algorithm what did it find"`, nhận `p399:005`. Lượt 2: `"… found biased against Black defendants what analysis revealed"`, nhận `p399:005` và `p399:006` | "biased against Black defendants… flagged as high risk at nearly twice the rate…" [p399:005][p399:006] | Grounded 2, citation 2/2. Query lượt 2 dùng cụm từ lấy từ chunk vừa đọc, không lấy từ gold. Đây là multi-hop hợp lệ. |
| q16 (dev, đã chọn) | Có đáp án | 1 lượt, nhận `p399:005`, kèm chunk lân cận `p399:006` | Đủ cả 2 ý [p399:005][p399:006] | Grounded 2. Đủ evidence chỉ trong 1 lượt nhờ neighbor expansion. |
| q05 (dev, baseline) | Câu tiếng Việt | `"stratified sampling definition"`, nhận `p76:001`, `p73:001`, `p76:002`, trượt `p73:000` | "chia tổng thể thành các nhóm con… chọn ngẫu nhiên từ từng nhóm" [p76:002][p73:001] | Grounded 1, vì thiếu ý "theo tỷ lệ kích thước của tầng". Query do agent viết lại đã làm trượt chunk chứa câu định nghĩa, trong khi câu hỏi gốc thì trúng. |
| q04 (dev, baseline) | Có đáp án | 1 lượt, nhận `p72:001`, `p72:002` | Đúng ý mailing list và thiên về người giàu [p72:001][p72:002] | Grounded 2, nhưng citation chỉ hợp lệ 1/2. `p72:002` nói về kết quả Landon/Roosevelt, không hỗ trợ khẳng định về cách chọn mẫu. ID có trong context nên code vẫn cho qua. |
| u02 (dev, cả hai cấu hình) | Thiếu bằng chứng | Baseline search 3 lượt với query gần như giống nhau, cấu hình được chọn search 1 lượt | Từ chối: "không nêu tổ chức điều tra hay công ty phát triển" | Từ chối đúng. Không nhắc ProPublica hay Northpointe, dù model nhiều khả năng biết đáp án ngoài đời. |
| q19 (test, đã chọn) | Có đáp án, có filter chương 8 | 1 lượt `"unrepresentative training sample algorithm unfair … bias"`, nhận `p399:009` (hạng 1), `p400:000` (hạng 2), `p399:003`. Context cuối: `p399:008–010` | Từ chối nhầm | Grounded 0. Xem phân tích bên dưới. |
| u04 (test, đã chọn) | Thiếu bằng chứng, tiếng Việt | `"Facebook fined how much after the 2019 user data leak"`, nhận `p394:003`, `p394:004`, … | Từ chối, nêu rõ tài liệu chỉ nhắc vụ lộ dữ liệu của 500 triệu user | Từ chối đúng. Không đưa số tiền phạt nào. |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - Trace retrieval nằm ở `results/retrieval/<split>-<tag>/traces.json`. Trace agent nằm ở `results/agent/<split>-<tag>/traces.json`, gồm từng tool call, query, kết quả, chunk lân cận, context cuối, các lượt gọi model (token, độ trễ, số lần thử lại), câu trả lời, citation và lý do dừng.
  - Nhãn chấm tay nằm trong `reviews.json`, metric trong `summary_metrics.json`, cùng thư mục.
  - Tổng chi phí API cho 5 lần chạy agent khoảng **0.145 USD**. Chi phí retrieval bằng 0 vì chạy local.
  - Code tính mọi input token theo giá thường, trong khi cached token rẻ hơn 10 lần, nên con số chi phí hơi cao hơn thực tế.
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - **Phân tích lỗi q19 trên test.** Ba yếu tố cùng gây ra lỗi:
    1. Agent viết lại query, làm `p400:000` tụt từ hạng 1 (với câu hỏi gốc) xuống hạng 2.
    2. Hit hạng 1 `p399:009` kéo theo 2 chunk lân cận. Ba chunk này chiếm 1416/2000 ký tự, nên khối `p400:000` (khoảng 590 ký tự) không còn đủ chỗ.
    3. Với `max_searches = 1`, agent không có cơ hội tìm lại.

    Baseline không gặp lỗi này, vì cả 3 lượt search của nó đều đưa `p400:000` lên hạng 1. Rủi ro này đã thấy từ dev: với N1 và budget 2000, có 11/16 câu (Dense) và 16/16 câu (Hybrid) bị đẩy một chunk top-3 ra khỏi context, nhưng chưa câu nào mất evidence. Rủi ro cũng đã được ghi vào `selection.json` trước khi chạy test. Mình không chỉnh gì sau khi thấy kết quả test.
  - **Đánh đổi.** Trên test, cấu hình được chọn rẻ hơn một nửa (0.0113 USD so với 0.0222 USD) và nhanh hơn khoảng 2.4 lần (2.2 s so với 5.5 s mỗi câu), có Hit@1 cao hơn (6/6 so với 4/6), nhưng chất lượng câu trả lời thấp hơn (1.67 so với 2.00) vì một câu bị từ chối nhầm. Trên dev, hai cấu hình gần như ngang nhau (2.00 so với 1.94).
  - **Tiêu chí chọn chưa tính đủ rủi ro.** Tiêu chí dựa vào Context Evidence Rate đo trên câu hỏi gốc, trong khi agent search bằng query tự viết. Một phiên bản tốt hơn nên đo Context trên chính các query của agent.
  - **Search thêm phần lớn là thừa.** Ở `dense-n0-s3` trên dev, 8 câu có search thêm thì 7 câu context lần đầu đã có evidence. Ở `hybrid-n1-s3`, cả 7/7 câu search thêm đều không cần thiết, và có câu lặp lại y hệt một query. Với câu thiếu bằng chứng, 3 lượt search không đổi được kết luận nhưng tốn khoảng 3 lần token.
  - **Query viết lại có lúc giúp, có lúc hại.** Nó giúp ở q16 (baseline tìm được `006` ở lượt 2), nhưng hại ở q05 dev (trượt chunk định nghĩa) và q19 test (tụt hạng).
  - **Hướng sửa cho lần sau** (chưa đánh giá, vì bộ test hiện tại đã bị nhìn): đóng gói context theo thứ tự ưu tiên, đưa tất cả hit top-k vào trước, rồi mới thêm chunk lân cận nếu còn budget; hoặc cho phép search lần 2 chỉ khi context lần đầu không có từ khóa của câu hỏi. Muốn đánh giá độc lập thì cần một bộ test mới.
  - **Giới hạn:**
    - Dev có 16 câu và test có 6 câu, nên mỗi câu chênh lệch tương ứng 6% trên dev và 17% trên test. Không đủ để kết luận cấu hình nào tốt hơn một cách tổng quát.
    - Chỉ có một người chấm. Lúc chấm, người chấm đã biết kết quả retrieval.
    - Cấu hình `hybrid-n1-s3` không được chấm câu trả lời bằng tay.
    - LLM không tất định, nên chạy lại có thể cho câu trả lời khác. Nhãn chỉ ứng với trace đã lưu.
    - Corpus chỉ có 2 chương, nên filter `chapter` chỉ có 2 giá trị. Ở cả 4 câu có filter, retriever không filter cũng ra đúng chương.
    - Câu q05 không kiểm tra thật sự được khả năng cross-lingual của BM25, vì câu hỏi có từ tiếng Anh trong ngoặc.
    - Trong một câu trả lời (q20, cấu hình được chọn), model chèn một từ tiếng Ả Rập ("حساس"). Lỗi này được ghi chú trong nhãn, không trừ điểm nội dung.

## Thử thách nâng cao (nếu thực hiện)

**Kết quả: 24/30 câu test đạt (80%), chưa đạt ngưỡng 90%.** Phần này vì vậy không đủ điều kiện nhận điểm thưởng. Mình vẫn trình bày đầy đủ vì quá trình làm có nhiều điều đáng ghi lại. Toàn bộ code và kết quả nằm trong thư mục [advanced/](../../advanced/).

- **Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản:**
  - Toàn bộ sách *Principles of Data Science* (OpenStax, Rice University, 2025), bản PDF web `Principles-of-Data-Science-WEB.pdf`, **569 trang**, giấy phép CC BY-NC-SA 4.0.
  - Đây là bản đầy đủ của chính cuốn sách dùng ở phần cơ bản, nhưng phần cơ bản chỉ dùng 2 chương. Sách có nhiều bảng, ảnh chụp màn hình output Python, sơ đồ và biểu đồ.
  - Số trang trong benchmark là **số trang PDF**. Số trang in trên sách bằng số trang PDF trừ 10.
  - File PDF không được commit, vì có thể tải lại từ nguồn công khai.

- **Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý:**
  - **Lọc trang có hình** ([profile_pdf.py](../../advanced/profile_pdf.py)): một trang được đánh dấu là có nội dung trực quan khi có ảnh lớn (trên 3% diện tích trang) hoặc có dòng chú thích bắt đầu bằng `Figure x.y` / `Table x.y`. Kết quả là **249 trang**.
    - Ban đầu mình còn đếm cả hình vector (drawings) và ra 387 trang. Kiểm tra trang 115 và 131 thì thấy phần lớn là công thức toán và khung Example, nên mình bỏ tín hiệu này.
  - **Index** ([build_index.py](../../advanced/build_index.py)) dùng một model duy nhất là `google/embeddinggemma-2` (768 chiều), bật thêm bộ mã hóa ảnh mà Day 6 đã tắt. Model này đưa text và ảnh vào cùng một không gian vector. Có 3 loại vector:
    - **text:** 2970 chunk (500 ký tự, chồng lấn 50) trên 560 trang có chữ;
    - **image:** 1 vector cho mỗi ảnh trang, chỉ gồm ảnh, render ở 110 dpi, cho 249 trang có hình;
    - **interleaved:** ảnh trang cộng 2000 ký tự đầu của trang trong cùng một input, cho 249 trang.

    Toàn bộ chạy trên CPU, không tốn API, mất khoảng 80 phút (khoảng 8 giây cho mỗi ảnh trang).
  - **Lúc trả lời**, agent gửi **ảnh gốc của trang** (`detail: high`) kèm 1500 ký tự text của trang cho gpt-5.4-mini. Bảng, biểu đồ và sơ đồ được model đọc trực tiếp từ ảnh.
  - **Chuyển hướng giữa chừng:** kế hoạch ban đầu là nhờ vision model viết mô tả cho từng trang có hình, rồi index phần mô tả ([describe_pages.py](../../advanced/describe_pages.py)). Thử trên 3 trang, mô tả bảng chính xác, nhưng histogram bị đọc lệch nửa cột. Sau đó mình chuyển sang nhúng thẳng ảnh vì miễn phí. Kết quả retrieval bên dưới cho thấy cách này đủ tốt, nên bước mô tả không được chạy cho cả sách.
  - **Phần chưa xử lý:**
    - Không cắt riêng vùng hình; mỗi trang được xử lý nguyên khổ.
    - Trang không có hình chỉ được tìm qua text.
    - Công thức toán render dạng hình bị mất khi trích text.
    - Hình nằm trên trang không có chú thích và không có ảnh lớn thì không có vector ảnh.

- **Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev:**
  - [advanced/benchmark/questions.json](../../advanced/benchmark/questions.json): **test 30 câu** (6 bảng, 5 ảnh/sơ đồ, 5 biểu đồ, 4 multi-hop, 4 text, 6 thiếu bằng chứng) và **dev 18 câu** (3 câu mỗi nhóm). Dev và test chia trước khi thử nghiệm.
  - Mỗi câu có `expected_answer`, `grading` (tiêu chí đạt, kèm khoảng sai số với giá trị đọc từ biểu đồ) và `evidence`. Mỗi evidence gồm số trang, loại (`table`, `chart`, `image`, `text`, `equation`) và `quote` hoặc `locator`, ví dụ "Figure 9.7, row Lettuce, column Agrifun".
  - Câu thiếu bằng chứng có ghi chú cách kiểm tra rằng sách không có thông tin đó.
  - [check_benchmark.py](../../advanced/benchmark/check_benchmark.py) kiểm tra số câu mỗi nhóm, kiểm tra `quote` có thật trên đúng trang, và kiểm tra test không dùng 3 trang đã dùng để thử prompt mô tả ảnh (27, 55, 431).
  - Hai chỗ sách tự mâu thuẫn được tránh khi đặt câu hỏi:
    - Table 2.3 ghi outlier ở tháng 6/1990, còn phần lời ghi tháng 3;
    - Figure 8.10 ghi "500 participants", nhưng cộng các nhóm lại được 600.
  - **Retrieval trên dev** ([eval_retrieval_adv.py](../../advanced/eval_retrieval_adv.py)) tính Hit@k theo trang. Câu multi-hop phải có đủ mọi trang evidence mới tính là trúng.

    | Phương pháp (dev, 15 câu) | Hit@1 | Hit@3 | Hit@5 |
    |---|---|---|---|
    | text_dense | 10 | 12 | 14 |
    | bm25 | 10 | 12 | 13 |
    | image | 10 | 13 | 14 |
    | interleaved | 11 | 14 | **15** |
    | hybrid_text (dense + BM25) | 10 | 11 | 13 |
    | hybrid_inter (dense + BM25 + interleaved) | 10 | 14 | **15** |

    Nhúng ảnh có tác dụng rõ. Ví dụ AD02 hỏi số liệu trong một ảnh chụp DataFrame: các phương pháp dùng text đều trượt, vì những con số đó không có trong text của trang.

    Ở bước này mình sửa một nhãn dev: AD12 từng yêu cầu thêm trang 205, nhưng trang này chỉ chứa định nghĩa residual và không cần để trả lời. Lý do sửa được ghi trong `note` của câu.
  - **Agent trên dev** ([agent_adv.py](../../advanced/agent_adv.py)): lấy top 5 trang mỗi lượt, tối đa 2 lượt search, model gpt-5.4-mini. Chấm đạt/không đạt bằng tay theo ADVANCED.md.

    | Lần chạy | Thay đổi | Đạt |
    |---|---|---|
    | `dev-inter-k5` | Chỉ dùng vector interleaved | 15/18 |
    | `dev-hinter-k5` | Hybrid ba nguồn | 15/18 |
    | `dev-hinter-k5-v2` | Thêm quy tắc vào prompt: không thay mục được hỏi bằng mục gần giống; gọi tên cột theo nhãn trục. Thêm một lượt cho model sửa citation sai định dạng | 16/18 |
    | `dev-hinter-k5-v3` | Mỗi citation bắt buộc có trường `reading`: model phải ghi lại chính xác thứ đã đọc trên ảnh | 17/18 |
    | `dev-hinter-k5-v4` | Sửa bộ kiểm citation, vì nó loại nhầm dạng gộp `[p204 text, p208 Table 4.16]` | 17/18 |

    Bài học chính ở v2 và v3: model dễ bỏ qua quy tắc viết trong prompt, nhưng một **trường bắt buộc** trong tool thì model phải điền. Trường `reading` cũng ghi lại vùng evidence của từng citation, ví dụ `"row Lettuce, column Agrifun: 4.3 tons/year"`.
  - **Cấu hình đã chốt** ([advanced/results/selection.json](../../advanced/results/selection.json)): `hybrid_inter`, top 5, 2 lượt search, v4.
    - Chọn hybrid thay vì `interleaved` vì hai bên ngang điểm, nhưng hybrid rẻ hơn và tìm được trên text của cả sách, còn `interleaved` chỉ biết 249 trang có hình.
    - Rủi ro đã ghi trước khi chạy test: AD07 (gọi tên cột cao nhất theo nhãn trục) đạt ở v3 nhưng trượt ở v4 với cùng prompt. Nguyên nhân là khi phần text của trang mô tả biểu đồ bằng lời, model có xu hướng chép lại lời thay vì đọc ảnh.
  - Trước khi chạy test, mình kiểm tra bằng mắt đáp án của 10 câu ảnh và biểu đồ trên PDF, và sửa lời mô tả đáp án của AT15 (thời điểm thấp nhất gần tháng 3–4/2020 hơn tháng 4–5). Tiêu chí chấm không đổi.

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | 6/6 | 100% | Đọc đúng mọi ô, kể cả ảnh chụp output `describe()` (AT03) |
| Ảnh, sơ đồ | 3/5 | 60% | AT09: cây Huffman, đi nhầm nhánh nên ra mã 11 thay vì 101. AT11: nói có 4 neuron ẩn thay vì 4 + 4 = 8, vì dựa vào chú thích "two hidden layers" thay vì đếm trên hình |
| Biểu đồ | 3/5 | 60% | AT12: trả lời x = 18 (≈ 0.27), tức chép P(18) = 0.271 trong lời giải bên trên, trong khi cột cao nhất trên hình là x = 19. AT14: trả lời 0.40 rồi tự cộng 0.07 + 0.01 = 0.08, câu trả lời tự mâu thuẫn |
| Multi-hop | 4/4 | 100% | AT17 lấy số từ bảng thay vì từ biểu đồ đường mà câu hỏi nhắc tới. Số vẫn đúng và bảng có số đó, nhưng câu này sát ranh giới |
| Text | 4/4 | 100% | |
| Thiếu bằng chứng | 4/6 | 67% | AT25: bịa ra 3.1 tấn cà rốt, lấy từ ô Asparagus, đúng kiểu lỗi "lấy hàng gần giống" mà dev đã nhắm sửa. AT28: bịa ra số người làm việc năm 1935, trong khi dữ liệu bắt đầu từ 1939 |
| **Toàn bộ** | **24/30** | **80%** | Chưa đạt ngưỡng 27/30 |

- **Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống:**
  - Trace: [advanced/results/agent/test-hinter-k5-v4/traces.json](../../advanced/results/agent/test-hinter-k5-v4/traces.json). Mỗi trace gồm từng lượt search (query, trang trả về, context), từng lượt gọi model (token, chi phí, độ trễ), câu trả lời, citation kèm `reading`, và các câu trả lời bị bộ kiểm citation bác.
  - Nhãn và lý do cho từng câu nằm trong `reviews.json` cùng thư mục. Tổng hợp theo nhóm: `grade_summary.json`, tạo bởi [grade_summary.py](../../advanced/grade_summary.py).
  - Có 3 câu mình chấm đạt nhưng ghi rõ là sát ranh giới: AT10 (ký hiệu "≤" trong câu trả lời bị hỏng thành ký tự lạ, nhưng thứ tự điều kiện đúng), AT16 (cite Figure 2.3, là bản đã bỏ outlier, thay vì Figure 2.2) và AT17. Nếu chấm chặt cả ba thì kết quả là 21/30.
  - **Retrieval không phải điểm nghẽn.** Trên test, context của agent có đủ trang evidence ở **24/24** câu có đáp án. Retrieval với câu hỏi gốc (`hybrid_inter`, [results/retrieval/test-baseline](../../advanced/results/retrieval/test-baseline/summary.json)) đạt Hit@1 19/24 và Hit@5 24/24. Cả 6 câu sai đều sai ở bước **đọc ảnh và lập luận**, không phải ở bước tìm trang.
  - **Chi phí và thời gian:** lần chạy test tốn 0.483 USD, khoảng 0.016 USD và 7.75 giây (trung vị) mỗi câu. Tổng chi phí API cho toàn bộ phần nâng cao, gồm thử mô tả ảnh, 5 lần chạy dev và 1 lần chạy test, là **2.10 USD**, ghi trong sổ chi phí [advanced/data/api_costs.jsonl](../../advanced/data/api_costs.jsonl). Code kiểm tra trần 3 USD trước mỗi lần gọi API.
  - Không có lỗi provider nào. Cấu hình `hybrid_inter` ở v1 bị chạy 2 lần (lần đầu bị ghi đè), nên sổ chi phí ghi nhiều hơn tổng của các thư mục kết quả.

- **Giới hạn, cách tái lập và đường dẫn benchmark:**
  - **Dev đánh giá quá cao hệ thống:** dev đạt 94%, test chỉ 80%. Dev có 3 câu mỗi nhóm và gần như không có các kiểu khó mà test có: lần theo nhánh cây, đếm phần tử trên hình, biểu đồ mà phần text bên cạnh có một con số khác. Dấu hiệu duy nhất là AD07, và đó cũng là rủi ro duy nhất được ghi trong `selection.json`.
  - **Ảnh trang bị thu nhỏ:** mỗi lượt gửi 5–7 ảnh trang nguyên khổ, nên chi tiết nhỏ như nhãn 0/1 trên cạnh cây Huffman hay nhãn trục dễ bị đọc sai.
  - **Dev được dùng 5 lần để chỉnh**, nên kết quả dev không còn là ước lượng độc lập.
  - **Test chỉ chạy đúng 1 lần.** Mình không chỉnh gì sau khi thấy kết quả. Muốn thử lại phải tạo một bộ test mới với độ khó tương đương. Làm test dễ hơn chỉ để vượt ngưỡng không phản ánh đúng năng lực của hệ thống.
  - Một người chấm. LLM không tất định, và nhãn ứng với trace đã lưu.
  - Cách chạy lại: xem mục "Phần nâng cao" trong README.
