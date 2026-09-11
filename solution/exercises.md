# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> 
### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature trong khoảng từ 0.1 đến 0.3. Chatbot hỗ trợ khách hàng ưu tiên hàng đầu tính chính xác, tính nhất quán trong chính sách và hạn chế tối đa ảo giác thông tin; mức nhiệt độ thấp này đảm bảo câu trả lời luôn bám sát dữ liệu chuẩn của doanh nghiệp và không đưa ra các phản hồi sai lệch ngẫu nhiên.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Với bảng giá đề bài ($0.010 cho GPT-4o và $0.0006 cho GPT-4o-mini trên 1K token output), GPT-4o đắt hơn GPT-4o-mini khoảng 16.67 lần (với 10.500K token/ngày, GPT-4o tốn $105/ngày còn mini chỉ tốn $6.3/ngày). Nên dùng GPT-4o: Khi cần xử lý các tác vụ suy luận phức tạp, phân tích báo cáo tài chính/pháp lý đa chiều hoặc viết code logic khó đòi hỏi độ chính xác tuyệt đối. Nên dùng GPT-4o-mini: Cho các tác vụ định tuyến tin nhắn, tóm tắt hội thoại cơ bản, chatbot phân loại ý định khách hàng hoặc trích xuất dữ liệu có cấu trúc đơn giản.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> 

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Số token thực tế đếm bằng tiktoken thường cao hơn công thức ước lượng số từ / 0.75 khoảng 40% đến 80% (tùy mật độ dấu tiếng Việt). Nguyên nhân là bộ mã hóa BPE (Byte-Pair Encoding) của OpenAI được huấn luyện tối ưu hóa trên kho ngữ liệu tiếng Anh; tiếng Việt là ngôn ngữ đa âm tiết có thanh dấu (Unicode nhiều byte) nên các từ tiếng Việt thường bị băm thành nhiều mảnh ký tự/byte rời rạc (subwords), dẫn đến số lượng token bị đội lên rất nhiều so với tiếng Anh cùng nghĩa.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các giao diện tương tác trực tiếp với con người (như chatbot trò chuyện, trợ lý ảo) vì nó làm giảm đáng kể Time-to-First-Token (TTFT), giúp người dùng không cảm thấy phải chờ đợi quá lâu trước phản hồi dài. Ngược lại, non-streaming phù hợp hơn cho các tác vụ chạy ngầm (background jobs), gọi API machine-to-machine, xử lý hàng loạt (batch processing), hoặc khi cần parse toàn bộ kết quả dạng JSON hoàn chỉnh trước khi lưu vào cơ sở dữ liệu.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giúp giãn cách thời gian giữa các lần thử lại ngày càng xa ($0.1s, 0.2s, 0.4s...$), tạo khoảng nghỉ cần thiết để server bên cung cấp dịch vụ kịp giải tỏa áp lực tải. Nếu hàng nghìn client cùng retry với một khoảng delay cố định (ví dụ luôn là 1 giây), toàn bộ các request này sẽ đồng loạt va đập vào server tại cùng một chu kỳ thời gian; điều này gây ra hiện tượng bão yêu cầu (thundering herd problem) làm hệ thống tiếp tục sập sâu hơn và không thể tự hồi phục.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> System prompt đã chọn: "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn, súc tích bằng tiếng Việt và luôn dùng ví dụ thực tế. "Giải thích lựa chọn: "Trả lời ngắn gọn, súc tích": Nhằm kiểm soát độ dài phản hồi, tiết kiệm token đầu ra và giữ cho thông tin tập trung, tránh câu trả lời dài dòng gây tràn ngữ cảnh CLI. "Bằng tiếng Việt": Ràng buộc trực tiếp ngôn ngữ phản hồi đồng nhất, ngăn chặn trường hợp mô hình tự động chuyển sang tiếng Anh khi gặp các thuật ngữ kỹ thuật trong câu hỏi.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất: Bộ nhớ hội thoại chỉ lưu cố định 3 lượt gần nhất (history[-6:]), khiến bot hoàn toàn quên ngữ cảnh, nhiệm vụ hoặc thỏa thuận được đặt ra từ các câu hỏi trước đó trong một phiên chat dài. Cách cải thiện: Triển khai cơ chế Tóm tắt ngữ cảnh (Summarization Memory). Cách làm: Khi lịch sử vượt quá 6 tin nhắn, thay vì cắt bỏ thẳng tay, gọi một model phụ (như gpt-4o-mini) tóm tắt ngắn gọn các thông tin then chốt của các tin nhắn cũ thành một đoạn tóm tắt ngắn và lưu vào system_prompt, sau đó chỉ ghép 2-3 tin nhắn gần nhất phía sau để giữ trọn vẹn mạch trao đổi mà vẫn tiết kiệm token.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
