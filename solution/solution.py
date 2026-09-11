"""
K4 — Ngày 1: Khám Phá LLM API (4 tiếng)
AICB-P1: AI Practical Competency Program, Phase 1

Hướng dẫn:
    1. Làm theo LAB_GUIDE.md — mỗi block có các bước chi tiết và checkpoint.
    2. Điền vào tất cả các chỗ đánh dấu TODO.
    3. KHÔNG đổi chữ ký hàm (tên hàm, tham số).
    4. Import OpenAI BÊN TRONG hàm (xem gợi ý) — nếu import ở đầu file,
       các bài test mock sẽ không hoạt động.
    5. Kiểm tra tiến độ:   pytest tests/test_part1.py -v  (từng phần)
       Chấm điểm tổng:     python grade.py
"""

import os
import time
from typing import Any, Callable

from dotenv import load_dotenv

# Nạp OPENAI_API_KEY từ file .env (copy .env.example thành .env và dán key vào)
load_dotenv()

# ---------------------------------------------------------------------------
# Bảng giá ước tính (USD / 1K token) — cập nhật nếu giá thay đổi
# ---------------------------------------------------------------------------
PRICING_PER_1K_TOKENS = {
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}

# Tên model có thể đổi qua .env — ví dụ khi dùng NVIDIA NIM miễn phí
# (xem LAB_GUIDE.md, Phụ lục B). Không đặt gì trong .env thì mặc định OpenAI.
OPENAI_MODEL = os.getenv("LAB_MODEL", "gpt-4o")
OPENAI_MINI_MODEL = os.getenv("LAB_MINI_MODEL", "gpt-4o-mini")


# ===========================================================================
# PART 1 — API CƠ BẢN (Block 1: phút 60–100)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 1.1 — Gọi GPT-4o
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi OpenAI Chat Completions API, trả về nội dung phản hồi + độ trễ.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start_time

    response_text = response.choices[0].message.content or ""
    return response_text, latency


# ---------------------------------------------------------------------------
# Task 1.2 — Gọi GPT-4o-mini
# ---------------------------------------------------------------------------
def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi API với model gpt-4o-mini — nhanh hơn và rẻ hơn.
    """
    return call_openai(
        prompt=prompt,
        model=OPENAI_MINI_MODEL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Task 1.3 — So sánh GPT-4o vs GPT-4o-mini
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Gọi cả hai model với cùng một prompt và trả về dict so sánh.
    """
    gpt4o_text, gpt4o_lat = call_openai(prompt, model=OPENAI_MODEL)
    mini_text, mini_lat = call_openai_mini(prompt)

    # Ước tính chi phí theo gợi ý: (số_từ / 0.75) / 1000 * đơn giá output
    cost_estimate = (
        (len(gpt4o_text.split()) / 0.75)
        / 1000
        * PRICING_PER_1K_TOKENS["gpt-4o"]["output"]
    )

    return {
        "gpt4o_response": gpt4o_text,
        "mini_response": mini_text,
        "gpt4o_latency": gpt4o_lat,
        "mini_latency": mini_lat,
        "gpt4o_cost_estimate": cost_estimate,
    }


# ===========================================================================
# PART 2 — SYSTEM PROMPT & TOKEN (Block 2: phút 100–140)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 2.1 — Chat với system prompt (persona)
# ---------------------------------------------------------------------------
def chat_with_system_prompt(
    system_prompt: str,
    user_prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi API với MESSAGES gồm 2 phần: system prompt và user prompt.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p if "top_p" in locals() else 0.9,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start_time

    response_text = response.choices[0].message.content or ""
    return response_text, latency


# ---------------------------------------------------------------------------
# Task 2.2 — Đếm token bằng tiktoken
# ---------------------------------------------------------------------------
def count_tokens(text: str, model: str = OPENAI_MODEL) -> int:
    """
    Đếm số token của một đoạn text bằng thư viện tiktoken có dự phòng fallback.
    """
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        # Fallback ước lượng dự phòng: 1 token ≈ 4 ký tự
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Task 2.3 — Ước tính chi phí chính xác
# ---------------------------------------------------------------------------
def estimate_cost(
    prompt: str, response: str, model: str = OPENAI_MODEL
) -> dict:
    """
    Tính chi phí một lượt gọi API dựa trên số token thật (đếm bằng count_tokens).
    """
    input_tokens = count_tokens(prompt, model=model)
    output_tokens = count_tokens(response, model=model)

    pricing = PRICING_PER_1K_TOKENS.get(
        model, PRICING_PER_1K_TOKENS.get("gpt-4o")
    )
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    total_cost = input_cost + output_cost

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


# ===========================================================================
# PART 3 — STREAMING & ĐỘ BỀN (Block 3: phút 150–190)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 3.1 — Chatbot streaming có lịch sử hội thoại
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Chatbot dòng lệnh tương tác dùng streaming và duy trì 3 lượt hội thoại gần nhất.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    history: list[dict] = []

    print("Bắt đầu chat (gõ 'quit' hoặc 'exit' để thoát):")
    while True:
        try:
            user_msg = input("\nUser: ").strip()
        except EOFError:
            break

        if user_msg.lower() in ("quit", "exit"):
            break

        if not user_msg:
            continue

        messages = history + [{"role": "user", "content": user_msg}]

        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            stream=True,
        )

        print("Assistant: ", end="", flush=True)
        assistant_reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            assistant_reply += delta
        print()

        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": assistant_reply})
        history = history[-6:]


# ---------------------------------------------------------------------------
# Task 3.2 — Retry với exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Gọi fn(). Nếu ném exception, thử lại tối đa max_retries lần với exponential backoff.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:
            if attempt >= max_retries:
                raise e
            delay = base_delay * (2**attempt)
            time.sleep(delay)
            attempt += 1


# ===========================================================================
# PART 4 — MINI-PROJECT: TRỢ LÝ CLI HOÀN CHỈNH (Block 4: phút 190–230)
# ===========================================================================
def run_assistant(
    persona: str,
    get_input: Callable[[], str] = None,
    max_turns: int = None,
) -> dict:
    """
    Trợ lý CLI hoàn chỉnh — ghép mọi thứ từ Part 1–3.
    """
    from openai import OpenAI

    if get_input is None:
        get_input = input

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    history: list[dict] = []
    num_turns = 0
    total_tokens = 0
    total_cost = 0.0

    while True:
        if max_turns is not None and num_turns >= max_turns:
            break

        try:
            user_msg = get_input()
        except EOFError:
            break

        if user_msg.strip().lower() in ("quit", "exit"):
            break

        messages = (
            [{"role": "system", "content": persona}]
            + history
            + [{"role": "user", "content": user_msg}]
        )

        # Gọi API qua retry_with_backoff
        stream = retry_with_backoff(
            lambda: client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                stream=True,
            )
        )

        reply_parts = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            reply_parts.append(delta)
        print()

        full_reply = "".join(reply_parts)
        num_turns += 1

        # Tính token và chi phí cho lượt này
        turn_cost_dict = estimate_cost(user_msg, full_reply, model=OPENAI_MODEL)
        total_tokens += (
            turn_cost_dict["input_tokens"] + turn_cost_dict["output_tokens"]
        )
        total_cost += turn_cost_dict["total_cost"]

        # Cập nhật lịch sử và giới hạn 3 lượt cuối (6 tin nhắn)
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": full_reply})
        history = history[-6:]

    return {
        "num_turns": num_turns,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "history": history,
    }


# ===========================================================================
# BONUS (không bắt buộc — cho bạn nào xong sớm)
# ===========================================================================
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Chạy compare_models cho từng prompt trong list.
    """
    results = []
    for prompt in prompts:
        cmp_data = compare_models(prompt)
        cmp_data["prompt"] = prompt
        results.append(cmp_data)
    return results


def format_comparison_table(results: list[dict]) -> str:
    """
    Định dạng kết quả batch_compare thành bảng text dễ đọc.
    """
    headers = [
        "Prompt",
        "GPT-4o Response",
        "Mini Response",
        "GPT-4o Latency",
        "Mini Latency",
    ]
    rows = []
    rows.append(
        f"{headers[0]:<40} | {headers[1]:<40} | {headers[2]:<40} | {headers[3]:<15} | {headers[4]:<15}"
    )
    rows.append("-" * 160)

    for res in results:
        p = (
            res.get("prompt", "")[:37] + "..."
            if len(res.get("prompt", "")) > 40
            else res.get("prompt", "")
        )
        g4o = (
            res.get("gpt4o_response", "")[:37] + "..."
            if len(res.get("gpt4o_response", "")) > 40
            else res.get("gpt4o_response", "")
        )
        mini = (
            res.get("mini_response", "")[:37] + "..."
            if len(res.get("mini_response", "")) > 40
            else res.get("mini_response", "")
        )
        g4o_lat = f"{res.get('gpt4o_latency', 0.0):.3f}s"
        mini_lat = f"{res.get('mini_latency', 0.0):.3f}s"

        rows.append(
            f"{p:<40} | {g4o:<40} | {mini:<40} | {g4o_lat:<15} | {mini_lat:<15}"
        )

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Entry point — demo chạy thật (cần OPENAI_API_KEY)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== So sánh model ===")
    result = compare_models(
        "Giải thích khác biệt giữa temperature và top_p trong một câu."
    )
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== Trợ lý CLI (gõ 'quit' để thoát) ===")
    stats = run_assistant(
        persona="Bạn là trợ giảng thân thiện của khóa AI, "
        "trả lời ngắn gọn bằng tiếng Việt.",
    )
    print("\n--- Thống kê phiên chat ---")
    for key, value in stats.items():
        if key != "history":
            print(f"{key}: {value}")