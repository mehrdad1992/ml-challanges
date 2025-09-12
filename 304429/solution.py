from openai import OpenAI
import requests
import json


class Solution:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key, base_url="https://turbo.torob.com/v1")

    # def solve_puzzle(self, text: str, type: str):
    #     response = self.client.chat.completions.create(
    #         model="gpt-4.1-mini",
    #         messages={
    #             "role": "system":
    #             "",
    #             "role": "user",
    #             "content": text,
    #         },
    #         temperature=0.1,
    #         max_tokens=1000,
    #     )
    #     msg = response.choices[0].message
    #     if msg.content:
    #         return msg.content.strip()

    def run(self, question: str) -> str:
        system_content = (
            "You are a puzzle resolver. "
            "You will receive puzzles as text that may contain math or URLs (text or image). "
            "Math puzzles can be solved directly. "
            "URL puzzles must be fetched using `fetch_url`. "
            "Return only the final concatenated number without explanations."
        )

        # Full conversation history
        history_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question},
        ]

        # Tools definition placed after system content
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "fetch_url",
                    "description": "Fetch the contents of a URL (text or image).",
                    "parameters": {
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"],
                    },
                },
            }
        ]

        while True:
            # GPT call
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=history_messages,
                tools=tools,
                temperature=0.1,
                max_tokens=1000,
            )

            msg = response.choices[0].message

            # Final answer returned
            if msg.content:
                return msg.content.strip()

            # Handle one tool call at a time
            if msg.tool_calls:
                tool_call = msg.tool_calls[0]
                if tool_call.function.name == "fetch_url":
                    args = json.loads(tool_call.function.arguments)
                    url = args["url"]

                    # Fetch content
                    try:
                        resp = requests.get(url)
                        resp.raise_for_status()
                        content = resp.text
                    except Exception as e:
                        content = f"Error fetching {url}: {e}"

                    # Determine puzzle type
                    if url.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                        user_instruction = (
                            f"Convert the image at {url} to a numeric answer."
                        )
                    else:
                        user_instruction = (
                            f"Extract the numeric answer from the text at {url}."
                        )

                    # ✅ Append messages in correct order
                    history_messages.append(msg)  # assistant message with tool_call
                    history_messages.append(
                        {"role": "user", "content": user_instruction}
                    )
                    history_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": content,
                        }
                    )

                    # Loop again → GPT sees the tool result
                    continue


def main():
    api_key = "trb-2380b1473118a24ba5-d8a1-4b44-b949-432713fdd5e9"
    solution = Solution(
        api_key=api_key,
    )

    url = input()
    resp = requests.get(url)
    question = resp.text

    print(solution.run(question))


if __name__ == "__main__":
    main()
