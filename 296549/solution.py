from openai import OpenAI
import requests
import io
import easyocr


class DivarContest:
    def __init__(self, api_token):
        self.api_token = api_token
        self.model = "gpt-4.1-mini"
        self.client = OpenAI(
            api_key=self.api_token, base_url="https://api.metisai.ir/openai/v1"
        )
        self.ocr_reader = easyocr.Reader(["en"])

    def launch_threads(self, num_threads, shared_counter):
        pass

    def capture_the_flag(self, question):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that should solve the puzzles. When solving puzzle use extract_text if needed to fetch text from image.",
            },
            {"role": "user", "content": f"calculate {question}. just print answer"},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "extract_text",
                    "description": "Extract text from an image using OCR.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image_base64": {
                                "type": "string",
                                "description": "Base64-encoded image data",
                            }
                        },
                        "required": ["image_base64"],
                    },
                },
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            max_tokens=100,
            temperature=0.1,
        )

        msg = response.choices[0].message
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call.function.name == "extract_text":
                    args = eval(tool_call.function.arguments)
                    text = self.extract_text(args["image_base64"])
                    messages.append(msg)
                    messages.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": text}
                    )
                    final_response = self.client.chat.completions.create(
                        model=self.model, messages=messages
                    )
                    return final_response.choices[0].message.content.strip()
        else:
            return msg.content.strip()
