from openai import OpenAI
from PIL import Image
import pytesseract
import requests
from io import BytesIO
import json
import re
from langdetect import detect


class DivarContest:
    def __init__(self):
        # self.api_token = "47afdefd-49d8-5f93-87db-4d504cb4fceb"
        # self.base_url = "https://arvancloudai.ir/gateway/models/GPT-4o-mini/4DFIB4L1R9JNtUPQEFoeal71p1s9m_xbTH7j68AaaRBgTVX3iNJiJC38c1WXkPyu8nOeGh6cYW7nV_nX5MOtdq2a_orn6ljkPfDyW-0SOFiHfTjk-Ssqt7se68Xs72eIC6tHSclHsYq6lLO_AH7Im2C6j47z4aVny9xTYztnHHpYIoWgZkDI--TAxTvNyX9ekHPZgDHr6ETXBpumyNYY7k7UXeI3P51zVZBa1JNZp9m2klvK2aI1ObDsLuA/v1"
        self.api_token = "tpsg-DQru38M36YSWpjzh8bXRIcpREv5lBMu"
        self.client = OpenAI(
            api_key=self.api_token, base_url="https://api.metisai.ir/openai/v1"
        )
        self.model = "gpt-4o-mini"

        # client = OpenAI(api_key=self.api_token, base_url=self.base_url)
        # models = client.models.list()
        # for model in models:
        #     print(model.id)

    def extract_text(self, url) -> str:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Referer": "https://google.com",
        }
        # url = "https://i.imgur.com/seCYob2.png"

        response = requests.get(url, headers=headers)
        img = Image.open(BytesIO(response.content))
        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )
        text = pytesseract.image_to_string(img)

        return text

    def decode_caeser(self, text: str, shift: int) -> str:
        decoded = ""
        for char in text:
            if char.isalpha():
                shifted_ord = ord(char) - shift
                if shifted_ord < 97:
                    shifted_ord += 26
                decoded += chr(shifted_ord)
            else:
                decoded += char

        return decoded

    def is_english_sentence(self, text: str) -> bool:
        words = re.findall(r"[A-Za-z']+", text)
        if len(words) < 3:
            return False
        try:
            return detect(text) == "en"
        except:
            return False

    def capture_the_flag(self, question):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that should solve the ceaser code puzzles. "
                "You are given a url containing image which you should extract its text first. "
                "for text extraction you must use the tool ```extract_text```. then you try to decode it with different shift windows."
                "for decoding you must use the tool ```decode_caeser```. you should consider which the decoded sentence with ordered shift is a proper english sentense or not"
                "Once you find a valid English sentence from Caesar decoding, STOP. "
                "Do not attempt any further shifts. Respond immediately with the decoded result. "
                "Never continue testing other shifts."
                "Then asnwer to that english question in short form",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                ],
            },
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
                            "url": {
                                "type": "string",
                                "description": "image url",
                            }
                        },
                        "required": ["url"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "decode_caeser",
                    "description": "decode caeser code",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "encrypted text with caser code",
                            },
                            "shift": {
                                "type": "integer",
                                "description": "the shift value which should use for each character in text to decode caeser encryption",
                            },
                        },
                        "required": ["text", "shift"],
                    },
                },
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            max_tokens=2000,
            temperature=0,
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call.function.name == "extract_text":
                    args = json.loads(tool_call.function.arguments)
                    ocr_text = self.extract_text(args["url"])

                    # ✅ Try Caesar shifts locally
                    decoded = None
                    for shift in range(26):
                        candidate = self.decode_caeser(ocr_text, shift)
                        if self.is_english_sentence(candidate):
                            decoded = candidate
                            break

                    messages.append(msg)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": ocr_text,
                        }
                    )

                    if decoded:
                        # Feed decoded sentence to LLM so it answers the question
                        messages.append(
                            {
                                "role": "user",
                                "content": f"Here is the decoded question: {decoded}. Please answer it briefly.",
                            }
                        )
                        final_response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            max_tokens=200,
                            temperature=0,
                        )
                        return final_response.choices[0].message.content.strip()

        return msg.content.strip()
