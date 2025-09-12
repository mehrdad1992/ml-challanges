from openai import OpenAI


class Solution:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key, base_url="https://turbo.torob.com/v1")

    def run(self, text_input: str) -> str:

        messages = [
            {
                "role": "system",
                "content": "You are an assistant for classification of products. "
                "You will receive the name of the shop and the name of the product delimitered by ```\n``` or newline character and you must determine its category."
                "There are 5 product categories: ```SMARTPHONE```, ```LAPTOP```, ```WATCH```, ```FLOWER```, ```CLOTH```"
                "And if you couldn't determine proper category you should consider its category as ```UNKNOWN```"
                "Consider your input text maybe in both english and persian languages",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_input},
                ],
            },
        ]

        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.1,
            max_tokens=100,
        )
        msg = response.choices[0].message.content.strip()

        return msg
