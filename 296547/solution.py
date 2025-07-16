from openai import OpenAI

class DivarContest:
    def __init__(self, api_token):
        self.api_token = api_token
        self.model = "gpt-4.1-mini"
        self.client = OpenAI(api_key=self.api_token, base_url="https://api.metisai.ir/openai/v1")

    def capture_the_flag(self, question):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": f"calculate {question}. just print answer"}],
            max_tokens=100,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()


