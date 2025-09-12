from openai import OpenAI


class XPathExtractor:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key, base_url="https://turbo.torob.com/v1")

    def run(self, available_link, unavailable_link):
        return {
            "name_xpath": "[Your result]",
            "final_price_xpath": "[Your result]",
            "availability_xpath": "[Your result]",
        }
