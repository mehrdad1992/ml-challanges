from solution import DivarContest

API_KEY="tpsg-DQru38M36YSWpjzh8bXRIcpREv5lBMu"


dc = DivarContest(api_token=API_KEY)
response = dc.capture_the_flag("Answer in one word, lowercase: what color is the sky?")
print(response)