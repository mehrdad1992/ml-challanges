from solution import DivarContest

API_KEY="tpsg-DQru38M36YSWpjzh8bXRIcpREv5lBMu"


dc = DivarContest(api_token=API_KEY)
response = dc.capture_the_flag("do what image says at { https://divar-contest.darkube.app/fyvkr93-public.png }")
print(response)