from solution import DivarContest
from PIL import Image
import pytesseract
from io import BytesIO
import requests


dc = DivarContest()
response = dc.capture_the_flag(
    "do what image says at { https://i.imgur.com/B9TXldY.png }"
)
print(response)
