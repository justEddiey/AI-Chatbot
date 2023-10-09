from flask import Flask
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('API_KEY')
app = Flask(__name__)