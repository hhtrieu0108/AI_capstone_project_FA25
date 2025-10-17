import os
from dotenv import load_dotenv


load_dotenv()
test = os.environ.get("GGDRIVE_CREDENTIALS")
print(test)