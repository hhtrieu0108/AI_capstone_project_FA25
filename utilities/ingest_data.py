import os
import json
import pickle
import logging
from dotenv import load_dotenv

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# ------------------ LOGGER CONFIG ------------------
logger = logging.getLogger("GoogleDriveIngestor")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

ch = logging.FileHandler('logs/gdrive_ingest.log', encoding='utf-8')
ch.setFormatter(formatter)
logger.addHandler(ch)

class GoogleDriveAuthenticator:

    def __init__(self, scope=None):
        self.scope = scope or SCOPES
        self.service = None

    def _save_creds(self, creds, path="key/token.pickle"):
        with open(path, "wb") as f:
            pickle.dump(creds, f)
        logger.info(f"Credentials saved to {path}.")

    def _load_creds(self, path="key/token.pickle"):
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            creds = pickle.load(f)
        logger.info(f"Credentials loaded from {path}.")

        return creds

    def authenticate(self):
        load_dotenv()

        logger.info("Starting authentication...")
        creds = self._load_creds()

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                logger.info("Credentials refreshed.")
            else:
                env_creds = os.environ.get('GGDRIVE_CREDENTIALS')
                client_config = json.loads(env_creds)
                
                flow = InstalledAppFlow.from_client_config(client_config, self.scope)
                creds = flow.run_local_server(port=0)
                logger.info("Authentication completed via OAuth flow.")
            self._save_creds(creds=creds)

        self.service = build('drive', 'v3', credentials=creds)
        logger.info("Google Drive service ready.")
    
    def get_ggdrive_service(self):
        if not self.service:
            raise RuntimeError("Call authenticate() first")
        return self.service

class GoogleDriveService:

    def __init__(self, service):
        self.service = service
    
    def list_files(self, page_size=10):
        if not self.service:
            raise RuntimeError("Call authenticate() first")

        results = self.service.files().list(pageSize=page_size, fields="files(id, name)").execute()
        items = results.get('files', [])

        if not items:
            logger.info("No files found.")
        else:
            logger.info(f"Found {len(items)} files:")
            for it in items:
                logger.info(f"{it['name']} ({it['id']})")

    def download_folder(self, folder_id: str, output_path: str):
        if not self.service:
            raise RuntimeError("Call authenticate() first")
        
        os.makedirs(output_path, exist_ok=True)
        query = f"'{folder_id}' in parents and trashed=false"
        page_token = None

        while True:
            response = self.service.files().list(
                q=query,
                spaces='drive',
                fields='nextPageToken, files(id, name, mimeType)',
                pageToken=page_token
            ).execute()

            for file in response.get('files', []):
                file_id = file['id']
                name = file['name']
                mime = file['mimeType']

                if mime == 'application/vnd.google-apps.folder':
                    subfolder_path = os.path.join(output_path, name)
                    self.download_folder(file_id, subfolder_path)
                else:
                    out_path = os.path.join(output_path, name)

                    if os.path.exists(out_path):
                        print(f"Skipping {name}, already exists.")
                        continue
                    
                    request = self.service.files().get_media(fileId=file_id)
                    fh = open(out_path, 'wb')
                    downloader = MediaIoBaseDownload(fh, request, chunksize=5*1024*1024)
                    done = False
                    while not done:
                        status, done = downloader.next_chunk()
                        print(f"Downloading {name} {int(status.progress() * 100)}%")
                    fh.close()

            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break


# Test
# if __name__ == "__main__":
#     ggdrive_auth = GoogleDriveAuthenticator(scope=SCOPES)
#     ggdrive_auth.authenticate()
#     service = ggdrive_auth.get_ggdrive_service()

#     ggdriver_ser = GoogleDriveService(service=service)

#     list_folder_id = ['12liJ0oGdAStAmX2NXAvlCvJ-_6h-enwo']
    
#     for folder_id in list_folder_id:
#         ggdriver_ser.download_folder(folder_id=folder_id, output_path='test/test_folder')