import os
import pillow_heif
from PIL import Image
from pathlib import Path
from roboflow import Roboflow
from unidecode import unidecode

from ingest_data import GoogleDriveAuthenticator, GoogleDriveService

from concurrent.futures import ThreadPoolExecutor

class DataPipelineProcesser:
    
    def __init__(self):
        pass

    def ingest_data(self, list_folder_id, output_path):
        scope = ['https://www.googleapis.com/auth/drive.readonly']

        ggdrive_auth = GoogleDriveAuthenticator(scope=scope)
        ggdrive_auth.authenticate()
        service = ggdrive_auth.get_ggdrive_service()

        ggdriver_ser = GoogleDriveService(service=service)

        for folder_id in list_folder_id:
            ggdriver_ser.download_folder(folder_id=folder_id, output_path=output_path)
    
    def processing_data(self, raw_path, processed_path):
        raw_path = Path(raw_path)
        processed_path = Path(processed_path)
        processed_path.mkdir(parents=True, exist_ok=True)

        for folder in raw_path.iterdir():
            if not folder.is_dir():
                continue

            clean_folder_name = unidecode(folder.name).replace(" ", "_")
            dest_folder = processed_path / clean_folder_name
            dest_folder.mkdir(parents=True, exist_ok=True)

            for file_path in folder.iterdir():
                if not file_path.is_file():
                    continue

                base_name = file_path.stem.replace(" ", "_")
                new_name = f"{clean_folder_name}_{base_name}.jpg"
                new_path = dest_folder / new_name

                if new_path.exists():
                    continue

                try:
                    img = Image.open(file_path)
                    rgb_img = img.convert("RGB")
                    rgb_img.save(new_path, "JPEG")
                    print(f"Processed {new_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

class RoboflowUploader:
    def __init__(self, api_key: str, workspace: str, project: str):
        self.rf = Roboflow(api_key=api_key)
        self.project = self.rf.workspace(workspace).project(project)
    
    def _upload_file(self, file_path):
        try:
            self.project.upload(image_path=str(file_path))
            print(f"Uploaded {file_path}")
        except Exception as e:
            print(f"Error uploading {file_path}: {e}")

    def upload_folder(self, folder_path: str, max_workers=8):
        folder_path = Path(folder_path)
        files = list(folder_path.rglob("*.jpg"))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(self._upload_file, files)

def main() -> None:
    pillow_heif.register_heif_opener()

    list_folder_id = ['12liJ0oGdAStAmX2NXAvlCvJ-_6h-enwo']
    output_path = 'data/raw'

    RAW_DATA_PATH = 'data/raw'
    PROCESSED_DATA_PATH = "data/processed_data"
    pipeline = DataPipelineProcesser()
    
    pipeline.ingest_data(list_folder_id=list_folder_id, output_path=output_path)
    pipeline.processing_data(raw_path=RAW_DATA_PATH, processed_path=PROCESSED_DATA_PATH)

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    workspace = "plant-xcvlc"
    project = "capstone-project-jlomf"

    uploader = RoboflowUploader(api_key, workspace, project)
    uploader.upload_folder(folder_path=PROCESSED_DATA_PATH)

if __name__ == "__main__":
    main()
