import os
from PIL import Image
from pathlib import Path
import pillow_heif
from roboflow import Roboflow

from ingest_data import GoogleDriveAuthenticator, GoogleDriveService

pillow_heif.register_heif_opener()

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

        # Duyệt từng folder
        for folder in raw_path.iterdir():
            if not folder.is_dir():
                continue

            clean_folder_name = folder.name.replace(" ", "_")
            dest_folder = processed_path / clean_folder_name
            dest_folder.mkdir(parents=True, exist_ok=True)

            for file_path in folder.iterdir():
                if not file_path.is_file():
                    continue

                base_name = file_path.stem.replace(" ", "_")
                new_name = f"{clean_folder_name}_{base_name}.jpg"
                new_path = dest_folder / new_name

                # Skip nếu file đã tồn tại
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
    
    def upload_folder(self, folder_path: str):
        folder_path = Path(folder_path)
        for file_path in folder_path.rglob("*.jpg"):
            try:
                self.project.version(1).upload(file_path=str(file_path))
                print(f"Uploaded {file_path}")
            except Exception as e:
                print(f"Error uploading {file_path}: {e}")

if __name__ == "__main__":
    list_folder_id = ['12liJ0oGdAStAmX2NXAvlCvJ-_6h-enwo']
    
    RAW_DATA_PATH = 'test/test_folder'
    PROCESSED_DATA_PATH = "test/processed_data"
    
    pipeline = DataPipelineProcesser()
    
    pipeline.processing_data(raw_path=RAW_DATA_PATH, processed_path=PROCESSED_DATA_PATH)
    
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    workspace = "plant-xcvlc"
    project = "capstone-project-jlomf"

    uploader = RoboflowUploader(api_key, workspace, project)
    uploader.upload_folder(folder_path=PROCESSED_DATA_PATH)