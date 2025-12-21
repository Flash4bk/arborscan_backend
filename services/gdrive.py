# services/gdrive.py
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from pathlib import Path
from typing import Optional
from config import (
    GOOGLE_DRIVE_CREDENTIALS,
    GOOGLE_DRIVE_ROOT_FOLDER,
    GOOGLE_DRIVE_UPLOADS_FOLDER,
    GOOGLE_DRIVE_MODELS_FOLDER,
)

SCOPES = ["https://www.googleapis.com/auth/drive"]


class GoogleDriveService:
    def __init__(self):
        credentials = service_account.Credentials.from_service_account_file(
            GOOGLE_DRIVE_CREDENTIALS,
            scopes=SCOPES,
        )
        self.service = build("drive", "v3", credentials=credentials)

        self.root_id = self._get_or_create_folder(GOOGLE_DRIVE_ROOT_FOLDER)
        self.uploads_id = self._get_or_create_folder(
            GOOGLE_DRIVE_UPLOADS_FOLDER, self.root_id
        )
        self.models_id = self._get_or_create_folder(
            GOOGLE_DRIVE_MODELS_FOLDER, self.root_id
        )

    def _get_or_create_folder(self, name: str, parent_id: Optional[str] = None) -> str:
        query = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        if parent_id:
            query += f" and '{parent_id}' in parents"

        results = (
            self.service.files()
            .list(q=query, fields="files(id, name)")
            .execute()
        )
        files = results.get("files", [])
        if files:
            return files[0]["id"]

        metadata = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if parent_id:
            metadata["parents"] = [parent_id]

        folder = (
            self.service.files()
            .create(body=metadata, fields="id")
            .execute()
        )
        return folder["id"]

    def upload_file(
        self,
        file_path: Path,
        drive_folder: str = "uploads",
        mime_type: str = "application/octet-stream",
    ) -> str:
        parent_id = self.uploads_id if drive_folder == "uploads" else self.models_id

        media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)

        file_metadata = {
            "name": file_path.name,
            "parents": [parent_id],
        }

        file = (
            self.service.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )

        return file["id"]
