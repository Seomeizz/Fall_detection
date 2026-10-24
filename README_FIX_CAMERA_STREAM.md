# Fix FallGuard camera shutdown and reconnect

## What this patch fixes

- A first Ctrl+C signals shutdown and the backend releases the webcam before waiting for open MJPEG viewers to finish.
- MJPEG requests now stop when the camera stops/errors and detect a disconnected client.
- The built-in browser page retries `/stream.mjpg` after a connection failure or backend restart.
- The Flutter CameraPage uses a new stream URL/key when backend uptime resets, the backend reconnects, or the camera returns online.
- The previous session's JPEG is cleared on camera restart to avoid displaying an old frame.

## Install into the project root

Unzip this archive into the root of the project, for example:

`F:\fall_detection\github\fall-detection-transformer`

Allow the files to overwrite their matching files, after making a backup if you have local changes:

- `backend\app\main.py`
- `backend\app\services\ai_camera.py`
- `backend\run_server.py` (new)
- `mobile_app\lib\main.dart`

## Start backend (important)

From CMD at the project root, with the project's virtual environment activated, run:

```cmd
python -u backend\run_server.py
```

Use this command instead of `python -u -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000`, because the custom runner stops the camera as soon as Ctrl+C is received.

When the log reports the server is running, open the phone app/browser as usual. After the backend is stopped and restarted, the built-in browser page and updated Flutter app should reconnect the MJPEG stream without requiring you to close and reopen the tab.

## If you use the Flutter app

After replacing `mobile_app\lib\main.dart`, rebuild and reinstall the app using your usual Flutter build process. If you only open the backend's web page at `http://<laptop-ip>:8000/`, updating the backend files is enough for the browser retry logic.
