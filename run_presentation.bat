@echo off
echo Installing required packages...
pip install -r requirements.txt

echo Starting Presentation App...
python presentation_app.py

pause
