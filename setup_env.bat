@echo off
echo Creating Python virtual environment...
python -m venv android_env

echo.
echo Activating virtual environment...
call android_env\Scripts\activate.bat

echo.
echo Installing required packages...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Setup complete! Virtual environment is activated.
echo To activate in the future, run: android_env\Scripts\activate.bat

