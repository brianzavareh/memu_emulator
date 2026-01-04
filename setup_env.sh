#!/bin/bash
echo "Creating Python virtual environment..."
python3 -m venv android_env

echo ""
echo "Activating virtual environment..."
source android_env/bin/activate

echo ""
echo "Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete! Virtual environment is activated."
echo "To activate in the future, run: source android_env/bin/activate"

