# 方式1：直接运行
python -m src.main

# 方式2：uvicorn
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload