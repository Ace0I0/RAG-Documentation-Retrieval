
# 1. Select the base image
FROM python:3.12-slim-bookworm

# 2. Set the working directory
WORKDIR /app

# 3. Copy Python dependencies
COPY requirements.txt .

# 4. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the application source code
COPY app/ ./app/

# 6. Document the application port
EXPOSE 8000

# 7. Define the startup command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
