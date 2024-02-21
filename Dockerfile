# Use Python 3.10 as base image
FROM python:3.10

# Set label metadata
LABEL app="RAGChatWatsonx"

# Set working directory
WORKDIR /app

# Install dependencies first for caching
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy app code 
COPY . .

# Expose ports
EXPOSE 8000 

# Environment variables  
ENV PYTHONUNBUFFERED=1

# Run the app
CMD uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_app.py