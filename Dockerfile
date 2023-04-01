# Use the official Python base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

RUN pip install --upgrade pip

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY python/async_app_fan.py .

# Expose the port the application will run on (if needed)
EXPOSE 8000

# Run the command to start the application
CMD ["python", "async_app_fan.py"]
