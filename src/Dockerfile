# Official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Expose port 8000 to the outside world
EXPOSE 8000

# Command to run your Flask app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
