# Disease Risk Prediction API - 

## Project Overview

The system uses a Random Forest classifier to predict the risk of heart disease based on patient vitals (age, cholesterol, resting blood pressure, and maximum heart rate). It features an automated data-validation pipeline using Pydantic to ensure incoming requests are structured correctly before hitting the model.

## Features
* **Machine Learning:** Scikit-Learn Random Forest Classifier trained on simulated patient data.
* **REST API:** Built with FastAPI for high performance and automatic Swagger UI documentation.
* **Data Validation:** Strict input schemas enforced by Pydantic.
* **Containerization:** Fully dockerized for consistent cross-platform execution.
* **Cloud-Ready:** Designed for deployment on AWS EC2 .

## Repository Structure

```text
disease-risk-/
├── model/                  # Directory containing the compiled .joblib model
├── src/
│   ├── train.py            # Script to generate synthetic data, train, and export the model
│   └── main.py             # FastAPI application and endpoint definitions
├── requirements.txt        # Python dependencies
├── Dockerfile              # Instructions for building the Docker container
└── README.md               # Project documentation
```

## Prerequisites
* Python 3.10+
* Docker Desktop (or Docker Engine)
* Git

---

## Local Development & Testing

### 1. Environment Setup
Clone the repository and set up a Python virtual environment:
```bash
git clone (https://github.com/Philip-Lucky/Disease-Risk-Prediction)
cd disease-risk-mlops
```

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Before running the API, you must train the model and generate the `.joblib` artifact.
```bash
cd src
python train.py
```
*Expected output: `Model saved successfully to ../model/disease_risk_model.joblib`*

### 3. Start the API Server
Run the FastAPI application locally using Uvicorn. **Ensure you are in the `src` directory** when running this command so the file paths resolve correctly:
```bash
uvicorn main:app --reload
```
Navigate to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser to interact with the API via the automated Swagger UI.

---

## Docker Containerization

To eliminate the "it works on my machine" problem, run the application inside a Docker container.

### 1. Build the Image
From the root directory of the project, build the Docker image:
```bash
docker build -t disease-risk-api .
```

### 2. Run the Container
Launch the container, mapping your local port 8000 to the container's port 8000:
```bash
docker run -d -p 8000:8000 --name ml_api_container disease-risk-api
```
The API is now running in an isolated Linux environment. You can test it again at `http://127.0.0.1:8000/docs`.

To stop the container:
```bash
docker stop ml_api_container
```

---

##  AWS EC2 Deployment (Overview)

This containerized setup is ready for an AWS EC2 instance. 

1. Launch an Ubuntu EC2 instance and open Port 8000 in your Security Group.
2. SSH into the instance and install Docker.
3. Clone this repository to the instance.
4. Run the Docker build and run commands exactly as shown in the Docker section above.
5. Your API will be accessible at `http://<your-ec2-public-ip>:8000/docs`.

---
